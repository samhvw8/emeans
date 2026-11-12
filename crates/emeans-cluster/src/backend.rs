//! Which assignment backend a run uses, and how to build it.
//!
//! [`Backend`] names a backend; [`make_assigner`] builds it. Metal is part of
//! every macOS build; CUDA needs the `cuda` feature. A backend this build or
//! machine lacks is an error at [`make_assigner`], not at compile time, so
//! code that names `Backend::Cuda` builds everywhere.
//!
//! An outside crate plugs in its own backend with [`Backend::Custom`]: a name
//! and a constructor. If its assigner implements `emeans_core::core::Device`
//! and returns `Some(f(self))` from `Assigner::with_engine`, training and kNN
//! run the device loop on it; otherwise they use the host loop and the CPU
//! search.

use anyhow::Result;

use crate::{Assigner, CpuAssigner, CpuKernel};

/// A backend supplied by another crate: a name for reports and a constructor.
#[derive(Clone, Copy)]
pub struct CustomBackend {
    pub name: &'static str,
    pub make: fn() -> Result<Box<dyn Assigner>>,
}

impl std::fmt::Debug for CustomBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CustomBackend").field(&self.name).finish()
    }
}

/// Two custom backends are equal when their names are.
impl PartialEq for CustomBackend {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for CustomBackend {}

/// Which assignment backend to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu(CpuKernel),
    /// The Metal engine: dot products and argmin/argmax in one pass (macOS).
    MetalFused,
    /// The fused CUDA kernel and the device loop (feature `cuda`).
    Cuda,
    /// Opt-in: BF16 operands on tensor cores, FP32 accumulate; not exact (feature `cuda`).
    CudaBf16,
    /// A backend from another crate.
    Custom(CustomBackend),
}

impl Backend {
    /// The fastest CPU kernel for this machine.
    pub fn cpu() -> Backend {
        Backend::Cpu(CpuKernel::fastest())
    }

    /// The name [`FromStr`](std::str::FromStr) accepts for this backend
    /// (`cpu-accelerate`, `metal-fused`, `cuda`, ...).
    pub fn name(&self) -> String {
        match self {
            Backend::Cpu(k) => k.name().into(),
            Backend::MetalFused => "metal-fused".into(),
            Backend::Cuda => "cuda".into(),
            Backend::CudaBf16 => "cuda-bf16".into(),
            Backend::Custom(c) => c.name.into(),
        }
    }

    /// Names [`FromStr`](std::str::FromStr) accepts, for help texts.
    pub const NAMES: &'static [&'static str] = &[
        "cpu",
        "cpu-blocked",
        "cpu-gemm",
        "cpu-accelerate",
        "cpu-fused",
        "metal",
        "cuda",
        "cuda-bf16",
    ];
}

impl std::str::FromStr for Backend {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "cpu" => Backend::cpu(),
            "cpu-blocked" => Backend::Cpu(CpuKernel::Blocked),
            "cpu-gemm" => Backend::Cpu(CpuKernel::Gemm),
            "cpu-accelerate" => Backend::Cpu(CpuKernel::Accelerate),
            "cpu-fused" => Backend::Cpu(CpuKernel::Fused),
            "metal" | "metal-fused" => Backend::MetalFused,
            "cuda" | "cuda-fused" => Backend::Cuda,
            "cuda-bf16" => Backend::CudaBf16,
            "cpu-candle-gemm" | "metal-unfused" | "cuda-unfused" => anyhow::bail!(
                "{s:?} is a candle baseline; it lives in the emeans-bench crate (feature `baselines`)"
            ),
            _ => anyhow::bail!("unknown backend {s:?} ({})", Backend::NAMES.join(", ")),
        })
    }
}

/// Build an assigner for `backend`. Fails when this build or machine lacks it
/// (no Metal device, no `cuda` feature, no CUDA device).
pub fn make_assigner(backend: Backend) -> Result<Box<dyn Assigner>> {
    match backend {
        Backend::Cpu(k) => Ok(Box::new(CpuAssigner::new(k))),
        Backend::Custom(c) => (c.make)(),
        Backend::MetalFused => metal(),
        Backend::Cuda => cuda(false),
        Backend::CudaBf16 => cuda(true),
    }
}

#[cfg(target_os = "macos")]
fn metal() -> Result<Box<dyn Assigner>> {
    Ok(Box::new(emeans_metal::MetalCore::new()?))
}

#[cfg(not(target_os = "macos"))]
fn metal() -> Result<Box<dyn Assigner>> {
    anyhow::bail!("the Metal backend is macOS only")
}

#[cfg(feature = "cuda")]
fn cuda(bf16: bool) -> Result<Box<dyn Assigner>> {
    Ok(Box::new(if bf16 {
        emeans_cuda::CudaCore::new_bf16()?
    } else {
        emeans_cuda::CudaCore::new()?
    }))
}

#[cfg(not(feature = "cuda"))]
fn cuda(_bf16: bool) -> Result<Box<dyn Assigner>> {
    anyhow::bail!("the CUDA backend needs the `cuda` feature")
}
