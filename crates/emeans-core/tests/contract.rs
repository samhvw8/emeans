//! Layer-1 contract: every CPU kernel against the f64 reference. The GPU
//! backends run the same check in their own crates.

use emeans_core::core::contract::check;
use emeans_core::{CpuAssigner, CpuKernel};

#[test]
fn cpu_backends_pass_the_contract() {
    let mut kernels = vec![CpuKernel::Blocked, CpuKernel::Fused];
    if cfg!(target_os = "macos") {
        kernels.push(CpuKernel::Accelerate);
    } else {
        kernels.push(CpuKernel::Gemm);
    }
    for kernel in kernels {
        let rep = check(&mut CpuAssigner::new(kernel)).unwrap();
        eprintln!("{rep:?}");
        assert!(rep.top1_rows > 0 && rep.topk_rows > 0);
    }
}

/// The production batch shape (16,384 x 768, k 1024 and 1000, exact duplicate
/// centroids and rows) on the default CPU kernels: `cpu-fused` (x86 default)
/// and Accelerate (macOS default). Release builds only.
#[test]
fn cpu_passes_the_contract_at_the_production_shape() {
    use emeans_core::core::contract::check_production;
    // About 4 minutes unoptimized (the f64 reference at 16,384 x 1024 x 768);
    // `cargo test --release` runs it.
    if cfg!(debug_assertions) {
        eprintln!("skipping in a debug build; run with --release");
        return;
    }
    let mut kernels = vec![CpuKernel::Fused];
    if cfg!(target_os = "macos") {
        kernels.push(CpuKernel::Accelerate);
    }
    for kernel in kernels {
        let rep = check_production(&mut CpuAssigner::new(kernel)).unwrap();
        eprintln!("{rep:?}");
        assert!(rep.top1_rows > 0 && rep.topk_rows > 0);
    }
}
