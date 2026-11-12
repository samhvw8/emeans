//! Shared parts of the emeans benchmarks: backend names, including the candle
//! baselines behind the `baselines` feature.
//!
//! | Name | Backend | Needs |
//! |---|---|---|
//! | any name `emeans::Backend` parses | that backend | — |
//! | `cpu-candle-gemm` | candle CPU matmul per row block, then a rayon argmax | `baselines` |
//! | `metal-unfused` | candle Metal `matmul` + `argmax` | `baselines`, macOS |
//! | `cuda-unfused` | candle CUDA `matmul` + `argmax` | `baselines`, `cuda` |
//!
//! The baselines are [`emeans::Backend::Custom`] backends with no engine, so
//! training with them runs the host loop.

#[cfg(feature = "baselines")]
pub mod baselines;

use anyhow::Result;
use emeans::Backend;

/// Parse a backend name, the candle baselines included.
pub fn backend(name: &str) -> Result<Backend> {
    #[cfg(feature = "baselines")]
    if let Some(b) = baselines::backend(name) {
        return Ok(b);
    }
    if matches!(name, "cpu-candle-gemm" | "metal-unfused" | "cuda-unfused") {
        anyhow::bail!("{name:?} is a candle baseline: build with --features baselines");
    }
    name.parse()
}
