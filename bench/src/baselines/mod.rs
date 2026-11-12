//! candle-based baselines (feature `baselines`).

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(target_os = "macos")]
pub mod metal;

use anyhow::Result;
use candle_core::{Device, Tensor};
use emeans::{Assigner, Backend, CustomBackend, Metric, Rows};
use emeans_core::assign::check_shapes;
use rayon::prelude::*;

/// The baseline called `name`, or None.
pub fn backend(name: &str) -> Option<Backend> {
    let make: fn() -> Result<Box<dyn Assigner>> = match name {
        "cpu-candle-gemm" => || Ok(Box::new(CandleGemm::default())),
        #[cfg(target_os = "macos")]
        "metal-unfused" => || Ok(Box::new(metal::MetalUnfused::new()?)),
        #[cfg(feature = "cuda")]
        "cuda-unfused" => || Ok(Box::new(cuda::CudaAssigner::new()?)),
        _ => return None,
    };
    let name = match name {
        "cpu-candle-gemm" => "cpu-candle-gemm",
        "metal-unfused" => "metal-unfused",
        _ => "cuda-unfused",
    };
    Some(Backend::Custom(CustomBackend { name, make }))
}

/// dots (r x k) = x (r x d) * c (k x d)^T with candle's CPU matmul.
pub fn candle_gemm_nt(x: &[f32], c: &[f32], d: usize, out: &mut [f32]) -> Result<()> {
    let (r, k) = (x.len() / d, c.len() / d);
    let xt = Tensor::from_slice(x, (r, d), &Device::Cpu)?;
    let ct = Tensor::from_slice(c, (k, d), &Device::Cpu)?;
    let v = xt.matmul(&ct.t()?)?.flatten_all()?.to_vec1::<f32>()?;
    out.copy_from_slice(&v);
    Ok(())
}

/// The CPU candle-GEMM assigner (`cpu-candle-gemm`): candle's matmul per row
/// block, then a rayon argmax. Before the library split this was
/// `CpuKernel::CandleGemm`; `CpuKernel::Gemm` makes the same `gemm` call
/// without candle.
#[derive(Default)]
pub struct CandleGemm {
    gather: Vec<f32>,
    dots: Vec<f32>,
}

impl Assigner for CandleGemm {
    fn name(&self) -> String {
        "cpu-candle-gemm".into()
    }

    fn assign(
        &mut self,
        rows: Rows<'_>,
        d: usize,
        centroids: &[f32],
        metric: Metric,
        labels: &mut [u32],
        scores: &mut [f32],
    ) -> Result<()> {
        let k = check_shapes(&rows, d, centroids, labels, scores)?;
        let n = rows.len(d);
        if n == 0 {
            return Ok(());
        }
        let cnorm: Vec<f32> = match metric {
            Metric::L2 => centroids
                .chunks(d)
                .map(|c| c.iter().map(|v| v * v).sum())
                .collect(),
            Metric::InnerProduct => Vec::new(),
        };
        let block = ((16usize << 20) / k).clamp(256, 16384);
        for b0 in (0..n).step_by(block) {
            let b1 = (b0 + block).min(n);
            let sub = rows.slice(b0, b1, d);
            self.gather.clear();
            for i in 0..b1 - b0 {
                self.gather.extend_from_slice(sub.row(i, d));
            }
            self.dots.resize((b1 - b0) * k, 0.0);
            candle_gemm_nt(&self.gather, centroids, d, &mut self.dots)?;
            let dots = &self.dots;
            labels[b0..b1]
                .par_iter_mut()
                .zip(scores[b0..b1].par_iter_mut())
                .enumerate()
                .for_each(|(i, (l, s))| {
                    let row = &dots[i * k..(i + 1) * k];
                    let (mut bj, mut bv) = (0usize, f32::NAN);
                    for (j, &v) in row.iter().enumerate() {
                        let v = match metric {
                            Metric::InnerProduct => v,
                            Metric::L2 => cnorm[j] - 2.0 * v,
                        };
                        let better = match metric {
                            Metric::InnerProduct => j == 0 || v > bv,
                            Metric::L2 => j == 0 || v < bv,
                        };
                        if better {
                            (bj, bv) = (j, v);
                        }
                    }
                    *l = bj as u32;
                    *s = match metric {
                        Metric::InnerProduct => bv,
                        Metric::L2 => {
                            let xn: f32 = sub.row(i, d).iter().map(|a| a * a).sum();
                            (bv + xn).max(0.0)
                        }
                    };
                });
        }
        Ok(())
    }
}
