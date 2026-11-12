//! CUDA baseline (features `baselines` + `cuda`): candle CUDA matmul, then a
//! candle argmax/argmin.
//!
//! The same unfused path as the Metal baseline, on candle's CUDA device 0:
//! the data set stays on the GPU after `bind_data`, each batch sends row ids and
//! centroids, and the `m x k` dot matrix is built in blocks of at most
//! `budget` floats. The fused engine is `emeans_cuda::CudaCore`.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

use emeans_core::assign::{Assigner, Metric, Rows, check_shapes};

struct Bound {
    ptr: usize,
    len: usize,
    d: usize,
    tensor: Tensor,
}

/// CUDA assigner (candle device 0), matmul + argmax.
pub struct CudaAssigner {
    device: Device,
    bound: Option<Bound>,
    /// Max floats of the `m x k` dot matrix per call.
    budget: usize,
}

impl CudaAssigner {
    pub fn new() -> Result<Self> {
        let device = Device::new_cuda(0).context("no CUDA device")?;
        Ok(Self {
            device,
            bound: None,
            budget: 64 << 20,
        })
    }

    fn data_tensor(&self, data: &[f32], d: usize) -> Option<&Tensor> {
        self.bound
            .as_ref()
            .filter(|b| b.ptr == data.as_ptr() as usize && b.len == data.len() && b.d == d)
            .map(|b| &b.tensor)
    }

    fn rows_on_device(&self, rows: Rows<'_>, d: usize) -> Result<Tensor> {
        let n = rows.len(d);
        Ok(match rows {
            Rows::Dense(x) => Tensor::from_slice(x, (n, d), &self.device)?,
            Rows::Indexed { data, ids } => match self.data_tensor(data, d) {
                Some(t) => t.index_select(&Tensor::from_slice(ids, n, &self.device)?, 0)?,
                None => {
                    let mut g = vec![0f32; n * d];
                    for (i, dst) in g.chunks_mut(d).enumerate() {
                        dst.copy_from_slice(rows.row(i, d));
                    }
                    Tensor::from_slice(&g, (n, d), &self.device)?
                }
            },
        })
    }
}

impl Assigner for CudaAssigner {
    fn name(&self) -> String {
        "cuda-unfused".into()
    }

    fn bind_data(&mut self, data: &[f32], d: usize) -> Result<()> {
        if self.data_tensor(data, d).is_some() {
            return Ok(());
        }
        self.bound = None;
        let tensor = Tensor::from_slice(data, (data.len() / d, d), &self.device)?;
        self.bound = Some(Bound {
            ptr: data.as_ptr() as usize,
            len: data.len(),
            d,
            tensor,
        });
        Ok(())
    }

    fn unbind_data(&mut self) {
        self.bound = None;
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
        let ct = Tensor::from_slice(centroids, (k, d), &self.device)?;
        let ctt = ct.t()?;
        let cn = match metric {
            Metric::L2 => Some(ct.sqr()?.sum_keepdim(1)?.t()?),
            Metric::InnerProduct => None,
        };
        let block = (self.budget / k).max(1);
        for b in (0..n).step_by(block) {
            let e = (b + block).min(n);
            let xb = self.rows_on_device(rows.slice(b, e, d), d)?;
            let dots = xb.matmul(&ctt)?;
            let (lab, sc) = match &cn {
                None => (dots.argmax_keepdim(1)?, dots.max_keepdim(1)?),
                Some(cn) => {
                    let s = dots.affine(-2.0, 0.0)?.broadcast_add(cn)?;
                    let best = s.min_keepdim(1)?;
                    let xn = xb.sqr()?.sum_keepdim(1)?;
                    (s.argmin_keepdim(1)?, (best + xn)?.relu()?)
                }
            };
            labels[b..e].copy_from_slice(&lab.flatten_all()?.to_vec1::<u32>()?);
            scores[b..e].copy_from_slice(&sc.flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(())
    }
}
