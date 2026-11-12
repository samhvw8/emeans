//! Metal baseline (macOS): candle `index_select` + `matmul` + `argmax`/`argmin`.
//! It has no engine, so training with it uses the host loop (one sync per batch).

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use emeans_core::assign::{Assigner, Metric, Rows, check_shapes};

struct Bound {
    ptr: usize,
    len: usize,
    d: usize,
    tensor: Tensor,
}

/// The unfused candle Metal assigner (`metal-unfused`).
pub struct MetalUnfused {
    device: Device,
    bound: Option<Bound>,
    /// Max floats of the `m x k` dot matrix per call.
    budget: usize,
}

impl MetalUnfused {
    pub fn new() -> Result<Self> {
        Ok(Self {
            device: Device::new_metal(0).context("no Metal device")?,
            bound: None,
            budget: 16 << 20,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn unfused(
        device: &Device,
        bound: &Option<Bound>,
        budget: usize,
        rows: Rows<'_>,
        d: usize,
        centroids: &[f32],
        metric: Metric,
        labels: &mut [u32],
        scores: &mut [f32],
    ) -> Result<()> {
        let n = rows.len(d);
        let k = centroids.len() / d;
        let ct = Tensor::from_slice(centroids, (k, d), device)?;
        let cn = match metric {
            Metric::L2 => Some(ct.sqr()?.sum_keepdim(1)?.t()?),
            Metric::InnerProduct => None,
        };
        let block = (budget / k).max(1);
        for b in (0..n).step_by(block) {
            let e = (b + block).min(n);
            let part = rows.slice(b, e, d);
            let xb = match part {
                Rows::Indexed { data, ids }
                    if bound.as_ref().is_some_and(|t| {
                        t.ptr == data.as_ptr() as usize && t.len == data.len() && t.d == d
                    }) =>
                {
                    let idx = Tensor::from_slice(ids, e - b, device)?;
                    bound.as_ref().unwrap().tensor.index_select(&idx, 0)?
                }
                _ => {
                    let mut g = Vec::with_capacity((e - b) * d);
                    for i in 0..e - b {
                        g.extend_from_slice(part.row(i, d));
                    }
                    Tensor::from_slice(&g, (e - b, d), device)?
                }
            };
            let dots = xb.matmul(&ct.t()?)?;
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

impl Assigner for MetalUnfused {
    fn name(&self) -> String {
        "metal-unfused".into()
    }

    fn bind_data(&mut self, data: &[f32], d: usize) -> Result<()> {
        // Always a fresh copy: the same pointer and length may hold other data
        // now (the caller may have freed and reallocated).
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
        check_shapes(&rows, d, centroids, labels, scores)?;
        if rows.is_empty(d) {
            return Ok(());
        }
        Self::unfused(
            &self.device,
            &self.bound,
            self.budget,
            rows,
            d,
            centroids,
            metric,
            labels,
            scores,
        )
    }
}
