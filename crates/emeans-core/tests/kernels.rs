//! The CPU kernels agree with each other (labels equal except near-ties) and
//! with an exact f64 brute force.

mod common;

use emeans_core::assign::assign_all;
use emeans_core::{Assigner, CpuAssigner, CpuKernel, Metric, Rows};

fn brute(x: &[f32], c: &[f32], d: usize, ip: bool) -> Vec<u32> {
    x.chunks(d)
        .map(|r| {
            let mut best = (0u32, if ip { f64::NEG_INFINITY } else { f64::INFINITY });
            for (j, cj) in c.chunks(d).enumerate() {
                let s = common::exact(r, cj, ip);
                if (ip && s > best.1) || (!ip && s < best.1) {
                    best = (j as u32, s);
                }
            }
            best.0
        })
        .collect()
}

#[test]
fn cpu_kernels_match_brute_force() {
    for &(n, d, k) in &[(1003, 13, 37), (517, 64, 5), (2000, 768, 70), (300, 3, 1)] {
        for ip in [false, true] {
            let x = common::mixture(n, d, 9, 0.7, ip, (n + d) as u32);
            let c: Vec<f32> = x[..k * d].iter().map(|v| v * 0.9 + 0.01).collect();
            let want = brute(&x, &c, d, ip);
            let metric = if ip { Metric::InnerProduct } else { Metric::L2 };
            let mut kernels = vec![CpuKernel::Blocked, CpuKernel::Fused];
            if cfg!(target_os = "macos") {
                kernels.push(CpuKernel::Accelerate);
            } else {
                kernels.push(CpuKernel::Gemm);
            }
            for kern in kernels {
                let mut a = CpuAssigner::new(kern);
                let ids: Vec<u32> = (0..n as u32).rev().collect();
                let (l, s) = assign_all(
                    &mut a,
                    Rows::Indexed {
                        data: &x,
                        ids: &ids,
                    },
                    d,
                    &c,
                    metric,
                    400,
                )
                .unwrap();
                let l: Vec<u32> = l.into_iter().rev().collect();
                let s: Vec<f32> = s.into_iter().rev().collect();
                assert_eq!(
                    common::non_ties(&x, &c, d, &l, &want, ip, 1e-5),
                    0,
                    "{kern:?} n={n} d={d} k={k} ip={ip}"
                );
                for (i, r) in x.chunks(d).enumerate() {
                    let e = common::exact(r, &c[l[i] as usize * d..(l[i] as usize + 1) * d], ip);
                    assert!(
                        (s[i] as f64 - e).abs() <= 1e-3 * e.abs().max(1.0),
                        "{kern:?} score {} vs {e}",
                        s[i]
                    );
                }
                let _ = a.name();
            }
        }
    }
}
