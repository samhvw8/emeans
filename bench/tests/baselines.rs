//! The candle baselines give the CPU labels (ties excepted). Feature `baselines`.
#![cfg(feature = "baselines")]

use emeans::assign::assign_all;
use emeans::{CpuAssigner, CpuKernel, Metric, Rows, make_assigner};

#[test]
fn baselines_match_the_cpu_labels() {
    let (n, d, k) = (1500, 64, 40);
    let x = emeans::core::contract::mixture(n, d, 9, 0.6, true, 5);
    let c: Vec<f32> = x[..k * d].iter().map(|v| v * 0.95).collect();
    for metric in [Metric::InnerProduct, Metric::L2] {
        let mut cpu = CpuAssigner::new(CpuKernel::Blocked);
        let (want, wscore) = assign_all(&mut cpu, Rows::Dense(&x), d, &c, metric, 1 << 16).unwrap();
        for name in ["cpu-candle-gemm", "metal-unfused", "cuda-unfused"] {
            let Ok(backend) = emeans_bench::backend(name) else {
                eprintln!("skipping {name}: not in this build");
                continue;
            };
            let mut a = match make_assigner(backend) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("skipping {name}: {e:#}");
                    continue;
                }
            };
            assert_eq!(a.name(), name);
            let ids: Vec<u32> = (0..n as u32).rev().collect();
            a.bind_data(&x, d).unwrap();
            let rows = Rows::Indexed {
                data: &x,
                ids: &ids,
            };
            let (l, s) = assign_all(a.as_mut(), rows, d, &c, metric, 1 << 16).unwrap();
            let mut flips = 0;
            for (p, &i) in ids.iter().enumerate() {
                let i = i as usize;
                if l[p] != want[i] {
                    // Only a near-tie may flip.
                    assert!(
                        (s[p] - wscore[i]).abs() <= 1e-4 * wscore[i].abs().max(1.0),
                        "{name} {metric:?} row {i}: {} vs {}",
                        s[p],
                        wscore[i]
                    );
                    flips += 1;
                }
            }
            assert!(flips <= n / 100, "{name} {metric:?}: {flips} labels differ");
        }
    }
}
