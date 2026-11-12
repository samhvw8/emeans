//! Time exact all-pairs kNN (the dedup step) on synthetic unit vectors, Metal vs CPU,
//! and check that both give the same neighbours (ties excepted).
//!
//! cargo run --release -p emeans-bench --bin knn_bench -- [n] [d] [k] [repeats]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this benchmark needs Metal (macOS)");
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    mac::main()
}

#[cfg(target_os = "macos")]
mod mac {
    use std::time::Instant;

    use emeans::synth::{NestedSpec, nested};
    use emeans::{Backend, CpuKernel, knn_self};

    pub fn main() -> anyhow::Result<()> {
        let args: Vec<usize> = std::env::args()
            .skip(1)
            .map(|a| a.parse().unwrap())
            .collect();
        let n = *args.first().unwrap_or(&16_000);
        let d = *args.get(1).unwrap_or(&768);
        let k = *args.get(2).unwrap_or(&10);
        let reps = *args.get(3).unwrap_or(&3);
        let data = nested(&NestedSpec {
            n,
            d,
            ..Default::default()
        });
        let x = data.x;
        println!(
            "n={n} d={d} k={k}, median of {reps} runs (first run includes shader compile for metal)"
        );
        let mut results = Vec::new();
        for (name, b) in [
            ("metal", Backend::MetalFused),
            ("cpu-accelerate", Backend::Cpu(CpuKernel::Accelerate)),
        ] {
            let mut times = Vec::new();
            let mut last = None;
            for _ in 0..reps {
                let t = Instant::now();
                let nb = knn_self(&x, d, k, b)?;
                times.push(t.elapsed().as_secs_f64());
                last = Some(nb);
            }
            let first = times[0];
            times.sort_by(f64::total_cmp);
            println!(
                "| {name} | {:.3} s (first {:.3} s) | {:.0} GFLOP/s |",
                times[times.len() / 2],
                first,
                2.0 * (n as f64) * (n as f64) * d as f64 / times[times.len() / 2] / 1e9
            );
            results.push(last.unwrap());
        }
        // Metal kernel alone: one compiled MetalKnn reused across runs.
        let t = Instant::now();
        let mk = emeans::metal::MetalKnn::new()?;
        let compile = t.elapsed().as_secs_f64();
        let mut times = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            mk.search(&x, &x, d, k, true)?;
            times.push(t.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        let med = times[times.len() / 2];
        println!(
            "| metal (compiled once) | {med:.3} s | {:.0} GFLOP/s | shader compile {compile:.3} s |",
            2.0 * (n as f64) * (n as f64) * d as f64 / med / 1e9
        );
        // Agreement: same id sets, or differing only where scores tie within 1e-5.
        let (a, b) = (&results[0], &results[1]);
        let mut differ = 0usize;
        let mut max_score_diff = 0f32;
        for i in 0..n {
            let (ia, sa) = a.row(i);
            let (ib, sb) = b.row(i);
            for t in 0..k {
                max_score_diff = max_score_diff.max((sa[t] - sb[t]).abs());
                if ia[t] != ib[t] && (sa[t] - sb[t]).abs() > 1e-5 {
                    differ += 1;
                }
            }
        }
        println!(
            "rank slots with a different id and score gap > 1e-5: {differ}; max score diff {max_score_diff:.2e}"
        );

        // Whole dedup step on the same rows plus injected bursts.
        let w = emeans::synth::with_bursts(&x, d, n / 20, (1, 8), 0.143, 3);
        let m = w.x.len() / d;
        let cfg = emeans::DedupConfig::default();
        for (name, b) in [
            ("cpu-accelerate", Backend::Cpu(CpuKernel::Accelerate)),
            ("metal", Backend::MetalFused),
        ] {
            let t = Instant::now();
            let r = emeans::dedup(&w.x, d, &emeans::DedupInput::default(), &cfg, b)?;
            let secs = t.elapsed().as_secs_f64();
            let ari = emeans::metrics::ari(&w.truth, &r.group_of);
            println!(
                "dedup {name}: {m} rows -> {} groups in {secs:.3} s; ARI vs injected bursts {ari:.4}; linked pairs {}; split components {}",
                r.n_groups(),
                r.linked_pairs,
                r.split_components
            );
        }
        Ok(())
    }
}
