//! Layer 2 on every backend this build has (CPU kernels, Metal on macOS, CUDA
//! with the `cuda` feature and a device): same seed, same final objective.

use emeans::{Assigner, Backend, Config, CpuAssigner, CpuKernel, Emeans, make_assigner};

/// Layer 2 on each backend with the same seed: the final objective of each run.
fn train_all_backends(spherical: bool, revival: bool) -> Vec<(String, f64, usize)> {
    let (n, d, k) = (20_000, 64, 64);
    let x = emeans::core::contract::mixture(n, d, 64, 0.15, spherical, 99);
    let mut cfg = Config::new(d, k);
    cfg.spherical = spherical;
    cfg.seed = 7;
    if !revival {
        cfg.unused_threshold = Some(0.0);
    }
    let mut backends: Vec<Box<dyn Assigner>> = vec![
        Box::new(CpuAssigner::new(CpuKernel::Blocked)),
        Box::new(CpuAssigner::new(CpuKernel::Fused)),
    ];
    if !cfg!(target_os = "macos") {
        backends.push(Box::new(CpuAssigner::new(CpuKernel::Gemm)));
    }
    // GPU backends this build has; skipped when the machine lacks the device.
    for b in [Backend::MetalFused, Backend::Cuda] {
        match make_assigner(b) {
            Ok(a) => backends.push(a),
            Err(e) => eprintln!("skipping {b:?}: {e:#}"),
        }
    }
    let mut out = Vec::new();
    for a in backends {
        let name = a.name();
        let mut m = Emeans::with_assigner(cfg.clone(), a).unwrap();
        m.train(&x).unwrap();
        let obj = m.assign(&x).unwrap().mean_objective(spherical);
        eprintln!(
            "spherical={spherical} revival={revival} {name}: final {obj} epochs {:?} splits {}",
            m.obj(),
            m.n_splits()
        );
        out.push((name, obj, m.n_splits()));
    }
    out
}

/// Same seed, same final objective on every backend (1e-4 relative).
///
/// Without revival the loop has no chaotic step, so the backends agree to
/// float noise. With revival a split makes twin centroids 1e-4 apart, their
/// rows become near-ties, and one flipped label can move the run to another
/// optimum (two CPU kernels do it too: 7.53 vs 9.19 on the Euclidean set).
/// The spherical set is checked with revival on; it stays on one optimum.
#[test]
fn backends_train_to_the_same_objective() {
    for (spherical, revival) in [(true, true), (true, false), (false, false)] {
        let runs = train_all_backends(spherical, revival);
        let base = runs[0].1;
        for (name, obj, _) in &runs {
            assert!(
                (obj - base).abs() <= 1e-4 * base.abs(),
                "spherical={spherical} revival={revival} {name}: objective {obj} vs cpu {base}"
            );
        }
    }
}
