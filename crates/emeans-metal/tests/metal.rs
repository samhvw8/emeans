//! The Metal engine gives the CPU labels (ties excepted). The candle unfused
//! baseline is checked in `emeans-bench` (feature `baselines`).
#![cfg(target_os = "macos")]

mod common;

use emeans_core::assign::assign_all;
use emeans_core::{Assigner, Config, CpuAssigner, CpuKernel, Emeans, Metric, Rows};
use emeans_metal::{FUSED_TILES, MetalCore};

fn metal_or_skip(tile: usize) -> Option<MetalCore> {
    match MetalCore::with_tile(tile) {
        Ok(a) => Some(a),
        Err(e) => {
            eprintln!("skipping: no Metal device ({e})");
            None
        }
    }
}

#[test]
fn metal_labels_equal_cpu_labels() {
    for &(n, d, k) in &[
        (1000, 768, 64),
        (777, 768, 100),
        (4099, 16, 37),
        (513, 13, 70),
        (90, 768, 1),
        (2000, 20, 300),
    ] {
        for ip in [true, false] {
            let x = common::mixture(n, d, 11, 0.6, ip, (3 * n + k) as u32);
            let c: Vec<f32> = x[..k * d].iter().map(|v| v * 0.95).collect();
            let metric = if ip { Metric::InnerProduct } else { Metric::L2 };
            let mut cpu = CpuAssigner::new(CpuKernel::Blocked);
            let (want, wscore) =
                assign_all(&mut cpu, Rows::Dense(&x), d, &c, metric, 1 << 16).unwrap();
            for tile in 0..FUSED_TILES.len() {
                let Some(mut m) = metal_or_skip(tile) else {
                    return;
                };
                let ids: Vec<u32> = (0..n as u32).map(|i| (i * 7919) % n as u32).collect();
                for bound in [false, true] {
                    if bound {
                        m.bind_data(&x, d).unwrap();
                    }
                    let (l, s) = assign_all(
                        &mut m,
                        Rows::Indexed {
                            data: &x,
                            ids: &ids,
                        },
                        d,
                        &c,
                        metric,
                        700,
                    )
                    .unwrap();
                    let mut lw = vec![0u32; n];
                    let mut sw = vec![0f32; n];
                    for (p, &i) in ids.iter().enumerate() {
                        lw[i as usize] = l[p];
                        sw[i as usize] = s[p];
                    }
                    let bad = common::non_ties(&x, &c, d, &lw, &want, ip, 1e-5);
                    assert_eq!(
                        bad,
                        0,
                        "{} n={n} d={d} k={k} ip={ip} bound={bound}",
                        m.name()
                    );
                    for i in 0..n {
                        if lw[i] == want[i] {
                            // L2 scores come from ||x||^2 + ||c||^2 - 2 x.c: the error scales with the norms.
                            let r = &x[i * d..(i + 1) * d];
                            let cj = &c[lw[i] as usize * d..(lw[i] as usize + 1) * d];
                            let mag = if ip {
                                1.0
                            } else {
                                r.iter().map(|v| v * v).sum::<f32>() as f64
                                    + cj.iter().map(|v| v * v).sum::<f32>() as f64
                            };
                            let tol = 1e-5 * mag.max(1.0) + 1e-4 * (wscore[i].abs() as f64);
                            assert!(
                                ((sw[i] - wscore[i]).abs() as f64) <= tol,
                                "{} score {} vs {}",
                                m.name(),
                                sw[i],
                                wscore[i]
                            );
                        }
                    }
                    m.unbind_data();
                }
                let (l, _) = assign_all(&mut m, Rows::Dense(&x), d, &c, metric, 1 << 16).unwrap();
                assert_eq!(
                    common::non_ties(&x, &c, d, &l, &want, ip, 1e-5),
                    0,
                    "{} dense",
                    m.name()
                );
            }
        }
    }
}

#[test]
fn metal_training_matches_cpu_training() {
    let (n, d, k) = (6000, 64, 32);
    let x = common::mixture(n, d, 24, 0.5, true, 77);
    let mut cfg = Config::new(d, k);
    cfg.spherical = true;
    let mut cpu =
        Emeans::with_assigner(cfg.clone(), Box::new(CpuAssigner::new(CpuKernel::Blocked))).unwrap();
    cpu.train(&x).unwrap();
    let Some(f) = metal_or_skip(0) else {
        return;
    };
    let mut gpu = Emeans::with_assigner(cfg, Box::new(f)).unwrap();
    gpu.train(&x).unwrap();
    let (a, b) = (
        cpu.assign(&x).unwrap().mean_objective(true),
        gpu.assign(&x).unwrap().mean_objective(true),
    );
    assert!((a - b).abs() <= 0.01 * a, "cpu {a} vs metal {b}");
}

/// `bind_shared` reads the data in place (no copy), so the engine must own it:
/// the caller hands over an `Arc` and cannot free or change the data while it is
/// bound. The engine keeps it alive after the caller drops its handle, gives the
/// CPU labels from it, and lets it go on `unbind_data`. (`bind_data` takes a
/// borrow and copies, so the borrow ending early is harmless.)
#[test]
fn bind_shared_owns_the_data_it_reads_in_place() {
    let (n, d, k) = (3000, 768, 300);
    let x = common::mixture(n, d, 11, 0.6, true, 41);
    let c: Vec<f32> = x[..k * d].iter().map(|v| v * 0.95).collect();
    let mut cpu = CpuAssigner::new(CpuKernel::Blocked);
    let (want, _) = assign_all(
        &mut cpu,
        Rows::Dense(&x),
        d,
        &c,
        Metric::InnerProduct,
        1 << 16,
    )
    .unwrap();
    let Some(mut m) = metal_or_skip(0) else {
        return;
    };
    let data = std::sync::Arc::new(x);
    let weak = std::sync::Arc::downgrade(&data);
    m.bind_shared(data.clone(), d).unwrap();
    let ids: Vec<u32> = (0..n as u32).map(|i| (i * 7919) % n as u32).collect();
    let rows = Rows::Indexed {
        data: &data,
        ids: &ids,
    };
    let (l, _) = assign_all(&mut m, rows, d, &c, Metric::InnerProduct, 1 << 16).unwrap();
    let x2 = data.clone();
    drop(data);
    let mut lw = vec![0u32; n];
    for (p, &i) in ids.iter().enumerate() {
        lw[i as usize] = l[p];
    }
    assert_eq!(
        common::non_ties(&x2, &c, d, &lw, &want, true, 1e-5),
        0,
        "labels from the bound data"
    );
    let (ptr, len) = (x2.as_ptr(), x2.len());
    drop(x2);
    assert!(
        weak.upgrade().is_some(),
        "the engine must keep the data alive while it is bound"
    );
    // GPU work after the caller's last handle is gone: the engine reads the
    // data it owns. SAFETY: the assertion above shows the allocation is alive
    // (the engine holds the Arc until `unbind_data` below), and nothing writes it.
    let kept = unsafe { std::slice::from_raw_parts(ptr, len) };
    let rows = Rows::Indexed {
        data: kept,
        ids: &ids,
    };
    let (l2, _) = assign_all(&mut m, rows, d, &c, Metric::InnerProduct, 1 << 16).unwrap();
    assert_eq!(l2, l, "labels after the caller dropped its handles");
    m.unbind_data();
    assert!(
        weak.upgrade().is_none(),
        "unbind_data must release the data"
    );
}

/// `bind_data` copies, and its cache key is (pointer, length). A caller that
/// frees the data and gets a new buffer of the same size at the same address
/// must not be served the old copy. Writing new contents into the same `Vec`
/// gives that state (same pointer, same length, other data) every time.
#[test]
fn rebinding_changed_data_reads_the_new_data() {
    let (n, d, k) = (1500, 64, 40);
    let x = common::mixture(n, d, 9, 0.6, true, 5);
    let y = common::mixture(n, d, 9, 0.6, true, 6);
    let c: Vec<f32> = x[..k * d].iter().map(|v| v * 0.95).collect();
    let mut cpu = CpuAssigner::new(CpuKernel::Blocked);
    let want = |cpu: &mut CpuAssigner, v: &[f32]| {
        assign_all(cpu, Rows::Dense(v), d, &c, Metric::InnerProduct, 1 << 16)
            .unwrap()
            .0
    };
    let (want_x, want_y) = (want(&mut cpu, &x), want(&mut cpu, &y));
    assert!(
        common::non_ties(&y, &c, d, &want_x, &want_y, true, 1e-5) > n / 4,
        "precondition: the two data sets have different labels"
    );
    let ids: Vec<u32> = (0..n as u32).collect();
    {
        let Some(mut m) = metal_or_skip(0) else {
            return;
        };
        let mut buf = x.clone();
        m.bind_data(&buf, d).unwrap();
        let indexed = |m: &mut MetalCore, buf: &[f32]| {
            let rows = Rows::Indexed {
                data: buf,
                ids: &ids,
            };
            assign_all(m, rows, d, &c, Metric::InnerProduct, 1 << 16)
                .unwrap()
                .0
        };
        let l = indexed(&mut m, &buf);
        assert_eq!(common::non_ties(&x, &c, d, &l, &want_x, true, 1e-5), 0);
        // New data at the same address, bound again.
        buf.copy_from_slice(&y);
        m.bind_data(&buf, d).unwrap();
        let l = indexed(&mut m, &buf);
        assert_eq!(
            common::non_ties(&y, &c, d, &l, &want_y, true, 1e-5),
            0,
            "{}: bind_data served the copy of the old data",
            m.name()
        );
        // New data at the same address, read as dense rows without a new bind.
        buf.copy_from_slice(&x);
        let (l, _) = assign_all(
            &mut m,
            Rows::Dense(&buf),
            d,
            &c,
            Metric::InnerProduct,
            1 << 16,
        )
        .unwrap();
        assert_eq!(
            common::non_ties(&x, &c, d, &l, &want_x, true, 1e-5),
            0,
            "{}: dense rows were read from the copy of the old data",
            m.name()
        );
    }
}

/// A buffer from `Device::bind` on shared data is a no-copy view. It must keep
/// the data alive by itself: after `unbind_data` and the caller's drop, and
/// while work that reads it is still queued.
#[test]
fn a_view_of_shared_data_keeps_it_alive() {
    use emeans_core::core::Device;
    let (n, d) = (3000, 768);
    let x = common::mixture(n, d, 11, 0.6, true, 43);
    let want: Vec<f32> = x.chunks(d).map(|r| r.iter().map(|v| v * v).sum()).collect();
    let Some(mut m) = metal_or_skip(0) else {
        return;
    };
    let data = std::sync::Arc::new(x);
    let weak = std::sync::Arc::downgrade(&data);
    m.bind_shared(data.clone(), d).unwrap();
    let mut data = Some(data);
    let copied = std::env::var_os("EMEANS_METAL_COPY").is_some();
    let core = &mut m;
    (|| -> anyhow::Result<()> {
        let data = data.take().unwrap();
        let view = core.bind(&data)?;
        Assigner::unbind_data(core);
        drop(data);
        assert!(
            copied || weak.upgrade().is_some(),
            "the view outlived the data it reads"
        );
        let mut out = core.zeros_f32(n)?;
        core.row_norms(&view, n, d, &mut out)?;
        drop(view);
        assert!(
            copied || weak.upgrade().is_some(),
            "queued work reads data that was freed"
        );
        let got = core.read_f32(&out, 0..n)?;
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-4 * w.max(1.0), "norm {g} vs {w}");
        }
        Ok(())
    })()
    .unwrap();
    assert!(
        weak.upgrade().is_none(),
        "the data must be released once no view and no queued work uses it"
    );
}
