//! Layer-1 contract on the CUDA engine, against the f64 reference. Every test
//! skips when no CUDA device is present.

use emeans_core::core::contract::check;
use emeans_cuda::CudaCore;

/// The engine, or None (and a note) when this machine has no CUDA device.
fn device() -> Option<CudaCore> {
    match CudaCore::new() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping: {e:#}");
            None
        }
    }
}

#[test]
fn cuda_passes_the_contract() {
    let Some(mut dev) = device() else { return };
    let rep = check(&mut dev).unwrap();
    eprintln!("{rep:?}");
    assert!(rep.top1_rows > 0 && rep.topk_rows > 0);
}

/// The production batch shape on CUDA: the default (screened) `top1` and the
/// FP32 kernel alone.
#[test]
fn cuda_passes_the_contract_at_the_production_shape() {
    use emeans_core::core::contract::check_production;
    let Some(mut dev) = device() else { return };
    for screen in [true, false] {
        dev.set_screen(screen);
        let rep = check_production(&mut dev).unwrap();
        eprintln!("screen={screen} {rep:?}");
    }
}

/// The default CUDA `top1` screens with FP16 and checks in FP32. It must give
/// the plain FP32 kernel's labels and scores bit for bit, also on adversarial
/// near-ties: groups of 32 centroids within 1e-6 (relative) of each other,
/// exact duplicates, and rows scaled by 1e3, 1e-3 and 1e-15. Prints the share
/// of rows the screen leaves to the FP32 kernel.
#[test]
fn cuda_screen_is_bit_equal_to_fp32_on_near_ties() {
    use emeans_core::assign::Metric;
    use emeans_core::core::{Device, Mat, Select};
    let Some(mut dev) = device() else { return };
    // Deterministic noise in [-1, 1).
    let noise = |i: u64| {
        let h = i.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ 0xD1B5_4A32_D192_ED03;
        (h.wrapping_mul(0x94D0_49BB_1331_11EB) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    for (n, d, k) in [(4000, 768, 512), (3000, 72, 256)] {
        // First half: groups of 32 centroids within 1e-6 (every row there goes
        // to the FP32 fallback). Second half: pairs within 1e-6 and exact
        // duplicate pairs, which the FP32 check must decide.
        let (crowds, pairs) = (k / 2 / 32, k / 2 / 2);
        let groups = crowds + pairs;
        let bases = emeans_core::core::contract::mixture(groups, d, groups, 0.0, true, 21);
        let mut c = Vec::with_capacity(k * d);
        for j in 0..k {
            let (g, first, dup) = if j < k / 2 {
                (j / 32, j % 32 == 0, j % 32 == 31)
            } else {
                let q = (j - k / 2) / 2;
                (crowds + q, j % 2 == 0, j % 2 == 1 && q % 2 == 1)
            };
            let b = &bases[g * d..(g + 1) * d];
            if dup {
                let at = if j < k / 2 { j - 31 } else { j - 1 };
                let src = c[at * d..(at + 1) * d].to_vec();
                c.extend(src);
            } else if first {
                c.extend_from_slice(b);
            } else {
                c.extend(
                    b.iter().enumerate().map(|(t, &v)| {
                        (v as f64 * (1.0 + 1e-6 * noise((j * d + t) as u64))) as f32
                    }),
                );
            }
        }
        let rows = emeans_core::core::contract::mixture(n, d, groups, 0.05, true, 21);
        for metric in [Metric::InnerProduct, Metric::L2] {
            // Rows near the group bases (same seed: same centers), at four scales.
            let x: Vec<f32> = rows
                .chunks(d)
                .enumerate()
                .flat_map(|(i, r)| {
                    let s = [1.0f32, 1e3, 1e-3, 1e-15][i % 4];
                    r.iter().map(move |&v| v * s).collect::<Vec<_>>()
                })
                .collect();
            let xb = dev.upload_f32(&x).unwrap();
            let cb = dev.upload_f32(&c).unwrap();
            let mut cn = dev.zeros_f32(k).unwrap();
            dev.row_norms(&cb, k, d, &mut cn).unwrap();
            let perm: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 5) % n as u32).collect();
            let ids = dev.upload_u32(&perm).unwrap();
            let mut outs = Vec::new();
            for screen in [true, false] {
                dev.set_screen(screen);
                let mut lab = dev.zeros_u32(n).unwrap();
                let mut sco = dev.zeros_f32(n).unwrap();
                for _ in 0..2 {
                    dev.top1(
                        Mat {
                            host: &x,
                            dev: &xb,
                            d,
                        },
                        Select {
                            ids: Some(&ids),
                            rows: 0..n,
                        },
                        &cb,
                        Some(&cn),
                        k,
                        metric,
                        &mut lab,
                        &mut sco,
                    )
                    .unwrap();
                }
                let fallback = if screen {
                    dev.screen_fallback_rows().unwrap()
                } else {
                    0
                };
                let l = dev.read_u32(&lab, 0..n).unwrap();
                let s: Vec<u32> = dev
                    .read_f32(&sco, 0..n)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                outs.push((l, s, fallback));
            }
            eprintln!(
                "near-ties n={n} d={d} k={k} {metric:?}: {} of {n} rows on the FP32 fallback",
                outs[0].2
            );
            assert!(
                outs[0].2 < n as u32,
                "{metric:?} d={d}: the pairs must reach the FP32 check"
            );
            assert_eq!(outs[0].0, outs[1].0, "{metric:?} d={d}: labels");
            assert_eq!(outs[0].1, outs[1].1, "{metric:?} d={d}: scores");
        }
    }
}

/// The opt-in BF16 path is not exact: operands round to 8 mantissa bits.
/// It must stay close: a label may differ from the f64 reference only when
/// the two scores are within 1e-2 (relative), and scores within 1e-2.
#[test]
fn cuda_bf16_is_close_to_the_reference() {
    use emeans_core::assign::{Metric, assign_all};
    use emeans_core::{Assigner, Rows};
    let mut a = match CudaCore::new_bf16() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skipping: {e:#}");
            return;
        }
    };
    for metric in [Metric::InnerProduct, Metric::L2] {
        let (n, d, k) = (3000, 768, 300);
        let x = emeans_core::core::contract::mixture(n, d, 50, 0.5, true, 4);
        let c: Vec<f32> = x[..k * d].iter().map(|v| v * 0.9).collect();
        a.bind_data(&x, d).unwrap();
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
            1 << 16,
        )
        .unwrap();
        let mut flips = 0;
        for (p, &i) in ids.iter().enumerate() {
            let r = &x[i as usize * d..(i as usize + 1) * d];
            let (want, ws) = emeans_core::core::reference::top1(r, d, &c, metric);
            let got = l[p] as usize;
            let gs = emeans_core::core::reference::score(r, &c[got * d..(got + 1) * d], metric);
            if got != want[0] as usize {
                flips += 1;
                assert!(
                    (gs - ws[0]).abs() <= 1e-2 * ws[0].abs().max(1.0),
                    "{metric:?} row {i}: {gs} vs {}",
                    ws[0]
                );
            }
            assert!(
                (s[p] as f64 - gs).abs() <= 1e-2 * gs.abs().max(1.0),
                "{metric:?} score {} vs {gs}",
                s[p]
            );
        }
        eprintln!("bf16 {metric:?}: {flips} of {n} labels differ from f64 (all within 1e-2)");
    }
}
