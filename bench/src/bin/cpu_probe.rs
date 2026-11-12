#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]
//! macOS probe: where the Accelerate top-1 time goes (gather, GEMM, argmax), and tiled variants.
//!
//! cargo run --release -p emeans-bench --bin cpu_probe -- --k 1024 --batch 16384 --reps 7

use std::time::Instant;

use rayon::prelude::*;

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| default.to_string())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time(name: &str, reps: usize, mut f: impl FnMut()) {
    f();
    let mut t = Vec::new();
    for _ in 0..reps {
        let s = Instant::now();
        f();
        t.push(s.elapsed().as_secs_f64() * 1e3);
    }
    println!("{name:<40} {:8.3} ms", median(t));
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "macos")]
fn sgemm(x: &[f32], c: &[f32], r: usize, k: usize, d: usize, out: &mut [f32]) {
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            101,
            111,
            112,
            r as i32,
            k as i32,
            d as i32,
            1.0,
            x.as_ptr(),
            d as i32,
            c.as_ptr(),
            d as i32,
            0.0,
            out.as_mut_ptr(),
            k as i32,
        )
    }
}

fn argmax(dots: &[f32], k: usize, lab: &mut [u32], sc: &mut [f32]) {
    for (i, (l, s)) in lab.iter_mut().zip(sc.iter_mut()).enumerate() {
        let row = &dots[i * k..(i + 1) * k];
        let mut b = f32::NEG_INFINITY;
        let mut bj = 0;
        for (j, &v) in row.iter().enumerate() {
            if v > b {
                b = v;
                bj = j;
            }
        }
        *l = bj as u32;
        *s = b;
    }
}

/// Branch-free argmax over 8 lanes (lane j holds columns j mod 8); ties keep the lowest column.
fn argmax_v(dots: &[f32], k: usize, lab: &mut [u32], sc: &mut [f32]) {
    const L: usize = 8;
    for (i, (l, s)) in lab.iter_mut().zip(sc.iter_mut()).enumerate() {
        let row = &dots[i * k..(i + 1) * k];
        let mut best = [f32::NEG_INFINITY; L];
        let mut bi = [0u32; L];
        let full = k / L * L;
        let mut j = 0;
        while j < full {
            let v: &[f32; L] = row[j..j + L].try_into().unwrap();
            for t in 0..L {
                let m = v[t] > best[t];
                best[t] = if m { v[t] } else { best[t] };
                bi[t] = if m { (j + t) as u32 } else { bi[t] };
            }
            j += L;
        }
        let (mut b, mut bj) = (f32::NEG_INFINITY, 0u32);
        for t in 0..L {
            if best[t] > b || (best[t] == b && bi[t] < bj) {
                b = best[t];
                bj = bi[t];
            }
        }
        for (jj, &v) in row.iter().enumerate().skip(full) {
            if v > b {
                b = v;
                bj = jj as u32;
            }
        }
        *l = bj;
        *s = b;
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("cpu_probe needs macOS (Accelerate)");
}

#[cfg(target_os = "macos")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = arg(&args, "--n", "200000").parse().unwrap();
    let d: usize = arg(&args, "--d", "768").parse().unwrap();
    let k: usize = arg(&args, "--k", "1024").parse().unwrap();
    let b: usize = arg(&args, "--batch", "16384").parse().unwrap();
    let reps: usize = arg(&args, "--reps", "7").parse().unwrap();
    let x = emeans::core::contract::mixture(n, d, 500, 0.5, true, 5);
    let c = x[..k * d].to_vec();
    let ids: Vec<u32> = (0..b as u32).map(|i| (i * 7919 + 1) % n as u32).collect();
    let mut g = vec![0f32; b * d];
    let mut dots = vec![0f32; b * k];
    let mut lab = vec![0u32; b];
    let mut sc = vec![0f32; b];
    println!(
        "n={n} d={d} k={k} batch={b} threads={}",
        rayon::current_num_threads()
    );
    time("gather (rayon)", reps, || {
        g.par_chunks_mut(d).zip(&ids).for_each(|(dst, &i)| {
            let i = i as usize;
            dst.copy_from_slice(&x[i * d..(i + 1) * d])
        })
    });
    time("sgemm whole batch", reps, || {
        sgemm(&g, &c, b, k, d, &mut dots)
    });
    time("argmax whole batch (rayon)", reps, || {
        dots.par_chunks(k * 64)
            .zip(lab.par_chunks_mut(64).zip(sc.par_chunks_mut(64)))
            .for_each(|(dd, (l, s))| argmax(dd, k, l, s))
    });
    time("argmax_v whole batch (rayon)", reps, || {
        dots.par_chunks(k * 64)
            .zip(lab.par_chunks_mut(64).zip(sc.par_chunks_mut(64)))
            .for_each(|(dd, (l, s))| argmax_v(dd, k, l, s))
    });
    let mut lab2 = vec![0u32; b];
    let mut sc2 = vec![0f32; b];
    dots.par_chunks(k * 64)
        .zip(lab2.par_chunks_mut(64).zip(sc2.par_chunks_mut(64)))
        .for_each(|(dd, (l, s))| argmax(dd, k, l, s));
    dots.par_chunks(k * 64)
        .zip(lab.par_chunks_mut(64).zip(sc.par_chunks_mut(64)))
        .for_each(|(dd, (l, s))| argmax_v(dd, k, l, s));
    assert!(lab == lab2 && sc == sc2, "argmax_v differs");
    #[cfg(target_os = "macos")]
    {
        let mut ct = vec![0f32; d * k];
        for j in 0..k {
            for t in 0..d {
                ct[t * k + j] = c[j * d + t];
            }
        }
        time("sgemm whole batch, B as d x k (NoTrans)", reps, || unsafe {
            cblas_sgemm(
                101,
                111,
                111,
                b as i32,
                k as i32,
                d as i32,
                1.0,
                g.as_ptr(),
                d as i32,
                ct.as_ptr(),
                k as i32,
                0.0,
                dots.as_mut_ptr(),
                k as i32,
            )
        });
        time("sgemm whole batch (again)", reps, || {
            sgemm(&g, &c, b, k, d, &mut dots)
        });
    }
    for nb in [2usize, 4, 8] {
        let blk = b.div_ceil(nb);
        time(
            &format!("pipelined {nb} blocks (gather/argmax || sgemm)"),
            reps,
            || {
                // stage i: sgemm block i on a helper thread while rayon gathers i+1 and argmaxes i-1.
                let mut gbufs: Vec<Vec<f32>> = (0..2).map(|_| vec![0f32; blk * d]).collect();
                let mut dbufs: Vec<Vec<f32>> = (0..2).map(|_| vec![0f32; blk * k]).collect();
                let gat = |dst: &mut [f32], bi: usize| {
                    let id = &ids[bi * blk..((bi + 1) * blk).min(b)];
                    dst[..id.len() * d]
                        .par_chunks_mut(d)
                        .zip(id)
                        .for_each(|(o, &i)| {
                            let i = i as usize;
                            o.copy_from_slice(&x[i * d..(i + 1) * d])
                        });
                };
                gat(&mut gbufs[0], 0);
                for bi in 0..=nb {
                    let (g0, g1) = gbufs.split_at_mut(1);
                    let (d0, d1) = dbufs.split_at_mut(1);
                    let (gcur, gnext) = if bi % 2 == 0 {
                        (&mut g0[0], &mut g1[0])
                    } else {
                        (&mut g1[0], &mut g0[0])
                    };
                    let (dcur, dprev) = if bi % 2 == 0 {
                        (&mut d0[0], &mut d1[0])
                    } else {
                        (&mut d1[0], &mut d0[0])
                    };
                    let r = if bi < nb { (b - bi * blk).min(blk) } else { 0 };
                    let cc = &c;
                    std::thread::scope(|sc_| {
                        if r > 0 {
                            let gcur: &Vec<f32> = gcur;
                            sc_.spawn(move || {
                                sgemm(&gcur[..r * d], cc, r, k, d, &mut dcur[..r * k])
                            });
                        }
                        if bi + 1 < nb {
                            gat(gnext, bi + 1);
                        }
                        if bi > 0 {
                            let p = bi - 1;
                            let r0 = p * blk;
                            let rp = (b - r0).min(blk);
                            dprev[..rp * k]
                                .par_chunks(k * 64)
                                .zip(
                                    lab[r0..r0 + rp]
                                        .par_chunks_mut(64)
                                        .zip(sc[r0..r0 + rp].par_chunks_mut(64)),
                                )
                                .for_each(|(dd, (l, s))| argmax_v(dd, k, l, s));
                        }
                    });
                }
            },
        );
    }
    for blk in [256usize, 512, 1024, 2048, 4096, 8192] {
        time(&format!("sgemm serial blocks of {blk}"), reps, || {
            for (gg, dd) in g.chunks(blk * d).zip(dots.chunks_mut(blk * k)) {
                sgemm(gg, &c, gg.len() / d, k, d, dd)
            }
        });
    }
    for threads in [1usize, 2, 3, 4, 6, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for tile in [256usize, 512, 1024, 2048] {
            let mut bufs: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
            time(
                &format!("fused tile {tile} x {threads} thr (gather+gemm+argmax)"),
                reps,
                || {
                    pool.install(|| {
                        lab.par_chunks_mut(tile)
                            .zip(sc.par_chunks_mut(tile))
                            .zip(ids.par_chunks(tile))
                            .for_each_init(
                                || (vec![0f32; tile * d], vec![0f32; tile * k]),
                                |(gb, db), ((l, s), id)| {
                                    let r = id.len();
                                    for (dst, &i) in gb.chunks_mut(d).zip(id) {
                                        let i = i as usize;
                                        dst.copy_from_slice(&x[i * d..(i + 1) * d]);
                                    }
                                    sgemm(&gb[..r * d], &c, r, k, d, &mut db[..r * k]);
                                    argmax(&db[..r * k], k, l, s);
                                },
                            )
                    })
                },
            );
            bufs.clear();
        }
    }
}
