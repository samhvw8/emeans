//! CUDA engine binding rules. Skips when no CUDA device is present.

mod common;

use emeans_core::assign::assign_all;
use emeans_core::{Assigner, CpuAssigner, CpuKernel, Metric, Rows};
use emeans_cuda::CudaCore;

/// `bind_data` copies the data it is given: binding again after the caller
/// wrote new values into the same buffer (same pointer and length) must read
/// the new values, not the copy of the old ones.
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
    let mut g = match CudaCore::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e:#}");
            return;
        }
    };
    let mut buf = x.clone();
    let indexed = |g: &mut CudaCore, buf: &[f32]| {
        let rows = Rows::Indexed {
            data: buf,
            ids: &ids,
        };
        assign_all(g, rows, d, &c, Metric::InnerProduct, 1 << 16)
            .unwrap()
            .0
    };
    g.bind_data(&buf, d).unwrap();
    let l = indexed(&mut g, &buf);
    assert_eq!(common::non_ties(&x, &c, d, &l, &want_x, true, 1e-5), 0);
    // New data at the same address, bound again.
    buf.copy_from_slice(&y);
    g.bind_data(&buf, d).unwrap();
    let l = indexed(&mut g, &buf);
    assert_eq!(
        common::non_ties(&y, &c, d, &l, &want_y, true, 1e-5),
        0,
        "bind_data served the copy of the old data"
    );
    g.unbind_data();
}
