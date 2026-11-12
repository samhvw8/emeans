//! Layer-1 contract on the Metal engine, against the f64 reference.
#![cfg(target_os = "macos")]

use emeans_core::core::contract::check;

#[test]
fn metal_passes_the_contract() {
    use emeans_metal::{FUSED_TILES, MetalCore};
    for tile in 0..FUSED_TILES.len() {
        let mut m = match MetalCore::with_tile(tile) {
            Ok(m) => m,
            Err(e) if format!("{e:#}").contains("no Metal device") => {
                eprintln!("skipping: {e:#}");
                return;
            }
            Err(e) => panic!("{e:#}"),
        };
        let rep = check(&mut m).unwrap();
        eprintln!("{rep:?}");
    }
}

/// The production batch shape on the engine's own tile choice (64 x 128 x 32,
/// one centroid part at 16,384 rows), with exact duplicate centroids and rows.
#[test]
fn metal_passes_the_contract_at_the_production_shape() {
    use emeans_core::core::contract::check_production;
    use emeans_metal::MetalCore;
    let mut m = match MetalCore::new() {
        Ok(m) => m,
        Err(e) if format!("{e:#}").contains("no Metal device") => {
            eprintln!("skipping: {e:#}");
            return;
        }
        Err(e) => panic!("{e:#}"),
    };
    let rep = check_production(&mut m).unwrap();
    eprintln!("{rep:?}");
    assert!(rep.top1_rows > 0 && rep.topk_rows > 0);
}
