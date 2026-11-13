//! Write the cross-language fixture (see `emeans_bindings_common::fixture`).
//!
//! Usage: `cargo run --release -p emeans-bindings-common --example write_fixture -- <dir>`

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: write_fixture <dir>"))?;
    emeans_bindings_common::fixture::write_fixture(std::path::Path::new(&dir))
}
