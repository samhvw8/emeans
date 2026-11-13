//! The shared library's install name: `libemeans` with the ABI version, so
//! `c:dist` can ship it as `libemeans.dylib` / `libemeans.so.1` and a linked
//! program finds it by that name (through its rpath).

fn main() {
    let abi = 1; // EMEANS_ABI_VERSION in src/lib.rs
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!(
            "cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libemeans.dylib,-compatibility_version,{abi}"
        ),
        Ok("linux") => println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libemeans.so.{abi}"),
        _ => {}
    }
    println!("cargo:rerun-if-changed=build.rs");
}
