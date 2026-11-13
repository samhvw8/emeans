fn main() {
    napi_build::setup();
    // macOS writes the output's absolute path as the library's install name;
    // a shipped addon must not carry the builder's home folder.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/emeans.node");
    }
}
