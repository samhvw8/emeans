//! e-means for the browser and Node.js.
//!
//! [`api`] is the binding API in plain Rust. On `wasm32-unknown-unknown` the
//! `web` module exports it to JavaScript with wasm-bindgen: typed arrays in
//! and out, options as plain objects, errors as thrown `Error`s.

pub mod api;

#[cfg(target_family = "wasm")]
mod web;
