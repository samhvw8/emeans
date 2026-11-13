---
type: Design
title: The C API — a first-class public ABI over the plain binding layer
description: "crates/emeans-ffi as a public C library: the functions (one-shot, trained model, hierarchy, bytes), the result mechanism, errors, ownership and threading, ABI versioning and stability, and the dist tree with pkg-config and CMake packaging."
resource: crates/emeans-ffi/
tags: [c-api, ffi, abi, packaging, cmake, pkg-config, bindings]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T10:00:00Z'
sources:
  - id: ffi
    resource: crates/emeans-ffi/src/lib.rs
    title: The C API source; cbindgen turns its doc comments into the header
  - id: header
    resource: crates/emeans-ffi/include/emeans.h
    title: The generated header and its contract comment
  - id: dist
    resource: crates/emeans-ffi/dist.sh
    title: The dist tree builder (mise run c:dist)
  - id: quickstart
    resource: examples/c/quickstart.c
    title: The C quickstart and test (mise run c:test)
---

# Shape

`crates/emeans-ffi` (not published to crates.io) wraps `emeans-bindings-common`, the same plain
layer as the Python and Node bindings ([language bindings](/design/bindings.md)), so the C API
follows the same config, shape and backend rules. Its doc comments become the header
`include/emeans.h` (cbindgen 0.29.4, C99 comments); the contract (status, ownership, data,
configs, threads, stability) is the header's top comment, set in `cbindgen.toml`.[^ffi]

| Area | Functions |
|---|---|
| Version, errors | `emeans_abi_version`, `emeans_version`, `emeans_last_error`, `emeans_string_free` |
| Configs, backends | `emeans_default_config(kind)`, `emeans_backends` |
| One-shot | `emeans_train`, `emeans_assign`, `emeans_dedup` |
| Trained model (`EmeansModel`) | `emeans_model_train` (model + the train result), `_assign` (the model's metric and backend), `_objective` (exact, one assign pass), `_centroids` (borrowed view), `_info`, `_save`/`_load` (file), `_to_bytes`/`_from_bytes`, `_free` |
| Hierarchy (`EmeansHierarchy`) | `emeans_auto_cluster`, `emeans_hierarchy_add`, `_info`, `_save`/`_load` (directory), `_to_bytes`/`_from_bytes` (one blob), `_free` |
| Results (`EmeansResult`) | `emeans_result_array(r, name, &EmeansArray)`, `emeans_result_json`, `emeans_result_free` |

**One result mechanism.** Every call that returns data returns an `EmeansResult`: named arrays
(each an `EmeansArray {data, len, dtype}` view with dtype `EMEANS_F32`, `U32`, `I32` or `U8`) and
a JSON object of scalars. A new output is a new array name or JSON key, which the stability policy
allows within an ABI version, so the API grows without new structs.

**The model handle** came with the C API (2026-11-13): `train → model → assign, save, load` without
the hierarchy, for plain k-means users. The bytes functions (`*_to_bytes`, `*_from_bytes`) build
on `Emeans::to_bytes` / `from_bytes_with` and `Hierarchy::to_bytes` / `from_bytes` (added for the
WebAssembly package) and use the same layouts as the files.

# Errors, ownership, threads

- Status `EMEANS_OK` 0, `EMEANS_INVALID_ARGUMENT` 1 (NULL with a length, overflow, misalignment,
  bad UTF-8 or JSON, a NULL out-pointer: checked before any work), `EMEANS_ERROR` 2 (the library
  refused or failed), `EMEANS_PANIC` 3 (`catch_unwind` around every entry point).
- The message is thread-local, cleared at the start of every call. Out-pointers are written only
  on success, so a failed call leaves nothing to free.
- The caller owns models, hierarchies, results and `char**` strings and frees each once; views
  (`EmeansArray`, result JSON, the error message, the version) are borrowed.
- A model is read-only after training (its assigner sits behind a mutex), so threads may share
  it. `emeans_hierarchy_add` must not overlap other use of its hierarchy.

# Stability

`EMEANS_ABI_VERSION` 1 (`#define` in the header, `emeans_abi_version()` at run time). Within one
ABI version, existing functions, structs, constants and array names keep their meaning; additions
are allowed. A breaking change raises the ABI version together with the library's major version.
The shared library's install name / SONAME carries the ABI version (`libemeans.so.1`; macOS
`@rpath/libemeans.dylib` with compatibility version 1), set in `build.rs`. Before 1.0 a minor
release may raise the ABI version. `bind:go:test` and `c:test` fail on a stale header
(`cbindgen --verify`).

# Packaging

`mise run c:dist` (`crates/emeans-ffi/dist.sh [DIR] [cargo args]`) builds once with
`--print native-static-libs` and writes a relocatable tree:[^dist]

| Path | What |
|---|---|
| `lib/libemeans.a` | static library (Rust's `libemeans_ffi.a` renamed) |
| `lib/libemeans.dylib` / `lib/libemeans.so.1` + `libemeans.so` | shared library |
| `include/emeans.h` | the header |
| `lib/pkgconfig/emeans.pc` | `prefix=${pcfiledir}/../..`; `Libs.private` = rustc's native-static-libs |
| `lib/cmake/emeans/emeansConfig.cmake`, `emeansConfigVersion.cmake` | imported targets `emeans::emeans` (shared) and `emeans::emeans_static` (with the system libraries); compatible within one 0.x minor, later within one major |

The C library is named `libemeans`, not `libemeans_ffi`: the crate name is an implementation
detail, and no other C library uses the name. Install steps for macOS and Linux are in
`crates/emeans-ffi/README.md`.

# Tests

| Check | Where | macOS arm64 | Linux x86-64 |
|---|---|---|---|
| C API unit tests: error codes, cleared errors, NULL out-pointers, model train/assign, bytes and file round trips bit-identical, corrupt bytes refused | `crates/emeans-ffi/src/lib.rs` tests | pass | pass |
| Common layer = facade, incl. model handle and both bytes round trips | `bindings/common/tests/fixture.rs` | pass | pass |
| C quickstart with `-Wall -Wextra -Werror`: pkg-config + `cc` (shared), CMake shared, CMake static; checks bytes round trips and error codes | `mise run c:test` | pass | see the log |
| Go binding over the same ABI against the fixture | `mise run bind:go:test` | pass | pass |

[^ffi]: `crates/emeans-ffi/src/lib.rs` and `cbindgen.toml`.
[^dist]: `crates/emeans-ffi/dist.sh`; the `c:dist` and `c:test` tasks in `mise.toml`.
