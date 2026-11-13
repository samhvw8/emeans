# emeans C API

The public C API of [emeans](https://github.com/samhvw8/emeans): e-means clustering (mini-batch
k-means with scaling laws) on the CPU, Apple GPUs (Metal) and NVIDIA GPUs (CUDA). Use it from C
or C++, or wrap it from any language with a C FFI. The Go binding (`bindings/go`) is the
reference wrapper. The guide for a new language is `.okf/recipes/adding-a-language-binding.md`.

The header, [`include/emeans.h`](include/emeans.h), documents every function. Its top comment
states the rules: status codes, the thread-local error message, who frees what, shapes and units,
configs, thread safety and ABI stability.

## Build and install

Needs Rust (`mise install` in the repo root gives the pinned toolchain, plus cmake and cbindgen).

```sh
mise run c:dist            # or: crates/emeans-ffi/dist.sh dist [--features cuda]
```

This writes a relocatable tree:

```
dist/include/emeans.h
dist/lib/libemeans.a                      static library
dist/lib/libemeans.dylib                  macOS shared library (install name @rpath/libemeans.dylib)
dist/lib/libemeans.so.1, libemeans.so     Linux shared library (SONAME libemeans.so.1)
dist/lib/pkgconfig/emeans.pc
dist/lib/cmake/emeans/emeansConfig.cmake, emeansConfigVersion.cmake
```

`dist.sh` builds with `--remap-path-prefix=$HOME=~` and without debug info, so the libraries
hold no path from your home folder; `mise run c:test` checks that.

The file names say `libemeans`, but cargo builds the crate `emeans-ffi` as `libemeans_ffi`;
`dist.sh` renames it. The Rust crate `emeans` is unrelated to these files.

Install system-wide, or keep the tree anywhere:

```sh
# macOS
sudo cp -R dist/include/* /usr/local/include/ && sudo cp -R dist/lib/* /usr/local/lib/
# Linux
sudo cp -R dist/include/* /usr/local/include/ && sudo cp -R dist/lib/* /usr/local/lib/ && sudo ldconfig
```

`emeans.pc` and the CMake files find the library relative to their own location, so a tree that
is copied or moved keeps working.

## Use it

pkg-config (shared library):

```sh
export PKG_CONFIG_PATH=/path/to/dist/lib/pkgconfig      # not needed under /usr/local
cc app.c $(pkg-config --cflags --libs emeans) -Wl,-rpath,/path/to/dist/lib -o app
```

Static linking: link `dist/lib/libemeans.a` and the system libraries that
`pkg-config --static --libs emeans` lists (`Libs.private`: Metal, Foundation, Accelerate and objc
on macOS; `-lm -ldl -lpthread` and friends on Linux).

CMake:

```cmake
find_package(emeans 0.1 REQUIRED)                 # -DCMAKE_PREFIX_PATH=/path/to/dist
target_link_libraries(app PRIVATE emeans::emeans) # or emeans::emeans_static
```

## Example

[`examples/c/quickstart.c`](../../examples/c/quickstart.c) trains a model, assigns, saves and
loads it as bytes, clusters with automatic k, adds new rows through a hierarchy loaded from bytes,
and shows the error handling. `mise run c:test` builds it with `cc` through pkg-config and with
CMake (shared and static), then runs all three.

```c
EmeansModel *m;
EmeansResult *r;
if (emeans_model_train(x, n, d, 32, "{\"spherical\": true}", &m, &r) != EMEANS_OK) {
  fprintf(stderr, "emeans: %s\n", emeans_last_error());
  return 1;
}
EmeansArray c;
emeans_result_array(r, "centroids", &c);   /* float, k * d, borrowed from r */
emeans_result_free(r);
emeans_model_free(m);
```

## API at a glance

| Area | Functions |
|---|---|
| Version and errors | `emeans_abi_version`, `emeans_version`, `emeans_last_error`, `emeans_string_free` |
| Configs and backends | `emeans_default_config`, `emeans_backends` |
| One-shot | `emeans_train`, `emeans_assign`, `emeans_dedup` |
| Trained model | `emeans_model_train`, `_assign`, `_centroids`, `_info`, `_save`, `_load`, `_to_bytes`, `_from_bytes`, `_free` |
| Automatic k, incremental | `emeans_auto_cluster`, `emeans_hierarchy_add`, `_info`, `_save`, `_load`, `_to_bytes`, `_from_bytes`, `_free` |
| Results | `emeans_result_array`, `emeans_result_json`, `emeans_result_free` |

## Stability

`EMEANS_ABI_VERSION` is 1, and `emeans_abi_version()` returns the loaded library's version.
Within one ABI version, existing functions, structs, constants and result array names keep their
meaning; new functions, arrays and JSON keys may be added. A change that breaks a compiled caller
raises the ABI version and the library's major version (the shared library's SONAME follows the
ABI version). Before 1.0 the API is not frozen: a 0.x minor release may raise the ABI version.

## Limits

- float32 only; metrics are squared L2 and inner product (cosine for unit rows).
- Built and tested on macOS arm64 (CPU, Metal) and Linux x86-64 (CPU). CUDA builds from source
  (`--features cuda`) and is not tested through the C API. Windows is not built.
- No prebuilt packages yet.
