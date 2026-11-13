#!/usr/bin/env bash
# Build the emeans C library and lay it out as an installable tree:
#   dist/lib/libemeans.a, libemeans.dylib (macOS) or libemeans.so.1 + .so (Linux)
#   dist/include/emeans.h
#   dist/lib/pkgconfig/emeans.pc          (pkg-config --cflags --libs emeans)
#   dist/lib/cmake/emeans/emeansConfig.cmake, emeansConfigVersion.cmake
#                                          (find_package(emeans) -> emeans::emeans, emeans::emeans_static)
# The tree is relocatable: copy it anywhere (e.g. /usr/local) and point
# PKG_CONFIG_PATH / CMAKE_PREFIX_PATH at it.
# Usage: crates/emeans-ffi/dist.sh [DIST_DIR] [extra cargo args, e.g. --features cuda]
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
dist="$(cd "$root" && mkdir -p "${1:-dist}" && cd "${1:-dist}" && pwd)"
shift || true
target="${CARGO_TARGET_DIR:-$root/target/c}"
# Shipped binaries must not carry the builder's home folder (panic locations, debug info).
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=$HOME=~"
# The macOS linker also records the paths of the objects it linked (debug map):
# no debug info in the shipped shared library.
export CARGO_PROFILE_RELEASE_STRIP=debuginfo
abi=$(sed -n 's/^pub const EMEANS_ABI_VERSION: u32 = \([0-9]*\);/\1/p' "$root/crates/emeans-ffi/src/lib.rs")
version=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' "$root/Cargo.toml")

# One build; the static library's native dependencies come from rustc itself.
native=$(cd "$root" && CARGO_TARGET_DIR="$target" CARGO_PROFILE_RELEASE_DEBUG=0 \
  cargo rustc --release -q -p emeans-ffi --lib "$@" -- --print native-static-libs 2>&1 \
  | sed -n 's/.*native-static-libs: //p' | tail -1)
[ -n "$native" ] || { echo "dist: no native-static-libs line from rustc" >&2; exit 1; }
# Drop repeats, keep "-framework X" pairs together.
native=$(echo "$native" | awk '{for(i=1;i<=NF;i++){t=$i; if(t=="-framework"){t=t" "$(++i)}; if(!seen[t]++) out=out (out?" ":"") t}; print out}')

rm -rf "$dist/lib" "$dist/include"
mkdir -p "$dist/lib/pkgconfig" "$dist/lib/cmake/emeans" "$dist/include"
cp "$root/crates/emeans-ffi/include/emeans.h" "$dist/include/"
cp "$target/release/libemeans_ffi.a" "$dist/lib/libemeans.a"
case "$(uname -s)" in
  Darwin) shared=libemeans.dylib
          cp "$target/release/libemeans_ffi.dylib" "$dist/lib/$shared" ;;
  Linux)  shared=libemeans.so.$abi
          cp "$target/release/libemeans_ffi.so" "$dist/lib/$shared"
          ln -sf "$shared" "$dist/lib/libemeans.so" ;;
  *) echo "dist: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac

cat > "$dist/lib/pkgconfig/emeans.pc" <<PC
prefix=\${pcfiledir}/../..
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: emeans
Description: e-means clustering on CPU, Metal and CUDA (C API, ABI $abi)
Version: $version
URL: https://github.com/samhvw8/emeans
Cflags: -I\${includedir}
Libs: -L\${libdir} -lemeans
Libs.private: $native
PC

cmake_native=$(echo "$native" | awk '{for(i=1;i<=NF;i++){t=$i; if(t=="-framework"){t=t" "$(++i)}; printf "%s\"%s\"", (n++?" ":""), t}}')
cat > "$dist/lib/cmake/emeans/emeansConfig.cmake" <<CM
# find_package(emeans) -> emeans::emeans (shared) and emeans::emeans_static.
get_filename_component(_emeans_prefix "\${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(emeans_VERSION $version)
set(emeans_ABI_VERSION $abi)
if(NOT TARGET emeans::emeans)
  add_library(emeans::emeans SHARED IMPORTED)
  set_target_properties(emeans::emeans PROPERTIES
    IMPORTED_LOCATION "\${_emeans_prefix}/lib/$shared"
    IMPORTED_SONAME "$( [ "$shared" = libemeans.dylib ] && echo @rpath/libemeans.dylib || echo "$shared")"
    INTERFACE_INCLUDE_DIRECTORIES "\${_emeans_prefix}/include")
  add_library(emeans::emeans_static STATIC IMPORTED)
  set_target_properties(emeans::emeans_static PROPERTIES
    IMPORTED_LOCATION "\${_emeans_prefix}/lib/libemeans.a"
    INTERFACE_INCLUDE_DIRECTORIES "\${_emeans_prefix}/include"
    INTERFACE_LINK_LIBRARIES "$(echo $cmake_native | sed 's/" "/;/g; s/"//g')")
endif()
unset(_emeans_prefix)
CM
cat > "$dist/lib/cmake/emeans/emeansConfigVersion.cmake" <<CM
# Compatible within one ABI version; before 1.0, within one minor version.
set(PACKAGE_VERSION $version)
if(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
elseif("$version" MATCHES "^0\\\\.")
  string(REGEX MATCH "^0\\\\.[0-9]+" _have "$version")
  string(REGEX MATCH "^0\\\\.[0-9]+" _want "\${PACKAGE_FIND_VERSION}")
  if(_have STREQUAL _want OR NOT PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  else()
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
  endif()
else()
  string(REGEX MATCH "^[0-9]+" _have "$version")
  string(REGEX MATCH "^[0-9]+" _want "\${PACKAGE_FIND_VERSION}")
  if(_have STREQUAL _want)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  else()
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
  endif()
endif()
if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
endif()
CM
echo "dist: emeans $version (ABI $abi) -> $dist"
echo "dist: static link needs: $native"
