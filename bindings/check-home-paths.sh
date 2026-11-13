#!/usr/bin/env bash
# Fail when a shipped binary holds an absolute home path (/Users/<name>/...,
# /home/<name>/...): panic locations and debug info record source paths, and
# a public artifact must not carry the builder's user name or folders.
# The builds that ship use RUSTFLAGS="--remap-path-prefix=$HOME=~".
# Usage: bindings/check-home-paths.sh FILE...
set -euo pipefail
bad=0
for f in "$@"; do
  [ -f "$f" ] || { echo "check-home-paths: missing $f" >&2; exit 2; }
  n=$( (LC_ALL=C grep -a -o -E '/(Users|home)/[A-Za-z0-9._-]+' "$f" || true) | wc -l | tr -d ' ')
  if [ "$n" != 0 ]; then
    echo "check-home-paths: $f holds $n home path(s), e.g.:" >&2
    (LC_ALL=C grep -a -o -E '/(Users|home)/[A-Za-z0-9._/-]+' "$f" || true) | sort -u | head -3 >&2
    bad=1
  else
    echo "check-home-paths: $f clean"
  fi
done
exit $bad
