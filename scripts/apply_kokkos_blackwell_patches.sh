#!/usr/bin/env bash
# Apply the NVIDIA Blackwell (sm_100, e.g. B200) patches to the bundled Kokkos 4.4.0 submodule.
# Kokkos 4.4.0 has no BLACKWELL100 architecture and no compute-capability-10 case in its
# block-size deduction; Kokkos >= 4.7 supports both natively. Run once after
# `git submodule update`, before configuring with -DKokkos_ARCH_BLACKWELL100=ON.
# Idempotent: patches that are already applied are skipped. Run from the AthenaK root.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kk="$root/kokkos"
want="4.4.00"
have="$(git -C "$kk" describe --tags 2>/dev/null || echo unknown)"
if [ "$have" != "$want" ]; then
  echo "kokkos submodule is $have, these patches target $want; refusing to apply" >&2
  exit 1
fi
for p in "$root"/patches/kokkos-4.4.0/blackwell_arch.patch \
         "$root"/patches/kokkos-4.4.0/blackwell_runtime.patch; do
  name="$(basename "$p")"
  if git -C "$kk" apply -p2 --reverse --check "$p" 2>/dev/null; then
    echo "already applied: $name"
  elif git -C "$kk" apply -p2 --check "$p"; then
    git -C "$kk" apply -p2 "$p"
    echo "applied: $name"
  else
    echo "cannot apply $name cleanly; inspect kokkos/ for local changes" >&2
    exit 1
  fi
done
