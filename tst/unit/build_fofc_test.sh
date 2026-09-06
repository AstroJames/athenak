#!/bin/bash
# Reuse an existing Makefile-generator AthenaK build; compile/link only, never execute.
set -euo pipefail
build_dir=$(realpath "${1:?Usage: build_fofc_test.sh BUILD_DIR}")
source_dir=$(cd "$(dirname "$0")/../.." && pwd)
cd "$build_dir/src"
read -r -a link_args < CMakeFiles/athena.dir/link.txt
compiler=${link_args[0]}
read -r -a defines <<< "$(sed -n 's/^CXX_DEFINES = //p' CMakeFiles/athena.dir/flags.make)"
read -r -a includes <<< "$(sed -n 's/^CXX_INCLUDES = //p' CMakeFiles/athena.dir/flags.make)"
read -r -a options <<< "$(sed -n 's/^CXX_FLAGS = //p' CMakeFiles/athena.dir/flags.make)"
"$compiler" "${defines[@]}" "${includes[@]}" "${options[@]}" \
  -c "$source_dir/tst/unit/fofc_cascade.cpp" -o fofc_cascade.o
# CMake quotes paths containing hyphens. This helper supports paths without whitespace.
for q in "${!link_args[@]}"; do
  arg=${link_args[q]//\"/}
  case "$arg" in
    CMakeFiles/athena.dir/main.cpp.o) arg=fofc_cascade.o ;;
    athena) arg=fofc_cascade ;;
    -Wl,--dependency-file,*) arg=-Wl,--dependency-file,fofc_cascade.link.d ;;
  esac
  link_args[q]=$arg
done
"${link_args[@]}"
