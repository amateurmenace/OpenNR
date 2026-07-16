#!/bin/sh
# Syntax/type-check the CUDA kernels WITHOUT a CUDA toolkit: clang parses each
# whole file (host code AND kernel bodies) in host-only CUDA mode against
# test/cuda_shim.cuh. Catches undeclared identifiers, type errors and brace
# slips in the textual ports — it does NOT prove the kernels compute anything;
# real verification still needs NVIDIA hardware.
#
# BOTH plugins' kernels are checked. Speak's was missing until 2026-07-16: this
# script only ever parsed Hush's CudaKernel.cu, so SpeakCudaKernel.cu — the one
# file in the project that NOTHING on a Mac compiles — had never been checked by
# CI at all, through ten Speak commits.
set -e
cd "$(dirname "$0")"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
printf '// empty shim — declarations come from cuda_shim.cuh (-include)\n' > "$tmpdir/cuda_runtime.h"
# Glob, not a hardcoded list: the same script serves the Hush repo, the Speak
# repo, and any tree with both (a missing hardcoded file would abort the loop).
for f in ../plugin/*.cu; do
  clang++ -x cuda --cuda-host-only -nocudainc -nocudalib -fsyntax-only -std=c++14 \
    -include cuda_shim.cuh -I"$tmpdir" -I../plugin "$f"
  echo "  ok: $f"
done
echo "CUDA SYNTAX CHECK OK"
