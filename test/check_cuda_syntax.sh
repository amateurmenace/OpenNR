#!/bin/sh
# Syntax/type-check CudaKernel.cu WITHOUT a CUDA toolkit: clang parses the
# whole file (host code AND kernel bodies) in host-only CUDA mode against
# test/cuda_shim.cuh. Catches undeclared identifiers, type errors and brace
# slips in the textual port — it does NOT prove the kernels compute anything;
# real verification still needs NVIDIA hardware.
set -e
cd "$(dirname "$0")"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
printf '// empty shim — declarations come from cuda_shim.cuh (-include)\n' > "$tmpdir/cuda_runtime.h"
clang++ -x cuda --cuda-host-only -nocudainc -nocudalib -fsyntax-only -std=c++14 \
  -include cuda_shim.cuh -I"$tmpdir" -I../plugin ../plugin/CudaKernel.cu
echo "CUDA SYNTAX CHECK OK"
