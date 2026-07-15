# OpenNR — project guide

Free, MIT-licensed spatio-temporal noise reduction OpenFX plugin for DaVinci
Resolve (works in the free edition — that's the whole point; Resolve's built-in
NR is Studio-only). Runs on the Color, Edit and Fusion pages. GPU: Metal
(macOS), CUDA + OpenCL (Windows/Linux), plus a CPU reference path.

## The cardinal rule

`plugin/nr_core.h` is the **single source of truth** for the algorithm. The
three GPU kernels (`MetalKernel.mm`, `CudaKernel.cu`, `OpenCLKernel.cpp`) are
line-by-line ports of it. **Any change to the math must be applied to all four
files** and then verified with the parity test. Constants, histogram layouts,
gate formulas, even loop order — keep them textually parallel; the parity test
asserts agreement to ~5e-3 max / 1e-4 mean (typically ~2e-5).

Shared plumbing lives in `plugin/NRParams.h`: the NRParams struct (all fields
4 bytes — it is passed by value to CUDA/OpenCL kernels and via setBytes to
Metal, and re-declared inside the Metal and OpenCL kernel source strings — a
field added in one place must be added in ALL FOUR struct declarations) and
the stats-buffer slot layout (`NR_STATS_*`).

## Build & test (run all of this before calling any change done)

```sh
# CPU algorithm suite — synthetic scenes with known noise, PSNR gates,
# estimator accuracy (iid + correlated), slow-motion no-ghosting regression,
# identity checks, view renders (out_view*.ppm for eyeballing)
cd test
c++ -O2 -std=c++14 -I../plugin test_denoise.cpp -o test_denoise && ./test_denoise

# GPU parity — runs the REAL RunMetalNR entry point vs the CPU reference
c++ -O2 -std=c++14 -I../plugin test_metal.mm ../plugin/MetalKernel.mm \
    -framework Metal -framework Foundation -o test_metal && ./test_metal

# Offline-compile the Metal kernel source (Resolve compiles it at plugin load —
# a syntax error would otherwise only appear on users' machines)
awk '/^static const char\* kKernelSource = R"MSL\(/{flag=1;next} /^\)MSL";/{flag=0} flag' \
    ../plugin/MetalKernel.mm > /tmp/opennr.metal
xcrun -sdk macosx metal -c /tmp/opennr.metal -o /tmp/opennr.air

# OpenCL kernel compile check: extract the R"CLC(...)CLC" block the same way
# and build it with a small clCreateProgramWithSource/clBuildProgram harness.

# Plugin build (universal arm64+x86_64, macOS 11.0 deployment target)
cd ../plugin && make

# Benchmark (README performance table comes from this)
cd ../test
c++ -O2 -std=c++14 -I../plugin bench_metal.mm ../plugin/MetalKernel.mm \
    -framework Metal -framework Foundation -o bench_metal && ./bench_metal
```

CUDA (`CudaKernel.cu`) cannot be compiled or tested on this Mac — it is kept
as a faithful textual port; treat it as unverified until CI builds it.

## Release

```sh
./build_release.sh   # builds, signs, packages release/*.pkg and *.zip
```

Version bump checklist (all four places):
1. `plugin/OpenNRPlugin.cpp` — `kPluginVersionMinor` and the version in
   `kPluginDescription`
2. `plugin/Info.plist` — both version strings
3. `build_release.sh` — `VERSION=`
4. `CHANGELOG.md` — new entry (used as GitHub release notes)

Signing: the script auto-detects a "Developer ID Application" identity
(Stephen's team: 6M536MV7GT) for the bundle, a "Developer ID Installer"
identity for the pkg (not present yet — pkg ships unsigned), and a notarytool
keychain profile named `opennr-notary` for notarization (not stored yet; set
up with `xcrun notarytool store-credentials opennr-notary --apple-id ...
--team-id 6M536MV7GT --password <app-specific password>`).

GitHub: repo `amateurmenace/OpenNR`; releases hold the pkg/zip artifacts
(`gh release create vX.Y.Z release/*.pkg release/*.zip`). The portfolio page
in `site/index.html` links to `releases/latest`.

## Distribution gotchas (learned the hard way)

- **Always set the deployment target.** Without `-mmacosx-version-min` the
  binary inherits the SDK minimum (was 15.5) and silently fails to load on
  older Macs; Resolve logs "OFX - plugin not supported on this platform, is
  being skipped". The Makefile pins 11.0.
- **Resolve caches OFX scans** in `~/Library/Application Support/Blackmagic
  Design/DaVinci Resolve/OFXPluginCacheV2.xml`. Delete it to force a rescan
  after swapping binaries.
- **Quarantine**: hand-copied downloaded bundles carry `com.apple.quarantine`
  and get silently blocked — `sudo xattr -dr com.apple.quarantine <bundle>`.
  Installer-pkg payloads don't have this problem.
- **Test in Resolve without admin**: launch Resolve with `OFX_PLUGIN_PATH`
  pointing at a directory containing the bundle; Resolve honors it and the
  cache records the scan result (`status="0"` = loaded). Same plugin identifier
  at a higher version wins over the installed copy.

## Algorithm summary (v1.2)

1. **Noise profiling** (`estimateSigma` / NoiseEst+FinalizeStats kernels):
   strided 2×2 sampling into six 256-bin histograms — fine |Laplacian| (Y, C),
   coarse |Laplacian| on 2×2 block means (catches spatially-correlated noise),
   |temporal diff| vs a distinct neighbor frame (the primary estimator for
   video; immune to spatial correlation). Robust statistics: medians, with the
   temporal estimate taken as median if it agrees with the unbiased 20th
   percentile (×1.4) else the quantile (motion robustness), rejected entirely
   if > 3.5× the Laplacian family or ≤ 0.0015 (duplicate frame). Calibration
   constants are derived in comments and verified by tests. Two sigma pairs
   result: σ_S (spatial filtering) and σ_T (temporal gating).
2. **Temporal merge**: per-pixel 3×3 patch mean |diff| against ±1/±2 frames;
   **hard-knee gate** — full weight below lo = 1.128·σ_T (expected pure-noise
   difference), smoothstep to exactly 0 at lo + (0.4+2.6·mt)·σ_T. No tail = no
   ghosting by construction (v1.1's Gaussian tail caused "ghost soup" on real
   footage). Chroma weight is multiplied by the luma gate. Outputs YCbCr +
   effective sample count (effN) per pixel.
3. **Spatial NLM**: 3×3 patches over a ±R window on the temporal result,
   h = 1.15σ (luma) / 2.0σ (chroma) with σ_eff = σ_S/√effN, distances
   bias-corrected by 2σ², edge-aware h reduction (Preserve Detail), luma-guided
   chroma, max-neighbor-weight center trick. Faster mode = single-pixel
   bilateral with spatial falloff.
4. **Views**: result / split / noise-only / **Noise Analysis HUD** (5×7 pixel
   font in `kFont[27]`, order "0-9.%ACEFILMOPRSTUY space"; meters, histogram
   from the stats buffer, region rectangle) / **Temporal Activity** heat map
   from effN. HUD panel is 300×134 local units scaled by max(1, H/540).

## UI philosophy

Numbered steps teach the pipeline (1 Measure → 2 Temporal → 3 Spatial →
4 Inspect). Every control's tooltip says what it does, when to touch it, and
which direction to push it. Per-stage Enable toggles exist so users can see
each stage's contribution. The viewer doubles as the scope because OFX can't
draw charts in the Inspector — measurements are rendered into the image by the
same kernels that use them, so the display can never disagree with the filter.

## Repo layout

- `plugin/` — nr_core.h (reference), NRParams.h, OpenNRPlugin.{h,cpp} (OFX
  host plumbing/params), 3 GPU kernel files, Makefile, Info.plist
- `ofx/` — vendored OpenFX 1.4 headers + Support C++ wrapper (BSD, don't edit)
- `test/` — test_denoise.cpp (CPU suite), test_metal.mm (parity),
  bench_metal.mm
- `installer/` — double-click install scripts for mac/windows (zip fallback)
- `site/` — portfolio/landing page (self-contained HTML)
- `docs/images/` — README/site imagery (from test renders)
- `.github/workflows/build.yml` — macOS CI (tests + build + artifacts)
