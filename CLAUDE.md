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

# GPU parity — runs the REAL RunMetalNR entry point vs the CPU reference.
# Tolerance modes: strict (max<5e-3, mean<1e-4); sparseOK for shift-search
# selection races; hudOK for scope panels (display math like
# int(sqrt(..)+0.5) bar heights flips ±1px under GPU fast-math — mean<5e-5
# and <=400 pixels, any magnitude).
c++ -O2 -std=c++14 -I../plugin test_metal.mm ../plugin/MetalKernel.mm \
    -framework Metal -framework Foundation -o test_metal && ./test_metal

# Offline-compile the Metal kernel source (Resolve compiles it at plugin load —
# a syntax error would otherwise only appear on users' machines)
awk '/^static const char\* kKernelSource = R"MSL\(/{flag=1;next} /^\)MSL";/{flag=0} flag' \
    ../plugin/MetalKernel.mm > /tmp/opennr.metal
xcrun -sdk macosx metal -c /tmp/opennr.metal -o /tmp/opennr.air

# OpenCL kernel compile check: extract the R"CLC(...)CLC" block the same way
# and build it with a small clCreateProgramWithSource/clBuildProgram harness
# (test/compile_opencl.cpp).

# Describe smoke test — runs load/describe/describeInContext against the real
# .ofx in a minimal OFX host. A describe-time crash is INVISIBLE in Resolve
# (the plugin silently never appears), so run this after touching any param
# definitions:
c++ -O2 -std=c++14 -I../ofx/include test_describe.cpp -o test_describe
./test_describe ../plugin/OpenNR.ofx.bundle/Contents/MacOS/OpenNR.ofx

# Plugin build (universal arm64+x86_64, macOS 11.0 deployment target)
cd ../plugin && make

# Benchmark (README performance table comes from this)
cd ../test
c++ -O2 -std=c++14 -I../plugin bench_metal.mm ../plugin/MetalKernel.mm \
    -framework Metal -framework Foundation -o bench_metal && ./bench_metal
```

CUDA (`CudaKernel.cu`) cannot be RUN on this Mac — it is kept as a faithful
textual port; treat it as unverified until it has run on real NVIDIA
hardware (the user is testing this on Windows: local CMake build with
`-DHUSH_ENABLE_CUDA=ON`; flip the CI default only after parity-style
verification there). It CAN be syntax/type-checked without a toolkit —
`./test/check_cuda_syntax.sh` parses the whole file (kernel bodies included)
via clang's host-only CUDA mode against `test/cuda_shim.cuh`; CI runs it on
every push. Run it after any CudaKernel.cu edit.

**Golden policy:** the CPU suite pins default-output goldens (PSNR/meanAbs).
Changes that keep defaults bit-exact must not touch them. An intentional
default-path improvement re-pins the golden with a comment AND a CHANGELOG
entry explaining why (precedent: v3.2 two-scale residual re-pinned the full
static golden 40.927→40.948; the spatial goldens stayed bit-exact because
the 0.9·coarse factor only engages on correlated noise).

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

GitHub: repo `amateurmenace/Hush-OpenNR` (renamed from `OpenNR` 2026-07-15 — old
github.com URLs redirect, but the old Pages URL `github.io/OpenNR/` does
NOT: anything hard-linking Pages assets must use `github.io/Hush-OpenNR/`);
releases hold the pkg/zip artifacts
(`gh release create vX.Y.Z release/*.pkg release/*.zip`). The portfolio page
in `site/index.html` links to `releases/latest`.

**Windows binaries** are built by the `windows` CI job (CMakeLists.txt, MSVC,
OpenCL via vcpkg; version string parsed from build_release.sh). After a
release-worthy push goes green: `gh run download <id> --name OpenNR-Windows`
then `gh release upload vX.Y.Z OpenNR-X.Y.Z-Windows.zip`. The zip contains the
bundle (Contents/Win64/OpenNR.ofx), the .bat installer and INSTALL-WINDOWS.txt
drag instructions. CUDA stays OFF (`HUSH_ENABLE_CUDA`) until someone verifies
the port on NVIDIA hardware — the Windows build renders via OpenCL, which
Resolve uses on NVIDIA/AMD/Intel when CUDA isn't advertised. Do NOT add
`ofxsHWNDInteract.cpp` to any build — it needs a header that doesn't ship.

## Screenshot / orientation gotcha

OFX buffers are **bottom-up**; the test harness writes PPMs top-down (buffer
row 0 first). Since v2.1 the HUD/scopes anchor in *display* space
(`yd = H-1-y`), so in raw PPM/PNG test renders the panels appear at the
bottom, vertically mirrored — that is correct. To produce screenshots that
look like Resolve, flip the render vertically (`sips --flip vertical`).
Never flip a stale pre-fix PNG (that's how a mirrored HUD once shipped on
the website).

The website is built reproducibly: edit `site/index.template.html`, run
`python3 site/build_site.py` (bakes base64 images into `site/index.html` and
syncs `docs/` for GitHub Pages) then `python3 site/build_embed.py` (the
scoped Squarespace fragment; images load from the Pages assets URLs). Never
hand-edit the baked outputs. Image tokens live in BOTH scripts
(`B64_AFTER/B64_BEFORE/B64_SCM/B64_SCMO/B64_SCEQ/B64_SCSNR/B64_WM`); new
JS-referenced element ids must be added to build_embed.py's `ids` prefix
list or the embed's interactions silently break. The per-scope 720p shots
are regenerated with `test/site_shots.cpp` (see its header for the sips
conversion lines).

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

## Algorithm summary (v3.2)

1. **Noise profiling** (`estimateInput` / NoiseEst+FinalizeStats kernels):
   strided 2×2 sampling into 256-bin histograms — fine |Laplacian| (Y, C),
   coarse |Laplacian| on 2×2 block means (catches spatially-correlated noise),
   |temporal diff| vs a distinct neighbor frame (the primary estimator for
   video; immune to spatial correlation), plus a 16-luma-bin brightness-
   dependent gain curve (Q35 per bin, confidence-weighted, 3-tap smoothed).
   Robust statistics with motion/duplicate guards; calibration constants
   derived in comments and verified by tests. Exactly-flat samples (letterbox
   bars, crushed blacks — zero Laplacian in all channels) are skipped (v3.1).
   Two sigma pairs result: σ_S (spatial) and σ_T (temporal gating).
   **Lock Profile** freezes a raw clip-aggregated snapshot (region-aware since
   v3.2); Auto Profile Adjust multiplies locked values at USE time, so the
   trim keeps working while locked.
2. **Temporal merge**: per-pixel 3×3 patch mean |diff| against ±1/±2 frames,
   shift-search **Motion Tracking** (±2 px, 9 candidates, 1% acceptance
   margin, steeper roll-off for shifted winners), **firefly zapper**
   (3-frame temporal median, three tests must agree). **Hard-knee gate** —
   full weight below lo = 1.128·σ_T, smoothstep to exactly 0 at
   lo + (0.4+2.6·mt)·σ_T; no tail = no ghosting by construction. v3.2 adds
   **Ghost Guard**: a second knee on the SIGNED patch mean (noise cancels
   signed, σ_mean = √2σ/3; knee start 1.128σ_T ≈ 2.4σ_mean, span
   thrMul·σ_T/2) — catches slow coherent motion the magnitude gate can't
   see, ~0.01 dB cost on static. Chroma slaved to the luma gate. Outputs
   YCbCr + effective sample count (effN).
3. **Residual re-measurement** on the merged image — at TWO scales since
   v3.2 (fine + even-aligned 2×2 block Laplacian, ry = max(fine, 0.9·coarse),
   capped at σ_S): compression blotch that survives the merge is larger than
   a pixel and invisible to the fine estimator alone.
4. **Spatial Noise EQ** on the residual measurement: fine NLM band (3×3
   patches, ±R≤10, h = 1.15σ·hMul, bias-corrected, edge-aware Preserve
   Detail), medium band (2×2 block-mean ring, 3–8 px), coarse luma band
   (4×4 blocks, to ~47 px) and the chroma blotch pass (to ~23 px) — band
   corrections clipped to noise scale. Sliders >100 widen tolerances/reach
   (amounts cap at 1); ≤100 is bit-compatible with earlier releases.
   **Detail Rescue** cores the fine band: correction clamped to
   k = σ(2+6(1−r)) — crank strengths without blur.
5. **Refine**: shadow desat, luma texture re-injection, gradient-aware
   deband, deterministic film grain; then **Global Blend** (plain final
   original↔result crossfade, identity short-circuit at 0).
6. **Views + scopes**: views are full-image modes (result/split/input/
   after-temporal/noise-removed with noise-adaptive soft-knee gain/HUD/
   activity/SNR/matte); **scopes are per-step overlay panels** (Measurements
   top-left, Noise EQ top-right, Motion mini-map bottom-right) that composite
   over ANY view and never write alpha. 5×7 font in `kFont[43]`
   ("0-9.%A-Z+ -|="), all text at 2× glyph scale on opaque panels (1-px
   strokes decimated at fit zoom), sqrt-scaled histogram. Panel scale
   max(1, H/540).

## UI philosophy

Numbered steps teach the pipeline (1 Measure → 2 Temporal → 3 Spatial →
4 Refine → 5 Inspect). Tooltips are 1–2 sentences (v3.1 cut them down from
paragraphs). Per-stage Enable toggles show each stage's contribution;
**Auto Setup** writes measured settings into the visible sliders (one-undo
edit block); **Clean Slate** zeroes everything for fully-manual builds.
Scope checkboxes live in the step they explain, and the EQ scope auto-shows
on the first manual touch of an EQ slider (once per instance,
`m_EqScopeShown`). The viewer doubles as the scope because OFX can't draw
charts in the Inspector — measurements are rendered into the image by the
same kernels that use them, so the display can never disagree with the
filter. The plugin is color-space agnostic by design: it measures noise in
whatever space the host hands it (per brightness band) instead of assuming
one — don't add hidden colorimetric conversions.

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
