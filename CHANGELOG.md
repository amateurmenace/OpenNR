# OpenNR changelog

## 3.0.0 — 2026-07-15

The intelligence release: the plugin now measures your footage and sets
itself up.

### Auto Setup — the headline
- **One button dials in the whole plugin.** *Auto Setup (Analyze Footage)*
  (right under Strength) measures several frames spread across the clip —
  noise level, chroma character, spatial correlation, camera motion — then
  writes the best settings **into the visible sliders**, so you can see what
  it chose and dial anything in or back from there. It is not a mode: after
  Auto Setup everything is ordinary manual state.
- A read-only **Analysis** line reports what it did, e.g. *"Analyzed 5 frames
  · noise 3.2%Y / 4.1%C (noisy) · steady camera · profile locked"*.
- One **Cmd+Z** undoes the entire setup (it is applied as a single edit
  block), and a **Revert Auto Setup** button restores the prior values as
  belt-and-braces.
- Auto Setup touches denoise parameters only — it never changes your Step 4
  look choices (grain, texture, desaturation, debanding) or the View.

### New in the pipeline
- **Motion Tracking (Step 2, on by default).** Before gating each neighbour
  frame, the temporal stage now tries shifting its patch by up to ±2 px and
  matches on the best alignment — slow pans and handheld drift keep their
  across-frames averaging instead of degrading to spatial-only. The hard-knee
  gate is unchanged (still zero past the knee — still ghost-proof), and a
  shifted match must pass a *steeper* gate, so motion beyond the search reach
  is protected exactly as before. Costs ~13% at defaults; free on locked-off
  footage; toggle off to reclaim it.
- **Firefly Removal (Step 2, on by default).** Single-frame impulses — hot
  pixels, sensor fireflies — are clipped to the 3-frame temporal median
  before the merge. Three tests must all agree before a pixel is touched
  (temporal spike, neighbour-frame agreement, spatial outlier), so moving
  detail and thin fast-moving structures are left alone.
- **Noise EQ (Step 3).** The spatial stage is now three bands with their own
  strengths: **Fine** (the pixel-scale NLM pass, 100 = exactly v2.1),
  **Medium** (3–8 px noise clumps from heavy compression, off by default) and
  **Coarse Luma** (16–32 px brightness stains, off by default). Band
  corrections are computed on block means with clipped magnitudes, so each
  band removes its own scale and structure is never smeared by more than a
  noise-sized amount. Chroma Blotch Reduction is unchanged and remains the
  coarse band's chroma path.
- **Lock Profile (Step 1).** Snapshots the measured noise profile (both
  sigma pairs + the 32-bin brightness curve, aggregated robustly across
  frames of the clip) so every frame filters against the same numbers.
  Survives save/reload; the analysis HUD shows **LOCKED** while active (its
  histogram stays live for comparison).
- **Deband (Step 4).** Gradient-aware debanding: banding-tolerance ring
  smoothing plus a triangular micro-dither. Real edges are rejected by the
  tolerance itself and stay untouched.
- **Matte: Noisiness (Step 5 view).** Writes the normalized noise-dominance
  map into RGB *and alpha* so you can key it downstream and treat noisy
  areas differently in later nodes.

### Fixed
- Refine-only configurations (grain / texture / desaturation / debanding with
  both NR stages at zero) now render — v2.1 wrongly short-circuited them as
  identity.

## 2.1.0 — 2026-07-14

- **New name: Hush Open NR.** The effect now appears in Resolve under
  Effects → OpenFX → Filters → **Hush → Hush Open NR**. (The internal plugin
  identifier is unchanged, so existing projects keep working; the bundle
  filename also stays `OpenNR.ofx.bundle`.)
- **Windows binaries.** Each release now ships `OpenNR-x.y.z-Windows.zip`,
  built by CI (MSVC, OpenCL render path — NVIDIA/AMD/Intel). Drag the bundle
  into `C:\Program Files\Common Files\OFX\Plugins` or use the included .bat.
- **Drag the noise-sample region in the viewer.** With Noise Profile set to
  Automatic (From Region), the yellow rectangle is now a live on-screen
  control: drag inside to move it, drag a corner handle to resize. Enable the
  OpenFX overlay in the viewer's on-screen-controls menu. Built on the OFX
  Draw Suite.


## 2.0.0 — 2026-07-14

The full-suite release, built from field feedback ("spatial NR doesn't seem to
be doing much", "so much chroma noise", "strength does nothing past one").

### Fixed
- **Strength above 1.0 was a mathematical no-op** — it multiplied blend
  amounts that were already clamped at 1. New semantics: below 1.0 Strength
  fades the effect; above 1.0 the filters widen what they treat as noise
  (filter strengths and the temporal gate scale up), so 2.0 genuinely removes
  more than 1.0.
- **Spatial NR under-filtered after temporal NR.** It assumed temporal
  averaging removed sqrt(N) of the noise — but compression correlates noise
  across frames, so far less is actually removed. v2 measures the residual
  noise on the temporally-merged image itself and filters against that
  measurement. Self-truing, no assumptions.

### Added
- **Brightness-dependent noise profiling**: 16 luma bins, each with its own
  robust noise estimate (confidence-weighted, smoothed) — shadows get the
  stronger filtering they need without touching clean highlights. Visualized
  as a curve in the Noise Analysis view.
- **Chroma Blotch Reduction**: a sparse large-radius (up to 16 px) brightness-
  guided chroma pass that reaches the big soft color stains 4:2:0 compression
  leaves behind — the normal search radius physically can't span them.
- **Step 4 · Refine The Finish**: shadow desaturation with range control (the
  classic sat-vs-lum move), luma texture re-injection (keep the natural grain
  energy, lose the color noise), and synthesized **film grain** — soft
  value-noise, midtone-weighted like real stock, size and color controls,
  animated per frame, deterministic.
- **New views**: Input, After Temporal (mid-pipeline), and an **SNR Map**
  (per-pixel signal-to-noise heat map — magenta where noise dominates).
- **Noise Analysis HUD v2**: input AND residual noise readouts (see what
  temporal NR actually accomplished), effective-frames-averaged readout, SNR
  gain in dB, and the noise-vs-brightness curve.
- Search radius extended to 8; chroma strength default raised to 100; per-
  stage strength sliders now scale filter aggressiveness as well as blend.


## 1.2.0 — 2026-07-14

### Fixed
- **Ghosting on motion ("ghost soup").** v1.1's temporal gate was a Gaussian
  falloff with a fat tail: pixels several noise-widths different still blended
  at 30–40%, which smeared slow motion, soft gradients, and motion blur into
  visible trails. The gate is now a **hard knee**: full weight while a
  difference is explainable by noise, smoothstep to **exactly zero** past the
  motion threshold — mismatched pixels can never bleed in, by construction.
  Chroma is additionally slaved to the luma gate (color trails were part of
  the soup). In the new slow-drift regression test, temporal NR now *gains*
  4.3 dB over spatial-only where v1.1 lost quality.
- The temporal noise estimator is stricter about motion: unbiased median and
  20th-percentile estimates must agree, otherwise only the static quantile is
  trusted — camera motion can no longer widen the gate by inflating the
  measured noise.

### Changed
- Motion Threshold default lowered (40 → 30) and its tooltip now explains the
  hard-zero behavior.
- macOS builds are signed with a Developer ID certificate.

## 1.1.0 — 2026-07-14

Field-testing release. Fixes the two launch-blocking issues found on real
footage and real machines, and rebuilds the UI around transparency.

### Fixed
- **Auto noise profiling did nothing on real footage.** v1.0 measured noise
  with a fine-detail (Laplacian) estimator only, which badly underestimates
  the spatially-correlated noise that debayering, chroma subsampling and
  compression produce — the measured level came out near zero and every
  filter threshold collapsed with it. v1.1 measures noise the right way for
  video: median |frame-to-frame difference| (immune to spatial correlation,
  motion-robust via a low-quantile cross-check and a spatial clamp, guarded
  against duplicate frames at clip boundaries), plus fine- and coarse-scale
  spatial estimators. Temporal and spatial stages are now calibrated by
  separate, appropriate estimates.
- **Plugin failed to load on other Macs.** The v1.0 binary carried the build
  machine's SDK minimum (macOS 15.5), so older systems silently skipped it.
  Builds now target macOS 11.0 (Intel + Apple Silicon) and the bundle is
  explicitly ad-hoc signed. If you hand-copy a downloaded bundle instead of
  using the .pkg, clear quarantine: `sudo xattr -dr com.apple.quarantine
  /Library/OFX/Plugins/OpenNR.ofx.bundle`.

### Changed
- Controls reorganized into numbered steps: **1 · Measure The Noise**,
  **2 · Temporal NR**, **3 · Spatial NR**, **4 · Inspect & Compare**.
- The two profiling checkboxes were replaced by a single **Noise Profile**
  dropdown — Automatic (Whole Frame) / Automatic (From Region) / Manual —
  with correct enabling/greying of the dependent controls.
- Every control has a longer, plain-English tooltip (what it does, when to
  touch it, which direction to push it).

### Added
- **Noise Analysis view**: live on-screen readout of the measured spatial and
  temporal noise levels (luma and chroma) with meter bars, the noise
  histogram with its median marked, and the measurement-region rectangle.
- **Temporal Activity view**: heat map of where across-frame averaging is
  active (green) versus motion-protected (red).
- **Enable toggles** for the temporal and spatial stages, to A/B each stage's
  contribution.
- **Auto Profile Adjust** (×0.25–×4) to fine-tune the automatic measurement.
- Split view moved next to the other inspection views.

## 1.0.0 — 2026-07-14

Initial release: GPU spatio-temporal noise reduction (Metal / CUDA / OpenCL)
with automatic noise profiling, 3/5-frame motion-adaptive temporal NR,
noise-adaptive NLM spatial NR with luma/chroma split, region profiling,
noise-only and split views. Validated against synthetic footage (+13 dB PSNR
static, +9 dB under motion at σ=0.05) with GPU/CPU parity tests.
