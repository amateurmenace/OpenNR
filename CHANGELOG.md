# OpenNR changelog

## 3.3.0 — 2026-07-15

More powerful and more efficient: the motion search quadruples its reach,
the temporal stack grows to 7 frames, a second spatial pass takes on
severe noise, chroma finally gets per-channel treatment — and locked
profiles render measurably faster.

### Added
- **Hierarchical motion search (~±8 px, was ±2).** Motion Tracking now runs
  a two-level search: a step-4 coarse grid scored on 2×2 block means (so a
  4 px step sees structure instead of aliasing on texture), then a
  converging ±1 refine walk. Real pans keep their temporal averaging: 4 and
  6 px/frame pans beat spatial-only by ~4 dB where v3.2 gated them off
  entirely; at a 1.5 px pan even the ±2-frame neighbours now re-aim. All
  the ghost-proofing survives untouched — knee start, the 1.7× steeper
  roll-off for shifted winners, the 1% acceptance margin (10% for the far
  coarse jumps: measured, anything looser leaked 0.07 dB of slow-drift
  quality), and Ghost Guard on the winner's signed mean. Same toggle, no
  new UI; tracking-off path is bit-exact with v3.2.
- **7 Frames** option on Number of Frames (append-only — existing projects
  keep their choice). +1.4 dB on static heavy noise, effective sample count
  measured at 6.7; under motion the outer frames simply gate off. Auto
  Setup picks 7 only for noisy/severe clips from a steady camera. The
  effective-frames histogram grew 32→64 bins (the old range saturated at
  4.875 — a 5-frame-era assumption).
- **Deep Clean (Step 3 checkbox, off by default).** A gentle fine-NLM
  pre-pass at 0.6× h into a second buffer before the main spatial stage;
  the residual is re-measured on its output so the main pass adapts to
  what is actually left. Corrections hard-clamped to noise size —
  structure-safe by construction, hence a checkbox and not another slider.
  +1.6 dB on correlated/compressed noise, +2.8 dB on heavy iid; costs
  ~3 ms at HD. Auto Setup enables it for the severe class. The After
  Temporal view keeps showing the true temporal result.
- **Per-channel chroma profiles.** Every chroma estimate (input fine +
  coarse, temporal, residual ×2) now measures Cb and Cr separately; the
  temporal chroma gate runs per channel against its own knee, and the fine
  NLM band normalizes each channel by its own sigma inside a pooled weight
  (at equal sigmas this is algebraically the old formula — every pinned
  golden stayed within its ±0.02 dB tolerance, no re-pins). Blue-channel
  night noise is the payoff: on σ_Cb = 3×σ_Cr footage the pipeline beats
  the combined treatment by +1.25 dB and cleans the Cb plane 18% better.
  Locked profiles carry the pairs (old project locks load with Cr = Cb).
- **Auto Setup suggests a sampling region.** The Analysis line now reports
  the flattest patch it found (lowest local variance minus expected noise
  at that brightness) — "flattest patch at 38%, 15%". Report-only: your
  From Region rectangle is never moved.

### Faster
- **Lock fast path.** With the profile locked and no scope/analysis view
  showing the live measurement, the input-estimation dispatches are
  skipped entirely (they were output-inert in that state — everything the
  filters use is locked or re-measured from the merged image). HD 127→152
  fps at v3.2 defaults on an M1 Max; the residual measurement stays live.
- **On-chip tile caching for the fine NLM loop** (Metal threadgroup / CUDA
  shared / OpenCL local memory). The (2R+1)²×10 overlapping reads per pixel
  now come from a cooperatively-staged tile. Bit-identical math. Modest on
  Apple Silicon (3–11%, the system cache was absorbing most of it); aimed
  squarely at the discrete-GPU CUDA/OpenCL paths, pending Windows
  verification.

### Evaluated and rejected (recorded so the ideas stay dead)
- **Sparse candidate search above radius 6** (checkerboard + weight
  compensation): lost 0.29 dB at R=8 / 0.20 dB at R=10 against a 0.1 dB
  budget — halving the averaged samples raises output noise variance in
  exactly the regime big radii serve. Dropped per its gate.
- **Half-float tmp buffer:** failed GPU parity in 11/51 cases (max diff
  3× tolerance) for a ~1.5% win — the A2 tile had already absorbed the
  bandwidth this was meant to save. Rejected per its gate.

### Compatibility
- Old locked profiles (HUSHLOCK1) load with Cr = Cb; project files keep
  their Number of Frames indices; defaults at 3.2 settings stay within
  ±0.02 dB of the pinned goldens (bit-exact everywhere except the
  per-channel chroma estimator's sampling differences, documented in the
  B5 commit); tracking-off temporal path is bit-exact.
- Windows note: the CUDA kernel is a faithful textual port, still awaiting
  verification on NVIDIA hardware; Windows renders via OpenCL as before.


## 3.2.0 — 2026-07-15

Field-review release: lock/region behavior fixed, slow-motion ghosting
attacked at the algorithm level, and the spatial stage taught to see the
noise it was told didn't exist.

### Fixed
- **Lock Profile no longer throws away your region setup.** Locking (and
  Auto Setup) used to re-measure the WHOLE frame with stock settings —
  so the profile you dialed in from a region was silently replaced and
  "the noise came back". Both now measure exactly what you set up: region
  honored, and the lock freezes it. **Auto Profile Adjust now trims a
  locked profile live** (the lock stores the raw measurement; Adjust is
  applied at use time). Unlocked + From Region still re-measures per frame
  — that is what the region mode is for; lock is the "stop changing" switch,
  and the tooltips now say so.
- **Spatial NR under-reported on compressed footage.** The residual noise
  (what the spatial stage filters against) was measured with a fine-scale
  estimator only — the correlated, blotchy residue compression leaves after
  temporal averaging is LARGER than a pixel and was invisible to it, so the
  spatial stage idled while the eye still saw noise. The residual is now
  measured at two scales (the v1.1 input-estimator lesson, applied to the
  residual; coarse blocks even-aligned to match 4:2:0 chroma siting).
  Correlated-noise PSNR: +2.1 dB at defaults. Bit-exact on clean/iid
  footage where the fine estimator was already right.

### Added
- **Ghost Guard (Step 2, on by default) — the slow-motion smear fix.** The
  magnitude gate cannot distinguish subtle coherent motion from noise of the
  same size; that is exactly where slow-motion ghosting lives. Ghost Guard
  adds a second knee on the SIGNED patch mean: noise differences cancel
  toward zero when averaged with their signs, coherent motion does not — so
  creeping movement gets its weight cut ~3x earlier while pure noise sails
  through. Measured cost on static footage: ~0.01 dB. The Motion Threshold
  tooltip now spells out the ghost-fixing path (lower threshold, keep guard
  on, check the Motion Map scope — moving things should read red).
- **Clean Slate (All Off) button** — replaces Revert Auto Setup. Zeroes
  every processing control so the node passes the image through untouched:
  the fully-manual starting point the defaults never gave you. One undo
  restores what you had (Auto Setup's single-undo behavior is unchanged).
- **Global Blend** — a plain final crossfade original ↔ processed result,
  after everything else (grain included). 0 = untouched (and the node
  short-circuits to identity).

### Changed
- Auto Setup reports "(from region)" when it measured your region, and sets
  Ghost Guard on.
- The full-pipeline static golden re-pinned 40.927 → 40.948 dB (the
  two-scale residual measures the merge's own slight correlation); the
  spatial-stage goldens were unaffected (bit-exact).

## 3.1.0 — 2026-07-15

The power & transparency release, built from field feedback: "only when I
cranked every value could I get close to Resolve's built-in NR" — and the
cranked result was smoother *and blurrier*. v3.1 fixes both halves.

### More power
- **Detail Rescue (Step 3) — smoothing without blur.** After the spatial
  stage, anything it changed by more than a noise-sized amount is put back.
  Crank Luma Strength / EQ Fine as hard as you like: flats and faces go
  buttery while edges physically cannot blur (the worst case is bounded by
  construction). At full crank it recovers +3.5 dB on the test scene. Auto
  Setup raises it with the noise class.
- **Every range extended, and the dead zones fixed.** Strength to 3.0,
  spatial Luma/Chroma to 150, temporal strengths to 125 (matching neighbours
  may now outweigh the current frame), Motion Threshold to 150, Search Radius
  to 10, EQ Fine to 300 — and above 150 it now genuinely smooths harder
  (before, the blend saturated and the top of the slider was a silent no-op).
  Band sliders (Clumps / Stains / Color Blotches) reach 150: the applied
  amount caps at 1, the extra drive widens tolerances and reach instead
  (color blotch reach up to ~23 px, luma stains to ~47 px).
- **Auto Setup is much bolder.** New per-class tables use the extended
  ranges: on the synthetic suite, auto now beats stock defaults by +2.8 dB
  (moderate), +5.0 dB (noisy), +2.0 dB (blotchy) — every class gated by
  tests to never be worse. The temporal gate stays deliberately cautious
  (sweeps showed wide gates buy ~nothing on static shots and visibly hurt
  the moment anything moves).
- **Letterbox bars and crushed blacks no longer sabotage the measurement.**
  Exactly-flat samples (bit-identical to all neighbours — real sensor pixels
  never are) are skipped by every estimator. Letterboxed footage used to
  collapse the measured sigma toward zero and quietly under-filter.

### More transparency
- **Per-step scopes, as checkboxes where you need them.** Step 1 "Scope:
  Measurements", Step 2 "Scope: Motion Map" (a live picture-in-picture
  mini-map — green = frames stacking, red = motion-protected), Step 3
  "Scope: Noise EQ". Scopes draw over ANY view, combine freely, and never
  touch alpha.
- **The Noise EQ finally explains itself.** The four band sliders moved into
  a labeled "Noise EQ · Cut Noise By Size" subgroup with plain-size names
  (Fine Grain ~1 px / Clumps 3–8 px / Stains 16 px+ / Color Blotches), and
  the first time you touch one, the EQ scope pops up in the viewer: one lane
  per band, the bar is how much you're cutting, the amber line is how much
  noise was measured at that size.
- **Analysis panel rebuilt.** 2× text on an opaque panel (the old 1-px font
  strokes decimated into garbage below 100% viewer zoom), plain-English
  labels (INPUT / RESIDUAL / AVG FRAMES / MEASURING LIVE vs PROFILE LOCKED),
  and a sqrt-scaled histogram so one dominant bin can't flatten the display.
- **Noise Removed view no longer looks like mistakes.** The gain now rides
  the measured noise level (clean footage still reads) and a soft knee
  compresses big legitimate changes — bokeh lights and motion no longer slam
  into hard black/white blobs.
- Every tooltip rewritten down to one or two sentences.

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
