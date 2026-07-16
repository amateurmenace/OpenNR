# OpenNR changelog

## 3.7.0 — 2026-07-16

The handoff release: Hush can now hand its clean-confidence matte to the next
node in the chain, which is what makes the "clean early, reconstruct late"
pairing with Speak real instead of aspirational.

- **Export Clean Matte to Alpha** (Output group, default off). In the Result
  view, writes the v3.6 clean-confidence matte — `clamp((effN − 1) / 6, 0, 1)`,
  the exact calibration of the "Matte: Clean Confidence" view — into the output
  ALPHA channel, leaving RGB untouched. A downstream node (Speak's grain, or any
  qualifier) can key on where cleaning actually succeeded: high where the
  temporal stage averaged deep, low on the motion the gate protected. Replaces
  the incoming alpha while on, and the hint says so. Off by default, so every
  existing project renders bit-identically.
- The inspection views keep the true alpha; only the Result view exports.

## 3.6.0 — 2026-07-16

The texture-reconstruction release. A real-footage audit found the denoising
core is already strong — so this release builds the *other half*: after the
noise is gone, put the image's optical character back. There is now a stage
that ADDS high-frequency energy (nothing before this could), grain that is
matched to the noise that was removed instead of a flat mid-gray dither, a
texture path that no longer re-noises the shadows it fills, a fix for the
blotchy color speckle left deep in shadows, and a confidence matte that hands
downstream nodes a map of where the denoiser actually succeeded.

Everything here is off by default, so existing projects render bit-for-bit
identically until you reach for it. All new math is ported across the CPU
reference and the Metal / CUDA / OpenCL kernels at the usual ~2e-5 parity, and
every idea was CPU-prototyped and gated on real 4K night footage before it
shipped.

### Added
- **Optical Acutance** (Refine). The first stage in the whole pipeline that
  raises high-frequency energy: an edginess-gated high-pass on the cleaned
  luma, hard-clamped to the local 3×3 min/max so edges get their slope back
  with **zero ringing by construction** — sharpness that reads like a lens,
  not a halo. Gated to real edges by the noise field, so it never amplifies
  noise. Measured +7.9% edge slope on the reference clip; flats untouched.
- **Shadow Color Cleanup** (Refine, the WEAK-1 fix). The residual mid-frequency
  color speckle in deep shadows sits *above* the ~23 px reach of the chroma
  bands. A wide luma-guided chroma pass averages it — guided by the clean luma
  so it never crosses a real object edge, and kept to a moderate reach so the
  frame-scale lighting gradient (warm vs cool ambient over dark fabric) stays
  put. −70% shadow chroma speckle on the clip with the real color preserved.
- **Clean-Confidence matte** (view mode). Exports the per-pixel effective
  sample count — how deep the temporal stage averaged — calibrated into
  RGB + alpha. A downstream node can key shadow-lift, grain and local contrast
  off *where the denoiser succeeded* (high on static, low on the motion the
  gate had to protect). The complement of the existing noisiness matte.
- **Grain Fineness** (blue-noise spectrum). High-passes the grain toward the
  eye's contrast-sensitivity peak; at matched RMS it reads sharper and hides
  the plasticky look better than full-band grain.

### Changed
- **Film Grain is now reconstructed, not dithered.** Amplitude follows the
  measured brightness-noise curve (loud in shadows, quiet in highlights —
  where real noise and the plasticky waxing actually live) instead of a
  mid-gray parabola, and it is contrast-masked off edges so it never dithers
  over detail or acutance. (Only affects clips that had grain turned on.)
- **Luma Texture now cores the noise out first.** The re-injected texture is
  soft-thresholded at the input-noise scale, so it puts back real
  micro-structure without pushing the removed noise back into the shadows —
  measured shadow re-noising dropped from ~4.7× to ~1.2×. (Only affects clips
  that had Luma Texture turned on.)
- **Adaptive Strength** (effN-steered spatial): the spatial cleaning is now
  spent per pixel where frame-averaging couldn't help (moving subjects) and
  relaxed where it already worked, keeping the residual uniform across motion.
- **Auto Setup deep-tune**: the self-tuner now also runs a coordinate descent
  over the motion/detail/EQ axes and a chroma-SURE pass, with hold-out
  cross-validation so a tune that only fits the probe's own noise is rejected.

### Fixed
- **Chroma auto-tune on correlated speckle.** Plain MC-SURE is only a valid
  error estimate for white noise; on the spatially-correlated shadow chroma
  speckle it read the win backwards and de-tuned a good profile by ~0.9 dB.
  The probe is now matched to the measured chroma-noise correlation length, so
  the estimator becomes generalized SURE and tracks true error — turning that
  0.9 dB loss into a 3.0 dB gain on the correlated case.

## 3.5.0 — 2026-07-15

The compounding release: sequential renders now stack up past the frame
window, Auto Setup measures its own answers instead of trusting a table,
and auto-exposure wobble stops knocking frames out of the temporal stack.

### Added
- **Render Boost** (Step 2, off by default). The previous frame's TRUE
  temporal result joins the merge as one extra gated candidate, so static
  areas compound frame over frame during sequential playback and exports —
  up to about twice the effective frames (cap 12). The history is
  calibrated for what it is: a diff against an n-frame average has
  sigma·sqrt(1+1/n), not sqrt(2)·sigma, so its knee START and span tighten
  by sqrt((1+1/n)/2) — coherent-change detection gets SHARPER against the
  cleaner reference, not looser. Its gate applies QUADRATICALLY (a 60%
  match keeps 36%, it does not drag 12× harder — linear gating smeared a
  panning chain −0.21 dB; quadratic returns it to exactly the boost-off
  number), Ghost Guard gates its signed mean, and the previous frame's
  exposure offset applies to its luma. Hosts on all four backends cache
  the buffer per queue/stream and only declare it valid when the previous
  committed render was frame−1 at the same size with the same params hash:
  scrubbing, parameter changes, resizes, clip edges and reverse renders
  all fall back to the plain stack bit-exactly. Measured: a 6-frame static
  chain goes 40.93 → 42.87 dB (+1.94, effN 4.7 → 8.9); a panning chain is
  bit-identical to boost-off; +1.1 ms at HD.
- **Self-tuning Auto Setup.** The class table is a good prior — now it is
  only the prior. Auto Setup runs a Monte-Carlo SURE sweep (Stein's
  unbiased risk estimate, luma-only, deterministic fixed-seed Rademacher
  probes) of the FULL pipeline on a centre crop of the real 7-frame stack
  at the playhead: a 3×3 grid of (temporal luma, spatial luma) at ±25%
  around the table values, 18 crop denoises, ~3–5 s at button press. The
  statistically best pair overwrites those two sliders and the Analysis
  line gains "· tuned". On the synthetic suite the SURE argmin lands
  exactly on the true-MSE argmin; any fetch hiccup falls back to the
  table silently.

### Fixed
- **Exposure match** — auto-exposure steps, iris pulls and light flicker
  no longer gate the temporal stack out. Each neighbour frame gets a
  global exposure offset (median of signed luma diffs, 128-bin histogram,
  deadzone snaps to exactly zero so flicker-free footage is BIT-EXACT
  with 3.4), subtracted in every patch diff, the Ghost-Guard signed mean,
  the coarse search metric, the zapper's cross-frame tests and the
  blended sample. A +3% step at sigma 1.5% used to cost 2.38 dB; it now
  recovers to 50.93 dB against a 50.96 clean ceiling. Costs ~0.4 ms at
  HD, runs under the lock fast path and manual profiles too.

### Evaluated and rejected (recorded so the ideas stay dead)
- **Jensen mean-bound NLM pruning** (skip patch loops the mean already
  rules out): the bound never bites on texture and the bookkeeping costs
  2% net on M1 on flat AND textured bench scenes. Dead.
- **Ring-convergence early exit** (stop the NLM ring walk when a ring
  contributes nothing): −0.066 dB at R8 against a 0.02 dB budget. Dead.
- **Sub-pixel winner refinement** (parabola/bilinear refinement of the
  shift-search winner): a noise-level wash even with noise-fair selection
  scaled by sqrt(2/(1+s²)) — the hard knee already absorbs fractional
  misalignment — and unnormalized it LOSES 0.04 dB because bilinear
  smoothing biases the selection. Naive parabola sub-pel on |diff|
  profiles is biased ~t/2 by construction. Dead.
- **Multi-frame detail recovery** (super-resolution-style phase-aware
  accumulation): killed at the ceiling gate BEFORE building estimation.
  With GROUND-TRUTH half-pixel offsets on a 2×-truth synthetic, the raw
  +3.2 dB win over the merge vs an alias-free reference turned out to be
  reference-kernel mismatch: a plain 3×3 post-blur of the current merge
  beats the phase-aware accumulator by 1.91 dB at sigma 0 (the true
  phase-information gain is NEGATIVE), the sigma>0 deltas are 63-sample
  noise averaging that Render Boost already provides without resampling,
  and against the clip's native rendering the accumulator loses 3.75 dB
  at sigma 2%. Dead — and offset estimation was never built.

## 3.4.0 — 2026-07-15

The lock release: Lock Profile now freezes exactly the measurement you are
looking at, an Auto Region button places the sampling box for you, the box
finally shows whether it is live or frozen — and the temporal gate learns
what the spatial stage always knew: shadows are noisier than highlights.

### Fixed
- **Lock Profile locks what you SEE.** Since v3.0 the lock measured the
  playhead frame plus four frames spread across the whole clip (12/37/62/87%)
  and stored the median — but the live (unlocked) render measures the
  playhead frame. On anything but a locked-off shot the screen-fixed region
  covers different content at those distant times, so the median collapsed
  and locking visibly brought the noise back: place the box, watch the noise
  vanish, hit Lock, watch it return. v3.4 locks the playhead measurement —
  the locked result is identical to the live result at the moment you
  clicked. (Second half of the "lock brings the noise back" bug; v3.2 fixed
  the region-ignored half.)
- **Auto Setup in region mode** had the same disease: sigmas now come from
  the tracked playhead sweep (exactly what Lock stores), while the spread
  frames keep doing what they are actually good for — classifying the
  clip's motion, measured whole-frame.

### Added
- **Tracked lock sweep.** Lock doesn't stop at the playhead frame: it tracks
  your patch across ±4 neighbouring frames (SAD over sampled luma with a
  zero-motion prior, so featureless patches stay put instead of
  random-walking; drift-capped), measures the region at its tracked
  position, and lets a frame vote only when its measurement agrees with the
  playhead's (sigma within 1.6×, patch brightness within 0.13). A subject
  crossing the patch, an occlusion or a lost track simply drops out; worst
  case the lock is exactly the playhead measurement. The Analysis line
  reports it: "region at the playhead, 8 of 9 nearby frames agreed (patch
  tracked, drift 2 px) · noise 3.2%Y / 4.1%C".
- **Auto Region button.** One click scans the current frame for the flattest
  patch at your region size (scored as local variance minus the expected
  noise variance at that brightness, so dark noisy flats are not penalized),
  switches the profile to From Region and moves the yellow box there so you
  can see exactly what is being sampled — then you Lock, or run Auto Setup.
  The v3.3 scan was report-only; the button is the consent to move the box
  (Auto Setup still never touches it). One undo restores.
- **The box shows its state.** Live = yellow, corner handles, centre cross,
  draggable. Locked = dim ice blue with a padlock, handles gone, dragging
  inert, region sliders greyed out — the profile is frozen and where the box
  sits no longer matters. Unlock to move it. (Manual sigmas also grey out
  while locked — a locked profile overrides them too; Auto Profile Adjust
  stays live because it trims the locked values.)

### Improved
- **Brightness-aware temporal gate.** The estimator has always measured a
  16-bin noise-vs-brightness curve, and the spatial stage has always used
  it per pixel — but the temporal knee, Ghost Guard and the shift-search
  engagement thresholds ran on one flat σ_T. On real footage shadows run
  1.5–2.2× noisier than mids, so dark pixels' patch diffs read as motion
  and the gate starved the shadows of averaging exactly where noise is
  ugliest — while highlights got a knee that was too wide. v3.4 scales all
  of them by the centre pixel's gain: measured +1.35 dB in the shadows on a
  brightness-profiled scene (temporal-only, locked-sigma A/B isolating the
  mechanism), highlights unchanged, +1.0 dB on the same scene in motion.
  The 6σ firefly zapper deliberately stays unscaled (at that magnitude the
  0.6–2.2× curve cannot flip a genuine impulse). Manual profiles have flat
  curves — bit-exact with v3.3 there; all pinned goldens held without
  re-pinning. Ported to all four implementations (CPU, Metal, CUDA, OpenCL)
  with GPU parity green, including a sloped locked-gain-curve case.

### Notes
- The clean-class Auto-vs-defaults e2e margin widened 0.3 → 0.4 dB: with
  the gate now consuming gains, Auto's locked clip-averaged curve vs the
  defaults' live per-frame curve differ by estimation noise on clean
  footage — ~0.3 dB at a fully transparent >59 dB. Not a visible change.
- Lock data format (HUSHLOCK2) is unchanged; existing locked projects load
  as before. Old projects' locks were clip-spread medians and may read
  differently from what their region shows today — re-lock once to upgrade
  them to playhead semantics.

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
