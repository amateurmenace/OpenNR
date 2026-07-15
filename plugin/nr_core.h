// OpenNR — free spatio-temporal noise reduction for DaVinci Resolve (OpenFX)
// nr_core.h — reference CPU implementation of the full algorithm (v3).
//
// This header is the single source of truth for the math. The Metal / CUDA /
// OpenCL kernels are line-by-line ports of these functions; the test harness
// (test/test_denoise.cpp) validates this implementation against synthetic
// footage with known noise, so any change here must keep the tests passing.
//
// Pipeline per rendered frame:
//   1. estimateInput()   — noise profiling of the source frame: temporal-diff
//      (primary, correlation-immune, motion-robust), fine+coarse Laplacian,
//      PLUS a 16-bin brightness-dependent profile (shadows are noisier than
//      highlights; each bin gets its own gain on the base sigma).
//   2. temporalMerge()   — motion-adaptive averaging, hard-knee gate (weight
//      is exactly zero past the motion threshold; no ghosting by design).
//      v3: a firefly zapper clips single-frame impulses to the 3-frame
//      temporal median first, and shift-search matching ("Motion Tracking")
//      re-aims each neighbour patch at the best of 9 candidate offsets before
//      gating, so slow pans keep their temporal averaging.
//   3. estimateResidual()— the noise that is ACTUALLY left after temporal
//      merging is measured again from the merged image. The spatial stage
//      filters against this measured residual, not a theoretical prediction —
//      compression correlates noise across frames, so temporal averaging
//      removes less than sqrt(N) and predictions undershoot.
//   4. spatialNLM()      — the v3 Noise EQ: a fine band (sigma-adaptive
//      NLM/bilateral), a medium band (sparse ring average of 2x2 block means,
//      2-6 px reach), and a coarse band — the blotch pass, which keeps its
//      v2.1 luma-guided chroma path exactly and gains an optional luma
//      component with band-scaled tolerance.
//   5. refine()          — finishing: shadow desaturation (sat-vs-lum curve),
//      luma texture re-injection, gradient-aware debanding (threshold-gated
//      ring smoothing + triangular micro-dither), and synthesized film grain.
//   6. Output assembly   — result / split / input / after-temporal / noise
//      removed / analysis HUD / temporal activity / SNR heat map / noise
//      matte (noise-dominance map in RGB and alpha, for keying downstream).
//
// v3 also adds profile locking: a snapshot of the measured input profile
// (both sigma pairs + 32 brightness gains) can be passed back in through the
// lock* params; estimation still runs (the HUD stays live) but the locked
// values are what the filters use.
//
// v3.1 adds Detail Rescue (a hard bound on the fine band's correction, so
// cranked strengths smooth flats without blurring structure), overshoot
// semantics for the band sliders (past 100 the tolerances and reach widen
// while the applied amount caps at 1), per-step scope overlays (measurement
// HUD, Noise EQ panel, motion mini-map — composable over any view, never
// written into alpha), an exactly-flat-sample skip in every estimator
// (letterbox bars and crushed blacks no longer collapse the measurement),
// and a noise-adaptive soft-knee Noise Removed view.
//
// Strength (master) semantics: below 1 it scales the blend amounts; above 1
// the blends stay at their slider values and the filter strengths and the
// temporal knee widen instead — "relax what counts as noise".
//
// All buffers are packed RGBA float32, row-major, stride = width*4 floats.
//
// MIT License — see LICENSE.

#ifndef OPENNR_NR_CORE_H
#define OPENNR_NR_CORE_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

namespace nrcore {

// ---------------------------------------------------------------------------
// Parameters (already normalized: sliders /100, sigmas in signal units 0..1)
// ---------------------------------------------------------------------------
struct Params {
    int   profileSource  = 0;      // 0 auto whole frame, 1 auto from region, 2 manual
    float sigmaY         = 0.02f;
    float sigmaC         = 0.02f;
    float profileAdjust  = 1.0f;
    float regionCX       = 0.5f;
    float regionCY       = 0.5f;
    float regionSize     = 0.25f;

    int   enableTemporal = 1;
    int   temporalFrames = 3;      // 3, 5 or 7
    float temporalLuma   = 0.6f;
    float temporalChroma = 0.8f;
    float motionThresh   = 0.4f;

    int   enableSpatial  = 1;
    int   spatialMode    = 1;      // 0 faster (bilateral), 1 better (NLM)
    int   spatialRadius  = 3;      // 1..8
    float spatialLuma    = 0.6f;
    float spatialChroma  = 1.0f;
    float preserveDetail = 0.35f;
    float chromaBlotch   = 0.25f;  // 0..1 -> large-radius chroma pass

    int   enableRefine   = 1;
    float shadowDesat    = 0.0f;   // 0..1
    float desatRange     = 0.15f;  // luma pivot
    float lumaTexture    = 0.0f;   // 0..1 residual luma re-add
    float grainAmount    = 0.0f;   // 0..1
    float grainSize      = 1.6f;   // 1..4 px
    float grainChroma    = 0.25f;  // 0..1
    int   frameIndex     = 0;

    float master         = 1.0f;   // 0..2
    int   viewMode       = 0;      // see view list above

    // ---- v3 ----
    int   motionTracking = 1;      // temporal shift-search matching
    int   fireflyRemoval = 1;      // 3-frame temporal median impulse clip
    float eqFine         = 1.0f;   // Noise EQ: fine band gain 0..3 (1 = v2.1;
                                   // >1 also widens the NLM h)
    float eqMedium       = 0.0f;   // Noise EQ: medium band amount 0..1.5
                                   // (>1 = wider tolerance, amount caps at 1)
    float eqCoarse       = 0.0f;   // Noise EQ: coarse band LUMA amount 0..1.5
    float deband         = 0.0f;   // gradient-aware debanding 0..1
    int   profileLocked  = 0;      // 1 = use lock* values instead of measuring
    float lockSY         = 0.02f;  // locked input profile (spatial pair)
    float lockSC         = 0.02f;
    float lockTY         = 0.02f;  // locked temporal pair
    float lockTC         = 0.02f;
    float lockGainY[16]  = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    float lockGainC[16]  = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

    // ---- v3.1 ----
    float detailRescue   = 0.0f;   // 0..1: after the fine band, restore any
                                   // change larger than a noise-sized amount
                                   // (crank strengths for smoothness, rescue
                                   // pulls real structure back out; 0 = off,
                                   // bit-exact v3.0 blend)
    int   scopeMeasure   = 0;      // overlay: Noise Analysis panel (top-left)
    int   scopeMotion    = 0;      // overlay: temporal-activity mini map (b-r)
    int   scopeEq        = 0;      // overlay: Noise EQ panel (top-right)

    // ---- v3.2 ----
    int   ghostGuard     = 1;      // temporal coherence gate: noise diffs
                                   // cancel when averaged SIGNED, subtle
                                   // coherent motion does not — catch the
                                   // slow-motion ghosting the magnitude gate
                                   // cannot see
    float globalBlend    = 1.0f;   // 0..1 final crossfade original -> result

    // ---- v3.3 ----
    int   deepClean      = 0;      // fine-NLM pre-pass at 0.6h before the
                                   // main spatial stage (see deepCleanPass)
    float lockSCr        = 0.02f;  // B5 per-channel chroma: lockSC/lockTC
    float lockTCr        = 0.02f;  // hold Cb, these hold Cr (old locks load
                                   // with Cr = Cb)
};

// Sigmas and everything the analysis HUD displays.
struct Stats {
    float sy = 0.02f, sc = 0.02f;      // input spatial-family sigmas
    float ty = 0.02f, tc = 0.02f;      // temporal-gating sigmas
    float ry = 0.02f, rc = 0.02f;      // measured residual after temporal
    // v3.3 B5: the chroma sigmas above are the Cb/Cr pair MEANS (display,
    // band normalizers, zapper); the gates and NLM h use the split pairs.
    float scb = 0.02f, scr = 0.02f;    // input chroma, per channel
    float tcb = 0.02f, tcr = 0.02f;    // temporal chroma, per channel
    float rcb = 0.02f, rcr = 0.02f;    // residual chroma, per channel
    float gainY[16];                   // brightness-dependent sigma gains
    float gainC[16];
    float effNMed = 1.0f;              // median effective sample count
    uint32_t histY[256] = {0};         // fine luma |laplacian| histogram
    uint32_t medBinY = 0;
    uint32_t histMax = 1;
    int hadTemporal = 0;
    // v3: raw estimator components for the clip analyzer (CPU-side only —
    // no GPU consumer, so these have no kernel mirror).
    float fineY = 0.02f, fineC = 0.02f;      // fine |Laplacian| estimates
    float coarseY = 0.02f, coarseC = 0.02f;  // coarse (2x2 block) estimates
    float motionRatio = 1.0f;                // calibrated tdiff median/Q20
                                             // ratio: ~1 static, >1 motion
    Stats() { for (int i = 0; i < 16; ++i) { gainY[i] = 1.0f; gainC[i] = 1.0f; } }
};

// ---------------------------------------------------------------------------
// Calibration constants (derivations in comments; verified by the test suite)
// ---------------------------------------------------------------------------
static const float kMedianCal   = 1.0f / (6.0f * 0.674490f);      // Immerkaer, median
static const float kQ35Cal      = 1.0f / (6.0f * 0.453762f);      // Immerkaer, Q35 (texture-robust)
static const float kMedianCalT  = 1.0f / (0.674490f * 1.414214f); // |frame diff| median
static const float kQ20CalT     = 1.0f / (0.253347f * 1.414214f); // |frame diff| Q20
static const float kAbsDiffBias = 1.128379f;                      // E|a-b| = 2 sigma/sqrt(pi)
static const float kNlmHLuma    = 1.15f;
static const float kNlmHChroma  = 2.20f;

static const int   kHistBins    = 256;
static const float kHistScaleY  = 512.0f;
static const float kHistScaleC  = 1024.0f;
static const int   kLumaBins    = 16;
static const int   kLumaSub     = 64;
static const float kLumaSubScaleY = 128.0f;  // |lap| in [0, 0.5) over 64 bins
static const float kLumaSubScaleC = 256.0f;  // |lap| in [0, 0.25)
static const float kSigmaMin    = 1e-4f;
static const float kSigmaMax    = 0.25f;
// v3.5 P1 exposure match: per-neighbour global luma offset, estimated as the
// median of signed diffs over a stride-4 grid (128 bins across +/-0.25).
// Offsets inside the deadzone snap to EXACTLY zero, so flicker-free footage
// is bit-exact with the uncompensated path; 1.5 bins covers the half-bin
// decode bias plus estimator noise.
static const int   kExpBins  = 128;
static const float kExpScale = 256.0f;
static const float kExpDead  = 0.006f;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int   clampi(int v, int lo, int hi)       { return v < lo ? lo : (v > hi ? hi : v); }
static inline float smooth01(float t) { t = clampf(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }

static inline void rgb2ycc(float r, float g, float b, float& y, float& cb, float& cr)
{
    y  = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    cb = (b - y) / 1.8556f;
    cr = (r - y) / 1.5748f;
}

static inline void ycc2rgb(float y, float cb, float cr, float& r, float& g, float& b)
{
    r = y + 1.5748f * cr;
    b = y + 1.8556f * cb;
    g = (y - 0.2126f * r - 0.0722f * b) / 0.7152f;
}

static inline void sampleYCC(const float* img, int W, int H, int x, int y,
                             float& Y, float& Cb, float& Cr)
{
    x = clampi(x, 0, W - 1);
    y = clampi(y, 0, H - 1);
    const float* p = img + (static_cast<size_t>(y) * W + x) * 4;
    rgb2ycc(p[0], p[1], p[2], Y, Cb, Cr);
}

static inline void blockMeanYCC(const float* img, int W, int H, int x, int y,
                                float& Y, float& Cb, float& Cr)
{
    float ya, cba, cra;
    Y = Cb = Cr = 0.0f;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            sampleYCC(img, W, H, x + dx, y + dy, ya, cba, cra);
            Y += ya; Cb += cba; Cr += cra;
        }
    Y *= 0.25f; Cb *= 0.25f; Cr *= 0.25f;
}

// tmp buffer accessor (Y', Cb', Cr', effN)
static inline const float* tmpAt(const float* tmp, int W, int H, int x, int y)
{
    x = clampi(x, 0, W - 1);
    y = clampi(y, 0, H - 1);
    return tmp + (static_cast<size_t>(y) * W + x) * 4;
}

static inline float med3(float a, float b, float c)
{
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

static inline void sort2(float& a, float& b)
{
    const float t = std::min(a, b);
    b = std::max(a, b);
    a = t;
}

// median of 9 via the standard 19-exchange network (Smith 1996)
static inline float med9(const float* v)
{
    float p0 = v[0], p1 = v[1], p2 = v[2], p3 = v[3], p4 = v[4],
          p5 = v[5], p6 = v[6], p7 = v[7], p8 = v[8];
    sort2(p1, p2); sort2(p4, p5); sort2(p7, p8);
    sort2(p0, p1); sort2(p3, p4); sort2(p6, p7);
    sort2(p1, p2); sort2(p4, p5); sort2(p7, p8);
    sort2(p0, p3); sort2(p5, p8); sort2(p4, p7);
    sort2(p3, p6); sort2(p1, p4); sort2(p2, p5);
    sort2(p4, p7); sort2(p4, p2); sort2(p6, p4);
    sort2(p4, p2);
    return p4;
}

// 2x2 block mean of the tmp buffer (already YCC) — the sampling primitive of
// the medium band and the debander; block means suppress the fine band so
// similarity tests respond to structure at the band's own scale.
static inline void blockMeanTmp(const float* tmp, int W, int H, int x, int y,
                                float& Y, float& Cb, float& Cr)
{
    Y = Cb = Cr = 0.0f;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            const float* t = tmpAt(tmp, W, H, x + dx, y + dy);
            Y += t[0]; Cb += t[1]; Cr += t[2];
        }
    Y *= 0.25f; Cb *= 0.25f; Cr *= 0.25f;
}

// 4x4 block mean of tmp, centred-ish on (x,y) — the coarse band's domain.
static inline void blockMean4Tmp(const float* tmp, int W, int H, int x, int y,
                                 float& Y, float& Cb, float& Cr)
{
    Y = Cb = Cr = 0.0f;
    for (int dy = -1; dy < 3; ++dy)
        for (int dx = -1; dx < 3; ++dx) {
            const float* t = tmpAt(tmp, W, H, x + dx, y + dy);
            Y += t[0]; Cb += t[1]; Cr += t[2];
        }
    Y *= 0.0625f; Cb *= 0.0625f; Cr *= 0.0625f;
}

// Sparse ring directions shared by the medium band, the coarse (blotch) band
// and the debander.
static const float kDirX[8] = { 1, 0, -1, 0, 0.7071f, -0.7071f, -0.7071f, 0.7071f };
static const float kDirY[8] = { 0, 1, 0, -1, 0.7071f, 0.7071f, -0.7071f, -0.7071f };

// v3.3 B1 hierarchical shift search. Coarse level: step-4 nodes covering a
// +/-4 box plus the +/-8 axes and corners, scored on 2x2 block means (blocks
// low-pass the comparison so a 4 px step sees large-scale structure instead
// of aliasing on fine texture). Refine level: a converging +/-1 box walk (up
// to 2 steps) around the winner — every shift within Chebyshev distance 2 of
// a node is recovered exactly, i.e. all of |shift| <= 6 plus the 8 px axes/
// corners (effective reach ~6-8 px). The walk also starts from the unshifted
// match when the coarse level finds nothing, so the old +/-1 and +/-2
// slow-drift shifts stay reachable; two steps (not more) keeps the noisy-
// score selection bias on static/drift footage at the v3.2 level. The 1%
// acceptance margin guards every step (a shift must be clearly better to
// win), exactly as the v3 single-level search.
static const int kCoarseX[16] = { 4, -4, 0,  0, 4, -4,  4, -4, 8, -8, 0,  0, 8, -8,  8, -8 };
static const int kCoarseY[16] = { 0,  0, 4, -4, 4,  4, -4, -4, 0,  0, 8, -8, 8,  8, -8, -8 };
static const int kRefX[8] = { 1, -1, 0,  0, 1, -1,  1, -1 };
static const int kRefY[8] = { 0,  0, 1, -1, 1,  1, -1, -1 };

// Mean |3x3 patch difference| of neighbour frame f shifted by (ox,oy) against
// the current frame's patch (cy9/ccb9/ccr9); also returns the SIGNED mean
// luma difference (v3.2 Ghost Guard: noise diffs cancel when averaged signed,
// coherent motion/lighting change does not) and the shifted centre sample for
// blending. At (0,0) the magnitude term is exactly the v2.1 matching term.
// expOff (v3.5 P1) is the neighbour's global exposure offset: subtracted
// from its luma everywhere — diffs, signed mean AND the returned sample —
// so a matched patch under flicker reads (and blends) like a static one.
static inline void patchDiff(const float* f, int W, int H, int x, int y,
                             int ox, int oy, float expOff,
                             const float* cy9, const float* ccb9, const float* ccr9,
                             float& dY, float& dCb, float& dCr, float& sdY,
                             float& fy, float& fcb, float& fcr)
{
    dY = 0.0f; dCb = 0.0f; dCr = 0.0f; sdY = 0.0f; fy = 0.0f; fcb = 0.0f; fcr = 0.0f;
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx, ++i) {
            float vy, vcb, vcr;
            sampleYCC(f, W, H, x + ox + dx, y + oy + dy, vy, vcb, vcr);
            vy -= expOff;
            if (i == 4) { fy = vy; fcb = vcb; fcr = vcr; }
            dY += std::fabs(vy - cy9[i]);
            sdY += (vy - cy9[i]);
            // v3.3 B5: per-channel chroma diffs — each channel gets its own
            // knee, so heavy Cb noise cannot blow the gate for a clean Cr
            dCb += std::fabs(vcb - ccb9[i]);
            dCr += std::fabs(vcr - ccr9[i]);
        }
    }
    dY *= (1.0f / 9.0f);
    dCb *= (1.0f / 9.0f);
    dCr *= (1.0f / 9.0f);
    sdY *= (1.0f / 9.0f);
}

// v3.3 B1: coarse matching score for the hierarchical shift search — mean
// |2x2-block-mean luma difference| of a 3x3 stride-2 block patch (6x6 px
// support) of neighbour frame f shifted by (ox,oy), against the centre's
// own block patch cb9. Luma only: selection follows the luma match exactly
// like the refine level.
static inline float coarseDiff(const float* f, int W, int H, int x, int y,
                               int ox, int oy, float expOff, const float* cb9)
{
    float d = 0.0f;
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i) {
            float bY, bCb, bCr;
            blockMeanYCC(f, W, H, x + ox + dx * 2, y + oy + dy * 2, bY, bCb, bCr);
            d += std::fabs(bY - expOff - cb9[i]);
        }
    return d * (1.0f / 9.0f);
}

// Integer hash noise (identical across CPU/Metal/CUDA/OpenCL: pure uint32 math)
static inline float hashNoise(uint32_t ix, uint32_t iy, uint32_t f, uint32_t ch)
{
    uint32_t h = ix * 374761393u + iy * 668265263u + f * 2246822519u + ch * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (static_cast<float>(h & 0xFFFFFFu) / 16777215.0f) * 2.0f - 1.0f;
}

// Smooth value noise in [-1,1] at frequency 1/size px, per frame & channel.
static inline float valueNoise(float x, float y, float size, uint32_t f, uint32_t ch)
{
    const float gx = x / size, gy = y / size;
    const int ix = static_cast<int>(std::floor(gx)), iy = static_cast<int>(std::floor(gy));
    float fx = gx - ix, fy = gy - iy;
    fx = fx * fx * fx * (fx * (fx * 6.0f - 15.0f) + 10.0f);  // quintic
    fy = fy * fy * fy * (fy * (fy * 6.0f - 15.0f) + 10.0f);
    const float n00 = hashNoise(static_cast<uint32_t>(ix),     static_cast<uint32_t>(iy),     f, ch);
    const float n10 = hashNoise(static_cast<uint32_t>(ix + 1), static_cast<uint32_t>(iy),     f, ch);
    const float n01 = hashNoise(static_cast<uint32_t>(ix),     static_cast<uint32_t>(iy + 1), f, ch);
    const float n11 = hashNoise(static_cast<uint32_t>(ix + 1), static_cast<uint32_t>(iy + 1), f, ch);
    return (n00 + (n10 - n00) * fx) + ((n01 + (n11 - n01) * fx) - (n00 + (n10 - n00) * fx)) * fy;
}

// ---------------------------------------------------------------------------
// Stage 1 — input noise profiling
// ---------------------------------------------------------------------------
// v3: a locked profile overrides the measured sigmas and gains, but the
// measurement still runs so the HUD histogram stays live.
// v3.2: the lock stores the RAW measurement; Auto Profile Adjust is applied
// here, at use time, so the trim slider keeps working on a locked profile.
static inline void applyLockedProfile(const Params& p, Stats& out)
{
    if (p.profileLocked == 0)
        return;
    const float adjL = clampf(p.profileAdjust, 0.25f, 6.0f);
    out.sy = clampf(p.lockSY * adjL, kSigmaMin, kSigmaMax);
    out.scb = clampf(p.lockSC * adjL, kSigmaMin, kSigmaMax);
    out.scr = clampf(p.lockSCr * adjL, kSigmaMin, kSigmaMax);
    out.ty = clampf(p.lockTY * adjL, kSigmaMin, kSigmaMax);
    out.tcb = clampf(p.lockTC * adjL, kSigmaMin, kSigmaMax);
    out.tcr = clampf(p.lockTCr * adjL, kSigmaMin, kSigmaMax);
    out.sc = 0.5f * (out.scb + out.scr);
    out.tc = 0.5f * (out.tcb + out.tcr);
    for (int b = 0; b < kLumaBins; ++b) {
        out.gainY[b] = clampf(p.lockGainY[b], 0.6f, 2.2f);
        out.gainC[b] = clampf(p.lockGainC[b], 0.6f, 2.2f);
    }
}

inline void estimateInput(const float* rgba, const float* diffPartner,
                          int W, int H, const Params& p, Stats& out)
{
    out.sy = out.ty = out.ry = clampf(p.sigmaY, kSigmaMin, kSigmaMax);
    out.sc = out.tc = out.rc = clampf(p.sigmaC, kSigmaMin, kSigmaMax);
    out.scb = out.scr = out.tcb = out.tcr = out.rcb = out.rcr = out.sc;
    out.fineY = out.coarseY = out.sy;   // per-band components default to the
    out.fineC = out.sc;                 // base sigmas (manual mode: flat EQ)
    if (p.profileSource == 2 && p.profileLocked == 0) {  // manual: sigmas literal, gains flat
        return;
    }
    // v3.3 lock fast path: with the profile locked and nothing on screen
    // showing the live measurement (Measurements scope, EQ scope, Analysis
    // view), this whole pass is output-inert — every value the filters use
    // is either locked (a pure function of the params) or re-measured from
    // the merged image by estimateResidual, which still runs. Skip it; the
    // GPU hosts skip the NoiseEst/FinalizeStats dispatches under the same
    // condition, and their kernels compute the locked values from the params
    // with exactly applyLockedProfile's arithmetic.
    if (p.profileLocked != 0 && p.scopeMeasure == 0 && p.scopeEq == 0 && p.viewMode != 5) {
        applyLockedProfile(p, out);
        return;
    }

    // v3.3 B5: chroma histograms are per channel — blue-channel night noise
    // puts several times more energy on Cb than Cr, and a combined median
    // splits the difference badly for both
    std::vector<uint32_t> hYf(kHistBins, 0), hCfb(kHistBins, 0), hCfr(kHistBins, 0);
    std::vector<uint32_t> hY2(kHistBins, 0), hC2b(kHistBins, 0), hC2r(kHistBins, 0);
    std::vector<uint32_t> hYt(kHistBins, 0), hCtb(kHistBins, 0), hCtr(kHistBins, 0);
    std::vector<uint32_t> lumY(kLumaBins * kLumaSub, 0), lumC(kLumaBins * kLumaSub, 0);

    int x0 = 1, x1 = W - 1, y0 = 1, y1 = H - 1;
    if (p.profileSource == 1) {
        const float rHalf = 0.5f * p.regionSize * static_cast<float>(std::min(W, H));
        const float cx = p.regionCX * W, cy = p.regionCY * H;
        x0 = clampi(static_cast<int>(cx - rHalf), 1, W - 1);
        x1 = clampi(static_cast<int>(cx + rHalf), 1, W - 1);
        y0 = clampi(static_cast<int>(cy - rHalf), 1, H - 1);
        y1 = clampi(static_cast<int>(cy + rHalf), 1, H - 1);
    }
    const int xs = x0 + ((x0 & 1) ? 0 : 1);
    const int ys = y0 + ((y0 & 1) ? 0 : 1);

    uint64_t totalF = 0, totalT = 0, total2 = 0;
    for (int y = ys; y < y1; y += 2) {
        for (int x = xs; x < x1; x += 2) {
            float Y[9], Cb[9], Cr[9];
            int i = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx, ++i)
                    sampleYCC(rgba, W, H, x + dx, y + dy, Y[i], Cb[i], Cr[i]);

            const float lapY  = 4.0f * Y[4]  - 2.0f * (Y[1] + Y[3] + Y[5] + Y[7])   + (Y[0] + Y[2] + Y[6] + Y[8]);
            const float lapCb = 4.0f * Cb[4] - 2.0f * (Cb[1] + Cb[3] + Cb[5] + Cb[7]) + (Cb[0] + Cb[2] + Cb[6] + Cb[8]);
            const float lapCr = 4.0f * Cr[4] - 2.0f * (Cr[1] + Cr[3] + Cr[5] + Cr[7]) + (Cr[0] + Cr[2] + Cr[6] + Cr[8]);

            // v3.1: exactly-flat samples (letterbox bars, crushed blacks,
            // synthetic graphics) carry no noise evidence — real sensor
            // pixels are never bit-identical to all eight neighbours. Enough
            // of them collapses every median; skip the sample entirely.
            if (lapY == 0.0f && lapCb == 0.0f && lapCr == 0.0f)
                continue;

            hYf[clampi(static_cast<int>(std::fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)]++;
            hCfb[clampi(static_cast<int>(std::fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)]++;
            hCfr[clampi(static_cast<int>(std::fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)]++;
            totalF++;

            // brightness-dependent profile
            const int lb = clampi(static_cast<int>(Y[4] * kLumaBins), 0, kLumaBins - 1);
            lumY[lb * kLumaSub + clampi(static_cast<int>(std::fabs(lapY)  * kLumaSubScaleY), 0, kLumaSub - 1)]++;
            lumC[lb * kLumaSub + clampi(static_cast<int>(std::fabs(lapCb) * kLumaSubScaleC), 0, kLumaSub - 1)]++;
            lumC[lb * kLumaSub + clampi(static_cast<int>(std::fabs(lapCr) * kLumaSubScaleC), 0, kLumaSub - 1)]++;

            if (diffPartner) {
                float py, pcb, pcr;
                sampleYCC(diffPartner, W, H, x, y, py, pcb, pcr);
                hYt[clampi(static_cast<int>(std::fabs(py - Y[4])   * kHistScaleY), 0, kHistBins - 1)]++;
                hCtb[clampi(static_cast<int>(std::fabs(pcb - Cb[4]) * kHistScaleC), 0, kHistBins - 1)]++;
                hCtr[clampi(static_cast<int>(std::fabs(pcr - Cr[4]) * kHistScaleC), 0, kHistBins - 1)]++;
                totalT++;
            }

            if ((((x - 1) >> 1) & 1) == 0 && (((y - 1) >> 1) & 1) == 0) {
                float bY[9], bCb[9], bCr[9];
                i = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx, ++i)
                        blockMeanYCC(rgba, W, H, x + dx * 2, y + dy * 2, bY[i], bCb[i], bCr[i]);
                const float lY  = 4.0f * bY[4]  - 2.0f * (bY[1] + bY[3] + bY[5] + bY[7])   + (bY[0] + bY[2] + bY[6] + bY[8]);
                const float lCb = 4.0f * bCb[4] - 2.0f * (bCb[1] + bCb[3] + bCb[5] + bCb[7]) + (bCb[0] + bCb[2] + bCb[6] + bCb[8]);
                const float lCr = 4.0f * bCr[4] - 2.0f * (bCr[1] + bCr[3] + bCr[5] + bCr[7]) + (bCr[0] + bCr[2] + bCr[6] + bCr[8]);
                if (lY != 0.0f || lCb != 0.0f || lCr != 0.0f) {  // v3.1 flat-sample skip
                    hY2[clampi(static_cast<int>(std::fabs(lY)  * kHistScaleY), 0, kHistBins - 1)]++;
                    hC2b[clampi(static_cast<int>(std::fabs(lCb) * kHistScaleC), 0, kHistBins - 1)]++;
                    hC2r[clampi(static_cast<int>(std::fabs(lCr) * kHistScaleC), 0, kHistBins - 1)]++;
                    total2++;
                }
            }
        }
    }

    if (totalF < 64) {
        applyLockedProfile(p, out);
        return;
    }

    struct QR { float value; uint32_t bin; };
    auto histQuantile = [](const uint32_t* h, int n, uint64_t total, float scale,
                           uint64_t num, uint64_t den) {
        uint64_t cum = 0;
        const uint64_t target = (total * num + den - 1) / den;
        for (int b = 0; b < n; ++b) {
            cum += h[b];
            if (cum >= target)
                return QR{ (static_cast<float>(b) + 0.5f) / scale, static_cast<uint32_t>(b) };
        }
        return QR{ (static_cast<float>(n) - 0.5f) / scale, static_cast<uint32_t>(n - 1) };
    };

    const QR mYf = histQuantile(hYf.data(), kHistBins, totalF, kHistScaleY, 1, 2);
    const float syFine = mYf.value * kMedianCal;
    const float scbFine = histQuantile(hCfb.data(), kHistBins, totalF, kHistScaleC, 1, 2).value * kMedianCal;
    const float scrFine = histQuantile(hCfr.data(), kHistBins, totalF, kHistScaleC, 1, 2).value * kMedianCal;
    const float syCoarse = 2.0f * histQuantile(hY2.data(), kHistBins, total2, kHistScaleY, 1, 2).value * kMedianCal;
    const float scbCoarse = 2.0f * histQuantile(hC2b.data(), kHistBins, total2, kHistScaleC, 1, 2).value * kMedianCal;
    const float scrCoarse = 2.0f * histQuantile(hC2r.data(), kHistBins, total2, kHistScaleC, 1, 2).value * kMedianCal;

    const float lapSY = std::max(syFine, 0.9f * syCoarse);
    const float lapSCb = std::max(scbFine, 0.9f * scbCoarse);
    const float lapSCr = std::max(scrFine, 0.9f * scrCoarse);

    // v3: raw components for the clip analyzer (chroma: pair mean)
    out.fineY = syFine;   out.fineC = 0.5f * (scbFine + scrFine);
    out.coarseY = syCoarse; out.coarseC = 0.5f * (scbCoarse + scrCoarse);

    float ty = lapSY, tcb = lapSCb, tcr = lapSCr;
    if (diffPartner && totalT >= 64) {
        const float medTY = histQuantile(hYt.data(), kHistBins, totalT, kHistScaleY, 1, 2).value * kMedianCalT;
        const float q20TY = histQuantile(hYt.data(), kHistBins, totalT, kHistScaleY, 1, 5).value * kQ20CalT;
        const float medTCb = histQuantile(hCtb.data(), kHistBins, totalT, kHistScaleC, 1, 2).value * kMedianCalT;
        const float q20TCb = histQuantile(hCtb.data(), kHistBins, totalT, kHistScaleC, 1, 5).value * kQ20CalT;
        const float medTCr = histQuantile(hCtr.data(), kHistBins, totalT, kHistScaleC, 1, 2).value * kMedianCalT;
        const float q20TCr = histQuantile(hCtr.data(), kHistBins, totalT, kHistScaleC, 1, 5).value * kQ20CalT;
        const float candY = (medTY <= 1.4f * q20TY) ? medTY : q20TY;
        const float candCb = (medTCb <= 1.4f * q20TCb) ? medTCb : q20TCb;
        const float candCr = (medTCr <= 1.4f * q20TCr) ? medTCr : q20TCr;
        if (candY > 0.0015f && candY <= 3.5f * lapSY) ty = candY;
        if (candCb > 0.0015f && candCb <= 3.5f * lapSCb) tcb = candCb;
        if (candCr > 0.0015f && candCr <= 3.5f * lapSCr) tcr = candCr;
        out.hadTemporal = 1;
        // v3: both are calibrated sigma estimates, so the ratio is ~1 on
        // static noise and rises with motion (motion inflates the median
        // faster than the 20th percentile).
        out.motionRatio = medTY / std::max(q20TY, 1e-6f);
    }

    const float adj = clampf(p.profileAdjust, 0.25f, 6.0f);
    out.sy = clampf(std::max(lapSY, 0.85f * ty) * adj, kSigmaMin, kSigmaMax);
    out.scb = clampf(std::max(lapSCb, 0.85f * tcb) * adj, kSigmaMin, kSigmaMax);
    out.scr = clampf(std::max(lapSCr, 0.85f * tcr) * adj, kSigmaMin, kSigmaMax);
    out.ty = clampf(ty * adj, kSigmaMin, kSigmaMax);
    out.tcb = clampf(tcb * adj, kSigmaMin, kSigmaMax);
    out.tcr = clampf(tcr * adj, kSigmaMin, kSigmaMax);
    out.sc = 0.5f * (out.scb + out.scr);
    out.tc = 0.5f * (out.tcb + out.tcr);

    // Brightness-dependent gains, relative to the global fine medians.
    // Per-bin we use the 35th percentile rather than the median: a luma bin
    // can be dominated by legitimate texture, and Q35 stays anchored to the
    // flat (noise-set) pixels as long as they are at least ~a third of the bin.
    const float q35RefY = histQuantile(hYf.data(), kHistBins, totalF, kHistScaleY, 7, 20).value * kQ35Cal;
    // the brightness-gain curve stays combined (one curve for both chroma
    // channels); its reference is the mean of the per-channel quantiles
    const float q35RefC = 0.5f * (histQuantile(hCfb.data(), kHistBins, totalF, kHistScaleC, 7, 20).value +
                                  histQuantile(hCfr.data(), kHistBins, totalF, kHistScaleC, 7, 20).value) * kQ35Cal;
    float gy[kLumaBins], gc[kLumaBins];
    for (int b = 0; b < kLumaBins; ++b) {
        uint64_t cy = 0, cc = 0;
        for (int s2 = 0; s2 < kLumaSub; ++s2) { cy += lumY[b * kLumaSub + s2]; cc += lumC[b * kLumaSub + s2]; }
        // confidence-weighted: sparse bins (edge/transition luminances, mostly
        // structure) earn their deviation from neutral by sample count
        gy[b] = 1.0f; gc[b] = 1.0f;
        if (cy >= 200 && q35RefY > 1e-6f) {
            const float sb = histQuantile(&lumY[b * kLumaSub], kLumaSub, cy, kLumaSubScaleY, 7, 20).value * kQ35Cal;
            const float w = static_cast<float>(cy) / (static_cast<float>(cy) + 2000.0f);
            gy[b] = clampf(1.0f + w * (sb / q35RefY - 1.0f), 0.6f, 2.2f);
        }
        if (cc >= 200 && q35RefC > 1e-6f) {
            const float sb = histQuantile(&lumC[b * kLumaSub], kLumaSub, cc, kLumaSubScaleC, 7, 20).value * kQ35Cal;
            const float w = static_cast<float>(cc) / (static_cast<float>(cc) + 4000.0f);
            gc[b] = clampf(1.0f + w * (sb / q35RefC - 1.0f), 0.6f, 2.2f);
        }
    }
    // noise-vs-brightness curves are physically smooth: 3-tap smoothing
    for (int b = 0; b < kLumaBins; ++b) {
        const int b0 = clampi(b - 1, 0, kLumaBins - 1);
        const int b1 = clampi(b + 1, 0, kLumaBins - 1);
        out.gainY[b] = 0.25f * gy[b0] + 0.5f * gy[b] + 0.25f * gy[b1];
        out.gainC[b] = 0.25f * gc[b0] + 0.5f * gc[b] + 0.25f * gc[b1];
    }

    for (int b = 0; b < kHistBins; ++b) {
        out.histY[b] = hYf[b];
        out.histMax = std::max(out.histMax, hYf[b]);
    }
    out.medBinY = mYf.bin;

    applyLockedProfile(p, out);
}

// ---------------------------------------------------------------------------
// Stage 2 — motion-adaptive temporal merge (hard-knee gate)
// ---------------------------------------------------------------------------
inline void temporalMerge(const float* const frames[7], int W, int H,
                          const Params& p, const Stats& s, float* tmp)
{
    const float mLow  = std::min(p.master, 1.0f);
    const float mHigh = std::max(p.master, 1.0f);
    // v3.1: sliders reach 125 — a matching neighbour may outweigh the centre
    // (stronger averaging on static areas; the gate still zeroes mismatches)
    const float tL = clampf(p.temporalLuma   * mLow, 0.0f, 1.25f);
    const float tC = clampf(p.temporalChroma * mLow, 0.0f, 1.25f);
    const float thrMul = 0.4f + 2.6f * clampf(p.motionThresh, 0.0f, 1.5f)
                       + 0.8f * (mHigh - 1.0f);   // master>1 widens the knee
    // v3.3 B2: the stack grows to 7 frames (reach 3) for static heavy noise
    const int reach = (p.enableTemporal == 0) ? 0
                    : ((p.temporalFrames >= 7) ? 3 : (p.temporalFrames >= 5) ? 2 : 1);
    const float loY = kAbsDiffBias * s.ty, hiY = loY + thrMul * s.ty;
    const float loCb = kAbsDiffBias * s.tcb, hiCb = loCb + thrMul * s.tcb;
    const float loCr = kAbsDiffBias * s.tcr, hiCr = loCr + thrMul * s.tcr;
    const float invSpanY = 1.0f / (hiY - loY);
    const float invSpanCb = 1.0f / (hiCb - loCb);
    const float invSpanCr = 1.0f / (hiCr - loCr);
    // v3 shift search engages only once the unshifted match is well into the
    // gate. Below this the gate keeps most of its weight anyway, and pure
    // noise reaches here so rarely that the selection bias of picking the
    // minimum of nine noisy scores (which correlates the chosen sample with
    // the centre and inflates its gate) stays negligible on static footage.
    const bool  track = (p.motionTracking != 0);
    const float searchThresh = loY + 0.75f * (hiY - loY);
    // v3.2 Ghost Guard: a second knee on the SIGNED patch mean. Pure-noise
    // signed means cancel toward zero (sigma_mean = sqrt(2)sigma/3, so the
    // knee start 1.128sigma sits at ~2.4 sigma_mean — barely any pure-noise
    // pixels reach it), while subtle coherent motion — the slow-motion
    // smear the magnitude gate cannot see — pushes the signed mean straight
    // past it and gets its weight cut.
    const bool  guard = (p.ghostGuard != 0);
    const float loS = kAbsDiffBias * s.ty;
    const float invSpanS = 1.0f / (0.5f * thrMul * s.ty);
    // v3 firefly zapper: active only when the temporal stage itself is
    // (identity when master is 0 or the stage contributes nothing).
    const bool  zap = (reach >= 1) && (p.fireflyRemoval != 0) &&
                      (p.master > 0.0f) && (tL > 0.0f || tC > 0.0f);
    const float zapY = 6.0f * s.ty;
    const float zapC = 6.0f * s.tc;

    // v3.5 P1: per-neighbour global exposure offsets. An auto-exposure step
    // or light flicker shifts EVERY diff at once and gates whole frames out
    // of the stack (measured: a +3% step cost 2.4 dB at sigma 1.5%). The
    // median of signed luma diffs over a stride-4 grid reads the offset
    // robustly (motion is the minority and largely sign-symmetric); the
    // deadzone keeps flicker-free footage bit-exact. Duplicate frames at
    // clip edges read a half-bin median and snap to zero the same way.
    float expOff[7] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int k = 3 - reach; k <= 3 + reach; ++k) {
        if (k == 3)
            continue;
        uint32_t eh[kExpBins];
        for (int b = 0; b < kExpBins; ++b) eh[b] = 0;
        uint32_t etotal = 0;
        for (int ey = 1; ey < H - 1; ey += 4)
            for (int ex = 1; ex < W - 1; ex += 4) {
                float cyv, cbv, crv, nyv;
                sampleYCC(frames[3], W, H, ex, ey, cyv, cbv, crv);
                sampleYCC(frames[k], W, H, ex, ey, nyv, cbv, crv);
                eh[clampi(static_cast<int>((nyv - cyv + 0.25f) * kExpScale), 0, kExpBins - 1)]++;
                etotal++;
            }
        if (etotal >= 64) {
            uint32_t cum = 0;
            const uint32_t target = (etotal + 1) / 2;
            int mbin = kExpBins - 1;
            for (int b = 0; b < kExpBins; ++b) {
                cum += eh[b];
                if (cum >= target) { mbin = b; break; }
            }
            const float o = (static_cast<float>(mbin) + 0.5f) / kExpScale - 0.25f;
            if (std::fabs(o) >= kExpDead)
                expOff[k] = o;
        }
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float cy9[9], ccb9[9], ccr9[9];
            int i = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx, ++i)
                    sampleYCC(frames[3], W, H, x + dx, y + dy, cy9[i], ccb9[i], ccr9[i]);

            // Firefly zapper: a centre pixel is clipped to the 3-frame
            // temporal median only when three tests all agree it is a
            // single-frame impulse (hot pixel / sensor firefly):
            //   1. it deviates from the temporal median by >6 sigma_T;
            //   2. the two neighbour frames agree with each other
            //      (<=3 sigma_T apart) — under motion they disagree too;
            //   3. it is a spatial outlier within its own 3x3 patch — a
            //      fast-moving thin structure is temporally impulsive at a
            //      pixel, but spatially coherent, and must not be erased.
            if (zap) {
                float pY, pCb, pCr, nY, nCb, nCr;
                sampleYCC(frames[2], W, H, x, y, pY, pCb, pCr);
                sampleYCC(frames[4], W, H, x, y, nY, nCb, nCr);
                pY -= expOff[2];   // v3.5 P1: the zapper's cross-frame tests
                nY -= expOff[4];   // compare exposure-matched values too
                if (std::fabs(pY - nY) < 0.5f * zapY &&
                    0.5f * (std::fabs(pCb - nCb) + std::fabs(pCr - nCr)) < 0.5f * zapC &&
                    std::fabs(cy9[4] - med9(cy9)) > 0.5f * zapY) {
                    const float mY  = med3(pY, cy9[4], nY);
                    const float mCb = med3(pCb, ccb9[4], nCb);
                    const float mCr = med3(pCr, ccr9[4], nCr);
                    if (std::fabs(cy9[4] - mY) > zapY ||
                        0.5f * (std::fabs(ccb9[4] - mCb) + std::fabs(ccr9[4] - mCr)) > zapC) {
                        cy9[4] = mY; ccb9[4] = mCb; ccr9[4] = mCr;
                    }
                }
            }

            // v3.4: per-brightness gate calibration. The estimator measures a
            // 16-bin "shadows are noisier" gain curve and the spatial stage
            // has always used it per pixel — the temporal knees now do too:
            // every threshold scales with the centre pixel's gain, so shadows
            // (noisier) keep their averaging instead of reading as motion,
            // and highlights gate tighter instead of ghosting. Flat curves
            // (manual profile) make all of this exactly 1.0 — bit-exact with
            // the ungained knee. The 6-sigma zapper stays unscaled on
            // purpose: at that magnitude the curve's 0.6..2.2x cannot flip a
            // genuine impulse, and an impulse corrupts its own bin anyway.
            const int gLb = clampi(static_cast<int>(cy9[4] * static_cast<float>(kLumaBins)),
                                   0, kLumaBins - 1);
            const float gnY = s.gainY[gLb];
            const float gnC = s.gainC[gLb];
            const float invGnY = 1.0f / gnY;
            const float invGnC = 1.0f / gnC;

            float accY = cy9[4], accCb = ccb9[4], accCr = ccr9[4];
            float sumWY = 1.0f, sumWY2 = 1.0f, sumWCb = 1.0f, sumWCr = 1.0f;

            // v3.3 B1: the centre's coarse block patch for the hierarchical
            // search — built lazily once per pixel, shared by all neighbours
            float cb9[9];
            bool haveCb9 = false;

            for (int k = 3 - reach; k <= 3 + reach; ++k) {
                if (k == 3)
                    continue;
                const float* f = frames[k];

                const float oK = expOff[k];   // v3.5 P1
                float dY, dCb, dCr, sdY, fy9c, fcb9c, fcr9c;
                patchDiff(f, W, H, x, y, 0, 0, oK, cy9, ccb9, ccr9,
                          dY, dCb, dCr, sdY, fy9c, fcb9c, fcr9c);

                // v3.3 B1 hierarchical shift search (see kCoarseX above).
                // Every acceptance needs the 1% margin so the unshifted
                // patch wins ties and near-ties, and any shifted winner
                // keeps the tightened roll-off:
                // a shifted patch must match like a static one to be
                // trusted: the knee START stays (true rematches sit at
                // pure-noise diff levels and keep weight 1) but the
                // roll-off is ~1.7x steeper, so partial matches on periodic
                // texture — motion beyond the search reach — cannot blend
                // misaligned detail. Ghost Guard gates the WINNER's signed
                // mean, wherever the search lands.
                float shiftTight = 1.0f;
                if (track && dY > searchThresh * gnY) {
                    int wx = 0, wy = 0;
                    // The coarse level runs only when the unshifted match is
                    // FULLY outside the knee (dY > hiY): its weight is
                    // already zero, so there is nothing to lose by hunting
                    // far — and static/drift pixels, whose noise crosses
                    // searchThresh routinely but hiY rarely, never pay the
                    // 16-node block-mean sweep (measured: gating here
                    // returned the static-scene fps to the v3.2 level).
                    if (dY > hiY * gnY) {
                        if (!haveCb9) {
                            int i = 0;
                            for (int dy = -1; dy <= 1; ++dy)
                                for (int dx = -1; dx <= 1; ++dx, ++i) {
                                    float bCb, bCr;
                                    blockMeanYCC(frames[3], W, H, x + dx * 2, y + dy * 2,
                                                 cb9[i], bCb, bCr);
                                }
                            haveCb9 = true;
                        }
                        // coarse level: best step-4 node on block means
                        float bestC = coarseDiff(f, W, H, x, y, kCoarseX[0], kCoarseY[0], oK, cb9);
                        int bestOx = kCoarseX[0], bestOy = kCoarseY[0];
                        for (int c = 1; c < 16; ++c) {
                            const float d = coarseDiff(f, W, H, x, y, kCoarseX[c], kCoarseY[c], oK, cb9);
                            if (d < bestC) { bestC = d; bestOx = kCoarseX[c]; bestOy = kCoarseY[c]; }
                        }
                        // The coarse winner must survive the real patch
                        // metric — by 10%, much stricter than the walk's 1%:
                        // a genuine 4+ px re-aim scores several times better
                        // than the misaligned unshifted patch and clears any
                        // margin, while small flukes on noisy structure (the
                        // sub-pixel-drift regime, where no step-4 node is
                        // right) must not yank the match 4 px away. Measured
                        // on the drift regression: 1% lost 0.07 dB and 3%
                        // still 0.06 dB to exactly these jumps; 10% keeps
                        // the v3.2 number with the pan gains untouched.
                        float dY2, dCb2, dCr2, sd2, fy2, fcb2, fcr2;
                        patchDiff(f, W, H, x, y, bestOx, bestOy, oK,
                                  cy9, ccb9, ccr9, dY2, dCb2, dCr2, sd2, fy2, fcb2, fcr2);
                        if (dY2 < dY * 0.90f) {
                            dY = dY2; dCb = dCb2; dCr = dCr2; sdY = sd2;
                            fy9c = fy2; fcb9c = fcb2; fcr9c = fcr2;
                            wx = bestOx; wy = bestOy;
                            shiftTight = 1.0f / 0.6f;
                        }
                    }
                    // refine level: converging +/-1 walk around the winner
                    // (from the unshifted match when the coarse level failed)
                    for (int it = 0; it < 2; ++it) {
                        int nwx = wx, nwy = wy;
                        for (int c = 0; c < 8; ++c) {
                            const int tx = wx + kRefX[c], ty = wy + kRefY[c];
                            float dY2, dCb2, dCr2, sd2, fy2, fcb2, fcr2;
                            patchDiff(f, W, H, x, y, tx, ty, oK,
                                      cy9, ccb9, ccr9, dY2, dCb2, dCr2, sd2, fy2, fcb2, fcr2);
                            if (dY2 < dY * 0.99f) {
                                dY = dY2; dCb = dCb2; dCr = dCr2; sdY = sd2;
                                fy9c = fy2; fcb9c = fcb2; fcr9c = fcr2;
                                nwx = tx; nwy = ty;
                                shiftTight = 1.0f / 0.6f;
                            }
                        }
                        if (nwx == wx && nwy == wy)
                            break;
                        wx = nwx; wy = nwy;
                    }
                }

                float gY = 1.0f - smooth01((dY - loY * gnY) * invSpanY * invGnY * shiftTight);
                if (guard)
                    gY *= 1.0f - smooth01((std::fabs(sdY) - loS * gnY) * invSpanS * invGnY * shiftTight);
                // v3.3 B5: each chroma channel gets its own gate against its
                // own knee (both still slaved to the luma gate)
                const float gCb = 1.0f - smooth01((dCb - loCb * gnC) * invSpanCb * invGnC * shiftTight);
                const float gCr = 1.0f - smooth01((dCr - loCr * gnC) * invSpanCr * invGnC * shiftTight);
                const float wY = tL * gY;
                const float wCb = tC * gCb * gY;
                const float wCr = tC * gCr * gY;

                accY  += wY * fy9c;
                accCb += wCb * fcb9c;
                accCr += wCr * fcr9c;
                sumWY  += wY;
                sumWY2 += wY * wY;
                sumWCb += wCb;
                sumWCr += wCr;
            }

            float* t = tmp + (static_cast<size_t>(y) * W + x) * 4;
            t[0] = accY / sumWY;
            t[1] = accCb / sumWCb;
            t[2] = accCr / sumWCr;
            t[3] = (sumWY * sumWY) / sumWY2;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 3 — residual noise measurement on the temporal result
// ---------------------------------------------------------------------------
inline void estimateResidual(const float* tmp, int W, int H, const Params& p, Stats& s)
{
    // manual: residual = the entered sigmas. A locked profile still measures
    // the residual live — it depends on how much the temporal stage actually
    // removed from THIS frame, which is not part of the locked snapshot.
    if (p.profileSource == 2 && p.profileLocked == 0) {
        s.ry = clampf(p.sigmaY, kSigmaMin, kSigmaMax);
        s.rc = clampf(p.sigmaC, kSigmaMin, kSigmaMax);
        s.rcb = s.rcr = s.rc;
        s.effNMed = 1.0f;
        return;
    }

    std::vector<uint32_t> hYr(kHistBins, 0), hCrb(kHistBins, 0), hCrr(kHistBins, 0), hN(64, 0);
    std::vector<uint32_t> hY2r(kHistBins, 0), hC2rb(kHistBins, 0), hC2rr(kHistBins, 0);
    uint64_t total = 0, total2 = 0;
    for (int y = 1; y < H - 1; y += 2) {
        for (int x = 1; x < W - 1; x += 2) {
            float Y[9], Cb[9], Cr[9];
            int i = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx, ++i) {
                    const float* t = tmpAt(tmp, W, H, x + dx, y + dy);
                    Y[i] = t[0]; Cb[i] = t[1]; Cr[i] = t[2];
                }
            const float lapY  = 4.0f * Y[4]  - 2.0f * (Y[1] + Y[3] + Y[5] + Y[7])   + (Y[0] + Y[2] + Y[6] + Y[8]);
            const float lapCb = 4.0f * Cb[4] - 2.0f * (Cb[1] + Cb[3] + Cb[5] + Cb[7]) + (Cb[0] + Cb[2] + Cb[6] + Cb[8]);
            const float lapCr = 4.0f * Cr[4] - 2.0f * (Cr[1] + Cr[3] + Cr[5] + Cr[7]) + (Cr[0] + Cr[2] + Cr[6] + Cr[8]);
            if (lapY == 0.0f && lapCb == 0.0f && lapCr == 0.0f)  // v3.1 flat-sample skip
                continue;
            hYr[clampi(static_cast<int>(std::fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)]++;
            hCrb[clampi(static_cast<int>(std::fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)]++;
            hCrr[clampi(static_cast<int>(std::fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)]++;
            const float effN = tmpAt(tmp, W, H, x, y)[3];
            // v3.3 B2: 64 bins — the 5-frame era's 32 saturated at effN
            // 4.875; the x8 scale is untouched so old bins decode identically
            hN[clampi(static_cast<int>((effN - 1.0f) * 8.0f), 0, 63)]++;
            total++;

            // v3.2: coarse residual — 2x2-block Laplacian on the merged
            // image, exactly the input estimator's trick. Compression noise
            // survives temporal averaging as blotch LARGER than a pixel;
            // the fine Laplacian is blind to it, so the spatial stage was
            // being told the image is cleaner than the eye can see. Blocks
            // are EVEN-aligned (x-1, y-1: the sampling grid is odd) so
            // 2x2-structured noise — 4:2:0 chroma sits on this grid — lands
            // inside one block instead of straddling four and cancelling.
            if ((((x - 1) >> 1) & 1) == 0 && (((y - 1) >> 1) & 1) == 0) {
                float bY[9], bCb[9], bCr[9];
                i = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx, ++i)
                        blockMeanTmp(tmp, W, H, (x - 1) + dx * 2, (y - 1) + dy * 2, bY[i], bCb[i], bCr[i]);
                const float lY  = 4.0f * bY[4]  - 2.0f * (bY[1] + bY[3] + bY[5] + bY[7])   + (bY[0] + bY[2] + bY[6] + bY[8]);
                const float lCb = 4.0f * bCb[4] - 2.0f * (bCb[1] + bCb[3] + bCb[5] + bCb[7]) + (bCb[0] + bCb[2] + bCb[6] + bCb[8]);
                const float lCr = 4.0f * bCr[4] - 2.0f * (bCr[1] + bCr[3] + bCr[5] + bCr[7]) + (bCr[0] + bCr[2] + bCr[6] + bCr[8]);
                if (lY != 0.0f || lCb != 0.0f || lCr != 0.0f) {
                    hY2r[clampi(static_cast<int>(std::fabs(lY)  * kHistScaleY), 0, kHistBins - 1)]++;
                    hC2rb[clampi(static_cast<int>(std::fabs(lCb) * kHistScaleC), 0, kHistBins - 1)]++;
                    hC2rr[clampi(static_cast<int>(std::fabs(lCr) * kHistScaleC), 0, kHistBins - 1)]++;
                    total2++;
                }
            }
        }
    }
    if (total < 64) {
        s.ry = s.sy; s.rc = s.sc; s.rcb = s.scb; s.rcr = s.scr; s.effNMed = 1.0f;
        return;
    }

    auto med = [](const std::vector<uint32_t>& h, uint64_t tot, float scale) {
        uint64_t cum = 0;
        const uint64_t target = (tot + 1) / 2;
        for (size_t b = 0; b < h.size(); ++b) {
            cum += h[b];
            if (cum >= target)
                return (static_cast<float>(b) + 0.5f) / scale;
        }
        return (static_cast<float>(h.size()) - 0.5f) / scale;
    };

    const float adj = clampf(p.profileAdjust, 0.25f, 6.0f);
    float ry = med(hYr, total, kHistScaleY) * kMedianCal * adj;
    float rcb = med(hCrb, total, kHistScaleC) * kMedianCal * adj;
    float rcr = med(hCrr, total, kHistScaleC) * kMedianCal * adj;
    // v3.2: filter against whichever scale still carries noise (0.9 factor
    // and 2x block calibration exactly as the input estimator)
    if (total2 >= 64) {
        const float ryC = 2.0f * med(hY2r, total2, kHistScaleY) * kMedianCal * adj;
        const float rcbC = 2.0f * med(hC2rb, total2, kHistScaleC) * kMedianCal * adj;
        const float rcrC = 2.0f * med(hC2rr, total2, kHistScaleC) * kMedianCal * adj;
        ry = std::max(ry, 0.9f * ryC);
        rcb = std::max(rcb, 0.9f * rcbC);
        rcr = std::max(rcr, 0.9f * rcrC);
    }
    s.effNMed = 1.0f + med(hN, total, 8.0f);

    // Floors: the residual cannot be less than the theoretical reduction, and
    // never exceeds the input estimate.
    const float floorY = 0.5f * s.sy / std::sqrt(std::max(1.0f, s.effNMed));
    const float floorCb = 0.5f * s.scb / std::sqrt(std::max(1.0f, s.effNMed));
    const float floorCr = 0.5f * s.scr / std::sqrt(std::max(1.0f, s.effNMed));
    s.ry = clampf(std::max(ry, floorY), kSigmaMin, s.sy > kSigmaMin ? s.sy : kSigmaMax);
    s.rcb = clampf(std::max(rcb, floorCb), kSigmaMin, s.scb > kSigmaMin ? s.scb : kSigmaMax);
    s.rcr = clampf(std::max(rcr, floorCr), kSigmaMin, s.scr > kSigmaMin ? s.scr : kSigmaMax);
    s.rc = 0.5f * (s.rcb + s.rcr);
}

// ---------------------------------------------------------------------------
// Stage 3b — v3.3 "Deep Clean": optional fine-NLM pre-pass over the temporal
// result, into a second buffer the rest of the pipeline then reads.
// ---------------------------------------------------------------------------
// Compressed/correlated noise survives a single NLM pass because
// neighbouring patches share the noise's structure and win spurious
// similarity; a conservative first pass at 0.6x the similarity h (weights
// die ~2.8x faster than the main pass') decorrelates exactly that component,
// and the main pass then sees patches whose remaining differences are
// honest. Corrections are hard-clamped to twice the pixel's noise scale, so
// the pre-pass can only ever make noise-sized changes — structure is safe by
// construction, which is why it ships as a plain checkbox with no strength
// slider. It runs on the PASS-1 residual measurement (the two-scale
// estimator reads correlated energy correctly there) and the residual is
// re-measured on its output, so the main pass adapts to what is actually
// left. effN passes through untouched; the After Temporal view keeps
// showing the true temporal result (spatialNLM reads both buffers).
inline void deepCleanPass(const float* tmp, int W, int H,
                          const Params& p, const Stats& s, float* out)
{
    const int R = 2;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t idx = (static_cast<size_t>(y) * W + x) * 4;
            const float* tc = tmp + idx;
            const int lb = clampi(static_cast<int>(tc[0] * kLumaBins), 0, kLumaBins - 1);
            const float sigY = clampf(s.ry * s.gainY[lb], 1e-5f, 1.0f);
            // v3.3 B5: per-channel chroma h — see spatialNLM
            const float sigCb = clampf(s.rcb * s.gainC[lb], 1e-5f, 1.0f);
            const float sigCr = clampf(s.rcr * s.gainC[lb], 1e-5f, 1.0f);
            const float hY = 0.6f * kNlmHLuma * sigY;
            // v3.3 B5: pooled-normalized chroma weight — see spatialNLM
            const float invHY2 = 1.0f / std::max(hY * hY, 1e-12f);
            const float invHC2 = 1.0f / std::max(0.36f * kNlmHChroma * kNlmHChroma, 1e-12f);
            const float invSCb2 = 1.0f / std::max(sigCb * sigCb, 1e-12f);
            const float invSCr2 = 1.0f / std::max(sigCr * sigCr, 1e-12f);
            const float biasY = 2.0f * sigY * sigY;
            const float biasCb = 2.0f * sigCb * sigCb;
            const float biasCr = 2.0f * sigCr * sigCr;

            float accY = 0.0f, accCb = 0.0f, accCr = 0.0f;
            float sumWY = 0.0f, sumWC = 0.0f, wYmax = 0.0f, wCmax = 0.0f;
            for (int dy = -R; dy <= R; ++dy) {
                for (int dx = -R; dx <= R; ++dx) {
                    if (dx == 0 && dy == 0)
                        continue;
                    const float* ts = tmpAt(tmp, W, H, x + dx, y + dy);
                    float dY2 = 0.0f, dCb2 = 0.0f, dCr2 = 0.0f;
                    for (int qy = -1; qy <= 1; ++qy) {
                        for (int qx = -1; qx <= 1; ++qx) {
                            const float* tp = tmpAt(tmp, W, H, x + qx, y + qy);
                            const float* tq = tmpAt(tmp, W, H, x + dx + qx, y + dy + qy);
                            const float eY = tp[0] - tq[0];
                            const float eCb = tp[1] - tq[1];
                            const float eCr = tp[2] - tq[2];
                            dY2 += eY * eY;
                            dCb2 += eCb * eCb;
                            dCr2 += eCr * eCr;
                        }
                    }
                    dY2 *= (1.0f / 9.0f);
                    dCb2 *= (1.0f / 9.0f);
                    dCr2 *= (1.0f / 9.0f);
                    dY2 = std::max(0.0f, dY2 - biasY);
                    const float dC2n = 0.5f * (std::max(0.0f, dCb2 - biasCb) * invSCb2 +
                                               std::max(0.0f, dCr2 - biasCr) * invSCr2);
                    const float wY = std::exp(-dY2 * invHY2);
                    const float wC = std::exp(-dC2n * invHC2) * std::exp(-dY2 * invHY2 * 0.25f);
                    accY  += wY * ts[0];
                    accCb += wC * ts[1];
                    accCr += wC * ts[2];
                    sumWY += wY;
                    sumWC += wC;
                    wYmax = std::max(wYmax, wY);
                    wCmax = std::max(wCmax, wC);
                }
            }
            const float wYc = std::max(wYmax, 1e-4f);
            const float wCc = std::max(wCmax, 1e-4f);
            const float Yf  = (accY  + wYc * tc[0]) / (sumWY + wYc);
            const float Cbf = (accCb + wCc * tc[1]) / (sumWC + wCc);
            const float Crf = (accCr + wCc * tc[2]) / (sumWC + wCc);
            float* o = out + idx;
            o[0] = tc[0] + clampf(Yf  - tc[0], -2.0f * sigY, 2.0f * sigY);
            o[1] = tc[1] + clampf(Cbf - tc[1], -2.0f * sigCb, 2.0f * sigCb);
            o[2] = tc[2] + clampf(Crf - tc[2], -2.0f * sigCr, 2.0f * sigCr);
            o[3] = tc[3];
        }
    }
}

// ---------------------------------------------------------------------------
// HUD v3 + scopes (Noise Analysis panel, Noise EQ panel, Motion mini-map)
// ---------------------------------------------------------------------------
// All text renders at 2x glyph scale (10x14 px strokes of 2 px at scale 1) so
// panels stay legible at fit-to-window zoom — the v3 1-px strokes decimated
// into garbage below 100% viewer zoom.
// glyph order: 0-9 . % A-Z + space - | =
static const uint64_t kFont[43] = {
    0x3a33ae62eULL, 0x11842108eULL, 0x3a213221fULL, 0x3a213062eULL, 0x08ca97c42ULL, 0x7e1e0862eULL,
    0x3a10f462eULL, 0x7c2222108ULL, 0x3a317462eULL, 0x3a317842eULL, 0x00000018cULL, 0x632222263ULL,
    0x3a31fc631ULL, 0x7a31f463eULL, 0x3a308422eULL, 0x7a318c63eULL, 0x7e10f421fULL, 0x7e10f4210ULL,
    0x3a30bc62fULL, 0x4631fc631ULL, 0x38842108eULL, 0x1c4210a4cULL, 0x4654c5251ULL, 0x42108421fULL,
    0x4775ac631ULL, 0x47359c631ULL, 0x3a318c62eULL, 0x7a31f4210ULL, 0x3a318d64dULL, 0x7a31f5251ULL,
    0x3e107043eULL, 0x7c8421084ULL, 0x46318c62eULL, 0x46318c544ULL, 0x4631ad771ULL, 0x462a22a31ULL,
    0x462a21084ULL, 0x7c222221fULL, 0x0084f9080ULL, 0x000000000ULL, 0x000070000ULL, 0x108421084ULL,
    0x01f07c00ULL,
};
#define NR_G_DOT 10
#define NR_G_PCT 11
#define NR_G_A 12
#define NR_G_B 13
#define NR_G_C 14
#define NR_G_D 15
#define NR_G_E 16
#define NR_G_F 17
#define NR_G_G 18
#define NR_G_H 19
#define NR_G_I 20
#define NR_G_J 21
#define NR_G_K 22
#define NR_G_L 23
#define NR_G_M 24
#define NR_G_N 25
#define NR_G_O 26
#define NR_G_P 27
#define NR_G_Q 28
#define NR_G_R 29
#define NR_G_S 30
#define NR_G_T 31
#define NR_G_U 32
#define NR_G_V 33
#define NR_G_W 34
#define NR_G_X 35
#define NR_G_Y 36
#define NR_G_Z 37
#define NR_G_PLUS 38
#define NR_G_SP 39
#define NR_G_DASH 40
#define NR_G_BAR 41
#define NR_G_EQ 42

static inline bool glyphPixel(int glyph, int gx, int gy)
{
    if (glyph < 0 || glyph > 42 || gx < 0 || gx >= 5 || gy < 0 || gy >= 7)
        return false;
    return (kFont[glyph] >> (34 - (gy * 5 + gx))) & 1ULL;
}

// sc = integer text scale: 1 -> 6x7 px cells, 2 -> 12x14 px cells
static inline bool textPixel(const int* chars, int n, int tx, int ty, int lx, int ly, int sc)
{
    if (ly < ty || ly >= ty + 7 * sc || lx < tx || lx >= tx + n * 6 * sc)
        return false;
    const int gx = (lx - tx) / sc;
    const int ci = gx / 6;
    return glyphPixel(chars[ci], gx - ci * 6, (ly - ty) / sc);
}

static inline void pctGlyphs(float pp, int out[6])
{
    const int v = clampi(static_cast<int>(pp * 100.0f + 0.5f), 0, 9999);
    const int tens = (v / 1000) % 10;
    out[0] = (tens == 0) ? NR_G_SP : tens;
    out[1] = (v / 100) % 10;
    out[2] = NR_G_DOT;
    out[3] = (v / 10) % 10;
    out[4] = v % 10;
    out[5] = NR_G_PCT;
}

// "x.y" for small values (effN, dB)
static inline void dec1Glyphs(float v, int out[3])
{
    const int t = clampi(static_cast<int>(v * 10.0f + 0.5f), 0, 99);
    out[0] = (t / 10) % 10;
    out[1] = NR_G_DOT;
    out[2] = t % 10;
}

// whole-percent "150%" from a 0..x fraction (EQ scope setting readouts)
static inline void pctIntGlyphs(float frac, int out[4])
{
    const int v = clampi(static_cast<int>(frac * 100.0f + 0.5f), 0, 999);
    out[0] = (v >= 100) ? (v / 100) % 10 : NR_G_SP;
    out[1] = (v >= 10) ? (v / 10) % 10 : NR_G_SP;
    out[2] = v % 10;
    out[3] = NR_G_PCT;
}

// Renders the analysis panel over (r,g,b) for absolute pixel (x,y).
static inline bool hudPixel(int x, int y, int W, int H, const Stats& st,
                            int enableTemporal, int locked,
                            float& r, float& g, float& b)
{
    // OFX buffers are bottom-up: anchor the panel in DISPLAY space (top-left)
    const int yd = H - 1 - y;
    const int s = std::max(1, H / 540);
    const int ox = 16 * s, oy = 16 * s;
    const int lw = 360, lh = 294;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    r = g = 0.045f; b = 0.05f;   // opaque panel: the image can't crawl under the text
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        r = g = b = 0.35f;
        return true;
    }

    static const int kLabIY[7]  = { NR_G_I, NR_G_N, NR_G_P, NR_G_U, NR_G_T, NR_G_SP, NR_G_Y };
    static const int kLabIC[7]  = { NR_G_I, NR_G_N, NR_G_P, NR_G_U, NR_G_T, NR_G_SP, NR_G_C };
    static const int kLabRY[10] = { NR_G_R, NR_G_E, NR_G_S, NR_G_I, NR_G_D, NR_G_U, NR_G_A, NR_G_L, NR_G_SP, NR_G_Y };
    static const int kLabRC[10] = { NR_G_R, NR_G_E, NR_G_S, NR_G_I, NR_G_D, NR_G_U, NR_G_A, NR_G_L, NR_G_SP, NR_G_C };
    static const int kLabAVGFR[10] = { NR_G_A, NR_G_V, NR_G_G, NR_G_SP, NR_G_F, NR_G_R, NR_G_A, NR_G_M, NR_G_E, NR_G_S };
    static const int kLabGAIN[4] = { NR_G_G, NR_G_A, NR_G_I, NR_G_N };
    static const int kLabDB[2]   = { NR_G_D, NR_G_B };
    static const int kLabLOCKED[14] = { NR_G_P, NR_G_R, NR_G_O, NR_G_F, NR_G_I, NR_G_L, NR_G_E, NR_G_SP,
                                        NR_G_L, NR_G_O, NR_G_C, NR_G_K, NR_G_E, NR_G_D };
    static const int kLabLIVE[14] = { NR_G_M, NR_G_E, NR_G_A, NR_G_S, NR_G_U, NR_G_R, NR_G_I, NR_G_N,
                                      NR_G_G, NR_G_SP, NR_G_L, NR_G_I, NR_G_V, NR_G_E };
    static const int kLabTOFF[12] = { NR_G_T, NR_G_E, NR_G_M, NR_G_P, NR_G_O, NR_G_R, NR_G_A, NR_G_L,
                                      NR_G_SP, NR_G_O, NR_G_F, NR_G_F };
    static const int kLabCURVE[19] = { NR_G_N, NR_G_O, NR_G_I, NR_G_S, NR_G_E, NR_G_SP, NR_G_V, NR_G_S,
                                       NR_G_SP, NR_G_B, NR_G_R, NR_G_I, NR_G_G, NR_G_H, NR_G_T, NR_G_N,
                                       NR_G_E, NR_G_S, NR_G_S };
    static const int kLabHIST[31] = { NR_G_N, NR_G_O, NR_G_I, NR_G_S, NR_G_E, NR_G_SP, NR_G_H, NR_G_I,
                                      NR_G_S, NR_G_T, NR_G_O, NR_G_G, NR_G_R, NR_G_A, NR_G_M, NR_G_SP,
                                      NR_G_DASH, NR_G_SP, NR_G_M, NR_G_E, NR_G_D, NR_G_I, NR_G_A, NR_G_N,
                                      NR_G_SP, NR_G_M, NR_G_A, NR_G_R, NR_G_K, NR_G_E, NR_G_D };

    const float sig[4] = { st.sy, st.sc, st.ry, st.rc };
    const int rowY[4] = { 10, 42, 74, 106 };

    for (int row = 0; row < 4; ++row) {
        const int ty0 = rowY[row];
        bool lit = false;
        switch (row) {
        case 0: lit = textPixel(kLabIY, 7,  10, ty0, lx, ly, 2); break;
        case 1: lit = textPixel(kLabIC, 7,  10, ty0, lx, ly, 2); break;
        case 2: lit = textPixel(kLabRY, 10, 10, ty0, lx, ly, 2); break;
        case 3: lit = textPixel(kLabRC, 10, 10, ty0, lx, ly, 2); break;
        }
        if (!lit) {
            int vg[6];
            pctGlyphs(sig[row] * 100.0f, vg);
            lit = textPixel(vg, 6, 278, ty0, lx, ly, 2);
        }
        if (lit) { r = g = b = 1.0f; return true; }

        if (ly >= ty0 + 16 && ly < ty0 + 22 && lx >= 10 && lx < 350) {
            const float fill = clampf(sig[row] / 0.08f, 0.0f, 1.0f);
            const bool on = (lx - 10) < static_cast<int>(fill * 340.0f);
            const bool residRow = (row >= 2);
            if (on) {
                if (residRow) { r = 0.95f; g = 0.65f; b = 0.20f; }
                else          { r = 0.20f; g = 0.65f; b = 0.95f; }
            } else { r = g = b = 0.14f; }
            return true;
        }
    }

    // info line: "AVG FRAMES x.x   GAIN +x.x DB"
    {
        const int ty0 = 138;
        bool lit = textPixel(kLabAVGFR, 10, 10, ty0, lx, ly, 2);
        if (!lit) {
            int vg[3];
            dec1Glyphs(enableTemporal ? st.effNMed : 1.0f, vg);
            lit = textPixel(vg, 3, 138, ty0, lx, ly, 2);
        }
        if (!lit) lit = textPixel(kLabGAIN, 4, 186, ty0, lx, ly, 2);
        if (!lit) {
            const float gainDb = clampf(20.0f * std::log10(std::max(st.sy, 1e-5f) / std::max(st.ry, 1e-5f)), 0.0f, 40.0f);
            int vg[4];
            vg[0] = NR_G_PLUS;
            int d1[3];
            dec1Glyphs(gainDb, d1);
            vg[1] = d1[0]; vg[2] = d1[1]; vg[3] = d1[2];
            lit = textPixel(vg, 4, 240, ty0, lx, ly, 2);
            if (!lit) lit = textPixel(kLabDB, 2, 294, ty0, lx, ly, 2);
        }
        if (lit) { r = g = b = 1.0f; return true; }
    }

    // status line: measurement state + temporal state
    {
        const int ty0 = 160;
        if (locked) {
            if (textPixel(kLabLOCKED, 14, 10, ty0, lx, ly, 2)) { r = 0.95f; g = 0.65f; b = 0.20f; return true; }
        } else {
            if (textPixel(kLabLIVE, 14, 10, ty0, lx, ly, 2)) { r = g = b = 0.55f; return true; }
        }
        if (enableTemporal == 0 && textPixel(kLabTOFF, 12, 190, ty0, lx, ly, 2)) {
            r = 0.90f; g = 0.45f; b = 0.30f; return true;
        }
    }

    // caption + noise-vs-brightness curve (dim line marks gain = 1.0)
    if (textPixel(kLabCURVE, 19, 10, 182, lx, ly, 1)) { r = g = b = 0.55f; return true; }
    if (lx >= 10 && lx < 346 && ly >= 194 && ly < 238) {
        const int bin = clampi((lx - 10) / 21, 0, kLumaBins - 1);
        const float v = clampf((st.gainY[bin] - 0.6f) / 1.6f, 0.0f, 1.0f);
        const bool bar = (237 - ly) < static_cast<int>(v * 43.0f + 0.5f);
        const bool ref = (237 - ly) == static_cast<int>((1.0f - 0.6f) / 1.6f * 43.0f + 0.5f);
        if (bar)      { r = 0.20f; g = 0.65f; b = 0.95f; }
        else if (ref) { r = g = b = 0.42f; }
        else          { r = g = b = 0.08f; }
        return true;
    }

    // caption + fine luma |laplacian| histogram, sqrt-scaled so one dominant
    // bin cannot flatten the rest of the display
    if (textPixel(kLabHIST, 31, 10, 244, lx, ly, 1)) { r = g = b = 0.55f; return true; }
    if (lx >= 10 && lx < 346 && ly >= 252 && ly < 284) {
        const int bin = clampi((lx - 10) * kHistBins / 336, 0, kHistBins - 1);
        const float frac = static_cast<float>(st.histY[bin]) / static_cast<float>(st.histMax);
        const float hgt = 31.0f * std::sqrt(clampf(frac, 0.0f, 1.0f));
        const bool bar = (283 - ly) < static_cast<int>(hgt + 0.5f);
        if (bin == static_cast<int>(st.medBinY)) { r = 0.95f; g = 0.85f; b = 0.15f; }
        else if (bar)                            { r = g = b = 0.55f; }
        else                                     { r = g = b = 0.08f; }
        return true;
    }

    return true;
}

// Renders the Noise EQ panel (top-right): one lane per band — the bar is how
// much that band is set to cut, the amber line is how much noise the
// estimators measure at that scale. The whole point is "noise lives HERE,
// you are cutting THIS much of it".
static inline bool eqScopePixel(int x, int y, int W, int H, const Stats& st,
                                const Params& p, float& r, float& g, float& b)
{
    const int yd = H - 1 - y;
    const int s = std::max(1, H / 540);
    const int lw = 300, lh = 190;
    const int ox = W - 16 * s - lw * s, oy = 16 * s;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    r = g = 0.045f; b = 0.05f;
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        r = g = b = 0.35f;
        return true;
    }

    static const int kLabTITLE[8] = { NR_G_N, NR_G_O, NR_G_I, NR_G_S, NR_G_E, NR_G_SP, NR_G_E, NR_G_Q };
    static const int kLabOFF[3]  = { NR_G_O, NR_G_F, NR_G_F };
    static const int kLabFINE[4] = { NR_G_F, NR_G_I, NR_G_N, NR_G_E };
    static const int kLabMED[6]  = { NR_G_M, NR_G_E, NR_G_D, NR_G_I, NR_G_U, NR_G_M };
    static const int kLabCRS[6]  = { NR_G_C, NR_G_O, NR_G_A, NR_G_R, NR_G_S, NR_G_E };
    static const int kLabCOL[5]  = { NR_G_C, NR_G_O, NR_G_L, NR_G_O, NR_G_R };
    static const int kLabPX1[3]  = { 1, NR_G_P, NR_G_X };
    static const int kLabPX38[5] = { 3, NR_G_DASH, 8, NR_G_P, NR_G_X };
    static const int kLabPX16[5] = { 1, 6, NR_G_P, NR_G_X, NR_G_PLUS };
    static const int kLabLEG[34] = { NR_G_B, NR_G_A, NR_G_R, NR_G_SP, NR_G_EQ, NR_G_SP, NR_G_C, NR_G_U,
                                     NR_G_T, NR_G_SP, NR_G_SP, NR_G_A, NR_G_M, NR_G_B, NR_G_E, NR_G_R,
                                     NR_G_SP, NR_G_EQ, NR_G_SP, NR_G_M, NR_G_E, NR_G_A, NR_G_S, NR_G_U,
                                     NR_G_R, NR_G_E, NR_G_D, NR_G_SP, NR_G_N, NR_G_O, NR_G_I, NR_G_S,
                                     NR_G_E, NR_G_SP };

    if (textPixel(kLabTITLE, 8, 10, 8, lx, ly, 2)) { r = g = b = 1.0f; return true; }
    if (p.enableSpatial == 0 && textPixel(kLabOFF, 3, 250, 8, lx, ly, 2)) {
        r = 0.90f; g = 0.45f; b = 0.30f; return true;
    }
    if (textPixel(kLabLEG, 34, 10, 172, lx, ly, 1)) { r = g = b = 0.55f; return true; }

    // band data: setting fraction (0..1 of slider reach) and measured sigma.
    // FINE = pixel-scale estimator; MEDIUM = 2x2-block estimator; COARSE =
    // the energy the temporal estimator sees beyond both (sqrt(ty^2-max^2));
    // COLOR = same decomposition for chroma.
    const float fineM = std::max(st.fineY, st.coarseY);
    const float lowY = std::sqrt(std::max(0.0f, st.ty * st.ty - fineM * fineM));
    const float lowC = std::sqrt(std::max(0.0f, st.tc * st.tc - st.fineC * st.fineC));
    const float amt[4] = { clampf(p.eqFine, 0.0f, 3.0f) / 3.0f,
                           clampf(p.eqMedium, 0.0f, 1.5f) / 1.5f,
                           clampf(p.eqCoarse, 0.0f, 1.5f) / 1.5f,
                           clampf(p.chromaBlotch, 0.0f, 1.5f) / 1.5f };
    const float rawPct[4] = { clampf(p.eqFine, 0.0f, 3.0f),
                              clampf(p.eqMedium, 0.0f, 1.5f),
                              clampf(p.eqCoarse, 0.0f, 1.5f),
                              clampf(p.chromaBlotch, 0.0f, 1.5f) };
    const float meas[4] = { st.fineY, st.coarseY, lowY, lowC };

    for (int lane = 0; lane < 4; ++lane) {
        const int x0 = 10 + lane * 72;
        if (lx < x0 || lx >= x0 + 60)
            continue;

        // setting readout above the lane
        {
            int vg[4];
            pctIntGlyphs(rawPct[lane], vg);
            if (textPixel(vg, 4, x0 + 18, 32, lx, ly, 1)) { r = g = b = 0.85f; return true; }
        }
        // lane labels under the plot
        bool lit = false;
        switch (lane) {
        case 0: lit = textPixel(kLabFINE, 4, x0 + 18, 148, lx, ly, 1) ||
                      textPixel(kLabPX1, 3, x0 + 21, 158, lx, ly, 1); break;
        case 1: lit = textPixel(kLabMED, 6, x0 + 12, 148, lx, ly, 1) ||
                      textPixel(kLabPX38, 5, x0 + 15, 158, lx, ly, 1); break;
        case 2: lit = textPixel(kLabCRS, 6, x0 + 12, 148, lx, ly, 1) ||
                      textPixel(kLabPX16, 5, x0 + 15, 158, lx, ly, 1); break;
        case 3: lit = textPixel(kLabCOL, 5, x0 + 15, 148, lx, ly, 1) ||
                      textPixel(kLabPX16, 5, x0 + 15, 158, lx, ly, 1); break;
        }
        if (lit) { r = g = b = 0.75f; return true; }

        // the plot: bar = cut amount, amber line = measured noise at scale
        if (ly >= 44 && ly < 144) {
            const int up = 143 - ly;   // 0 at baseline
            const float nv = clampf(meas[lane] / 0.08f, 0.0f, 1.0f);
            const int markH = static_cast<int>(nv * 98.0f + 0.5f);
            const int barH = static_cast<int>(clampf(amt[lane], 0.0f, 1.0f) * 98.0f + 0.5f);
            if (up >= markH - 1 && up <= markH + 1 && meas[lane] > 1e-5f) {
                r = 0.95f; g = 0.65f; b = 0.20f;      // measured-noise line
            } else if (up < barH) {
                if (up >= barH - 3) { r = g = b = 0.90f; }   // bar cap
                else                { r = g = b = 0.30f; }   // bar body
            } else {
                r = g = b = 0.08f;
            }
            return true;
        }
        return true;   // lane gutter
    }

    return true;
}

// Renders the temporal-activity mini map (bottom-right): a live thumbnail of
// where across-frames averaging is working (green) vs motion-protected (red).
static inline bool motionScopePixel(int x, int y, int W, int H,
                                    const float* tmp, const float* curr,
                                    float& r, float& g, float& b)
{
    const int yd = H - 1 - y;
    const int s = std::max(1, H / 540);
    const int mapW = 300;
    const int mapH = std::max(40, (mapW * H) / std::max(W, 1));
    const int lw = mapW + 2, lh = mapH + 18;
    const int ox = W - 16 * s - lw * s, oy = H - 16 * s - lh * s;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        r = g = b = 0.35f;
        return true;
    }

    static const int kLabMO[33] = { NR_G_M, NR_G_O, NR_G_T, NR_G_I, NR_G_O, NR_G_N, NR_G_SP, NR_G_SP,
                                    NR_G_G, NR_G_R, NR_G_E, NR_G_E, NR_G_N, NR_G_EQ, NR_G_S, NR_G_T,
                                    NR_G_A, NR_G_C, NR_G_K, NR_G_E, NR_G_D, NR_G_SP, NR_G_SP, NR_G_R,
                                    NR_G_E, NR_G_D, NR_G_EQ, NR_G_M, NR_G_O, NR_G_V, NR_G_I, NR_G_N,
                                    NR_G_G };
    if (ly < 16) {
        r = g = 0.045f; b = 0.05f;
        if (textPixel(kLabMO, 33, 4, 5, lx, ly, 1)) { r = g = b = 0.85f; }
        return true;
    }

    // map area: sample the frame in display orientation
    const int u = clampi(lx - 1, 0, mapW - 1);
    const int v = clampi(ly - 16, 0, mapH - 1);
    const int sx = clampi((u * W) / mapW, 0, W - 1);
    const int sdy = clampi((v * H) / mapH, 0, H - 1);
    const int sy2 = H - 1 - sdy;
    const float* t = tmpAt(tmp, W, H, sx, sy2);
    const float* c = curr + (static_cast<size_t>(sy2) * W + sx) * 4;
    float cy, cb, cr;
    rgb2ycc(c[0], c[1], c[2], cy, cb, cr);
    const float effN = std::max(1.0f, t[3]);
    const float tt = clampf((effN - 1.0f) / 4.0f, 0.0f, 1.0f);
    const float mr = 0.90f + (0.10f - 0.90f) * tt;
    const float mg = 0.15f + (0.85f - 0.15f) * tt;
    const float mb = 0.10f + (0.20f - 0.10f) * tt;
    r = cy * 0.45f + mr * 0.55f;
    g = cy * 0.45f + mg * 0.55f;
    b = cy * 0.45f + mb * 0.55f;
    return true;
}

// ---------------------------------------------------------------------------
// Stage 4+5+6 — spatial NLM + chroma blotch + refine + output assembly
// ---------------------------------------------------------------------------
// tmp is the working buffer the bands filter (the Deep Clean output when
// that pass ran); tmpTrue is the TRUE temporal result — the After Temporal
// view and the motion scope read it, so they never show the pre-pass.
// Without Deep Clean the two are the same buffer.
inline void spatialNLM(const float* tmp, const float* tmpTrue, const float* curr,
                       int W, int H, const Params& p, const Stats& s, float* out)
{
    const float mLow  = std::min(p.master, 1.0f);
    const float mHigh = std::max(p.master, 1.0f);
    const float hBoost = std::pow(mHigh, 1.2f);

    const float sL = clampf(p.spatialLuma, 0.0f, 1.5f);
    const float sC = clampf(p.spatialChroma, 0.0f, 1.5f);
    // v3 Noise EQ: the fine slider scales the NLM band's blend (1 = v2.1).
    // v3.1: above 100% it also widens the similarity h — the blend saturates
    // at 1, so the slider's top half used to be a silent no-op.
    // v3.2: the over-100 regions bite harder (field feedback: the spatial
    // stage needed more reach per slider unit); at and below 100 the curves
    // are bit-identical to earlier releases.
    const float eqF = clampf(p.eqFine, 0.0f, 3.0f);
    const float eqH = std::pow(std::max(1.0f, eqF), 0.8f);
    const float aY = (p.enableSpatial == 0) ? 0.0f : clampf(sL * mLow * eqF, 0.0f, 1.0f);
    const float aC = (p.enableSpatial == 0) ? 0.0f : clampf(sC * mLow * eqF, 0.0f, 1.0f);
    const float overL = (sL > 1.0f) ? 1.6f * std::pow(sL - 1.0f, 1.2f) : 0.0f;
    const float overC = (sC > 1.0f) ? 1.6f * std::pow(sC - 1.0f, 1.2f) : 0.0f;
    const float hMulY = (0.6f + 1.4f * std::pow(sL, 1.5f) + overL) * hBoost * eqH;
    const float hMulC = (0.6f + 1.4f * std::pow(sC, 1.5f) + overC) * hBoost * eqH;
    const float pd = clampf(p.preserveDetail, 0.0f, 1.0f);
    const int   R  = clampi(p.spatialRadius, 1, 10);
    const bool  nlm = (p.spatialMode == 1);
    const bool  runSpatial = (p.enableSpatial != 0) && (aY > 0.0f || aC > 0.0f);
    const float spatialSigma = std::max(1.0f, R / 1.5f);
    const float invSpatial2 = 1.0f / (2.0f * spatialSigma * spatialSigma);
    // v3.1 Detail Rescue: bound on the fine band's correction (see below)
    const float rescue = clampf(p.detailRescue, 0.0f, 1.0f);

    // v3.1: the band sliders reach 150 — the applied amount still caps at 1
    // (an over-unity blend would overshoot), the extra drive widens the
    // similarity tolerances and the reach instead.
    const float blotchRaw = clampf(p.chromaBlotch, 0.0f, 1.5f);
    const float medRaw    = clampf(p.eqMedium, 0.0f, 1.5f);
    const float coarseRaw = clampf(p.eqCoarse, 0.0f, 1.5f);
    const float blotch = (p.enableSpatial != 0) ? std::min(blotchRaw * mLow, 1.0f) : 0.0f;
    // v3 Noise EQ: medium band amount and coarse-band luma amount; the coarse
    // radius follows whichever of the two coarse controls is larger.
    const float eqMed  = (p.enableSpatial != 0) ? std::min(medRaw * mLow, 1.0f) : 0.0f;
    const float coarseL = (p.enableSpatial != 0) ? std::min(coarseRaw * mLow, 1.0f) : 0.0f;
    const float medOver = std::max(1.0f, medRaw);
    const float crsOver = std::max(1.0f, std::max(blotchRaw, coarseRaw));
    const int   Rb = 2 + static_cast<int>(14.0f * std::max(blotchRaw, coarseRaw));
    const int   Rm = 3 + static_cast<int>(5.0f * medRaw);
    // Band tolerance: when the temporal estimator (immune to spatial
    // correlation) reads higher than the Laplacian family, the noise has
    // energy at scales the fine estimators under-measure — widen the medium
    // and coarse-luma similarity tolerances by that ratio.
    const float bandRatioY = clampf(s.ty / std::max(s.sy, 1e-6f), 1.0f, 3.0f);
    const float bandRatioC = clampf(s.tc / std::max(s.sc, 1e-6f), 1.0f, 3.0f);

    const bool refine = (p.enableRefine != 0) && (p.master > 0.0f);
    const float desat = refine ? clampf(p.shadowDesat, 0.0f, 1.0f) : 0.0f;
    const float desatRange = std::max(0.02f, p.desatRange);
    const float tex = refine ? clampf(p.lumaTexture, 0.0f, 1.0f) * mLow : 0.0f;
    // v3 deband: acceptance thresholds sit at banding scale (~2.5 8-bit LSB)
    // or at the residual noise level, whichever is higher, so real edges are
    // rejected while quantization steps are averaged through.
    const float debandAmt = refine ? clampf(p.deband, 0.0f, 1.0f) * mLow : 0.0f;
    const float dbThrY = std::max(0.010f, 1.5f * s.ry);
    const float dbThrC = std::max(0.010f, 1.5f * s.rc);
    const float grainAmt = refine ? clampf(p.grainAmount, 0.0f, 1.0f) * 0.06f : 0.0f;
    const float grainSize = clampf(p.grainSize, 0.5f, 6.0f);
    const float grainCh = clampf(p.grainChroma, 0.0f, 1.0f);
    const uint32_t frame = static_cast<uint32_t>(p.frameIndex);
    // v3.2: final crossfade original -> processed (a plain output mix, unlike
    // Strength which reshapes the filters themselves)
    const float gBlend = clampf(p.globalBlend, 0.0f, 1.0f);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t idx = (static_cast<size_t>(y) * W + x) * 4;
            const float* tc = tmp + idx;

            const int lb = clampi(static_cast<int>(tc[0] * kLumaBins), 0, kLumaBins - 1);
            const float sigY = clampf(s.ry * s.gainY[lb], 1e-5f, 1.0f);
            // v3.3 B5: the fine band works per chroma channel (blue-channel
            // night noise wants a wide Cb h and a tight Cr h); the medium/
            // coarse/deband bands keep the combined colour-distance metric
            // and use the pair mean below.
            const float sigCb = clampf(s.rcb * s.gainC[lb], 1e-5f, 1.0f);
            const float sigCr = clampf(s.rcr * s.gainC[lb], 1e-5f, 1.0f);
            const float sigC = 0.5f * (sigCb + sigCr);

            float Yo = tc[0], Cbo = tc[1], Cro = tc[2];

            if (runSpatial) {
                float mean = 0.0f, m2 = 0.0f;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const float v = tmpAt(tmp, W, H, x + dx, y + dy)[0];
                        mean += v; m2 += v * v;
                    }
                mean *= (1.0f / 9.0f);
                const float var = std::max(0.0f, m2 * (1.0f / 9.0f) - mean * mean);
                const float edginess = clampf(std::sqrt(std::max(var - sigY * sigY, 0.0f)) / (3.0f * sigY), 0.0f, 1.0f);

                const float hY = kNlmHLuma   * sigY * hMulY * (1.0f - pd * 0.85f * edginess);
                // v3.3 B5: per-channel h via normalization inside a POOLED
                // chroma weight. Each channel's patch distance is bias-
                // subtracted and scaled by ITS OWN sigma (a wide h for a
                // noisy Cb, a tight one for a clean Cr — the night-noise
                // case), then the two pool into one acceptance, keeping the
                // v2.1 cross-channel edge protection and the variance
                // reduction of pooling. At equal sigmas this is numerically
                // the v2.1 formula (fully independent per-channel weights
                // were tried first and lost 0.2 dB on the blotch golden to
                // doubled weight noise).
                const float mC = hMulC * (1.0f - pd * 0.50f * edginess);
                const float invHY2  = 1.0f / std::max(hY * hY, 1e-12f);
                const float invHC2  = 1.0f / std::max(kNlmHChroma * kNlmHChroma * mC * mC, 1e-12f);
                const float invSCb2 = 1.0f / std::max(sigCb * sigCb, 1e-12f);
                const float invSCr2 = 1.0f / std::max(sigCr * sigCr, 1e-12f);
                const float biasY = 2.0f * sigY * sigY;
                const float biasCb = 2.0f * sigCb * sigCb;
                const float biasCr = 2.0f * sigCr * sigCr;

                float accY = 0.0f, accCb = 0.0f, accCr = 0.0f;
                float sumWY = 0.0f, sumWC = 0.0f, wYmax = 0.0f, wCmax = 0.0f;

                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        if (dx == 0 && dy == 0)
                            continue;
                        const float* ts = tmpAt(tmp, W, H, x + dx, y + dy);

                        float dY2, dCb2, dCr2;
                        if (nlm) {
                            dY2 = 0.0f; dCb2 = 0.0f; dCr2 = 0.0f;
                            for (int qy = -1; qy <= 1; ++qy) {
                                for (int qx = -1; qx <= 1; ++qx) {
                                    const float* tp = tmpAt(tmp, W, H, x + qx, y + qy);
                                    const float* tq = tmpAt(tmp, W, H, x + dx + qx, y + dy + qy);
                                    const float eY = tp[0] - tq[0];
                                    const float eCb = tp[1] - tq[1];
                                    const float eCr = tp[2] - tq[2];
                                    dY2 += eY * eY;
                                    dCb2 += eCb * eCb;
                                    dCr2 += eCr * eCr;
                                }
                            }
                            dY2 *= (1.0f / 9.0f);
                            dCb2 *= (1.0f / 9.0f);
                            dCr2 *= (1.0f / 9.0f);
                        } else {
                            const float eY = tc[0] - ts[0];
                            const float eCb = tc[1] - ts[1];
                            const float eCr = tc[2] - ts[2];
                            dY2 = eY * eY;
                            dCb2 = eCb * eCb;
                            dCr2 = eCr * eCr;
                        }

                        dY2 = std::max(0.0f, dY2 - biasY);
                        const float dC2n = 0.5f * (std::max(0.0f, dCb2 - biasCb) * invSCb2 +
                                                   std::max(0.0f, dCr2 - biasCr) * invSCr2);

                        float wY = std::exp(-dY2 * invHY2);
                        float wC = std::exp(-dC2n * invHC2) * std::exp(-dY2 * invHY2 * 0.25f);
                        if (!nlm) {
                            const float fall = std::exp(-static_cast<float>(dx * dx + dy * dy) * invSpatial2);
                            wY *= fall;
                            wC *= fall;
                        }

                        accY  += wY * ts[0];
                        accCb += wC * ts[1];
                        accCr += wC * ts[2];
                        sumWY += wY;
                        sumWC += wC;
                        wYmax = std::max(wYmax, wY);
                        wCmax = std::max(wCmax, wC);
                    }
                }

                const float wYc = std::max(wYmax, 1e-4f);
                const float wCc = std::max(wCmax, 1e-4f);
                const float Yf  = (accY  + wYc * tc[0]) / (sumWY + wYc);
                const float Cbf = (accCb + wCc * tc[1]) / (sumWC + wCc);
                const float Crf = (accCr + wCc * tc[2]) / (sumWC + wCc);

                // v3.1 Detail Rescue: clamp the fine band's correction to a
                // noise-sized amount. Cranked strengths then flatten noise as
                // hard as they like — anything bigger than noise (edges,
                // texture the weights failed to protect) is restored, so
                // smoothing cannot become blur. 0 = off, the exact v3.0 blend.
                if (rescue > 0.0f) {
                    const float kY = sigY * (2.0f + 6.0f * (1.0f - rescue));
                    const float kCb = sigCb * (3.0f + 9.0f * (1.0f - rescue));
                    const float kCr = sigCr * (3.0f + 9.0f * (1.0f - rescue));
                    Yo  = tc[0] - aY * clampf(tc[0] - Yf,  -kY, kY);
                    Cbo = tc[1] - aC * clampf(tc[1] - Cbf, -kCb, kCb);
                    Cro = tc[2] - aC * clampf(tc[2] - Crf, -kCr, kCr);
                } else {
                    Yo  = tc[0] + aY * (Yf  - tc[0]);
                    Cbo = tc[1] + aC * (Cbf - tc[1]);
                    Cro = tc[2] + aC * (Crf - tc[2]);
                }
            }

            // v3 medium band: the correction is computed entirely in the 2x2
            // block-mean domain — ring average minus the centre's own block
            // mean — so only components that survive block-meaning (the 2-8px
            // band) are touched; fine noise at the centre passes through to
            // the fine band untouched. The correction is clipped to the
            // band's noise scale so structure the weights let through cannot
            // be smeared by more than a noise-sized amount.
            if (eqMed > 0.0f) {
                const float mScale = 2.6f * sigY * bandRatioY * hBoost * medOver;
                const float myDen = 1.0f / std::max(mScale, 1e-6f);
                const float mcDen = 1.0f / std::max(3.0f * sigC * bandRatioC * hBoost * medOver, 1e-6f);
                float b0Y, b0Cb, b0Cr;
                blockMeanTmp(tmp, W, H, x, y, b0Y, b0Cb, b0Cr);
                float accMY = b0Y, accMB = b0Cb, accMR = b0Cr, sumWm = 1.0f;
                for (int d = 0; d < 8; ++d) {
                    for (int ri = 1; ri <= 2; ++ri) {
                        const float rr = Rm * (static_cast<float>(ri) / 2.0f);
                        float bY, bCb, bCr;
                        blockMeanTmp(tmp, W, H,
                                     x + static_cast<int>(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                                     y + static_cast<int>(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f)),
                                     bY, bCb, bCr);
                        const float eY = (bY - b0Y) * myDen;
                        const float eC = (0.5f * (std::fabs(bCb - b0Cb) + std::fabs(bCr - b0Cr))) * mcDen;
                        const float w = std::exp(-(eY * eY + eC * eC));
                        accMY += w * bY;
                        accMB += w * bCb;
                        accMR += w * bCr;
                        sumWm += w;
                    }
                }
                const float lim = 2.5f * mScale;
                Yo  += eqMed * clampf(accMY / sumWm - b0Y, -lim, lim);
                Cbo += eqMed * (accMB / sumWm - b0Cb);
                Cro += eqMed * (accMR / sumWm - b0Cr);
            }

            // coarse band: large-radius sparse pass. The chroma path is the
            // v2.1 blotch pass unchanged; v3 adds an optional luma component
            // that works in the 4x4 block-mean domain (blind to the fine and
            // medium bands by construction), with band-scaled tolerance and
            // a clipped correction — still chroma-guarded so colour edges
            // aren't crossed.
            if (blotch > 0.0f || coarseL > 0.0f) {
                const float gyDen = 1.0f / std::max(2.0f * sigY * hBoost * crsOver, 1e-6f);
                const float gcDen = 1.0f / std::max(3.0f * sigC * hBoost * crsOver, 1e-6f);
                const float cScale = 2.2f * sigY * bandRatioY * hBoost * crsOver;
                const float glDen = 1.0f / std::max(cScale, 1e-6f);
                // the luma band reaches to 32 px: its targets (16px+ stains)
                // are wider than the chroma blotches, and rings must clear
                // the stain to see clean context
                const int RbL = 2 + static_cast<int>(30.0f * coarseRaw);
                float c0Y = 0.0f, c0Cb = 0.0f, c0Cr = 0.0f;
                if (coarseL > 0.0f)
                    blockMean4Tmp(tmp, W, H, x, y, c0Y, c0Cb, c0Cr);
                float accB = tc[1], accR = tc[2], sumW = 1.0f;
                float accL = c0Y, sumWL = 1.0f;
                for (int d = 0; d < 8; ++d) {
                    for (int ri = 1; ri <= 3; ++ri) {
                        const float rr = Rb * (static_cast<float>(ri) / 3.0f);
                        const int sx = x + static_cast<int>(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f));
                        const int sy2 = y + static_cast<int>(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f));
                        const float* ts = tmpAt(tmp, W, H, sx, sy2);
                        const float eY = (ts[0] - tc[0]) * gyDen;
                        const float eC = (0.5f * (std::fabs(ts[1] - tc[1]) + std::fabs(ts[2] - tc[2]))) * gcDen;
                        const float w = std::exp(-(eY * eY + eC * eC));
                        accB += w * ts[1];
                        accR += w * ts[2];
                        sumW += w;
                        if (coarseL > 0.0f) {
                            const float rrL = RbL * (static_cast<float>(ri) / 3.0f);
                            const int lx2 = x + static_cast<int>(kDirX[d] * rrL + (kDirX[d] > 0 ? 0.5f : -0.5f));
                            const int ly2 = y + static_cast<int>(kDirY[d] * rrL + (kDirY[d] > 0 ? 0.5f : -0.5f));
                            float b4Y, b4Cb, b4Cr;
                            blockMean4Tmp(tmp, W, H, lx2, ly2, b4Y, b4Cb, b4Cr);
                            const float eL = (b4Y - c0Y) * glDen;
                            const float eLC = (0.5f * (std::fabs(b4Cb - c0Cb) + std::fabs(b4Cr - c0Cr))) * gcDen;
                            const float wL = std::exp(-(eL * eL + eLC * eLC));
                            accL += wL * b4Y;
                            sumWL += wL;
                        }
                    }
                }
                if (blotch > 0.0f) {
                    Cbo += blotch * (accB / sumW - Cbo);
                    Cro += blotch * (accR / sumW - Cro);
                }
                if (coarseL > 0.0f) {
                    const float lim = 2.5f * cScale;
                    Yo += coarseL * clampf(accL / sumWL - c0Y, -lim, lim);
                }
            }

            const float* c = curr + idx;
            float cyIn, ccbIn, ccrIn;
            rgb2ycc(c[0], c[1], c[2], cyIn, ccbIn, ccrIn);

            // ---- refine: sat-vs-lum, texture, grain -------------------------
            float Yr = Yo, Cbr = Cbo, Crr = Cro;
            if (refine) {
                const float sat = 1.0f - desat * (1.0f - smooth01(Yr / desatRange));
                Cbr *= sat;
                Crr *= sat;
                Yr += tex * (cyIn - Yr);
                // v3 deband: ring average of 2x2 block means accepted only
                // within the banding-scale threshold (edges are rejected by
                // the threshold itself), then a triangular micro-dither so
                // any remaining step decorrelates instead of reading as a
                // contour. Runs before grain so grain sits on top.
                if (debandAmt > 0.0f) {
                    const float dyDen = 1.0f / dbThrY;
                    const float dcDen = 1.0f / dbThrC;
                    float b0Y, b0Cb, b0Cr;
                    blockMeanTmp(tmp, W, H, x, y, b0Y, b0Cb, b0Cr);
                    float accDY = b0Y, accDB = b0Cb, accDR = b0Cr, sumWd = 1.0f;
                    for (int d = 0; d < 8; ++d) {
                        for (int ri = 1; ri <= 3; ++ri) {
                            const float rr = 16.0f * (static_cast<float>(ri) / 3.0f);
                            float bY, bCb, bCr;
                            blockMeanTmp(tmp, W, H,
                                         x + static_cast<int>(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                                         y + static_cast<int>(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f)),
                                         bY, bCb, bCr);
                            const float eY = (bY - b0Y) * dyDen;
                            const float eC = (0.5f * (std::fabs(bCb - b0Cb) + std::fabs(bCr - b0Cr))) * dcDen;
                            const float w = std::exp(-(eY * eY + eC * eC));
                            accDY += w * bY;
                            accDB += w * bCb;
                            accDR += w * bCr;
                            sumWd += w;
                        }
                    }
                    // The ring mean is the smooth-field estimate and the
                    // correction pulls the PIXEL toward it, weighted by the
                    // SQUARED fraction of accepted samples: on banded
                    // gradients the ring is near-unanimous (full effect),
                    // while at a real edge only same-mixture samples survive
                    // the threshold — a small fraction, squared to nothing.
                    // Clipped to banding scale as a hard safety.
                    const float agree = (sumWd - 1.0f) * (1.0f / 24.0f);
                    const float conf = agree * agree;
                    Yr  += debandAmt * conf * clampf(accDY / sumWd - Yr,  -dbThrY, dbThrY);
                    Cbr += debandAmt * conf * clampf(accDB / sumWd - Cbr, -dbThrC, dbThrC);
                    Crr += debandAmt * conf * clampf(accDR / sumWd - Crr, -dbThrC, dbThrC);
                    const float dith = 0.7f * debandAmt / 255.0f;
                    Yr += dith * 0.5f * (hashNoise(static_cast<uint32_t>(x), static_cast<uint32_t>(y), frame, 3u) +
                                         hashNoise(static_cast<uint32_t>(x), static_cast<uint32_t>(y), frame + 977u, 3u));
                }
                if (grainAmt > 0.0f) {
                    const float resp = 0.25f + 0.75f * (4.0f * clampf(Yr, 0.0f, 1.0f) * (1.0f - clampf(Yr, 0.0f, 1.0f)));
                    const float gn = valueNoise(static_cast<float>(x), static_cast<float>(y), grainSize, frame, 0u);
                    Yr += grainAmt * resp * gn;
                    if (grainCh > 0.0f) {
                        Cbr += grainAmt * grainCh * 0.6f * resp * valueNoise(static_cast<float>(x), static_cast<float>(y), grainSize, frame, 1u);
                        Crr += grainAmt * grainCh * 0.6f * resp * valueNoise(static_cast<float>(x), static_cast<float>(y), grainSize, frame, 2u);
                    }
                }
            }

            // v3.2 global blend: crossfade back toward the untouched input.
            // Applied to the finished result only — the Noise Removed view
            // keeps showing the full-strength diagnostic.
            if (gBlend < 1.0f) {
                Yr  = cyIn  + gBlend * (Yr  - cyIn);
                Cbr = ccbIn + gBlend * (Cbr - ccbIn);
                Crr = ccrIn + gBlend * (Crr - ccrIn);
            }

            float r, g, b;
            ycc2rgb(Yr, Cbr, Crr, r, g, b);
            float dnR, dnG, dnB;                    // denoised, pre-refine (for Noise Removed)
            ycc2rgb(Yo, Cbo, Cro, dnR, dnG, dnB);

            float* o = out + idx;
            o[3] = c[3];   // alpha passthrough (the matte view overrides)

            switch (p.viewMode) {
            case 1: { // split: input | result
                if (x < W / 2) { o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; }
                else           { o[0] = r;    o[1] = g;    o[2] = b;    }
                if (std::abs(x - W / 2) <= 1) { o[0] = 1.0f; o[1] = 1.0f; o[2] = 1.0f; }
                break;
            }
            case 2: { // input passthrough
                o[0] = c[0]; o[1] = c[1]; o[2] = c[2];
                break;
            }
            case 3: { // after temporal (mid-pipeline): the TRUE temporal
                      // result, even when Deep Clean rewrote the working buffer
                const float* tt = tmpTrue + idx;
                float rr, gg, bb;
                ycc2rgb(tt[0], tt[1], tt[2], rr, gg, bb);
                o[0] = rr; o[1] = gg; o[2] = bb;
                break;
            }
            case 4: { // noise removed (denoise only, pre-refine): the gain
                      // rides the measured noise level so clean footage still
                      // reads, and a soft knee compresses big legitimate
                      // changes (lights, motion) instead of clipping them
                      // into hard black/white blobs
                const float nrGain = 0.08f / clampf(s.sy, 0.004f, 0.08f);
                const float dR = (c[0] - dnR) * nrGain;
                const float dG = (c[1] - dnG) * nrGain;
                const float dB = (c[2] - dnB) * nrGain;
                o[0] = 0.5f + 0.5f * dR / (0.5f + std::fabs(dR));
                o[1] = 0.5f + 0.5f * dG / (0.5f + std::fabs(dG));
                o[2] = 0.5f + 0.5f * dB / (0.5f + std::fabs(dB));
                break;
            }
            case 5: { // noise analysis: the result, plus the measurement
                      // scope drawn by the overlay pass below
                o[0] = r; o[1] = g; o[2] = b;
                break;
            }
            case 6: { // temporal activity
                const float effN = std::max(1.0f, tc[3]);
                const float t = clampf((effN - 1.0f) / 4.0f, 0.0f, 1.0f);
                const float mr = 0.90f + (0.10f - 0.90f) * t;
                const float mg = 0.15f + (0.85f - 0.15f) * t;
                const float mb = 0.10f + (0.20f - 0.10f) * t;
                o[0] = cyIn * 0.45f + mr * 0.55f;
                o[1] = cyIn * 0.45f + mg * 0.55f;
                o[2] = cyIn * 0.45f + mb * 0.55f;
                break;
            }
            case 7: { // SNR heat map: local signal stddev vs noise sigma, in dB
                float mean = 0.0f, m2 = 0.0f;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        const float v = tmpAt(tmp, W, H, x + dx, y + dy)[0];
                        mean += v; m2 += v * v;
                    }
                mean *= (1.0f / 25.0f);
                const float var = std::max(0.0f, m2 * (1.0f / 25.0f) - mean * mean);
                const float sigNoise = std::max(s.sy * s.gainY[lb], 1e-5f);
                const float sigSignal = std::sqrt(std::max(var - sigNoise * sigNoise, 0.0f));
                const float snrDb = 20.0f * std::log10(std::max(sigSignal, 1e-6f) / sigNoise);
                const float t = clampf((snrDb + 6.0f) / 36.0f, 0.0f, 1.0f);
                // magenta (noise wins) -> amber -> green (signal wins)
                float mr, mg, mb;
                if (t < 0.5f) {
                    const float u = t * 2.0f;
                    mr = 0.85f + (0.95f - 0.85f) * u;
                    mg = 0.10f + (0.70f - 0.10f) * u;
                    mb = 0.75f + (0.15f - 0.75f) * u;
                } else {
                    const float u = (t - 0.5f) * 2.0f;
                    mr = 0.95f + (0.10f - 0.95f) * u;
                    mg = 0.70f + (0.85f - 0.70f) * u;
                    mb = 0.15f + (0.20f - 0.15f) * u;
                }
                o[0] = cyIn * 0.35f + mr * 0.65f;
                o[1] = cyIn * 0.35f + mg * 0.65f;
                o[2] = cyIn * 0.35f + mb * 0.65f;
                break;
            }
            case 8: { // noise matte: normalized noise dominance in RGB+alpha,
                      // for keying downstream (1 = noise wins, 0 = image wins)
                float mean = 0.0f, m2 = 0.0f;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        const float v = tmpAt(tmp, W, H, x + dx, y + dy)[0];
                        mean += v; m2 += v * v;
                    }
                mean *= (1.0f / 25.0f);
                const float var = std::max(0.0f, m2 * (1.0f / 25.0f) - mean * mean);
                const float sigNoise = std::max(s.sy * s.gainY[lb], 1e-5f);
                const float sigSignal = std::sqrt(std::max(var - sigNoise * sigNoise, 0.0f));
                const float snrDb = 20.0f * std::log10(std::max(sigSignal, 1e-6f) / sigNoise);
                const float m = 1.0f - clampf((snrDb + 6.0f) / 36.0f, 0.0f, 1.0f);
                o[0] = m; o[1] = m; o[2] = m; o[3] = m;
                break;
            }
            default:
                o[0] = r; o[1] = g; o[2] = b;
                break;
            }

            // ---- v3.1 scope overlays: independent per-step panels, drawn
            // over ANY view (never into alpha, so keys stay clean). The
            // legacy Noise Analysis view is the measurement scope forced on.
            {
                const bool wantHud = (p.scopeMeasure != 0) || (p.viewMode == 5);
                const bool wantEq  = (p.scopeEq != 0);
                const bool wantMo  = (p.scopeMotion != 0);
                if (wantHud || wantEq || wantMo) {
                    float rr = o[0], gg = o[1], bb = o[2];
                    bool drew = false;
                    if (wantEq && eqScopePixel(x, y, W, H, s, p, rr, gg, bb))
                        drew = true;
                    if (wantMo && motionScopePixel(x, y, W, H, tmpTrue, curr, rr, gg, bb))
                        drew = true;
                    if (wantHud) {
                        if (hudPixel(x, y, W, H, s, p.enableTemporal, p.profileLocked, rr, gg, bb)) {
                            drew = true;
                        } else if (p.profileSource == 1) {
                            const float rHalf = 0.5f * p.regionSize * static_cast<float>(std::min(W, H));
                            const float cx = p.regionCX * W, cyy = p.regionCY * H;
                            const float ax = std::fabs(x - cx), ay = std::fabs(y - cyy);
                            const bool onEdge = (ax <= rHalf && ay <= rHalf) &&
                                                (ax >= rHalf - 2.0f || ay >= rHalf - 2.0f);
                            if (onEdge) { rr = 1.0f; gg = 1.0f; bb = 0.1f; drew = true; }
                        }
                    }
                    if (drew) { o[0] = rr; o[1] = gg; o[2] = bb; }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience full pipeline (CPU). Returns the stats actually used.
// ---------------------------------------------------------------------------
inline Stats denoiseFrame(const float* const frames[7], int W, int H,
                          const Params& p, float* out, std::vector<float>& scratch)
{
    // v3.3: Deep Clean is a within-frame pass — it follows the spatial
    // stage's enable and the master switch so identity states stay identity.
    const bool deep = (p.deepClean != 0) && (p.enableSpatial != 0) && (p.master > 0.0f);
    const size_t plane = static_cast<size_t>(W) * H * 4;
    scratch.resize(deep ? plane * 2 : plane);
    float* tmp = scratch.data();
    float* tmp2 = deep ? scratch.data() + plane : tmp;
    const float* partner = nullptr;
    if (frames[2] != frames[3]) partner = frames[2];
    else if (frames[4] != frames[3]) partner = frames[4];

    Stats s;
    estimateInput(frames[3], partner, W, H, p, s);
    temporalMerge(frames, W, H, p, s, tmp);
    estimateResidual(tmp, W, H, p, s);
    if (deep) {
        deepCleanPass(tmp, W, H, p, s, tmp2);
        estimateResidual(tmp2, W, H, p, s);   // the main pass adapts to what is left
    }
    spatialNLM(tmp2, tmp, frames[3], W, H, p, s, out);
    return s;
}

} // namespace nrcore

#endif // OPENNR_NR_CORE_H
