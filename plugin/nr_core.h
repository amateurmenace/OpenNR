// OpenNR — free spatio-temporal noise reduction for DaVinci Resolve (OpenFX)
// nr_core.h — reference CPU implementation of the full algorithm (v2).
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
//   3. estimateResidual()— the noise that is ACTUALLY left after temporal
//      merging is measured again from the merged image. The spatial stage
//      filters against this measured residual, not a theoretical prediction —
//      compression correlates noise across frames, so temporal averaging
//      removes less than sqrt(N) and predictions undershoot.
//   4. spatialNLM()      — sigma-adaptive NLM/bilateral + a sparse LARGE
//      radius luma-guided chroma pass ("blotch") that reaches the big soft
//      chroma stains 4:2:0 subsampling produces.
//   5. refine()          — finishing: shadow desaturation (sat-vs-lum curve),
//      luma texture re-injection, and synthesized film grain (soft value
//      noise, brightness-response weighted, animated per frame).
//   6. Output assembly   — result / split / input / after-temporal / noise
//      removed / analysis HUD / temporal activity / SNR heat map.
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
    int   temporalFrames = 3;      // 3 or 5
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
};

// Sigmas and everything the analysis HUD displays.
struct Stats {
    float sy = 0.02f, sc = 0.02f;      // input spatial-family sigmas
    float ty = 0.02f, tc = 0.02f;      // temporal-gating sigmas
    float ry = 0.02f, rc = 0.02f;      // measured residual after temporal
    float gainY[16];                   // brightness-dependent sigma gains
    float gainC[16];
    float effNMed = 1.0f;              // median effective sample count
    uint32_t histY[256] = {0};         // fine luma |laplacian| histogram
    uint32_t medBinY = 0;
    uint32_t histMax = 1;
    int hadTemporal = 0;
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
inline void estimateInput(const float* rgba, const float* diffPartner,
                          int W, int H, const Params& p, Stats& out)
{
    out.sy = out.ty = out.ry = clampf(p.sigmaY, kSigmaMin, kSigmaMax);
    out.sc = out.tc = out.rc = clampf(p.sigmaC, kSigmaMin, kSigmaMax);
    if (p.profileSource == 2)   // manual: sigmas literal, gains flat
        return;

    std::vector<uint32_t> hYf(kHistBins, 0), hCf(kHistBins, 0);
    std::vector<uint32_t> hY2(kHistBins, 0), hC2(kHistBins, 0);
    std::vector<uint32_t> hYt(kHistBins, 0), hCt(kHistBins, 0);
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

            hYf[clampi(static_cast<int>(std::fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)]++;
            hCf[clampi(static_cast<int>(std::fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)]++;
            hCf[clampi(static_cast<int>(std::fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)]++;
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
                hCt[clampi(static_cast<int>(std::fabs(pcb - Cb[4]) * kHistScaleC), 0, kHistBins - 1)]++;
                hCt[clampi(static_cast<int>(std::fabs(pcr - Cr[4]) * kHistScaleC), 0, kHistBins - 1)]++;
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
                hY2[clampi(static_cast<int>(std::fabs(lY)  * kHistScaleY), 0, kHistBins - 1)]++;
                hC2[clampi(static_cast<int>(std::fabs(lCb) * kHistScaleC), 0, kHistBins - 1)]++;
                hC2[clampi(static_cast<int>(std::fabs(lCr) * kHistScaleC), 0, kHistBins - 1)]++;
                total2++;
            }
        }
    }

    if (totalF < 64)
        return;

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
    const float scFine = histQuantile(hCf.data(), kHistBins, totalF * 2, kHistScaleC, 1, 2).value * kMedianCal;
    const float syCoarse = 2.0f * histQuantile(hY2.data(), kHistBins, total2, kHistScaleY, 1, 2).value * kMedianCal;
    const float scCoarse = 2.0f * histQuantile(hC2.data(), kHistBins, total2 * 2, kHistScaleC, 1, 2).value * kMedianCal;

    const float lapSY = std::max(syFine, 0.9f * syCoarse);
    const float lapSC = std::max(scFine, 0.9f * scCoarse);

    float ty = lapSY, tc = lapSC;
    if (diffPartner && totalT >= 64) {
        const float medTY = histQuantile(hYt.data(), kHistBins, totalT, kHistScaleY, 1, 2).value * kMedianCalT;
        const float q20TY = histQuantile(hYt.data(), kHistBins, totalT, kHistScaleY, 1, 5).value * kQ20CalT;
        const float medTC = histQuantile(hCt.data(), kHistBins, totalT * 2, kHistScaleC, 1, 2).value * kMedianCalT;
        const float q20TC = histQuantile(hCt.data(), kHistBins, totalT * 2, kHistScaleC, 1, 5).value * kQ20CalT;
        const float candY = (medTY <= 1.4f * q20TY) ? medTY : q20TY;
        const float candC = (medTC <= 1.4f * q20TC) ? medTC : q20TC;
        if (candY > 0.0015f && candY <= 3.5f * lapSY) ty = candY;
        if (candC > 0.0015f && candC <= 3.5f * lapSC) tc = candC;
        out.hadTemporal = 1;
    }

    const float adj = clampf(p.profileAdjust, 0.25f, 4.0f);
    out.sy = clampf(std::max(lapSY, 0.85f * ty) * adj, kSigmaMin, kSigmaMax);
    out.sc = clampf(std::max(lapSC, 0.85f * tc) * adj, kSigmaMin, kSigmaMax);
    out.ty = clampf(ty * adj, kSigmaMin, kSigmaMax);
    out.tc = clampf(tc * adj, kSigmaMin, kSigmaMax);

    // Brightness-dependent gains, relative to the global fine medians.
    // Per-bin we use the 35th percentile rather than the median: a luma bin
    // can be dominated by legitimate texture, and Q35 stays anchored to the
    // flat (noise-set) pixels as long as they are at least ~a third of the bin.
    const float q35RefY = histQuantile(hYf.data(), kHistBins, totalF, kHistScaleY, 7, 20).value * kQ35Cal;
    const float q35RefC = histQuantile(hCf.data(), kHistBins, totalF * 2, kHistScaleC, 7, 20).value * kQ35Cal;
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
}

// ---------------------------------------------------------------------------
// Stage 2 — motion-adaptive temporal merge (hard-knee gate)
// ---------------------------------------------------------------------------
inline void temporalMerge(const float* const frames[5], int W, int H,
                          const Params& p, const Stats& s, float* tmp)
{
    const float mLow  = std::min(p.master, 1.0f);
    const float mHigh = std::max(p.master, 1.0f);
    const float tL = clampf(p.temporalLuma   * mLow, 0.0f, 1.0f);
    const float tC = clampf(p.temporalChroma * mLow, 0.0f, 1.0f);
    const float thrMul = 0.4f + 2.6f * clampf(p.motionThresh, 0.0f, 1.0f)
                       + 0.8f * (mHigh - 1.0f);   // master>1 widens the knee
    const int reach = (p.enableTemporal == 0) ? 0
                    : ((p.temporalFrames >= 5) ? 2 : 1);
    const float loY = kAbsDiffBias * s.ty, hiY = loY + thrMul * s.ty;
    const float loC = kAbsDiffBias * s.tc, hiC = loC + thrMul * s.tc;
    const float invSpanY = 1.0f / (hiY - loY);
    const float invSpanC = 1.0f / (hiC - loC);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float cy9[9], ccb9[9], ccr9[9];
            int i = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx, ++i)
                    sampleYCC(frames[2], W, H, x + dx, y + dy, cy9[i], ccb9[i], ccr9[i]);

            float accY = cy9[4], accCb = ccb9[4], accCr = ccr9[4];
            float sumWY = 1.0f, sumWY2 = 1.0f, sumWC = 1.0f;

            for (int k = 2 - reach; k <= 2 + reach; ++k) {
                if (k == 2)
                    continue;
                const float* f = frames[k];

                float dY = 0.0f, dC = 0.0f;
                float fy9c = 0.0f, fcb9c = 0.0f, fcr9c = 0.0f;
                i = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx, ++i) {
                        float fy, fcb, fcr;
                        sampleYCC(f, W, H, x + dx, y + dy, fy, fcb, fcr);
                        if (i == 4) { fy9c = fy; fcb9c = fcb; fcr9c = fcr; }
                        dY += std::fabs(fy - cy9[i]);
                        dC += 0.5f * (std::fabs(fcb - ccb9[i]) + std::fabs(fcr - ccr9[i]));
                    }
                }
                dY *= (1.0f / 9.0f);
                dC *= (1.0f / 9.0f);

                const float gY = 1.0f - smooth01((dY - loY) * invSpanY);
                const float gC = 1.0f - smooth01((dC - loC) * invSpanC);
                const float wY = tL * gY;
                const float wC = tC * gC * gY;   // chroma slaved to the luma gate

                accY  += wY * fy9c;
                accCb += wC * fcb9c;
                accCr += wC * fcr9c;
                sumWY  += wY;
                sumWY2 += wY * wY;
                sumWC  += wC;
            }

            float* t = tmp + (static_cast<size_t>(y) * W + x) * 4;
            t[0] = accY / sumWY;
            t[1] = accCb / sumWC;
            t[2] = accCr / sumWC;
            t[3] = (sumWY * sumWY) / sumWY2;
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 3 — residual noise measurement on the temporal result
// ---------------------------------------------------------------------------
inline void estimateResidual(const float* tmp, int W, int H, const Params& p, Stats& s)
{
    if (p.profileSource == 2) {   // manual: residual = the entered sigmas
        s.ry = clampf(p.sigmaY, kSigmaMin, kSigmaMax);
        s.rc = clampf(p.sigmaC, kSigmaMin, kSigmaMax);
        s.effNMed = 1.0f;
        return;
    }

    std::vector<uint32_t> hYr(kHistBins, 0), hCr(kHistBins, 0), hN(32, 0);
    uint64_t total = 0;
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
            hYr[clampi(static_cast<int>(std::fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)]++;
            hCr[clampi(static_cast<int>(std::fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)]++;
            hCr[clampi(static_cast<int>(std::fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)]++;
            const float effN = tmpAt(tmp, W, H, x, y)[3];
            hN[clampi(static_cast<int>((effN - 1.0f) * 8.0f), 0, 31)]++;
            total++;
        }
    }
    if (total < 64) {
        s.ry = s.sy; s.rc = s.sc; s.effNMed = 1.0f;
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

    const float adj = clampf(p.profileAdjust, 0.25f, 4.0f);
    float ry = med(hYr, total, kHistScaleY) * kMedianCal * adj;
    float rc = med(hCr, total * 2, kHistScaleC) * kMedianCal * adj;
    s.effNMed = 1.0f + med(hN, total, 8.0f);

    // Floors: the residual cannot be less than the theoretical reduction, and
    // never exceeds the input estimate.
    const float floorY = 0.5f * s.sy / std::sqrt(std::max(1.0f, s.effNMed));
    const float floorC = 0.5f * s.sc / std::sqrt(std::max(1.0f, s.effNMed));
    s.ry = clampf(std::max(ry, floorY), kSigmaMin, s.sy > kSigmaMin ? s.sy : kSigmaMax);
    s.rc = clampf(std::max(rc, floorC), kSigmaMin, s.sc > kSigmaMin ? s.sc : kSigmaMax);
}

// ---------------------------------------------------------------------------
// HUD v2 (Noise Analysis view)
// ---------------------------------------------------------------------------
// glyph order: 0-9 . % A C E F I L M O P R S T U Y B D G N + space
static const uint64_t kFont[32] = {
    0x3a33ae62eULL, 0x11842108eULL, 0x3a213221fULL, 0x3a213062eULL, 0x08ca97c42ULL, 0x7e1e0862eULL,
    0x3a10f462eULL, 0x7c2222108ULL, 0x3a317462eULL, 0x3a317842eULL, 0x00000018cULL, 0x632222263ULL,
    0x3a31fc631ULL, 0x3a308422eULL, 0x7e10f421fULL, 0x7e10f4210ULL, 0x38842108eULL, 0x42108421fULL,
    0x4775ac631ULL, 0x3a318c62eULL, 0x7a31f4210ULL, 0x7a31f5251ULL, 0x3e107043eULL, 0x7c8421084ULL,
    0x46318c62eULL, 0x462a21084ULL, 0x7a31f463eULL, 0x7a318c63eULL, 0x3a30bc62fULL, 0x47359c631ULL,
    0x0084f9080ULL, 0x000000000ULL,
};
#define NR_G_DOT 10
#define NR_G_PCT 11
#define NR_G_A 12
#define NR_G_C 13
#define NR_G_E 14
#define NR_G_F 15
#define NR_G_I 16
#define NR_G_L 17
#define NR_G_M 18
#define NR_G_O 19
#define NR_G_P 20
#define NR_G_R 21
#define NR_G_S 22
#define NR_G_T 23
#define NR_G_U 24
#define NR_G_Y 25
#define NR_G_B 26
#define NR_G_D 27
#define NR_G_G 28
#define NR_G_N 29
#define NR_G_PLUS 30
#define NR_G_SP 31

static inline bool glyphPixel(int glyph, int gx, int gy)
{
    if (glyph < 0 || glyph > 31 || gx < 0 || gx >= 5 || gy < 0 || gy >= 7)
        return false;
    return (kFont[glyph] >> (34 - (gy * 5 + gx))) & 1ULL;
}

static inline bool textPixel(const int* chars, int n, int tx, int ty, int lx, int ly)
{
    if (ly < ty || ly >= ty + 7 || lx < tx || lx >= tx + n * 6)
        return false;
    const int ci = (lx - tx) / 6;
    return glyphPixel(chars[ci], (lx - tx) - ci * 6, ly - ty);
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

// Renders the analysis panel over (r,g,b) for absolute pixel (x,y).
static inline bool hudPixel(int x, int y, int W, int H, const Stats& st,
                            int enableTemporal, float& r, float& g, float& b)
{
    const int s = std::max(1, H / 540);
    const int ox = 16 * s, oy = 16 * s;
    const int lw = 320, lh = 198;
    if (x < ox || y < oy || x >= ox + lw * s || y >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (y - oy) / s;

    r = r * 0.20f + 0.015f; g = g * 0.20f + 0.015f; b = b * 0.20f + 0.02f;
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        r = g = b = 0.35f;
        return true;
    }

    static const int kLabIY[4]  = { NR_G_I, NR_G_N, NR_G_SP, NR_G_Y };
    static const int kLabIC[4]  = { NR_G_I, NR_G_N, NR_G_SP, NR_G_C };
    static const int kLabRY[10] = { NR_G_R, NR_G_E, NR_G_S, NR_G_I, NR_G_D, NR_G_U, NR_G_A, NR_G_L, NR_G_SP, NR_G_Y };
    static const int kLabRC[10] = { NR_G_R, NR_G_E, NR_G_S, NR_G_I, NR_G_D, NR_G_U, NR_G_A, NR_G_L, NR_G_SP, NR_G_C };
    static const int kLabOFF[3] = { NR_G_O, NR_G_F, NR_G_F };
    static const int kLabEFFN[5] = { NR_G_E, NR_G_F, NR_G_F, NR_G_SP, NR_G_N };
    static const int kLabGAIN[4] = { NR_G_G, NR_G_A, NR_G_I, NR_G_N };
    static const int kLabDB[2]   = { NR_G_D, NR_G_B };

    const float sig[4] = { st.sy, st.sc, st.ry, st.rc };
    const int rowY[4] = { 6, 28, 50, 72 };

    for (int row = 0; row < 4; ++row) {
        const int ty0 = rowY[row];
        bool lit = false;
        switch (row) {
        case 0: lit = textPixel(kLabIY, 4,  8, ty0, lx, ly); break;
        case 1: lit = textPixel(kLabIC, 4,  8, ty0, lx, ly); break;
        case 2: lit = textPixel(kLabRY, 10, 8, ty0, lx, ly); break;
        case 3: lit = textPixel(kLabRC, 10, 8, ty0, lx, ly); break;
        }
        if (!lit) {
            int vg[6];
            pctGlyphs(sig[row] * 100.0f, vg);
            lit = textPixel(vg, 6, 252, ty0, lx, ly);
        }
        if (lit) { r = g = b = 1.0f; return true; }

        if (ly >= ty0 + 9 && ly < ty0 + 13 && lx >= 8 && lx < 288) {
            const float fill = clampf(sig[row] / 0.08f, 0.0f, 1.0f);
            const bool on = (lx - 8) < static_cast<int>(fill * 280.0f);
            const bool residRow = (row >= 2);
            if (on) {
                if (residRow) { r = 0.95f; g = 0.65f; b = 0.20f; }
                else          { r = 0.20f; g = 0.65f; b = 0.95f; }
            } else { r = g = b = 0.16f; }
            return true;
        }
    }

    // info line: "EFF N x.x    GAIN +x.x DB"
    {
        const int ty0 = 94;
        bool lit = textPixel(kLabEFFN, 5, 8, ty0, lx, ly);
        if (!lit) {
            int vg[3];
            dec1Glyphs(enableTemporal ? st.effNMed : 1.0f, vg);
            lit = textPixel(vg, 3, 44, ty0, lx, ly);
        }
        if (!lit) lit = textPixel(kLabGAIN, 4, 120, ty0, lx, ly);
        if (!lit) {
            const float gainDb = clampf(20.0f * std::log10(std::max(st.sy, 1e-5f) / std::max(st.ry, 1e-5f)), 0.0f, 40.0f);
            int vg[4];
            vg[0] = NR_G_PLUS;
            int d1[3];
            dec1Glyphs(gainDb, d1);
            vg[1] = d1[0]; vg[2] = d1[1]; vg[3] = d1[2];
            lit = textPixel(vg, 4, 150, ty0, lx, ly);
            if (!lit) lit = textPixel(kLabDB, 2, 178, ty0, lx, ly);
        }
        if (lit) { r = g = b = 1.0f; return true; }
        if (enableTemporal == 0 && textPixel(kLabOFF, 3, 220, ty0, lx, ly)) { r = g = b = 0.7f; return true; }
    }

    // noise vs brightness curve: 16 bars, box x [8,264), y [108,148);
    // dim line marks gain = 1.0 so a flat profile reads as flat
    if (lx >= 8 && lx < 264 && ly >= 108 && ly < 148) {
        const int bin = clampi((lx - 8) / 16, 0, kLumaBins - 1);
        const float v = clampf((st.gainY[bin] - 0.6f) / 1.6f, 0.0f, 1.0f);
        const bool bar = (147 - ly) < static_cast<int>(v * 39.0f + 0.5f);
        const bool ref = (147 - ly) == static_cast<int>((1.0f - 0.6f) / 1.6f * 39.0f + 0.5f);
        if (bar)      { r = 0.20f; g = 0.65f; b = 0.95f; }
        else if (ref) { r = g = b = 0.32f; }
        else          { r = g = b = 0.10f; }
        return true;
    }

    // fine luma |laplacian| histogram with median marker: y [156,192)
    if (lx >= 8 && lx < 268 && ly >= 156 && ly < 192) {
        const int bin = clampi((lx - 8) * kHistBins / 260, 0, kHistBins - 1);
        const float hgt = 35.0f * static_cast<float>(st.histY[bin]) / static_cast<float>(st.histMax);
        const bool bar = (191 - ly) < static_cast<int>(hgt + 0.5f);
        if (bin == static_cast<int>(st.medBinY)) { r = 0.95f; g = 0.85f; b = 0.15f; }
        else if (bar)                            { r = g = b = 0.55f; }
        else                                     { r = g = b = 0.10f; }
        return true;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Stage 4+5+6 — spatial NLM + chroma blotch + refine + output assembly
// ---------------------------------------------------------------------------
inline void spatialNLM(const float* tmp, const float* curr, int W, int H,
                       const Params& p, const Stats& s, float* out)
{
    const float mLow  = std::min(p.master, 1.0f);
    const float mHigh = std::max(p.master, 1.0f);
    const float hBoost = std::pow(mHigh, 1.2f);

    const float sL = clampf(p.spatialLuma, 0.0f, 1.0f);
    const float sC = clampf(p.spatialChroma, 0.0f, 1.0f);
    const float aY = (p.enableSpatial == 0) ? 0.0f : clampf(sL * mLow, 0.0f, 1.0f);
    const float aC = (p.enableSpatial == 0) ? 0.0f : clampf(sC * mLow, 0.0f, 1.0f);
    const float hMulY = (0.6f + 1.4f * std::pow(sL, 1.5f)) * hBoost;
    const float hMulC = (0.6f + 1.4f * std::pow(sC, 1.5f)) * hBoost;
    const float pd = clampf(p.preserveDetail, 0.0f, 1.0f);
    const int   R  = clampi(p.spatialRadius, 1, 8);
    const bool  nlm = (p.spatialMode == 1);
    const bool  runSpatial = (p.enableSpatial != 0) && (aY > 0.0f || aC > 0.0f);
    const float spatialSigma = std::max(1.0f, R / 1.5f);
    const float invSpatial2 = 1.0f / (2.0f * spatialSigma * spatialSigma);

    const float blotch = (p.enableSpatial != 0) ? clampf(p.chromaBlotch, 0.0f, 1.0f) * mLow : 0.0f;
    const int   Rb = 2 + static_cast<int>(14.0f * clampf(p.chromaBlotch, 0.0f, 1.0f));

    const bool refine = (p.enableRefine != 0) && (p.master > 0.0f);
    const float desat = refine ? clampf(p.shadowDesat, 0.0f, 1.0f) : 0.0f;
    const float desatRange = std::max(0.02f, p.desatRange);
    const float tex = refine ? clampf(p.lumaTexture, 0.0f, 1.0f) * mLow : 0.0f;
    const float grainAmt = refine ? clampf(p.grainAmount, 0.0f, 1.0f) * 0.06f : 0.0f;
    const float grainSize = clampf(p.grainSize, 0.5f, 6.0f);
    const float grainCh = clampf(p.grainChroma, 0.0f, 1.0f);
    const uint32_t frame = static_cast<uint32_t>(p.frameIndex);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t idx = (static_cast<size_t>(y) * W + x) * 4;
            const float* tc = tmp + idx;

            const int lb = clampi(static_cast<int>(tc[0] * kLumaBins), 0, kLumaBins - 1);
            const float sigY = clampf(s.ry * s.gainY[lb], 1e-5f, 1.0f);
            const float sigC = clampf(s.rc * s.gainC[lb], 1e-5f, 1.0f);

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
                const float hC = kNlmHChroma * sigC * hMulC * (1.0f - pd * 0.50f * edginess);
                const float invHY2  = 1.0f / std::max(hY * hY, 1e-12f);
                const float invHC2  = 1.0f / std::max(hC * hC, 1e-12f);
                const float biasY = 2.0f * sigY * sigY;
                const float biasC = 2.0f * sigC * sigC;

                float accY = 0.0f, accCb = 0.0f, accCr = 0.0f;
                float sumWY = 0.0f, sumWC = 0.0f, wYmax = 0.0f, wCmax = 0.0f;

                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        if (dx == 0 && dy == 0)
                            continue;
                        const float* ts = tmpAt(tmp, W, H, x + dx, y + dy);

                        float dY2, dC2;
                        if (nlm) {
                            dY2 = 0.0f; dC2 = 0.0f;
                            for (int qy = -1; qy <= 1; ++qy) {
                                for (int qx = -1; qx <= 1; ++qx) {
                                    const float* tp = tmpAt(tmp, W, H, x + qx, y + qy);
                                    const float* tq = tmpAt(tmp, W, H, x + dx + qx, y + dy + qy);
                                    const float eY = tp[0] - tq[0];
                                    const float eCb = tp[1] - tq[1];
                                    const float eCr = tp[2] - tq[2];
                                    dY2 += eY * eY;
                                    dC2 += 0.5f * (eCb * eCb + eCr * eCr);
                                }
                            }
                            dY2 *= (1.0f / 9.0f);
                            dC2 *= (1.0f / 9.0f);
                        } else {
                            const float eY = tc[0] - ts[0];
                            const float eCb = tc[1] - ts[1];
                            const float eCr = tc[2] - ts[2];
                            dY2 = eY * eY;
                            dC2 = 0.5f * (eCb * eCb + eCr * eCr);
                        }

                        dY2 = std::max(0.0f, dY2 - biasY);
                        dC2 = std::max(0.0f, dC2 - biasC);

                        float wY = std::exp(-dY2 * invHY2);
                        float wC = std::exp(-dC2 * invHC2) * std::exp(-dY2 * invHY2 * 0.25f);
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

                Yo  = tc[0] + aY * (Yf  - tc[0]);
                Cbo = tc[1] + aC * (Cbf - tc[1]);
                Cro = tc[2] + aC * (Crf - tc[2]);
            }

            // large-radius sparse chroma pass: reaches blotches NLM can't
            if (blotch > 0.0f) {
                static const float kDirX[8] = { 1, 0, -1, 0, 0.7071f, -0.7071f, -0.7071f, 0.7071f };
                static const float kDirY[8] = { 0, 1, 0, -1, 0.7071f, 0.7071f, -0.7071f, -0.7071f };
                const float gyDen = 1.0f / std::max(2.0f * sigY * hBoost, 1e-6f);
                const float gcDen = 1.0f / std::max(3.0f * sigC * hBoost, 1e-6f);
                float accB = tc[1], accR = tc[2], sumW = 1.0f;
                for (int d = 0; d < 8; ++d) {
                    for (int ri = 1; ri <= 3; ++ri) {
                        const float rr = Rb * (static_cast<float>(ri) / 3.0f);
                        const float* ts = tmpAt(tmp, W, H,
                                                x + static_cast<int>(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                                                y + static_cast<int>(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f)));
                        const float eY = (ts[0] - tc[0]) * gyDen;
                        const float eC = (0.5f * (std::fabs(ts[1] - tc[1]) + std::fabs(ts[2] - tc[2]))) * gcDen;
                        const float w = std::exp(-(eY * eY + eC * eC));
                        accB += w * ts[1];
                        accR += w * ts[2];
                        sumW += w;
                    }
                }
                Cbo += blotch * (accB / sumW - Cbo);
                Cro += blotch * (accR / sumW - Cro);
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

            float r, g, b;
            ycc2rgb(Yr, Cbr, Crr, r, g, b);
            float dnR, dnG, dnB;                    // denoised, pre-refine (for Noise Removed)
            ycc2rgb(Yo, Cbo, Cro, dnR, dnG, dnB);

            float* o = out + idx;

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
            case 3: { // after temporal (mid-pipeline)
                float rr, gg, bb;
                ycc2rgb(tc[0], tc[1], tc[2], rr, gg, bb);
                o[0] = rr; o[1] = gg; o[2] = bb;
                break;
            }
            case 4: { // noise removed (denoise only, pre-refine), amplified
                o[0] = 0.5f + (c[0] - dnR) * 4.0f;
                o[1] = 0.5f + (c[1] - dnG) * 4.0f;
                o[2] = 0.5f + (c[2] - dnB) * 4.0f;
                break;
            }
            case 5: { // noise analysis HUD + region rect
                float rr = r, gg = g, bb = b;
                if (!hudPixel(x, y, W, H, s, p.enableTemporal, rr, gg, bb) &&
                    p.profileSource == 1) {
                    const float rHalf = 0.5f * p.regionSize * static_cast<float>(std::min(W, H));
                    const float cx = p.regionCX * W, cyy = p.regionCY * H;
                    const float ax = std::fabs(x - cx), ay = std::fabs(y - cyy);
                    const bool onEdge = (ax <= rHalf && ay <= rHalf) &&
                                        (ax >= rHalf - 2.0f || ay >= rHalf - 2.0f);
                    if (onEdge) { rr = 1.0f; gg = 1.0f; bb = 0.1f; }
                }
                o[0] = rr; o[1] = gg; o[2] = bb;
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
            default:
                o[0] = r; o[1] = g; o[2] = b;
                break;
            }
            o[3] = c[3];
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience full pipeline (CPU). Returns the stats actually used.
// ---------------------------------------------------------------------------
inline Stats denoiseFrame(const float* const frames[5], int W, int H,
                          const Params& p, float* out, std::vector<float>& scratch)
{
    scratch.resize(static_cast<size_t>(W) * H * 4);
    const float* partner = nullptr;
    if (frames[1] != frames[2]) partner = frames[1];
    else if (frames[3] != frames[2]) partner = frames[3];

    Stats s;
    estimateInput(frames[2], partner, W, H, p, s);
    temporalMerge(frames, W, H, p, s, scratch.data());
    estimateResidual(scratch.data(), W, H, p, s);
    spatialNLM(scratch.data(), frames[2], W, H, p, s, out);
    return s;
}

} // namespace nrcore

#endif // OPENNR_NR_CORE_H
