// OpenNR — free spatio-temporal noise reduction for DaVinci Resolve (OpenFX)
// nr_core.h — reference CPU implementation of the full algorithm.
//
// This header is the single source of truth for the math. The Metal / CUDA /
// OpenCL kernels are line-by-line ports of these functions; the test harness
// (test/test_denoise.cpp) validates this implementation against synthetic
// footage with known noise, so any change here must keep the tests passing.
//
// Pipeline per rendered frame:
//   1. estimateSigma()  — automatic noise profiling with three estimators:
//        a. temporal:   median |frame difference| — the primary estimator for
//           video. Immune to spatial noise correlation (debayer, chroma
//           subsampling, compression), robust to motion via the median and a
//           spatial-estimate clamp.
//        b. spatial fine:   median |Laplacian| (Immerkaer) — catches
//           per-pixel noise; underestimates correlated noise.
//        c. spatial coarse: same on a 2x downsampled image — catches
//           spatially correlated (blotchy) noise the fine scale misses.
//      Produces two sigma pairs: sigmaS (spatial filtering) = max(fine,
//      0.9*coarse), and sigmaT (temporal gating) = temporal estimate clamped
//      to <= 1.5*sigmaS (so global motion cannot inflate it).
//   2. temporalMerge()  — motion-adaptive weighted average of up to 5 frames.
//   3. spatialNLM()     — non-local-means (or fast bilateral) on the temporal
//      result, sigma-adaptive, with separate luma/chroma strengths.
//   4. Output assembly — result / split / noise-removed / a rendered
//      Noise Analysis HUD (live sigma readouts, meters, histogram) / a
//      temporal-activity heat map.
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
    float sigmaY         = 0.02f;  // manual luma sigma (signal units)
    float sigmaC         = 0.02f;  // manual chroma sigma
    float profileAdjust  = 1.0f;   // 0.25..4 multiplier on the auto estimate
    float regionCX       = 0.5f;
    float regionCY       = 0.5f;
    float regionSize     = 0.25f;

    int   enableTemporal = 1;
    int   temporalFrames = 3;      // 3 or 5
    float temporalLuma   = 0.6f;
    float temporalChroma = 0.8f;
    float motionThresh   = 0.4f;

    int   enableSpatial  = 1;
    int   spatialMode    = 1;      // 0 = faster (bilateral), 1 = better (NLM)
    int   spatialRadius  = 3;
    float spatialLuma    = 0.45f;
    float spatialChroma  = 0.75f;
    float preserveDetail = 0.35f;

    float master         = 1.0f;   // 0..2
    int   viewMode       = 0;      // 0 result, 1 split, 2 noise, 3 analysis, 4 temporal map
};

// Sigma pairs plus the data the analysis HUD displays.
struct Stats {
    float sy = 0.02f;  // spatial-filtering sigmas
    float sc = 0.02f;
    float ty = 0.02f;  // temporal-gating sigmas
    float tc = 0.02f;
    uint32_t histY[256] = {0};  // fine luma |laplacian| histogram (for the HUD)
    uint32_t medBinY = 0;
    uint32_t histMax = 1;
    int hadTemporal = 0;
};

// ---------------------------------------------------------------------------
// Calibration constants (derivations in comments; verified by the test suite)
// ---------------------------------------------------------------------------
// Immerkaer operator M = [1 -2 1; -2 4 -2; 1 -2 1]: response std = 6*sigma
// for iid noise; median of |N(0,s)| = 0.674490*s.
static const float kMedianCal   = 1.0f / (6.0f * 0.674490f);      // 0.247100
// |A - B| of two iid frames: std sqrt(2)*sigma; median = 0.674490*sqrt(2)*sigma.
static const float kMedianCalT  = 1.0f / (0.674490f * 1.414214f); // 1.048360
// 20th percentile of |N(0,s)| = 0.253347*s (motion-robust temporal quantile:
// if at least a fifth of the sampled pixels are static, Q20 is noise-set).
static const float kQ20CalT     = 1.0f / (0.253347f * 1.414214f); // 2.791278
// E|a-b| for two iid N(mu,sigma) samples = 2*sigma/sqrt(pi).
static const float kAbsDiffBias = 1.128379f;
// NLM filter strength as a multiple of the effective noise sigma.
static const float kNlmHLuma    = 1.15f;
static const float kNlmHChroma  = 2.00f;

static const int   kHistBins    = 256;
static const float kHistScaleY  = 512.0f;
static const float kHistScaleC  = 1024.0f;
static const float kSigmaMin    = 1e-4f;
static const float kSigmaMax    = 0.25f;

// ---------------------------------------------------------------------------
// Color helpers (BT.709 full-range luma/chroma opponent space)
// ---------------------------------------------------------------------------
static inline float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

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

// 2x2 block mean in YCC, block origin (x, y)
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

// ---------------------------------------------------------------------------
// Stage 1 — automatic noise profiling
// ---------------------------------------------------------------------------
// diffPartner: a *different* frame (t-1 or t+1) for the temporal estimator,
// or nullptr when unavailable (stills, single-frame clips, temporal off).
inline Stats estimateSigma(const float* rgba, const float* diffPartner,
                           int W, int H, const Params& p)
{
    Stats out;
    out.sy = out.ty = clampf(p.sigmaY, kSigmaMin, kSigmaMax);
    out.sc = out.tc = clampf(p.sigmaC, kSigmaMin, kSigmaMax);
    if (p.profileSource == 2)   // manual
        return out;

    std::vector<uint32_t> hYf(kHistBins, 0), hCf(kHistBins, 0);
    std::vector<uint32_t> hY2(kHistBins, 0), hC2(kHistBins, 0);
    std::vector<uint32_t> hYt(kHistBins, 0), hCt(kHistBins, 0);

    int x0 = 1, x1 = W - 1, y0 = 1, y1 = H - 1;
    if (p.profileSource == 1) {
        const float rHalf = 0.5f * p.regionSize * static_cast<float>(std::min(W, H));
        const float cx = p.regionCX * W, cy = p.regionCY * H;
        x0 = clampi(static_cast<int>(cx - rHalf), 1, W - 1);
        x1 = clampi(static_cast<int>(cx + rHalf), 1, W - 1);
        y0 = clampi(static_cast<int>(cy - rHalf), 1, H - 1);
        y1 = clampi(static_cast<int>(cy + rHalf), 1, H - 1);
    }

    // Snap loop starts to the global 2x2 sampling lattice (odd coordinates) so
    // that region mode samples exactly the same pixels as the GPU kernels.
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

            // fine-scale Laplacian
            const float lapY  = 4.0f * Y[4]  - 2.0f * (Y[1] + Y[3] + Y[5] + Y[7])   + (Y[0] + Y[2] + Y[6] + Y[8]);
            const float lapCb = 4.0f * Cb[4] - 2.0f * (Cb[1] + Cb[3] + Cb[5] + Cb[7]) + (Cb[0] + Cb[2] + Cb[6] + Cb[8]);
            const float lapCr = 4.0f * Cr[4] - 2.0f * (Cr[1] + Cr[3] + Cr[5] + Cr[7]) + (Cr[0] + Cr[2] + Cr[6] + Cr[8]);
            hYf[clampi(static_cast<int>(std::fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)]++;
            hCf[clampi(static_cast<int>(std::fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)]++;
            hCf[clampi(static_cast<int>(std::fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)]++;
            totalF++;

            // temporal difference (per-pixel, center only)
            if (diffPartner) {
                float py, pcb, pcr;
                sampleYCC(diffPartner, W, H, x, y, py, pcb, pcr);
                hYt[clampi(static_cast<int>(std::fabs(py - Y[4])   * kHistScaleY), 0, kHistBins - 1)]++;
                hCt[clampi(static_cast<int>(std::fabs(pcb - Cb[4]) * kHistScaleC), 0, kHistBins - 1)]++;
                hCt[clampi(static_cast<int>(std::fabs(pcr - Cr[4]) * kHistScaleC), 0, kHistBins - 1)]++;
                totalT++;
            }

            // coarse-scale Laplacian on 2x2 block means, on a stride-4 sub-lattice
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
        return out;   // degenerate region: keep manual values

    struct QuantileResult { float value; uint32_t bin; };
    // quantile at cum >= ceil(total * num / den)
    auto histQuantile = [](const std::vector<uint32_t>& h, uint64_t total, float scale,
                           uint64_t num, uint64_t den) {
        uint64_t cum = 0;
        const uint64_t target = (total * num + den - 1) / den;
        for (int b = 0; b < kHistBins; ++b) {
            cum += h[b];
            if (cum >= target)
                return QuantileResult{ (static_cast<float>(b) + 0.5f) / scale, static_cast<uint32_t>(b) };
        }
        return QuantileResult{ static_cast<float>(kHistBins - 0.5f) / scale, kHistBins - 1 };
    };

    const QuantileResult mYf = histQuantile(hYf, totalF, kHistScaleY, 1, 2);
    const float syFine = mYf.value * kMedianCal;
    const float scFine = histQuantile(hCf, totalF * 2, kHistScaleC, 1, 2).value * kMedianCal;
    // coarse estimate maps back to full-res equivalent via the 2x factor:
    // iid noise gives coarse == fine (no boost); correlated noise boosts.
    const float syCoarse = 2.0f * histQuantile(hY2, total2, kHistScaleY, 1, 2).value * kMedianCal;
    const float scCoarse = 2.0f * histQuantile(hC2, total2 * 2, kHistScaleC, 1, 2).value * kMedianCal;

    // Laplacian-family spatial floor (underestimates correlated noise).
    const float lapSY = std::max(syFine, 0.9f * syCoarse);
    const float lapSC = std::max(scFine, 0.9f * scCoarse);

    // Temporal-difference estimator: measures the TRUE total per-pixel noise
    // (immune to spatial correlation). Motion robustness: the unbiased median
    // and 20th-percentile estimates agree on static footage; motion inflates
    // the median first. When they agree, use the (more precise) median; when
    // they disagree, the frame is partly in motion — trust only the static
    // quantile. If even Q20 vastly exceeds the Laplacian family, the whole
    // frame is moving and the temporal signal says nothing about noise.
    float ty = lapSY, tc = lapSC;
    if (diffPartner && totalT >= 64) {
        const float medTY = histQuantile(hYt, totalT, kHistScaleY, 1, 2).value * kMedianCalT;
        const float q20TY = histQuantile(hYt, totalT, kHistScaleY, 1, 5).value * kQ20CalT;
        const float medTC = histQuantile(hCt, totalT * 2, kHistScaleC, 1, 2).value * kMedianCalT;
        const float q20TC = histQuantile(hCt, totalT * 2, kHistScaleC, 1, 5).value * kQ20CalT;
        const float candY = (medTY <= 1.4f * q20TY) ? medTY : q20TY;
        const float candC = (medTC <= 1.4f * q20TC) ? medTC : q20TC;
        // candidates near zero mean the "partner" frame was a duplicate
        // (still, freeze frame, clip boundary) — no temporal information.
        if (candY > 0.0015f && candY <= 3.5f * lapSY)
            ty = candY;
        if (candC > 0.0015f && candC <= 3.5f * lapSC)
            tc = candC;
        out.hadTemporal = 1;
    }

    // Spatial filtering needs the total noise energy too; the vetted temporal
    // estimate feeds back as a floor so correlated noise is filtered properly.
    float sy = std::max(lapSY, 0.85f * ty);
    float sc = std::max(lapSC, 0.85f * tc);

    const float adj = clampf(p.profileAdjust, 0.25f, 4.0f);
    out.sy = clampf(sy * adj, kSigmaMin, kSigmaMax);
    out.sc = clampf(sc * adj, kSigmaMin, kSigmaMax);
    out.ty = clampf(ty * adj, kSigmaMin, kSigmaMax);
    out.tc = clampf(tc * adj, kSigmaMin, kSigmaMax);

    // HUD data
    for (int b = 0; b < kHistBins; ++b) {
        out.histY[b] = hYf[b];
        out.histMax = std::max(out.histMax, hYf[b]);
    }
    out.medBinY = mYf.bin;
    return out;
}

// ---------------------------------------------------------------------------
// Stage 2 — motion-adaptive temporal merge
// ---------------------------------------------------------------------------
// tmp per pixel: [Y', Cb', Cr', effN], effN = (sum w)^2 / sum w^2 in 1..5.
inline void temporalMerge(const float* const frames[5], int W, int H,
                          const Params& p, const Stats& s, float* tmp)
{
    const float tL = clampf(p.temporalLuma   * p.master, 0.0f, 1.0f);
    const float tC = clampf(p.temporalChroma * p.master, 0.0f, 1.0f);
    // Hard-knee gate: full weight while a patch difference is explainable by
    // noise (below lo = expected pure-noise difference), smoothstep to EXACT
    // ZERO at hi. No tail — mismatched pixels can never bleed in (no ghosting
    // beyond the knee, by construction).
    const float thrMul = 0.4f + 2.6f * clampf(p.motionThresh, 0.0f, 1.0f);
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

                const float tY01 = clampf((dY - loY) * invSpanY, 0.0f, 1.0f);
                const float gY = 1.0f - tY01 * tY01 * (3.0f - 2.0f * tY01);
                const float tC01 = clampf((dC - loC) * invSpanC, 0.0f, 1.0f);
                const float gC = 1.0f - tC01 * tC01 * (3.0f - 2.0f * tC01);
                const float wY = tL * gY;
                const float wC = tC * gC * gY;   // chroma fully slaved to the luma gate

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
// HUD rendering (Noise Analysis view) — 5x7 pixel font, meters, histogram.
// Panel local space is 300 x 134, scaled by s = max(1, H/540).
// ---------------------------------------------------------------------------
// glyph order: 0-9 . % A C E F I L M O P R S T U Y space
static const uint64_t kFont[27] = {
    0x3a33ae62eULL, 0x11842108eULL, 0x3a213221fULL, 0x3a213062eULL, 0x08ca97c42ULL, 0x7e1e0862eULL,
    0x3a10f462eULL, 0x7c2222108ULL, 0x3a317462eULL, 0x3a317842eULL, 0x00000018cULL, 0x632222263ULL,
    0x3a31fc631ULL, 0x3a308422eULL, 0x7e10f421fULL, 0x7e10f4210ULL, 0x38842108eULL, 0x42108421fULL,
    0x4775ac631ULL, 0x3a318c62eULL, 0x7a31f4210ULL, 0x7a31f5251ULL, 0x3e107043eULL, 0x7c8421084ULL,
    0x46318c62eULL, 0x462a21084ULL, 0x000000000ULL,
};
// glyph indices
#define NR_G_DOT   10
#define NR_G_PCT   11
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
#define NR_G_SP 26

static inline bool glyphPixel(int glyph, int gx, int gy)
{
    if (glyph < 0 || glyph > 26 || gx < 0 || gx >= 5 || gy < 0 || gy >= 7)
        return false;
    return (kFont[glyph] >> (34 - (gy * 5 + gx))) & 1ULL;
}

// Is panel-local pixel (lx,ly) lit by the string at (tx,ty)? chars: glyph ids,
// 6 px pitch.
static inline bool textPixel(const int* chars, int n, int tx, int ty, int lx, int ly)
{
    if (ly < ty || ly >= ty + 7 || lx < tx || lx >= tx + n * 6)
        return false;
    const int ci = (lx - tx) / 6;
    return glyphPixel(chars[ci], (lx - tx) - ci * 6, ly - ty);
}

// digits of a percentage value pp (= sigma*100) as "TO.dh%" with blank leading zero
static inline void pctGlyphs(float pp, int out[6])
{
    const int v = clampi(static_cast<int>(pp * 100.0f + 0.5f), 0, 9999); // hundredths of a percent
    const int tens = (v / 1000) % 10;
    out[0] = (tens == 0) ? NR_G_SP : tens;
    out[1] = (v / 100) % 10;
    out[2] = NR_G_DOT;
    out[3] = (v / 10) % 10;
    out[4] = v % 10;
    out[5] = NR_G_PCT;
}

// Renders the analysis panel over (r,g,b) for absolute pixel (x,y).
// Returns true if the pixel was inside the panel.
static inline bool hudPixel(int x, int y, int W, int H, const Stats& st,
                            int enableTemporal, float& r, float& g, float& b)
{
    const int s = std::max(1, H / 540);
    const int ox = 16 * s, oy = 16 * s;
    const int lw = 300, lh = 134;
    if (x < ox || y < oy || x >= ox + lw * s || y >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (y - oy) / s;

    // background + border
    r = r * 0.20f + 0.015f; g = g * 0.20f + 0.015f; b = b * 0.20f + 0.02f;
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        r = g = b = 0.35f;
        return true;
    }

    static const int kLabSY[9]  = { NR_G_S, NR_G_P, NR_G_A, NR_G_T, NR_G_I, NR_G_A, NR_G_L, NR_G_SP, NR_G_Y };
    static const int kLabSC[9]  = { NR_G_S, NR_G_P, NR_G_A, NR_G_T, NR_G_I, NR_G_A, NR_G_L, NR_G_SP, NR_G_C };
    static const int kLabTY[10] = { NR_G_T, NR_G_E, NR_G_M, NR_G_P, NR_G_O, NR_G_R, NR_G_A, NR_G_L, NR_G_SP, NR_G_Y };
    static const int kLabTC[10] = { NR_G_T, NR_G_E, NR_G_M, NR_G_P, NR_G_O, NR_G_R, NR_G_A, NR_G_L, NR_G_SP, NR_G_C };
    static const int kLabOFF[3] = { NR_G_O, NR_G_F, NR_G_F };

    const float sig[4] = { st.sy, st.sc, st.ty, st.tc };
    const int   rowY[4] = { 6, 30, 54, 78 };

    for (int row = 0; row < 4; ++row) {
        const int ty0 = rowY[row];
        // label
        bool lit = false;
        switch (row) {
        case 0: lit = textPixel(kLabSY, 9,  8, ty0, lx, ly); break;
        case 1: lit = textPixel(kLabSC, 9,  8, ty0, lx, ly); break;
        case 2: lit = textPixel(kLabTY, 10, 8, ty0, lx, ly); break;
        case 3: lit = textPixel(kLabTC, 10, 8, ty0, lx, ly); break;
        }
        // value (right side) or OFF for temporal rows when disabled
        const bool tOff = (row >= 2) && (enableTemporal == 0);
        if (!lit) {
            if (tOff) {
                lit = textPixel(kLabOFF, 3, 250, ty0, lx, ly);
            } else {
                int vg[6];
                pctGlyphs(sig[row] * 100.0f, vg);
                lit = textPixel(vg, 6, 232, ty0, lx, ly);
            }
        }
        if (lit) {
            r = g = b = 1.0f;
            return true;
        }
        // meter bar under the text: track [8, 268), fill by sigma/0.08
        if (ly >= ty0 + 9 && ly < ty0 + 13 && lx >= 8 && lx < 268) {
            const float fill = clampf(sig[row] / 0.08f, 0.0f, 1.0f);
            const bool on = (lx - 8) < static_cast<int>(fill * 260.0f);
            if (on && !tOff) { r = 0.20f; g = 0.65f; b = 0.95f; }
            else            { r = g = b = 0.16f; }
            return true;
        }
    }

    // histogram of fine luma |laplacian|: box x [8,268), y [102,130)
    if (lx >= 8 && lx < 268 && ly >= 102 && ly < 130) {
        const int bin = clampi((lx - 8) * kHistBins / 260, 0, kHistBins - 1);
        const float hgt = 27.0f * static_cast<float>(st.histY[bin]) / static_cast<float>(st.histMax);
        const bool bar = (129 - ly) < static_cast<int>(hgt + 0.5f);
        if (bin == static_cast<int>(st.medBinY)) { r = 0.95f; g = 0.85f; b = 0.15f; } // median marker
        else if (bar)                            { r = g = b = 0.55f; }
        else                                     { r = g = b = 0.10f; }
        return true;
    }

    return true; // inside panel, background already applied
}

// ---------------------------------------------------------------------------
// Stage 3 — noise-adaptive NLM / bilateral + output assembly
// ---------------------------------------------------------------------------
inline void spatialNLM(const float* tmp, const float* curr, int W, int H,
                       const Params& p, const Stats& s, float* out)
{
    const float aY = (p.enableSpatial == 0) ? 0.0f : clampf(p.spatialLuma   * p.master, 0.0f, 1.0f);
    const float aC = (p.enableSpatial == 0) ? 0.0f : clampf(p.spatialChroma * p.master, 0.0f, 1.0f);
    const float pd = clampf(p.preserveDetail, 0.0f, 1.0f);
    const int   R  = clampi(p.spatialRadius, 1, 5);
    const bool  nlm = (p.spatialMode == 1);
    const bool  runSpatial = (p.enableSpatial != 0) && (aY > 0.0f || aC > 0.0f);
    const float spatialSigma = std::max(1.0f, R / 1.5f);
    const float invSpatial2 = 1.0f / (2.0f * spatialSigma * spatialSigma);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t idx = (static_cast<size_t>(y) * W + x) * 4;
            const float* tc = tmp + idx;

            float Yo = tc[0], Cbo = tc[1], Cro = tc[2];

            if (runSpatial) {
                const float effN = std::max(1.0f, tc[3]);
                const float sigY = clampf(s.sy / std::sqrt(effN), 1e-5f, 1.0f);
                const float sigC = clampf(s.sc / std::sqrt(effN), 1e-5f, 1.0f);

                float mean = 0.0f, m2 = 0.0f;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int sx = clampi(x + dx, 0, W - 1);
                        const int sy2 = clampi(y + dy, 0, H - 1);
                        const float v = tmp[(static_cast<size_t>(sy2) * W + sx) * 4];
                        mean += v; m2 += v * v;
                    }
                }
                mean *= (1.0f / 9.0f);
                const float var = std::max(0.0f, m2 * (1.0f / 9.0f) - mean * mean);
                const float edginess = clampf(std::sqrt(std::max(var - sigY * sigY, 0.0f)) / (3.0f * sigY), 0.0f, 1.0f);

                const float hY = kNlmHLuma   * sigY * (1.0f - pd * 0.85f * edginess);
                const float hC = kNlmHChroma * sigC * (1.0f - pd * 0.50f * edginess);
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
                        const int sx = clampi(x + dx, 0, W - 1);
                        const int sy2 = clampi(y + dy, 0, H - 1);
                        const float* ts = tmp + (static_cast<size_t>(sy2) * W + sx) * 4;

                        float dY2, dC2;
                        if (nlm) {
                            dY2 = 0.0f; dC2 = 0.0f;
                            for (int qy = -1; qy <= 1; ++qy) {
                                for (int qx = -1; qx <= 1; ++qx) {
                                    const int px = clampi(x + qx, 0, W - 1);
                                    const int py = clampi(y + qy, 0, H - 1);
                                    const int qxs = clampi(sx + qx, 0, W - 1);
                                    const int qys = clampi(sy2 + qy, 0, H - 1);
                                    const float* tp = tmp + (static_cast<size_t>(py) * W + px) * 4;
                                    const float* tq = tmp + (static_cast<size_t>(qys) * W + qxs) * 4;
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

            float r, g, b;
            ycc2rgb(Yo, Cbo, Cro, r, g, b);

            const float* c = curr + idx;
            float* o = out + idx;

            switch (p.viewMode) {
            case 1: { // split: original | result
                if (x < W / 2) { o[0] = c[0]; o[1] = c[1]; o[2] = c[2]; }
                else           { o[0] = r;    o[1] = g;    o[2] = b;    }
                if (std::abs(x - W / 2) <= 1) { o[0] = 1.0f; o[1] = 1.0f; o[2] = 1.0f; }
                break;
            }
            case 2: { // removed noise, amplified, around mid gray
                o[0] = 0.5f + (c[0] - r) * 4.0f;
                o[1] = 0.5f + (c[1] - g) * 4.0f;
                o[2] = 0.5f + (c[2] - b) * 4.0f;
                break;
            }
            case 3: { // noise analysis: result + HUD + region rectangle
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
            case 4: { // temporal activity: red = protected, green = averaging
                const float effN = std::max(1.0f, tc[3]);
                const float t = clampf((effN - 1.0f) / 4.0f, 0.0f, 1.0f);
                const float mr = 0.90f + (0.10f - 0.90f) * t;
                const float mg = 0.15f + (0.85f - 0.15f) * t;
                const float mb = 0.10f + (0.20f - 0.10f) * t;
                float yl, cb2, cr2;
                rgb2ycc(c[0], c[1], c[2], yl, cb2, cr2);
                o[0] = yl * 0.45f + mr * 0.55f;
                o[1] = yl * 0.45f + mg * 0.55f;
                o[2] = yl * 0.45f + mb * 0.55f;
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
    // temporal-difference partner: nearest distinct neighbor frame (used by
    // the noise estimator even when temporal NR itself is disabled)
    const float* partner = nullptr;
    if (frames[1] != frames[2]) partner = frames[1];
    else if (frames[3] != frames[2]) partner = frames[3];
    const Stats s = estimateSigma(frames[2], partner, W, H, p);
    temporalMerge(frames, W, H, p, s, scratch.data());
    spatialNLM(scratch.data(), frames[2], W, H, p, s, out);
    return s;
}

} // namespace nrcore

#endif // OPENNR_NR_CORE_H
