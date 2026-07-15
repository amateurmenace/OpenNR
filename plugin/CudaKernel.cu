// OpenNR — CUDA implementation of the denoising pipeline (Windows/Linux, NVIDIA).
// Line-by-line port of plugin/nr_core.h; keep the two in sync.

#include <cuda_runtime.h>

#include <cstdio>
#include <map>

#include "NRParams.h"

#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif

namespace {

#define kMedianCal   0.247100f
#define kMedianCalT  1.048360f
#define kQ20CalT     2.791278f
#define kAbsDiffBias 1.128379f
#define kNlmHLuma    1.15f
#define kNlmHChroma  2.00f
#define kHistBins    256
#define kHistScaleY  512.0f
#define kHistScaleC  1024.0f
#define kSigmaMin    1e-4f
#define kSigmaMax    0.25f

#define H_YF   NR_STATS_HIST_YF
#define H_CF   NR_STATS_HIST_CF
#define H_Y2   NR_STATS_HIST_Y2
#define H_C2   NR_STATS_HIST_C2
#define H_YT   NR_STATS_HIST_YT
#define H_CT   NR_STATS_HIST_CT
#define S_SY   NR_STATS_SIGMA_SY
#define S_SC   NR_STATS_SIGMA_SC
#define S_TY   NR_STATS_SIGMA_TY
#define S_TC   NR_STATS_SIGMA_TC
#define S_MED  NR_STATS_MEDBIN_Y
#define S_HMAX NR_STATS_HISTMAX_Y

__device__ inline float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
__device__ inline int   clampi(int v, int lo, int hi)       { return v < lo ? lo : (v > hi ? hi : v); }

__device__ inline void rgb2ycc(float r, float g, float b, float& y, float& cb, float& cr)
{
    y  = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    cb = (b - y) / 1.8556f;
    cr = (r - y) / 1.5748f;
}

__device__ inline void ycc2rgb(float y, float cb, float cr, float& r, float& g, float& b)
{
    r = y + 1.5748f * cr;
    b = y + 1.8556f * cb;
    g = (y - 0.2126f * r - 0.0722f * b) / 0.7152f;
}

__device__ inline void sampleYCC(const float4* img, int W, int H, int x, int y,
                                 float& Y, float& Cb, float& Cr)
{
    x = clampi(x, 0, W - 1);
    y = clampi(y, 0, H - 1);
    const float4 p = img[y * W + x];
    rgb2ycc(p.x, p.y, p.z, Y, Cb, Cr);
}

__device__ inline void blockMeanYCC(const float4* img, int W, int H, int x, int y,
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

__device__ inline float4 sampleTmp(const float4* tmp, int W, int H, int x, int y)
{
    x = clampi(x, 0, W - 1);
    y = clampi(y, 0, H - 1);
    return tmp[y * W + x];
}

__device__ inline void loadSigmas(const NRParams& p, const unsigned int* stats,
                                  float& sy, float& sc, float& ty, float& tc)
{
    if (p.profileSource != 2) {
        sy = __uint_as_float(stats[S_SY]);
        sc = __uint_as_float(stats[S_SC]);
        ty = __uint_as_float(stats[S_TY]);
        tc = __uint_as_float(stats[S_TC]);
    } else {
        sy = ty = clampf(p.sigmaY, kSigmaMin, kSigmaMax);
        sc = tc = clampf(p.sigmaC, kSigmaMin, kSigmaMax);
    }
}

// ---------------------------------------------------------------------------
__global__ void NoiseEstKernel(NRParams p, int W, int H, const float4* curr,
                               const float4* partner, unsigned int* stats)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = 1 + gx * 2;
    const int y = 1 + gy * 2;
    if (x >= W - 1 || y >= H - 1)
        return;

    if (p.profileSource == 1) {
        const float rHalf = 0.5f * p.regionSize * (float)min(W, H);
        const float cx = p.regionCX * W, cy = p.regionCY * H;
        const int x0 = clampi((int)(cx - rHalf), 1, W - 1);
        const int x1 = clampi((int)(cx + rHalf), 1, W - 1);
        const int y0 = clampi((int)(cy - rHalf), 1, H - 1);
        const int y1 = clampi((int)(cy + rHalf), 1, H - 1);
        if (x < x0 || x >= x1 || y < y0 || y >= y1)
            return;
    }

    float Y[9], Cb[9], Cr[9];
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i)
            sampleYCC(curr, W, H, x + dx, y + dy, Y[i], Cb[i], Cr[i]);

    const float lapY  = 4.0f * Y[4]  - 2.0f * (Y[1] + Y[3] + Y[5] + Y[7])   + (Y[0] + Y[2] + Y[6] + Y[8]);
    const float lapCb = 4.0f * Cb[4] - 2.0f * (Cb[1] + Cb[3] + Cb[5] + Cb[7]) + (Cb[0] + Cb[2] + Cb[6] + Cb[8]);
    const float lapCr = 4.0f * Cr[4] - 2.0f * (Cr[1] + Cr[3] + Cr[5] + Cr[7]) + (Cr[0] + Cr[2] + Cr[6] + Cr[8]);

    atomicAdd(&stats[H_YF + clampi((int)(fabsf(lapY)  * kHistScaleY), 0, kHistBins - 1)], 1u);
    atomicAdd(&stats[H_CF + clampi((int)(fabsf(lapCb) * kHistScaleC), 0, kHistBins - 1)], 1u);
    atomicAdd(&stats[H_CF + clampi((int)(fabsf(lapCr) * kHistScaleC), 0, kHistBins - 1)], 1u);

    if (p.hasTemporalDiff != 0) {
        float py, pcb, pcr;
        sampleYCC(partner, W, H, x, y, py, pcb, pcr);
        atomicAdd(&stats[H_YT + clampi((int)(fabsf(py - Y[4])   * kHistScaleY), 0, kHistBins - 1)], 1u);
        atomicAdd(&stats[H_CT + clampi((int)(fabsf(pcb - Cb[4]) * kHistScaleC), 0, kHistBins - 1)], 1u);
        atomicAdd(&stats[H_CT + clampi((int)(fabsf(pcr - Cr[4]) * kHistScaleC), 0, kHistBins - 1)], 1u);
    }

    if ((gx & 1) == 0 && (gy & 1) == 0) {
        float bY[9], bCb[9], bCr[9];
        i = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx, ++i)
                blockMeanYCC(curr, W, H, x + dx * 2, y + dy * 2, bY[i], bCb[i], bCr[i]);
        const float lY  = 4.0f * bY[4]  - 2.0f * (bY[1] + bY[3] + bY[5] + bY[7])   + (bY[0] + bY[2] + bY[6] + bY[8]);
        const float lCb = 4.0f * bCb[4] - 2.0f * (bCb[1] + bCb[3] + bCb[5] + bCb[7]) + (bCb[0] + bCb[2] + bCb[6] + bCb[8]);
        const float lCr = 4.0f * bCr[4] - 2.0f * (bCr[1] + bCr[3] + bCr[5] + bCr[7]) + (bCr[0] + bCr[2] + bCr[6] + bCr[8]);
        atomicAdd(&stats[H_Y2 + clampi((int)(fabsf(lY)  * kHistScaleY), 0, kHistBins - 1)], 1u);
        atomicAdd(&stats[H_C2 + clampi((int)(fabsf(lCb) * kHistScaleC), 0, kHistBins - 1)], 1u);
        atomicAdd(&stats[H_C2 + clampi((int)(fabsf(lCr) * kHistScaleC), 0, kHistBins - 1)], 1u);
    }
}

// ---------------------------------------------------------------------------
struct QuantRes { float value; unsigned int bin; };

__device__ QuantRes histQuantile(const unsigned int* stats, int base, unsigned long long total,
                                 float scale, unsigned long long num, unsigned long long den)
{
    unsigned long long cum = 0;
    const unsigned long long target = (total * num + den - 1) / den;
    for (int b = 0; b < kHistBins; ++b) {
        cum += stats[base + b];
        if (cum >= target) {
            QuantRes r; r.value = ((float)b + 0.5f) / scale; r.bin = (unsigned int)b;
            return r;
        }
    }
    QuantRes r; r.value = ((float)kHistBins - 0.5f) / scale; r.bin = kHistBins - 1;
    return r;
}

__global__ void FinalizeStatsKernel(NRParams p, unsigned int* stats)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    unsigned long long totalF = 0, total2 = 0, totalT = 0;
    unsigned int hmax = 1;
    for (int b = 0; b < kHistBins; ++b) {
        totalF += stats[H_YF + b];
        total2 += stats[H_Y2 + b];
        totalT += stats[H_YT + b];
        hmax = max(hmax, stats[H_YF + b]);
    }

    float sy = clampf(p.sigmaY, kSigmaMin, kSigmaMax);
    float sc = clampf(p.sigmaC, kSigmaMin, kSigmaMax);
    float ty = sy, tc = sc;
    unsigned int medBin = 0;

    if (totalF >= 64) {
        const QuantRes mYf = histQuantile(stats, H_YF, totalF, kHistScaleY, 1, 2);
        medBin = mYf.bin;
        const float syFine = mYf.value * kMedianCal;
        const float scFine = histQuantile(stats, H_CF, totalF * 2, kHistScaleC, 1, 2).value * kMedianCal;
        const float syCoarse = 2.0f * histQuantile(stats, H_Y2, total2, kHistScaleY, 1, 2).value * kMedianCal;
        const float scCoarse = 2.0f * histQuantile(stats, H_C2, total2 * 2, kHistScaleC, 1, 2).value * kMedianCal;

        const float lapSY = fmaxf(syFine, 0.9f * syCoarse);
        const float lapSC = fmaxf(scFine, 0.9f * scCoarse);

        ty = lapSY;
        tc = lapSC;
        if (p.hasTemporalDiff != 0 && totalT >= 64) {
            const float medTY = histQuantile(stats, H_YT, totalT, kHistScaleY, 1, 2).value * kMedianCalT;
            const float q20TY = histQuantile(stats, H_YT, totalT, kHistScaleY, 1, 5).value * kQ20CalT;
            const float medTC = histQuantile(stats, H_CT, totalT * 2, kHistScaleC, 1, 2).value * kMedianCalT;
            const float q20TC = histQuantile(stats, H_CT, totalT * 2, kHistScaleC, 1, 5).value * kQ20CalT;
            const float candY = (medTY <= 1.4f * q20TY) ? medTY : q20TY;
            const float candC = (medTC <= 1.4f * q20TC) ? medTC : q20TC;
            if (candY > 0.0015f && candY <= 3.5f * lapSY) ty = candY;
            if (candC > 0.0015f && candC <= 3.5f * lapSC) tc = candC;
        }

        const float adj = clampf(p.profileAdjust, 0.25f, 4.0f);
        sy = clampf(fmaxf(lapSY, 0.85f * ty) * adj, kSigmaMin, kSigmaMax);
        sc = clampf(fmaxf(lapSC, 0.85f * tc) * adj, kSigmaMin, kSigmaMax);
        ty = clampf(ty * adj, kSigmaMin, kSigmaMax);
        tc = clampf(tc * adj, kSigmaMin, kSigmaMax);
    }

    stats[S_SY] = __float_as_uint(sy);
    stats[S_SC] = __float_as_uint(sc);
    stats[S_TY] = __float_as_uint(ty);
    stats[S_TC] = __float_as_uint(tc);
    stats[S_MED] = medBin;
    stats[S_HMAX] = hmax;
}

// ---------------------------------------------------------------------------
__global__ void TemporalKernel(NRParams p, int W, int H,
                               const float4* f0, const float4* f1, const float4* f2,
                               const float4* f3, const float4* f4,
                               const unsigned int* stats, float4* tmp)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    float sigSY, sigSC, sigTY, sigTC;
    loadSigmas(p, stats, sigSY, sigSC, sigTY, sigTC);
    const float tL = clampf(p.temporalLuma   * p.master, 0.0f, 1.0f);
    const float tC = clampf(p.temporalChroma * p.master, 0.0f, 1.0f);
    const float thrMul = 0.4f + 2.6f * clampf(p.motionThresh, 0.0f, 1.0f);
    const int reach = (p.enableTemporal == 0) ? 0 : ((p.temporalFrames >= 5) ? 2 : 1);
    const float loY = kAbsDiffBias * sigTY, hiY = loY + thrMul * sigTY;
    const float loC = kAbsDiffBias * sigTC, hiC = loC + thrMul * sigTC;
    const float invSpanY = 1.0f / (hiY - loY);
    const float invSpanC = 1.0f / (hiC - loC);

    const float4* frames[5] = { f0, f1, f2, f3, f4 };

    float cy9[9], ccb9[9], ccr9[9];
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i)
            sampleYCC(f2, W, H, x + dx, y + dy, cy9[i], ccb9[i], ccr9[i]);

    float accY = cy9[4], accCb = ccb9[4], accCr = ccr9[4];
    float sumWY = 1.0f, sumWY2 = 1.0f, sumWC = 1.0f;

    for (int k = 2 - reach; k <= 2 + reach; ++k) {
        if (k == 2)
            continue;
        const float4* f = frames[k];

        float dY = 0.0f, dC = 0.0f;
        float fyc = 0.0f, fcbc = 0.0f, fcrc = 0.0f;
        i = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx, ++i) {
                float fy, fcb, fcr;
                sampleYCC(f, W, H, x + dx, y + dy, fy, fcb, fcr);
                if (i == 4) { fyc = fy; fcbc = fcb; fcrc = fcr; }
                dY += fabsf(fy - cy9[i]);
                dC += 0.5f * (fabsf(fcb - ccb9[i]) + fabsf(fcr - ccr9[i]));
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

        accY  += wY * fyc;
        accCb += wC * fcbc;
        accCr += wC * fcrc;
        sumWY  += wY;
        sumWY2 += wY * wY;
        sumWC  += wC;
    }

    float4 t;
    t.x = accY / sumWY;
    t.y = accCb / sumWC;
    t.z = accCr / sumWC;
    t.w = (sumWY * sumWY) / sumWY2;
    tmp[y * W + x] = t;
}

// ---------------------------------------------------------------------------
// HUD (Noise Analysis view)
// glyph order: 0-9 . % A C E F I L M O P R S T U Y space
__device__ const unsigned long long kFont[27] = {
    0x3a33ae62eULL, 0x11842108eULL, 0x3a213221fULL, 0x3a213062eULL, 0x08ca97c42ULL, 0x7e1e0862eULL,
    0x3a10f462eULL, 0x7c2222108ULL, 0x3a317462eULL, 0x3a317842eULL, 0x00000018cULL, 0x632222263ULL,
    0x3a31fc631ULL, 0x3a308422eULL, 0x7e10f421fULL, 0x7e10f4210ULL, 0x38842108eULL, 0x42108421fULL,
    0x4775ac631ULL, 0x3a318c62eULL, 0x7a31f4210ULL, 0x7a31f5251ULL, 0x3e107043eULL, 0x7c8421084ULL,
    0x46318c62eULL, 0x462a21084ULL, 0x000000000ULL,
};
#define G_DOT 10
#define G_PCT 11
#define G_A 12
#define G_C 13
#define G_E 14
#define G_F 15
#define G_I 16
#define G_L 17
#define G_M 18
#define G_O 19
#define G_P 20
#define G_R 21
#define G_S 22
#define G_T 23
#define G_U 24
#define G_Y 25
#define G_SP 26

__device__ const int kLabSY[9]  = { G_S, G_P, G_A, G_T, G_I, G_A, G_L, G_SP, G_Y };
__device__ const int kLabSC[9]  = { G_S, G_P, G_A, G_T, G_I, G_A, G_L, G_SP, G_C };
__device__ const int kLabTY[10] = { G_T, G_E, G_M, G_P, G_O, G_R, G_A, G_L, G_SP, G_Y };
__device__ const int kLabTC[10] = { G_T, G_E, G_M, G_P, G_O, G_R, G_A, G_L, G_SP, G_C };
__device__ const int kLabOFF[3] = { G_O, G_F, G_F };

__device__ inline bool glyphPixel(int glyph, int gx, int gy)
{
    if (glyph < 0 || glyph > 26 || gx < 0 || gx >= 5 || gy < 0 || gy >= 7)
        return false;
    return (kFont[glyph] >> (34 - (gy * 5 + gx))) & 1ULL;
}

__device__ inline bool textPixel(const int* chars, int n, int tx, int ty, int lx, int ly)
{
    if (ly < ty || ly >= ty + 7 || lx < tx || lx >= tx + n * 6)
        return false;
    const int ci = (lx - tx) / 6;
    return glyphPixel(chars[ci], (lx - tx) - ci * 6, ly - ty);
}

__device__ inline void pctGlyphs(float pp, int* outg)
{
    const int v = clampi((int)(pp * 100.0f + 0.5f), 0, 9999);
    const int tens = (v / 1000) % 10;
    outg[0] = (tens == 0) ? G_SP : tens;
    outg[1] = (v / 100) % 10;
    outg[2] = G_DOT;
    outg[3] = (v / 10) % 10;
    outg[4] = v % 10;
    outg[5] = G_PCT;
}

__device__ inline bool hudPixel(int x, int y, int W, int H,
                                float sigSY, float sigSC, float sigTY, float sigTC,
                                unsigned int medBin, unsigned int hmax,
                                const unsigned int* stats, int enableTemporal,
                                float& r, float& g, float& b)
{
    const int s = max(1, H / 540);
    const int ox = 16 * s, oy = 16 * s;
    const int lw = 300, lh = 134;
    if (x < ox || y < oy || x >= ox + lw * s || y >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (y - oy) / s;

    r = r * 0.20f + 0.015f; g = g * 0.20f + 0.015f; b = b * 0.20f + 0.02f;
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        r = g = b = 0.35f;
        return true;
    }

    const float sigRow[4] = { sigSY, sigSC, sigTY, sigTC };
    const int rowY[4] = { 6, 30, 54, 78 };

    for (int row = 0; row < 4; ++row) {
        const int ty0 = rowY[row];
        bool lit = false;
        if (row == 0) lit = textPixel(kLabSY, 9,  8, ty0, lx, ly);
        if (row == 1) lit = textPixel(kLabSC, 9,  8, ty0, lx, ly);
        if (row == 2) lit = textPixel(kLabTY, 10, 8, ty0, lx, ly);
        if (row == 3) lit = textPixel(kLabTC, 10, 8, ty0, lx, ly);
        const bool tOff = (row >= 2) && (enableTemporal == 0);
        if (!lit) {
            if (tOff) {
                lit = textPixel(kLabOFF, 3, 250, ty0, lx, ly);
            } else {
                int vg[6];
                pctGlyphs(sigRow[row] * 100.0f, vg);
                lit = textPixel(vg, 6, 232, ty0, lx, ly);
            }
        }
        if (lit) {
            r = g = b = 1.0f;
            return true;
        }
        if (ly >= ty0 + 9 && ly < ty0 + 13 && lx >= 8 && lx < 268) {
            const float fill = clampf(sigRow[row] / 0.08f, 0.0f, 1.0f);
            const bool on = (lx - 8) < (int)(fill * 260.0f);
            if (on && !tOff) { r = 0.20f; g = 0.65f; b = 0.95f; }
            else             { r = g = b = 0.16f; }
            return true;
        }
    }

    if (lx >= 8 && lx < 268 && ly >= 102 && ly < 130) {
        const int bin = clampi((lx - 8) * kHistBins / 260, 0, kHistBins - 1);
        const float hgt = 27.0f * (float)stats[H_YF + bin] / (float)max(hmax, 1u);
        const bool bar = (129 - ly) < (int)(hgt + 0.5f);
        if (bin == (int)medBin) { r = 0.95f; g = 0.85f; b = 0.15f; }
        else if (bar)           { r = g = b = 0.55f; }
        else                    { r = g = b = 0.10f; }
        return true;
    }

    return true;
}

// ---------------------------------------------------------------------------
__global__ void SpatialNLMKernel(NRParams p, int W, int H,
                                 const float4* tmp, const float4* curr,
                                 const unsigned int* stats, float4* dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    float sigSY, sigSC, sigTY, sigTC;
    loadSigmas(p, stats, sigSY, sigSC, sigTY, sigTC);
    const float aY = (p.enableSpatial == 0) ? 0.0f : clampf(p.spatialLuma   * p.master, 0.0f, 1.0f);
    const float aC = (p.enableSpatial == 0) ? 0.0f : clampf(p.spatialChroma * p.master, 0.0f, 1.0f);
    const float pd = clampf(p.preserveDetail, 0.0f, 1.0f);
    const int   R  = clampi(p.spatialRadius, 1, 5);
    const bool  nlm = (p.spatialMode == 1);
    const bool  runSpatial = (p.enableSpatial != 0) && (aY > 0.0f || aC > 0.0f);
    const float spatialSigma = fmaxf(1.0f, (float)R / 1.5f);
    const float invSpatial2 = 1.0f / (2.0f * spatialSigma * spatialSigma);

    const float4 tc = tmp[y * W + x];
    float Yo = tc.x, Cbo = tc.y, Cro = tc.z;

    if (runSpatial) {
        const float effN = fmaxf(1.0f, tc.w);
        const float sigY = clampf(sigSY / sqrtf(effN), 1e-5f, 1.0f);
        const float sigC = clampf(sigSC / sqrtf(effN), 1e-5f, 1.0f);

        float pY[9], pCb[9], pCr[9];
        {
            int i = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx, ++i) {
                    const float4 v = sampleTmp(tmp, W, H, x + dx, y + dy);
                    pY[i] = v.x; pCb[i] = v.y; pCr[i] = v.z;
                }
        }

        float mean = 0.0f, m2 = 0.0f;
        for (int i = 0; i < 9; ++i) { mean += pY[i]; m2 += pY[i] * pY[i]; }
        mean *= (1.0f / 9.0f);
        const float var = fmaxf(0.0f, m2 * (1.0f / 9.0f) - mean * mean);
        const float edginess = clampf(sqrtf(fmaxf(var - sigY * sigY, 0.0f)) / (3.0f * sigY), 0.0f, 1.0f);

        const float hY = kNlmHLuma   * sigY * (1.0f - pd * 0.85f * edginess);
        const float hC = kNlmHChroma * sigC * (1.0f - pd * 0.50f * edginess);
        const float invHY2 = 1.0f / fmaxf(hY * hY, 1e-12f);
        const float invHC2 = 1.0f / fmaxf(hC * hC, 1e-12f);
        const float biasY = 2.0f * sigY * sigY;
        const float biasC = 2.0f * sigC * sigC;

        float accY = 0.0f, accCb = 0.0f, accCr = 0.0f;
        float sumWY = 0.0f, sumWC = 0.0f, wYmax = 0.0f, wCmax = 0.0f;

        for (int dy = -R; dy <= R; ++dy) {
            for (int dx = -R; dx <= R; ++dx) {
                if (dx == 0 && dy == 0)
                    continue;
                const int sx = clampi(x + dx, 0, W - 1);
                const int sy = clampi(y + dy, 0, H - 1);
                const float4 ts = tmp[sy * W + sx];

                float dY2, dC2;
                if (nlm) {
                    dY2 = 0.0f; dC2 = 0.0f;
                    int i = 0;
                    for (int qy = -1; qy <= 1; ++qy) {
                        for (int qx = -1; qx <= 1; ++qx, ++i) {
                            const float4 tq = sampleTmp(tmp, W, H, sx + qx, sy + qy);
                            const float eY = pY[i] - tq.x;
                            const float eCb = pCb[i] - tq.y;
                            const float eCr = pCr[i] - tq.z;
                            dY2 += eY * eY;
                            dC2 += 0.5f * (eCb * eCb + eCr * eCr);
                        }
                    }
                    dY2 *= (1.0f / 9.0f);
                    dC2 *= (1.0f / 9.0f);
                } else {
                    const float eY = tc.x - ts.x;
                    const float eCb = tc.y - ts.y;
                    const float eCr = tc.z - ts.z;
                    dY2 = eY * eY;
                    dC2 = 0.5f * (eCb * eCb + eCr * eCr);
                }

                dY2 = fmaxf(0.0f, dY2 - biasY);
                dC2 = fmaxf(0.0f, dC2 - biasC);

                float wY = expf(-dY2 * invHY2);
                float wC = expf(-dC2 * invHC2) * expf(-dY2 * invHY2 * 0.25f);
                if (!nlm) {
                    const float fall = expf(-(float)(dx * dx + dy * dy) * invSpatial2);
                    wY *= fall;
                    wC *= fall;
                }

                accY  += wY * ts.x;
                accCb += wC * ts.y;
                accCr += wC * ts.z;
                sumWY += wY;
                sumWC += wC;
                wYmax = fmaxf(wYmax, wY);
                wCmax = fmaxf(wCmax, wC);
            }
        }

        const float wYc = fmaxf(wYmax, 1e-4f);
        const float wCc = fmaxf(wCmax, 1e-4f);
        const float Yf  = (accY  + wYc * tc.x) / (sumWY + wYc);
        const float Cbf = (accCb + wCc * tc.y) / (sumWC + wCc);
        const float Crf = (accCr + wCc * tc.z) / (sumWC + wCc);

        Yo  = tc.x + aY * (Yf  - tc.x);
        Cbo = tc.y + aC * (Cbf - tc.y);
        Cro = tc.z + aC * (Crf - tc.z);
    }

    float r, g, b;
    ycc2rgb(Yo, Cbo, Cro, r, g, b);
    const float4 c = curr[y * W + x];
    float4 o;
    o.x = r; o.y = g; o.z = b; o.w = c.w;

    if (p.viewMode == 1) {
        if (x < W / 2) { o.x = c.x; o.y = c.y; o.z = c.z; }
        if (abs(x - W / 2) <= 1) { o.x = 1.0f; o.y = 1.0f; o.z = 1.0f; }
    } else if (p.viewMode == 2) {
        o.x = 0.5f + (c.x - r) * 4.0f;
        o.y = 0.5f + (c.y - g) * 4.0f;
        o.z = 0.5f + (c.z - b) * 4.0f;
    } else if (p.viewMode == 3) {
        float rr = r, gg = g, bb = b;
        if (!hudPixel(x, y, W, H, sigSY, sigSC, sigTY, sigTC,
                      stats[S_MED], stats[S_HMAX], stats, p.enableTemporal, rr, gg, bb)) {
            if (p.profileSource == 1) {
                const float rHalf = 0.5f * p.regionSize * (float)min(W, H);
                const float cx = p.regionCX * W, cyy = p.regionCY * H;
                const float ax = fabsf((float)x - cx), ay = fabsf((float)y - cyy);
                const bool onEdge = (ax <= rHalf && ay <= rHalf) &&
                                    (ax >= rHalf - 2.0f || ay >= rHalf - 2.0f);
                if (onEdge) { rr = 1.0f; gg = 1.0f; bb = 0.1f; }
            }
        }
        o.x = rr; o.y = gg; o.z = bb;
    } else if (p.viewMode == 4) {
        const float effN = fmaxf(1.0f, tc.w);
        const float t = clampf((effN - 1.0f) / 4.0f, 0.0f, 1.0f);
        const float hr = 0.90f + (0.10f - 0.90f) * t;
        const float hg = 0.15f + (0.85f - 0.15f) * t;
        const float hb = 0.10f + (0.20f - 0.10f) * t;
        float yl, cb2, cr2;
        rgb2ycc(c.x, c.y, c.z, yl, cb2, cr2);
        o.x = yl * 0.45f + hr * 0.55f;
        o.y = yl * 0.45f + hg * 0.55f;
        o.z = yl * 0.45f + hb * 0.55f;
    }

    dst[y * W + x] = o;
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

class Locker
{
public:
#ifdef _WIN64
    Locker() { InitializeCriticalSection(&m); }
    ~Locker() { DeleteCriticalSection(&m); }
    void Lock() { EnterCriticalSection(&m); }
    void Unlock() { LeaveCriticalSection(&m); }
    CRITICAL_SECTION m;
#else
    Locker() { pthread_mutex_init(&m, NULL); }
    ~Locker() { pthread_mutex_destroy(&m); }
    void Lock() { pthread_mutex_lock(&m); }
    void Unlock() { pthread_mutex_unlock(&m); }
    pthread_mutex_t m;
#endif
};

struct StreamResources
{
    float4* tmp = nullptr;
    unsigned int* stats = nullptr;
    int w = 0, h = 0;
};

} // namespace

void RunCudaNR(void* p_Stream, int p_Width, int p_Height, const NRParams& p_Params,
               const float* const p_Srcs[5], float* p_Dst)
{
    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);

    static std::map<void*, StreamResources> s_resources;
    static Locker s_locker;

    StreamResources res;
    s_locker.Lock();
    {
        StreamResources& r = s_resources[p_Stream];
        if (!r.tmp || r.w != p_Width || r.h != p_Height)
        {
            if (r.tmp)   cudaFree(r.tmp);
            if (r.stats) cudaFree(r.stats);
            cudaMalloc(&r.tmp, static_cast<size_t>(p_Width) * p_Height * sizeof(float4));
            cudaMalloc(&r.stats, NR_STATS_UINTS * sizeof(unsigned int));
            r.w = p_Width;
            r.h = p_Height;
        }
        res = r;
    }
    s_locker.Unlock();

    if (!res.tmp || !res.stats)
        return;

    // temporal-difference partner: nearest distinct neighbor frame
    NRParams params = p_Params;
    const float* partnerPtr = p_Srcs[2];
    if (p_Srcs[1] != p_Srcs[2])      partnerPtr = p_Srcs[1];
    else if (p_Srcs[3] != p_Srcs[2]) partnerPtr = p_Srcs[3];
    params.hasTemporalDiff = (partnerPtr != p_Srcs[2]) ? 1 : 0;

    const float4* srcs[5];
    for (int i = 0; i < 5; ++i)
        srcs[i] = reinterpret_cast<const float4*>(p_Srcs[i]);
    const float4* partner = reinterpret_cast<const float4*>(partnerPtr);
    float4* dst = reinterpret_cast<float4*>(p_Dst);

    const dim3 threads(16, 16, 1);
    const dim3 blocks((p_Width + threads.x - 1) / threads.x,
                      (p_Height + threads.y - 1) / threads.y, 1);

    if (params.profileSource != 2)
    {
        cudaMemsetAsync(res.stats, 0, NR_STATS_UINTS * sizeof(unsigned int), stream);
        const dim3 estBlocks((p_Width / 2 + threads.x - 1) / threads.x,
                             (p_Height / 2 + threads.y - 1) / threads.y, 1);
        NoiseEstKernel<<<estBlocks, threads, 0, stream>>>(params, p_Width, p_Height, srcs[2], partner, res.stats);
        FinalizeStatsKernel<<<1, 1, 0, stream>>>(params, res.stats);
    }

    TemporalKernel<<<blocks, threads, 0, stream>>>(params, p_Width, p_Height,
                                                   srcs[0], srcs[1], srcs[2], srcs[3], srcs[4],
                                                   res.stats, res.tmp);
    SpatialNLMKernel<<<blocks, threads, 0, stream>>>(params, p_Width, p_Height,
                                                     res.tmp, srcs[2], res.stats, dst);
}
