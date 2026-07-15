// OpenNR — CUDA implementation of the denoising pipeline (v2, Windows/Linux NVIDIA).
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
#define kQ35Cal      0.367299f
#define kMedianCalT  1.048360f
#define kQ20CalT     2.791278f
#define kAbsDiffBias 1.128379f
#define kNlmHLuma    1.15f
#define kNlmHChroma  2.20f
#define kHistBins    256
#define kHistScaleY  512.0f
#define kHistScaleC  1024.0f
#define kLumaBins    16
#define kLumaSub     64
#define kLumaSubScaleY 128.0f
#define kLumaSubScaleC 256.0f
#define kSigmaMin    1e-4f
#define kSigmaMax    0.25f

#define H_YF    NR_STATS_HIST_YF
#define H_CF    NR_STATS_HIST_CF
#define H_Y2    NR_STATS_HIST_Y2
#define H_C2    NR_STATS_HIST_C2
#define H_YT    NR_STATS_HIST_YT
#define H_CT    NR_STATS_HIST_CT
#define L_Y     NR_STATS_LUMA_Y
#define L_C     NR_STATS_LUMA_C
#define H_YR    NR_STATS_HIST_YR
#define H_CR    NR_STATS_HIST_CR
#define H_EN    NR_STATS_HIST_EFFN
#define S_SY    NR_STATS_SIGMA_SY
#define S_SC    NR_STATS_SIGMA_SC
#define S_TY    NR_STATS_SIGMA_TY
#define S_TC    NR_STATS_SIGMA_TC
#define S_RY    NR_STATS_SIGMA_RY
#define S_RC    NR_STATS_SIGMA_RC
#define S_MED   NR_STATS_MEDBIN_Y
#define S_HMAX  NR_STATS_HISTMAX_Y
#define S_GY    NR_STATS_GAINY
#define S_GC    NR_STATS_GAINC
#define S_ENMED NR_STATS_EFFN_MED

__device__ inline float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
__device__ inline int   clampi(int v, int lo, int hi)       { return v < lo ? lo : (v > hi ? hi : v); }
__device__ inline float smooth01f(float t)
{
    t = clampf(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

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

__device__ inline float med3(float a, float b, float c)
{
    return fmaxf(fminf(a, b), fminf(fmaxf(a, b), c));
}

__device__ inline void sort2(float& a, float& b)
{
    const float t = fminf(a, b);
    b = fmaxf(a, b);
    a = t;
}

// median of 9 via the standard 19-exchange network (Smith 1996)
__device__ inline float med9(const float* v)
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

// 2x2 block mean of the tmp buffer (already YCC)
__device__ inline void blockMeanTmp(const float4* tmp, int W, int H, int x, int y,
                                    float& Y, float& Cb, float& Cr)
{
    Y = Cb = Cr = 0.0f;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            const float4 t = sampleTmp(tmp, W, H, x + dx, y + dy);
            Y += t.x; Cb += t.y; Cr += t.z;
        }
    Y *= 0.25f; Cb *= 0.25f; Cr *= 0.25f;
}

// 4x4 block mean of tmp, centred-ish on (x,y)
__device__ inline void blockMean4Tmp(const float4* tmp, int W, int H, int x, int y,
                                     float& Y, float& Cb, float& Cr)
{
    Y = Cb = Cr = 0.0f;
    for (int dy = -1; dy < 3; ++dy)
        for (int dx = -1; dx < 3; ++dx) {
            const float4 t = sampleTmp(tmp, W, H, x + dx, y + dy);
            Y += t.x; Cb += t.y; Cr += t.z;
        }
    Y *= 0.0625f; Cb *= 0.0625f; Cr *= 0.0625f;
}

// v3 shift-search candidate offsets: centre first, then +/-1 and +/-2
__device__ const int kOffX[9] = { 0, 1, -1, 0, 0, 2, -2, 0, 0 };
__device__ const int kOffY[9] = { 0, 0, 0, 1, -1, 0, 0, 2, -2 };

// Mean |3x3 patch difference| of neighbour frame f shifted by (ox,oy) against
// the current frame's patch; also returns the shifted centre sample.
__device__ inline void patchDiff(const float4* f, int W, int H, int x, int y,
                                 int ox, int oy,
                                 const float* cy9, const float* ccb9, const float* ccr9,
                                 float& dY, float& dC, float& fy, float& fcb, float& fcr)
{
    dY = 0.0f; dC = 0.0f; fy = 0.0f; fcb = 0.0f; fcr = 0.0f;
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx, ++i) {
            float vy, vcb, vcr;
            sampleYCC(f, W, H, x + ox + dx, y + oy + dy, vy, vcb, vcr);
            if (i == 4) { fy = vy; fcb = vcb; fcr = vcr; }
            dY += fabsf(vy - cy9[i]);
            dC += 0.5f * (fabsf(vcb - ccb9[i]) + fabsf(vcr - ccr9[i]));
        }
    }
    dY *= (1.0f / 9.0f);
    dC *= (1.0f / 9.0f);
}

__device__ inline float hashNoise(unsigned int ix, unsigned int iy, unsigned int f, unsigned int ch)
{
    unsigned int h = ix * 374761393u + iy * 668265263u + f * 2246822519u + ch * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return ((float)(h & 0xFFFFFFu) / 16777215.0f) * 2.0f - 1.0f;
}

__device__ inline float valueNoise(float x, float y, float size, unsigned int f, unsigned int ch)
{
    const float gx = x / size, gy = y / size;
    const int ix = (int)floorf(gx), iy = (int)floorf(gy);
    float fx = gx - ix, fy = gy - iy;
    fx = fx * fx * fx * (fx * (fx * 6.0f - 15.0f) + 10.0f);
    fy = fy * fy * fy * (fy * (fy * 6.0f - 15.0f) + 10.0f);
    const float n00 = hashNoise((unsigned int)ix,       (unsigned int)iy,       f, ch);
    const float n10 = hashNoise((unsigned int)(ix + 1), (unsigned int)iy,       f, ch);
    const float n01 = hashNoise((unsigned int)ix,       (unsigned int)(iy + 1), f, ch);
    const float n11 = hashNoise((unsigned int)(ix + 1), (unsigned int)(iy + 1), f, ch);
    return (n00 + (n10 - n00) * fx) + ((n01 + (n11 - n01) * fx) - (n00 + (n10 - n00) * fx)) * fy;
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

    const int lb = clampi((int)(Y[4] * kLumaBins), 0, kLumaBins - 1);
    atomicAdd(&stats[L_Y + lb * kLumaSub + clampi((int)(fabsf(lapY)  * kLumaSubScaleY), 0, kLumaSub - 1)], 1u);
    atomicAdd(&stats[L_C + lb * kLumaSub + clampi((int)(fabsf(lapCb) * kLumaSubScaleC), 0, kLumaSub - 1)], 1u);
    atomicAdd(&stats[L_C + lb * kLumaSub + clampi((int)(fabsf(lapCr) * kLumaSubScaleC), 0, kLumaSub - 1)], 1u);

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

__device__ QuantRes histQuantile(const unsigned int* stats, int base, int n, unsigned long long total,
                                 float scale, unsigned long long num, unsigned long long den)
{
    unsigned long long cum = 0;
    const unsigned long long target = (total * num + den - 1) / den;
    for (int b = 0; b < n; ++b) {
        cum += stats[base + b];
        if (cum >= target) {
            QuantRes r; r.value = ((float)b + 0.5f) / scale; r.bin = (unsigned int)b;
            return r;
        }
    }
    QuantRes r; r.value = ((float)n - 0.5f) / scale; r.bin = n - 1;
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

    float gy[16], gc[16];
    for (int b = 0; b < 16; ++b) { gy[b] = 1.0f; gc[b] = 1.0f; }

    if (totalF >= 64) {
        const QuantRes mYf = histQuantile(stats, H_YF, kHistBins, totalF, kHistScaleY, 1, 2);
        medBin = mYf.bin;
        const float syFine = mYf.value * kMedianCal;
        const float scFine = histQuantile(stats, H_CF, kHistBins, totalF * 2, kHistScaleC, 1, 2).value * kMedianCal;
        const float syCoarse = 2.0f * histQuantile(stats, H_Y2, kHistBins, total2, kHistScaleY, 1, 2).value * kMedianCal;
        const float scCoarse = 2.0f * histQuantile(stats, H_C2, kHistBins, total2 * 2, kHistScaleC, 1, 2).value * kMedianCal;

        const float lapSY = fmaxf(syFine, 0.9f * syCoarse);
        const float lapSC = fmaxf(scFine, 0.9f * scCoarse);

        ty = lapSY;
        tc = lapSC;
        if (p.hasTemporalDiff != 0 && totalT >= 64) {
            const float medTY = histQuantile(stats, H_YT, kHistBins, totalT, kHistScaleY, 1, 2).value * kMedianCalT;
            const float q20TY = histQuantile(stats, H_YT, kHistBins, totalT, kHistScaleY, 1, 5).value * kQ20CalT;
            const float medTC = histQuantile(stats, H_CT, kHistBins, totalT * 2, kHistScaleC, 1, 2).value * kMedianCalT;
            const float q20TC = histQuantile(stats, H_CT, kHistBins, totalT * 2, kHistScaleC, 1, 5).value * kQ20CalT;
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

        const float q35RefY = histQuantile(stats, H_YF, kHistBins, totalF, kHistScaleY, 7, 20).value * kQ35Cal;
        const float q35RefC = histQuantile(stats, H_CF, kHistBins, totalF * 2, kHistScaleC, 7, 20).value * kQ35Cal;
        for (int b = 0; b < kLumaBins; ++b) {
            unsigned long long cy = 0, cc = 0;
            for (int s2 = 0; s2 < kLumaSub; ++s2) { cy += stats[L_Y + b * kLumaSub + s2]; cc += stats[L_C + b * kLumaSub + s2]; }
            if (cy >= 200 && q35RefY > 1e-6f) {
                const float sb = histQuantile(stats, L_Y + b * kLumaSub, kLumaSub, cy, kLumaSubScaleY, 7, 20).value * kQ35Cal;
                const float w = (float)cy / ((float)cy + 2000.0f);
                gy[b] = clampf(1.0f + w * (sb / q35RefY - 1.0f), 0.6f, 2.2f);
            }
            if (cc >= 200 && q35RefC > 1e-6f) {
                const float sb = histQuantile(stats, L_C + b * kLumaSub, kLumaSub, cc, kLumaSubScaleC, 7, 20).value * kQ35Cal;
                const float w = (float)cc / ((float)cc + 4000.0f);
                gc[b] = clampf(1.0f + w * (sb / q35RefC - 1.0f), 0.6f, 2.2f);
            }
        }
    }

    // v3: a locked profile overrides the measured sigmas and gains; the
    // histogram/medbin stay live so the HUD keeps showing the measurement.
    if (p.profileLocked != 0) {
        sy = clampf(p.lockSY, kSigmaMin, kSigmaMax);
        sc = clampf(p.lockSC, kSigmaMin, kSigmaMax);
        ty = clampf(p.lockTY, kSigmaMin, kSigmaMax);
        tc = clampf(p.lockTC, kSigmaMin, kSigmaMax);
    }

    stats[S_SY] = __float_as_uint(sy);
    stats[S_SC] = __float_as_uint(sc);
    stats[S_TY] = __float_as_uint(ty);
    stats[S_TC] = __float_as_uint(tc);
    stats[S_MED] = medBin;
    stats[S_HMAX] = hmax;
    for (int b = 0; b < kLumaBins; ++b) {
        const int b0 = clampi(b - 1, 0, kLumaBins - 1);
        const int b1 = clampi(b + 1, 0, kLumaBins - 1);
        stats[S_GY + b] = __float_as_uint(p.profileLocked != 0 ? clampf(p.lockGainY[b], 0.6f, 2.2f)
                                          : 0.25f * gy[b0] + 0.5f * gy[b] + 0.25f * gy[b1]);
        stats[S_GC + b] = __float_as_uint(p.profileLocked != 0 ? clampf(p.lockGainC[b], 0.6f, 2.2f)
                                          : 0.25f * gc[b0] + 0.5f * gc[b] + 0.25f * gc[b1]);
    }
}

// ---------------------------------------------------------------------------
__device__ inline void loadSigmasIn(const NRParams& p, const unsigned int* stats,
                                    float& sy, float& sc, float& ty, float& tc)
{
    if (p.profileSource != 2 || p.profileLocked != 0) {
        sy = __uint_as_float(stats[S_SY]);
        sc = __uint_as_float(stats[S_SC]);
        ty = __uint_as_float(stats[S_TY]);
        tc = __uint_as_float(stats[S_TC]);
    } else {
        sy = ty = clampf(p.sigmaY, kSigmaMin, kSigmaMax);
        sc = tc = clampf(p.sigmaC, kSigmaMin, kSigmaMax);
    }
}

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
    loadSigmasIn(p, stats, sigSY, sigSC, sigTY, sigTC);
    const float mLow  = fminf(p.master, 1.0f);
    const float mHigh = fmaxf(p.master, 1.0f);
    const float tL = clampf(p.temporalLuma   * mLow, 0.0f, 1.0f);
    const float tC = clampf(p.temporalChroma * mLow, 0.0f, 1.0f);
    const float thrMul = 0.4f + 2.6f * clampf(p.motionThresh, 0.0f, 1.0f)
                       + 0.8f * (mHigh - 1.0f);
    const int reach = (p.enableTemporal == 0) ? 0 : ((p.temporalFrames >= 5) ? 2 : 1);
    const float loY = kAbsDiffBias * sigTY, hiY = loY + thrMul * sigTY;
    const float loC = kAbsDiffBias * sigTC, hiC = loC + thrMul * sigTC;
    const float invSpanY = 1.0f / (hiY - loY);
    const float invSpanC = 1.0f / (hiC - loC);
    // v3 shift search engages only once the unshifted match is well into the
    // gate — high enough that pure noise almost never reaches it, which
    // also keeps GPU warps convergent on static footage (see nr_core.h).
    const bool  track = (p.motionTracking != 0);
    const float searchThresh = loY + 0.75f * (hiY - loY);
    const bool  zap = (reach >= 1) && (p.fireflyRemoval != 0) &&
                      (p.master > 0.0f) && (tL > 0.0f || tC > 0.0f);
    const float zapY = 6.0f * sigTY;
    const float zapC = 6.0f * sigTC;

    const float4* frames[5] = { f0, f1, f2, f3, f4 };

    float cy9[9], ccb9[9], ccr9[9];
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i)
            sampleYCC(f2, W, H, x + dx, y + dy, cy9[i], ccb9[i], ccr9[i]);

    // v3 firefly zapper — see nr_core.h for the three-test rationale
    if (zap) {
        float pY, pCb, pCr, nY, nCb, nCr;
        sampleYCC(f1, W, H, x, y, pY, pCb, pCr);
        sampleYCC(f3, W, H, x, y, nY, nCb, nCr);
        if (fabsf(pY - nY) < 0.5f * zapY &&
            0.5f * (fabsf(pCb - nCb) + fabsf(pCr - nCr)) < 0.5f * zapC &&
            fabsf(cy9[4] - med9(cy9)) > 0.5f * zapY) {
            const float mY  = med3(pY, cy9[4], nY);
            const float mCb = med3(pCb, ccb9[4], nCb);
            const float mCr = med3(pCr, ccr9[4], nCr);
            if (fabsf(cy9[4] - mY) > zapY ||
                0.5f * (fabsf(ccb9[4] - mCb) + fabsf(ccr9[4] - mCr)) > zapC) {
                cy9[4] = mY; ccb9[4] = mCb; ccr9[4] = mCr;
            }
        }
    }

    float accY = cy9[4], accCb = ccb9[4], accCr = ccr9[4];
    float sumWY = 1.0f, sumWY2 = 1.0f, sumWC = 1.0f;

    for (int k = 2 - reach; k <= 2 + reach; ++k) {
        if (k == 2)
            continue;
        const float4* f = frames[k];

        float dY, dC, fyc, fcbc, fcrc;
        patchDiff(f, W, H, x, y, 0, 0, cy9, ccb9, ccr9, dY, dC, fyc, fcbc, fcrc);

        // v3 shift search — see nr_core.h for the acceptance margin and the
        // tightened roll-off for shifted winners
        float shiftTight = 1.0f;
        if (track && dY > searchThresh) {
            for (int c = 1; c < 9; ++c) {
                float dY2, dC2, fy2, fcb2, fcr2;
                patchDiff(f, W, H, x, y, kOffX[c], kOffY[c],
                          cy9, ccb9, ccr9, dY2, dC2, fy2, fcb2, fcr2);
                if (dY2 < dY * 0.99f) {
                    dY = dY2; dC = dC2;
                    fyc = fy2; fcbc = fcb2; fcrc = fcr2;
                    shiftTight = 1.0f / 0.6f;
                }
            }
        }

        const float gY = 1.0f - smooth01f((dY - loY) * invSpanY * shiftTight);
        const float gC = 1.0f - smooth01f((dC - loC) * invSpanC * shiftTight);
        const float wY = tL * gY;
        const float wC = tC * gC * gY;

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
__global__ void ResidualEstKernel(NRParams p, int W, int H,
                                  const float4* tmp, unsigned int* stats)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = 1 + gx * 2;
    const int y = 1 + gy * 2;
    if (x >= W - 1 || y >= H - 1)
        return;

    float Y[9], Cb[9], Cr[9];
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i) {
            const float4 t = sampleTmp(tmp, W, H, x + dx, y + dy);
            Y[i] = t.x; Cb[i] = t.y; Cr[i] = t.z;
        }
    const float lapY  = 4.0f * Y[4]  - 2.0f * (Y[1] + Y[3] + Y[5] + Y[7])   + (Y[0] + Y[2] + Y[6] + Y[8]);
    const float lapCb = 4.0f * Cb[4] - 2.0f * (Cb[1] + Cb[3] + Cb[5] + Cb[7]) + (Cb[0] + Cb[2] + Cb[6] + Cb[8]);
    const float lapCr = 4.0f * Cr[4] - 2.0f * (Cr[1] + Cr[3] + Cr[5] + Cr[7]) + (Cr[0] + Cr[2] + Cr[6] + Cr[8]);
    atomicAdd(&stats[H_YR + clampi((int)(fabsf(lapY)  * kHistScaleY), 0, kHistBins - 1)], 1u);
    atomicAdd(&stats[H_CR + clampi((int)(fabsf(lapCb) * kHistScaleC), 0, kHistBins - 1)], 1u);
    atomicAdd(&stats[H_CR + clampi((int)(fabsf(lapCr) * kHistScaleC), 0, kHistBins - 1)], 1u);
    const float effN = sampleTmp(tmp, W, H, x, y).w;
    atomicAdd(&stats[H_EN + clampi((int)((effN - 1.0f) * 8.0f), 0, 31)], 1u);
}

__global__ void FinalizeResidualKernel(NRParams p, unsigned int* stats)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    unsigned long long total = 0;
    for (int b = 0; b < kHistBins; ++b)
        total += stats[H_YR + b];

    const float sy = __uint_as_float(stats[S_SY]);
    const float sc = __uint_as_float(stats[S_SC]);
    float ry = sy, rc = sc, enmed = 1.0f;

    if (total >= 64) {
        const float adj = clampf(p.profileAdjust, 0.25f, 4.0f);
        ry = histQuantile(stats, H_YR, kHistBins, total, kHistScaleY, 1, 2).value * kMedianCal * adj;
        rc = histQuantile(stats, H_CR, kHistBins, total * 2, kHistScaleC, 1, 2).value * kMedianCal * adj;
        enmed = 1.0f + histQuantile(stats, H_EN, 32, total, 8.0f, 1, 2).value;
        const float floorY = 0.5f * sy / sqrtf(fmaxf(1.0f, enmed));
        const float floorC = 0.5f * sc / sqrtf(fmaxf(1.0f, enmed));
        ry = clampf(fmaxf(ry, floorY), kSigmaMin, sy > kSigmaMin ? sy : kSigmaMax);
        rc = clampf(fmaxf(rc, floorC), kSigmaMin, sc > kSigmaMin ? sc : kSigmaMax);
    }

    stats[S_RY] = __float_as_uint(ry);
    stats[S_RC] = __float_as_uint(rc);
    stats[S_ENMED] = __float_as_uint(enmed);
}

// ---------------------------------------------------------------------------
// HUD — glyph order: 0-9 . % A C E F I L M O P R S T U Y B D G N + space K
__device__ const unsigned long long kFont[33] = {
    0x3a33ae62eULL, 0x11842108eULL, 0x3a213221fULL, 0x3a213062eULL, 0x08ca97c42ULL, 0x7e1e0862eULL,
    0x3a10f462eULL, 0x7c2222108ULL, 0x3a317462eULL, 0x3a317842eULL, 0x00000018cULL, 0x632222263ULL,
    0x3a31fc631ULL, 0x3a308422eULL, 0x7e10f421fULL, 0x7e10f4210ULL, 0x38842108eULL, 0x42108421fULL,
    0x4775ac631ULL, 0x3a318c62eULL, 0x7a31f4210ULL, 0x7a31f5251ULL, 0x3e107043eULL, 0x7c8421084ULL,
    0x46318c62eULL, 0x462a21084ULL, 0x7a31f463eULL, 0x7a318c63eULL, 0x3a30bc62fULL, 0x47359c631ULL,
    0x0084f9080ULL, 0x000000000ULL, 0x4654c5251ULL,
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
#define G_B 26
#define G_D 27
#define G_G 28
#define G_N 29
#define G_PLUS 30
#define G_SP 31
#define G_K 32

__device__ const int kLabIY[4]  = { G_I, G_N, G_SP, G_Y };
__device__ const int kLabIC[4]  = { G_I, G_N, G_SP, G_C };
__device__ const int kLabRY[10] = { G_R, G_E, G_S, G_I, G_D, G_U, G_A, G_L, G_SP, G_Y };
__device__ const int kLabRC[10] = { G_R, G_E, G_S, G_I, G_D, G_U, G_A, G_L, G_SP, G_C };
__device__ const int kLabOFF[3] = { G_O, G_F, G_F };
__device__ const int kLabEFFN[5] = { G_E, G_F, G_F, G_SP, G_N };
__device__ const int kLabGAIN[4] = { G_G, G_A, G_I, G_N };
__device__ const int kLabDB[2]   = { G_D, G_B };
__device__ const int kLabLOCKED[6] = { G_L, G_O, G_C, G_K, G_E, G_D };
__device__ const float kDirX[8] = { 1, 0, -1, 0, 0.7071f, -0.7071f, -0.7071f, 0.7071f };
__device__ const float kDirY[8] = { 0, 1, 0, -1, 0.7071f, 0.7071f, -0.7071f, -0.7071f };

__device__ inline bool glyphPixel(int glyph, int gx, int gy)
{
    if (glyph < 0 || glyph > 32 || gx < 0 || gx >= 5 || gy < 0 || gy >= 7)
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

__device__ inline void dec1Glyphs(float v, int* outg)
{
    const int t = clampi((int)(v * 10.0f + 0.5f), 0, 99);
    outg[0] = (t / 10) % 10;
    outg[1] = G_DOT;
    outg[2] = t % 10;
}

__device__ inline bool hudPixel(int x, int y, int W, int H,
                                float sy, float sc, float ry, float rc, float enmed,
                                unsigned int medBin, unsigned int hmax,
                                const unsigned int* stats, int enableTemporal,
                                int locked, float& r, float& g, float& b)
{
    const int yd = H - 1 - y;   // OFX buffers are bottom-up; panel anchors top-left on screen
    const int s = max(1, H / 540);
    const int ox = 16 * s, oy = 16 * s;
    const int lw = 320, lh = 198;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    r = r * 0.20f + 0.015f; g = g * 0.20f + 0.015f; b = b * 0.20f + 0.02f;
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        r = g = b = 0.35f;
        return true;
    }

    const float sig[4] = { sy, sc, ry, rc };
    const int rowY[4] = { 6, 28, 50, 72 };

    for (int row = 0; row < 4; ++row) {
        const int ty0 = rowY[row];
        bool lit = false;
        if (row == 0) lit = textPixel(kLabIY, 4,  8, ty0, lx, ly);
        if (row == 1) lit = textPixel(kLabIC, 4,  8, ty0, lx, ly);
        if (row == 2) lit = textPixel(kLabRY, 10, 8, ty0, lx, ly);
        if (row == 3) lit = textPixel(kLabRC, 10, 8, ty0, lx, ly);
        if (!lit) {
            int vg[6];
            pctGlyphs(sig[row] * 100.0f, vg);
            lit = textPixel(vg, 6, 252, ty0, lx, ly);
        }
        if (lit) { r = g = b = 1.0f; return true; }

        if (ly >= ty0 + 9 && ly < ty0 + 13 && lx >= 8 && lx < 288) {
            const float fill = clampf(sig[row] / 0.08f, 0.0f, 1.0f);
            const bool on = (lx - 8) < (int)(fill * 280.0f);
            const bool residRow = (row >= 2);
            if (on) {
                if (residRow) { r = 0.95f; g = 0.65f; b = 0.20f; }
                else          { r = 0.20f; g = 0.65f; b = 0.95f; }
            } else { r = g = b = 0.16f; }
            return true;
        }
    }

    {
        const int ty0 = 94;
        bool lit = textPixel(kLabEFFN, 5, 8, ty0, lx, ly);
        if (!lit) {
            int vg[3];
            dec1Glyphs(enableTemporal ? enmed : 1.0f, vg);
            lit = textPixel(vg, 3, 44, ty0, lx, ly);
        }
        if (!lit) lit = textPixel(kLabGAIN, 4, 120, ty0, lx, ly);
        if (!lit) {
            const float gainDb = clampf(20.0f * log10f(fmaxf(sy, 1e-5f) / fmaxf(ry, 1e-5f)), 0.0f, 40.0f);
            int vg[4];
            vg[0] = G_PLUS;
            int d1[3];
            dec1Glyphs(gainDb, d1);
            vg[1] = d1[0]; vg[2] = d1[1]; vg[3] = d1[2];
            lit = textPixel(vg, 4, 150, ty0, lx, ly);
            if (!lit) lit = textPixel(kLabDB, 2, 178, ty0, lx, ly);
        }
        if (lit) { r = g = b = 1.0f; return true; }
        if (enableTemporal == 0 && textPixel(kLabOFF, 3, 220, ty0, lx, ly)) { r = g = b = 0.7f; return true; }
        if (locked && textPixel(kLabLOCKED, 6, 246, ty0, lx, ly)) { r = 0.95f; g = 0.65f; b = 0.20f; return true; }
    }

    if (lx >= 8 && lx < 264 && ly >= 108 && ly < 148) {
        const int bin = clampi((lx - 8) / 16, 0, kLumaBins - 1);
        const float gain = __uint_as_float(stats[S_GY + bin]);
        const float v = clampf((gain - 0.6f) / 1.6f, 0.0f, 1.0f);
        const bool bar = (147 - ly) < (int)(v * 39.0f + 0.5f);
        const bool ref = (147 - ly) == (int)((1.0f - 0.6f) / 1.6f * 39.0f + 0.5f);
        if (bar)      { r = 0.20f; g = 0.65f; b = 0.95f; }
        else if (ref) { r = g = b = 0.32f; }
        else          { r = g = b = 0.10f; }
        return true;
    }

    if (lx >= 8 && lx < 268 && ly >= 156 && ly < 192) {
        const int bin = clampi((lx - 8) * kHistBins / 260, 0, kHistBins - 1);
        const float hgt = 35.0f * (float)stats[H_YF + bin] / (float)max(hmax, 1u);
        const bool bar = (191 - ly) < (int)(hgt + 0.5f);
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

    const bool manual = (p.profileSource == 2) && (p.profileLocked == 0);
    const float sy = manual ? clampf(p.sigmaY, kSigmaMin, kSigmaMax) : __uint_as_float(stats[S_SY]);
    const float sc = manual ? clampf(p.sigmaC, kSigmaMin, kSigmaMax) : __uint_as_float(stats[S_SC]);
    const float tyG = manual ? sy : __uint_as_float(stats[S_TY]);
    const float tcG = manual ? sc : __uint_as_float(stats[S_TC]);
    const float ryG = manual ? sy : __uint_as_float(stats[S_RY]);
    const float rcG = manual ? sc : __uint_as_float(stats[S_RC]);
    const float enmed = manual ? 1.0f : __uint_as_float(stats[S_ENMED]);

    const float mLow  = fminf(p.master, 1.0f);
    const float mHigh = fmaxf(p.master, 1.0f);
    const float hBoost = powf(mHigh, 1.2f);

    const float sL = clampf(p.spatialLuma, 0.0f, 1.0f);
    const float sC = clampf(p.spatialChroma, 0.0f, 1.0f);
    // v3 Noise EQ: the fine slider scales the NLM band's blend (1 = v2.1)
    const float eqF = clampf(p.eqFine, 0.0f, 2.0f);
    const float aY = (p.enableSpatial == 0) ? 0.0f : clampf(sL * mLow * eqF, 0.0f, 1.0f);
    const float aC = (p.enableSpatial == 0) ? 0.0f : clampf(sC * mLow * eqF, 0.0f, 1.0f);
    const float hMulY = (0.6f + 1.4f * powf(sL, 1.5f)) * hBoost;
    const float hMulC = (0.6f + 1.4f * powf(sC, 1.5f)) * hBoost;
    const float pd = clampf(p.preserveDetail, 0.0f, 1.0f);
    const int   R  = clampi(p.spatialRadius, 1, 8);
    const bool  nlm = (p.spatialMode == 1);
    const bool  runSpatial = (p.enableSpatial != 0) && (aY > 0.0f || aC > 0.0f);
    const float spatialSigma = fmaxf(1.0f, (float)R / 1.5f);
    const float invSpatial2 = 1.0f / (2.0f * spatialSigma * spatialSigma);

    const float blotch = (p.enableSpatial != 0) ? clampf(p.chromaBlotch, 0.0f, 1.0f) * mLow : 0.0f;
    // v3 Noise EQ: medium band amount and coarse-band luma amount
    const float eqMed  = (p.enableSpatial != 0) ? clampf(p.eqMedium, 0.0f, 1.0f) * mLow : 0.0f;
    const float coarseL = (p.enableSpatial != 0) ? clampf(p.eqCoarse, 0.0f, 1.0f) * mLow : 0.0f;
    const int   Rb = 2 + (int)(14.0f * fmaxf(clampf(p.chromaBlotch, 0.0f, 1.0f),
                                             clampf(p.eqCoarse, 0.0f, 1.0f)));
    const int   Rm = 3 + (int)(5.0f * clampf(p.eqMedium, 0.0f, 1.0f));
    // Band tolerance: see nr_core.h
    const float bandRatioY = clampf(tyG / fmaxf(sy, 1e-6f), 1.0f, 3.0f);
    const float bandRatioC = clampf(tcG / fmaxf(sc, 1e-6f), 1.0f, 3.0f);

    const bool refine = (p.enableRefine != 0) && (p.master > 0.0f);
    const float desat = refine ? clampf(p.shadowDesat, 0.0f, 1.0f) : 0.0f;
    const float desatRange = fmaxf(0.02f, p.desatRange);
    const float tex = refine ? clampf(p.lumaTexture, 0.0f, 1.0f) * mLow : 0.0f;
    // v3 deband thresholds — see nr_core.h
    const float debandAmt = refine ? clampf(p.deband, 0.0f, 1.0f) * mLow : 0.0f;
    const float dbThrY = fmaxf(0.010f, 1.5f * ryG);
    const float dbThrC = fmaxf(0.010f, 1.5f * rcG);
    const float grainAmt = refine ? clampf(p.grainAmount, 0.0f, 1.0f) * 0.06f : 0.0f;
    const float grainSize = clampf(p.grainSize, 0.5f, 6.0f);
    const float grainCh = clampf(p.grainChroma, 0.0f, 1.0f);
    const unsigned int frame = (unsigned int)p.frameIndex;

    const float4 tc = tmp[y * W + x];
    const int lb = clampi((int)(tc.x * kLumaBins), 0, kLumaBins - 1);
    const float gainYv = manual ? 1.0f : __uint_as_float(stats[S_GY + lb]);
    const float gainCv = manual ? 1.0f : __uint_as_float(stats[S_GC + lb]);
    const float sigY = clampf(ryG * gainYv, 1e-5f, 1.0f);
    const float sigC = clampf(rcG * gainCv, 1e-5f, 1.0f);

    float Yo = tc.x, Cbo = tc.y, Cro = tc.z;

    if (runSpatial) {
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

        const float hY = kNlmHLuma   * sigY * hMulY * (1.0f - pd * 0.85f * edginess);
        const float hC = kNlmHChroma * sigC * hMulC * (1.0f - pd * 0.50f * edginess);
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
                const float4 ts4 = sampleTmp(tmp, W, H, x + dx, y + dy);

                float dY2, dC2;
                if (nlm) {
                    dY2 = 0.0f; dC2 = 0.0f;
                    int i = 0;
                    for (int qy = -1; qy <= 1; ++qy) {
                        for (int qx = -1; qx <= 1; ++qx, ++i) {
                            const float4 tq = sampleTmp(tmp, W, H, x + dx + qx, y + dy + qy);
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
                    const float eY = tc.x - ts4.x;
                    const float eCb = tc.y - ts4.y;
                    const float eCr = tc.z - ts4.z;
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

                accY  += wY * ts4.x;
                accCb += wC * ts4.y;
                accCr += wC * ts4.z;
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

    // v3 medium band — see nr_core.h for the block-mean-domain rationale
    if (eqMed > 0.0f) {
        const float mScale = 2.6f * sigY * bandRatioY * hBoost;
        const float myDen = 1.0f / fmaxf(mScale, 1e-6f);
        const float mcDen = 1.0f / fmaxf(3.0f * sigC * bandRatioC * hBoost, 1e-6f);
        float b0Y, b0Cb, b0Cr;
        blockMeanTmp(tmp, W, H, x, y, b0Y, b0Cb, b0Cr);
        float accMY = b0Y, accMB = b0Cb, accMR = b0Cr, sumWm = 1.0f;
        for (int d = 0; d < 8; ++d) {
            for (int ri = 1; ri <= 2; ++ri) {
                const float rr = Rm * ((float)ri / 2.0f);
                float bY, bCb, bCr;
                blockMeanTmp(tmp, W, H,
                             x + (int)(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                             y + (int)(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f)),
                             bY, bCb, bCr);
                const float eY = (bY - b0Y) * myDen;
                const float eC = (0.5f * (fabsf(bCb - b0Cb) + fabsf(bCr - b0Cr))) * mcDen;
                const float w = expf(-(eY * eY + eC * eC));
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

    // coarse band — chroma path is the v2.1 blotch pass unchanged; the v3
    // luma component works on 4x4 block means (see nr_core.h)
    if (blotch > 0.0f || coarseL > 0.0f) {
        const float gyDen = 1.0f / fmaxf(2.0f * sigY * hBoost, 1e-6f);
        const float gcDen = 1.0f / fmaxf(3.0f * sigC * hBoost, 1e-6f);
        const float cScale = 2.2f * sigY * bandRatioY * hBoost;
        const float glDen = 1.0f / fmaxf(cScale, 1e-6f);
        const int RbL = 2 + (int)(30.0f * clampf(p.eqCoarse, 0.0f, 1.0f));
        float c0Y = 0.0f, c0Cb = 0.0f, c0Cr = 0.0f;
        if (coarseL > 0.0f)
            blockMean4Tmp(tmp, W, H, x, y, c0Y, c0Cb, c0Cr);
        float accB = tc.y, accR2 = tc.z, sumW = 1.0f;
        float accL = c0Y, sumWL = 1.0f;
        for (int d = 0; d < 8; ++d) {
            for (int ri = 1; ri <= 3; ++ri) {
                const float rr = Rb * ((float)ri / 3.0f);
                const int sx = x + (int)(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f));
                const int sy2 = y + (int)(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f));
                const float4 ts4 = sampleTmp(tmp, W, H, sx, sy2);
                const float eY = (ts4.x - tc.x) * gyDen;
                const float eC = (0.5f * (fabsf(ts4.y - tc.y) + fabsf(ts4.z - tc.z))) * gcDen;
                const float w = expf(-(eY * eY + eC * eC));
                accB += w * ts4.y;
                accR2 += w * ts4.z;
                sumW += w;
                if (coarseL > 0.0f) {
                    const float rrL = RbL * ((float)ri / 3.0f);
                    const int lx2 = x + (int)(kDirX[d] * rrL + (kDirX[d] > 0 ? 0.5f : -0.5f));
                    const int ly2 = y + (int)(kDirY[d] * rrL + (kDirY[d] > 0 ? 0.5f : -0.5f));
                    float b4Y, b4Cb, b4Cr;
                    blockMean4Tmp(tmp, W, H, lx2, ly2, b4Y, b4Cb, b4Cr);
                    const float eL = (b4Y - c0Y) * glDen;
                    const float eLC = (0.5f * (fabsf(b4Cb - c0Cb) + fabsf(b4Cr - c0Cr))) * gcDen;
                    const float wL = expf(-(eL * eL + eLC * eLC));
                    accL += wL * b4Y;
                    sumWL += wL;
                }
            }
        }
        if (blotch > 0.0f) {
            Cbo += blotch * (accB / sumW - Cbo);
            Cro += blotch * (accR2 / sumW - Cro);
        }
        if (coarseL > 0.0f) {
            const float lim = 2.5f * cScale;
            Yo += coarseL * clampf(accL / sumWL - c0Y, -lim, lim);
        }
    }

    const float4 c = curr[y * W + x];
    float cinY, cinCb, cinCr;
    rgb2ycc(c.x, c.y, c.z, cinY, cinCb, cinCr);

    float Yr = Yo, Cbr = Cbo, Crr = Cro;
    if (refine) {
        const float sat = 1.0f - desat * (1.0f - smooth01f(Yr / desatRange));
        Cbr *= sat;
        Crr *= sat;
        Yr += tex * (cinY - Yr);
        // v3 deband — see nr_core.h for the agreement-confidence rationale
        if (debandAmt > 0.0f) {
            const float dyDen = 1.0f / dbThrY;
            const float dcDen = 1.0f / dbThrC;
            float b0Y, b0Cb, b0Cr;
            blockMeanTmp(tmp, W, H, x, y, b0Y, b0Cb, b0Cr);
            float accDY = b0Y, accDB = b0Cb, accDR = b0Cr, sumWd = 1.0f;
            for (int d = 0; d < 8; ++d) {
                for (int ri = 1; ri <= 3; ++ri) {
                    const float rr = 16.0f * ((float)ri / 3.0f);
                    float bY, bCb, bCr;
                    blockMeanTmp(tmp, W, H,
                                 x + (int)(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                                 y + (int)(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f)),
                                 bY, bCb, bCr);
                    const float eY = (bY - b0Y) * dyDen;
                    const float eC = (0.5f * (fabsf(bCb - b0Cb) + fabsf(bCr - b0Cr))) * dcDen;
                    const float w = expf(-(eY * eY + eC * eC));
                    accDY += w * bY;
                    accDB += w * bCb;
                    accDR += w * bCr;
                    sumWd += w;
                }
            }
            const float agree = (sumWd - 1.0f) * (1.0f / 24.0f);
            const float conf = agree * agree;
            Yr  += debandAmt * conf * clampf(accDY / sumWd - Yr,  -dbThrY, dbThrY);
            Cbr += debandAmt * conf * clampf(accDB / sumWd - Cbr, -dbThrC, dbThrC);
            Crr += debandAmt * conf * clampf(accDR / sumWd - Crr, -dbThrC, dbThrC);
            const float dith = 0.7f * debandAmt / 255.0f;
            Yr += dith * 0.5f * (hashNoise((unsigned int)x, (unsigned int)y, frame, 3u) +
                                 hashNoise((unsigned int)x, (unsigned int)y, frame + 977u, 3u));
        }
        if (grainAmt > 0.0f) {
            const float yc = clampf(Yr, 0.0f, 1.0f);
            const float resp = 0.25f + 0.75f * (4.0f * yc * (1.0f - yc));
            const float gn = valueNoise((float)x, (float)y, grainSize, frame, 0u);
            Yr += grainAmt * resp * gn;
            if (grainCh > 0.0f) {
                Cbr += grainAmt * grainCh * 0.6f * resp * valueNoise((float)x, (float)y, grainSize, frame, 1u);
                Crr += grainAmt * grainCh * 0.6f * resp * valueNoise((float)x, (float)y, grainSize, frame, 2u);
            }
        }
    }

    float r, g, b;
    ycc2rgb(Yr, Cbr, Crr, r, g, b);
    float dnR, dnG, dnB;
    ycc2rgb(Yo, Cbo, Cro, dnR, dnG, dnB);
    float4 o;
    o.x = r; o.y = g; o.z = b; o.w = c.w;

    if (p.viewMode == 1) {
        if (x < W / 2) { o.x = c.x; o.y = c.y; o.z = c.z; }
        if (abs(x - W / 2) <= 1) { o.x = 1.0f; o.y = 1.0f; o.z = 1.0f; }
    } else if (p.viewMode == 2) {
        o.x = c.x; o.y = c.y; o.z = c.z;
    } else if (p.viewMode == 3) {
        float rr, gg, bb;
        ycc2rgb(tc.x, tc.y, tc.z, rr, gg, bb);
        o.x = rr; o.y = gg; o.z = bb;
    } else if (p.viewMode == 4) {
        o.x = 0.5f + (c.x - dnR) * 4.0f;
        o.y = 0.5f + (c.y - dnG) * 4.0f;
        o.z = 0.5f + (c.z - dnB) * 4.0f;
    } else if (p.viewMode == 5) {
        float rr = r, gg = g, bb = b;
        if (!hudPixel(x, y, W, H, sy, sc, ryG, rcG, enmed,
                      stats[S_MED], stats[S_HMAX], stats, p.enableTemporal,
                      p.profileLocked, rr, gg, bb)) {
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
    } else if (p.viewMode == 6) {
        const float effN = fmaxf(1.0f, tc.w);
        const float t = clampf((effN - 1.0f) / 4.0f, 0.0f, 1.0f);
        const float hr = 0.90f + (0.10f - 0.90f) * t;
        const float hg = 0.15f + (0.85f - 0.15f) * t;
        const float hb = 0.10f + (0.20f - 0.10f) * t;
        o.x = cinY * 0.45f + hr * 0.55f;
        o.y = cinY * 0.45f + hg * 0.55f;
        o.z = cinY * 0.45f + hb * 0.55f;
    } else if (p.viewMode == 7) {
        float mean = 0.0f, m2 = 0.0f;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
                const float v = sampleTmp(tmp, W, H, x + dx, y + dy).x;
                mean += v; m2 += v * v;
            }
        mean *= (1.0f / 25.0f);
        const float var = fmaxf(0.0f, m2 * (1.0f / 25.0f) - mean * mean);
        const float sigNoise = fmaxf(sy * gainYv, 1e-5f);
        const float sigSignal = sqrtf(fmaxf(var - sigNoise * sigNoise, 0.0f));
        const float snrDb = 20.0f * log10f(fmaxf(sigSignal, 1e-6f) / sigNoise);
        const float t = clampf((snrDb + 6.0f) / 36.0f, 0.0f, 1.0f);
        float hr, hg, hb;
        if (t < 0.5f) {
            const float u = t * 2.0f;
            hr = 0.85f + (0.95f - 0.85f) * u;
            hg = 0.10f + (0.70f - 0.10f) * u;
            hb = 0.75f + (0.15f - 0.75f) * u;
        } else {
            const float u = (t - 0.5f) * 2.0f;
            hr = 0.95f + (0.10f - 0.95f) * u;
            hg = 0.70f + (0.85f - 0.70f) * u;
            hb = 0.15f + (0.20f - 0.15f) * u;
        }
        o.x = cinY * 0.35f + hr * 0.65f;
        o.y = cinY * 0.35f + hg * 0.65f;
        o.z = cinY * 0.35f + hb * 0.65f;
    } else if (p.viewMode == 8) {
        // noise matte: normalized noise dominance in RGB+alpha
        float mean = 0.0f, m2 = 0.0f;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
                const float v = sampleTmp(tmp, W, H, x + dx, y + dy).x;
                mean += v; m2 += v * v;
            }
        mean *= (1.0f / 25.0f);
        const float var = fmaxf(0.0f, m2 * (1.0f / 25.0f) - mean * mean);
        const float sigNoise = fmaxf(sy * gainYv, 1e-5f);
        const float sigSignal = sqrtf(fmaxf(var - sigNoise * sigNoise, 0.0f));
        const float snrDb = 20.0f * log10f(fmaxf(sigSignal, 1e-6f) / sigNoise);
        const float m = 1.0f - clampf((snrDb + 6.0f) / 36.0f, 0.0f, 1.0f);
        o.x = m; o.y = m; o.z = m; o.w = m;
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
    const dim3 blocksHalf((p_Width / 2 + threads.x - 1) / threads.x,
                          (p_Height / 2 + threads.y - 1) / threads.y, 1);

    // a locked profile still runs estimation (live HUD) — FinalizeStats
    // overrides the sigma/gain slots with the locked values
    const bool autoProfile = (params.profileSource != 2) || (params.profileLocked != 0);

    if (autoProfile)
    {
        cudaMemsetAsync(res.stats, 0, NR_STATS_UINTS * sizeof(unsigned int), stream);
        NoiseEstKernel<<<blocksHalf, threads, 0, stream>>>(params, p_Width, p_Height, srcs[2], partner, res.stats);
        FinalizeStatsKernel<<<1, 1, 0, stream>>>(params, res.stats);
    }

    TemporalKernel<<<blocks, threads, 0, stream>>>(params, p_Width, p_Height,
                                                   srcs[0], srcs[1], srcs[2], srcs[3], srcs[4],
                                                   res.stats, res.tmp);

    if (autoProfile)
    {
        ResidualEstKernel<<<blocksHalf, threads, 0, stream>>>(params, p_Width, p_Height, res.tmp, res.stats);
        FinalizeResidualKernel<<<1, 1, 0, stream>>>(params, res.stats);
    }

    SpatialNLMKernel<<<blocks, threads, 0, stream>>>(params, p_Width, p_Height,
                                                     res.tmp, srcs[2], res.stats, dst);
}
