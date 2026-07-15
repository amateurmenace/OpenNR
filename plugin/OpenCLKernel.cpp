// OpenNR — OpenCL implementation of the denoising pipeline (v2).
// Line-by-line port of plugin/nr_core.h; keep the two in sync.

#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include <cstdio>
#include <map>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "NRParams.h"

static const char* kKernelSource = R"CLC(

typedef struct NRParams
{
    int   profileSource;
    float sigmaY;
    float sigmaC;
    float profileAdjust;
    float regionCX;
    float regionCY;
    float regionSize;
    int   hasTemporalDiff;

    int   enableTemporal;
    int   temporalFrames;
    float temporalLuma;
    float temporalChroma;
    float motionThresh;

    int   enableSpatial;
    int   spatialMode;
    int   spatialRadius;
    float spatialLuma;
    float spatialChroma;
    float preserveDetail;
    float chromaBlotch;

    int   enableRefine;
    float shadowDesat;
    float desatRange;
    float lumaTexture;
    float grainAmount;
    float grainSize;
    float grainChroma;
    int   frameIndex;

    float master;
    int   viewMode;

    int   motionTracking;
    int   fireflyRemoval;
    float eqFine;
    float eqMedium;
    float eqCoarse;
    float deband;
    int   profileLocked;
    float lockSY;
    float lockSC;
    float lockTY;
    float lockTC;
    float lockGainY[16];
    float lockGainC[16];

    float detailRescue;
    int   scopeMeasure;
    int   scopeMotion;
    int   scopeEq;

    int   ghostGuard;
    float globalBlend;

    int   deepClean;
    float lockSCr;
    float lockTCr;
} NRParams;

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
// v3.5 P1 exposure match — see nr_core.h
#define kExpBins     128
#define kExpScale    256.0f
#define kExpDead     0.006f

#define H_YF    0
#define H_CFB   256
#define H_CFR   512
#define H_Y2    768
#define H_C2B   1024
#define H_C2R   1280
#define H_YT    1536
#define H_CTB   1792
#define H_CTR   2048
#define L_Y     2304
#define L_C     3328
#define H_YR    4352
#define H_CRB   4608
#define H_CRR   4864
#define H_EN    5120
#define H_YR2   5184
#define H_CR2B  5440
#define H_CR2R  5696
#define S_SY    5952
#define S_SCB   5953
#define S_SCR   5954
#define S_TY    5955
#define S_TCB   5956
#define S_TCR   5957
#define S_RY    5958
#define S_RCB   5959
#define S_RCR   5960
#define S_MED   5961
#define S_HMAX  5962
#define S_GY    5963
#define S_GC    5979
#define S_ENMED 5995
#define S_FINEY 5996
#define S_FINEC 5997
#define S_CRSY  5998
#define H_EXP   5999
#define S_EXPOFF 6767

inline float smooth01f(float t)
{
    t = clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float3 rgb2ycc(float3 c)
{
    const float y = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    return (float3)(y, (c.z - y) / 1.8556f, (c.x - y) / 1.5748f);
}

inline float3 ycc2rgb(float3 v)
{
    const float r = v.x + 1.5748f * v.z;
    const float b = v.x + 1.8556f * v.y;
    const float g = (v.x - 0.2126f * r - 0.0722f * b) / 0.7152f;
    return (float3)(r, g, b);
}

inline float3 sampleYCC(__global const float4* img, int W, int H, int x, int y)
{
    x = clamp(x, 0, W - 1);
    y = clamp(y, 0, H - 1);
    return rgb2ycc(img[y * W + x].xyz);
}

inline float3 blockMeanYCC(__global const float4* img, int W, int H, int x, int y)
{
    float3 acc = (float3)(0.0f);
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
            acc += sampleYCC(img, W, H, x + dx, y + dy);
    return acc * 0.25f;
}

inline float4 sampleTmp(__global const float4* tmp, int W, int H, int x, int y)
{
    x = clamp(x, 0, W - 1);
    y = clamp(y, 0, H - 1);
    return tmp[y * W + x];
}

inline float med3(float a, float b, float c)
{
    return fmax(fmin(a, b), fmin(fmax(a, b), c));
}

inline void sort2(float* a, float* b)
{
    const float t = fmin(*a, *b);
    *b = fmax(*a, *b);
    *a = t;
}

// median of 9 via the standard 19-exchange network (Smith 1996)
inline float med9(const float* v)
{
    float p0 = v[0], p1 = v[1], p2 = v[2], p3 = v[3], p4 = v[4],
          p5 = v[5], p6 = v[6], p7 = v[7], p8 = v[8];
    sort2(&p1, &p2); sort2(&p4, &p5); sort2(&p7, &p8);
    sort2(&p0, &p1); sort2(&p3, &p4); sort2(&p6, &p7);
    sort2(&p1, &p2); sort2(&p4, &p5); sort2(&p7, &p8);
    sort2(&p0, &p3); sort2(&p5, &p8); sort2(&p4, &p7);
    sort2(&p3, &p6); sort2(&p1, &p4); sort2(&p2, &p5);
    sort2(&p4, &p7); sort2(&p4, &p2); sort2(&p6, &p4);
    sort2(&p4, &p2);
    return p4;
}

// 2x2 block mean of the tmp buffer (already YCC)
inline float3 blockMeanTmp(__global const float4* tmp, int W, int H, int x, int y)
{
    float3 acc = (float3)(0.0f);
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
            acc += sampleTmp(tmp, W, H, x + dx, y + dy).xyz;
    return acc * 0.25f;
}

// 4x4 block mean of tmp, centred-ish on (x,y)
inline float3 blockMean4Tmp(__global const float4* tmp, int W, int H, int x, int y)
{
    float3 acc = (float3)(0.0f);
    for (int dy = -1; dy < 3; ++dy)
        for (int dx = -1; dx < 3; ++dx)
            acc += sampleTmp(tmp, W, H, x + dx, y + dy).xyz;
    return acc * 0.0625f;
}

// v3.3 B1 hierarchical shift search offsets — see nr_core.h for the grid
// design, margins and the drift-bias rationale
__constant int kCoarseX[16] = { 4, -4, 0,  0, 4, -4,  4, -4, 8, -8, 0,  0, 8, -8,  8, -8 };
__constant int kCoarseY[16] = { 0,  0, 4, -4, 4,  4, -4, -4, 0,  0, 8, -8, 8,  8, -8, -8 };
__constant int kRefX[8] = { 1, -1, 0,  0, 1, -1,  1, -1 };
__constant int kRefY[8] = { 0,  0, 1, -1, 1,  1, -1, -1 };

// Mean |3x3 patch difference| of neighbour frame f shifted by (ox,oy) against
// the current frame's patch; also returns the shifted centre sample.
// expOff (v3.5 P1) — see nr_core.h
inline void patchDiff(__global const float4* f, int W, int H, int x, int y,
                      int ox, int oy, float expOff, const float3* c9,
                      float* dY, float* dCb, float* dCr, float* sdY, float3* fc)
{
    *dY = 0.0f; *dCb = 0.0f; *dCr = 0.0f; *sdY = 0.0f; *fc = (float3)(0.0f);
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx, ++i) {
            float3 v = sampleYCC(f, W, H, x + ox + dx, y + oy + dy);
            v.x -= expOff;
            if (i == 4) *fc = v;
            *dY += fabs(v.x - c9[i].x);
            *sdY += (v.x - c9[i].x);
            // v3.3 B5: per-channel chroma diffs — see nr_core.h
            *dCb += fabs(v.y - c9[i].y);
            *dCr += fabs(v.z - c9[i].z);
        }
    }
    *dY *= (1.0f / 9.0f);
    *dCb *= (1.0f / 9.0f);
    *dCr *= (1.0f / 9.0f);
    *sdY *= (1.0f / 9.0f);
}

// v3.3 B1: coarse matching score for the hierarchical shift search — mean
// |2x2-block-mean luma difference| of a 3x3 stride-2 block patch (6x6 px
// support); see nr_core.h.
inline float coarseDiff(__global const float4* f, int W, int H, int x, int y,
                        int ox, int oy, float expOff, const float* cb9)
{
    float d = 0.0f;
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i)
            d += fabs(blockMeanYCC(f, W, H, x + ox + dx * 2, y + oy + dy * 2).x - expOff - cb9[i]);
    return d * (1.0f / 9.0f);
}

inline float hashNoise(uint ix, uint iy, uint f, uint ch)
{
    uint h = ix * 374761393u + iy * 668265263u + f * 2246822519u + ch * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return ((float)(h & 0xFFFFFFu) / 16777215.0f) * 2.0f - 1.0f;
}

inline float valueNoise(float x, float y, float size, uint f, uint ch)
{
    const float gx = x / size, gy = y / size;
    const int ix = (int)floor(gx), iy = (int)floor(gy);
    float fx = gx - ix, fy = gy - iy;
    fx = fx * fx * fx * (fx * (fx * 6.0f - 15.0f) + 10.0f);
    fy = fy * fy * fy * (fy * (fy * 6.0f - 15.0f) + 10.0f);
    const float n00 = hashNoise((uint)ix,       (uint)iy,       f, ch);
    const float n10 = hashNoise((uint)(ix + 1), (uint)iy,       f, ch);
    const float n01 = hashNoise((uint)ix,       (uint)(iy + 1), f, ch);
    const float n11 = hashNoise((uint)(ix + 1), (uint)(iy + 1), f, ch);
    return (n00 + (n10 - n00) * fx) + ((n01 + (n11 - n01) * fx) - (n00 + (n10 - n00) * fx)) * fy;
}

__kernel void NoiseEstKernel(NRParams p, int W, int H,
                             __global const float4* curr,
                             __global const float4* partner,
                             volatile __global uint* stats)
{
    const int x = 1 + get_global_id(0) * 2;
    const int y = 1 + get_global_id(1) * 2;
    if (x >= W - 1 || y >= H - 1)
        return;

    if (p.profileSource == 1) {
        const float rHalf = 0.5f * p.regionSize * (float)min(W, H);
        const float cx = p.regionCX * W, cy = p.regionCY * H;
        const int x0 = clamp((int)(cx - rHalf), 1, W - 1);
        const int x1 = clamp((int)(cx + rHalf), 1, W - 1);
        const int y0 = clamp((int)(cy - rHalf), 1, H - 1);
        const int y1 = clamp((int)(cy + rHalf), 1, H - 1);
        if (x < x0 || x >= x1 || y < y0 || y >= y1)
            return;
    }

    float Y[9], Cb[9], Cr[9];
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i) {
            const float3 v = sampleYCC(curr, W, H, x + dx, y + dy);
            Y[i] = v.x; Cb[i] = v.y; Cr[i] = v.z;
        }

    const float lapY  = 4.0f * Y[4]  - 2.0f * (Y[1] + Y[3] + Y[5] + Y[7])   + (Y[0] + Y[2] + Y[6] + Y[8]);
    const float lapCb = 4.0f * Cb[4] - 2.0f * (Cb[1] + Cb[3] + Cb[5] + Cb[7]) + (Cb[0] + Cb[2] + Cb[6] + Cb[8]);
    const float lapCr = 4.0f * Cr[4] - 2.0f * (Cr[1] + Cr[3] + Cr[5] + Cr[7]) + (Cr[0] + Cr[2] + Cr[6] + Cr[8]);

    // v3.1: exactly-flat samples carry no noise evidence — skip entirely
    if (lapY == 0.0f && lapCb == 0.0f && lapCr == 0.0f)
        return;

    atomic_inc(&stats[H_YF + clamp((int)(fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)]);
    atomic_inc(&stats[H_CFB + clamp((int)(fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)]);
    atomic_inc(&stats[H_CFR + clamp((int)(fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)]);

    const int lb = clamp((int)(Y[4] * kLumaBins), 0, kLumaBins - 1);
    atomic_inc(&stats[L_Y + lb * kLumaSub + clamp((int)(fabs(lapY)  * kLumaSubScaleY), 0, kLumaSub - 1)]);
    atomic_inc(&stats[L_C + lb * kLumaSub + clamp((int)(fabs(lapCb) * kLumaSubScaleC), 0, kLumaSub - 1)]);
    atomic_inc(&stats[L_C + lb * kLumaSub + clamp((int)(fabs(lapCr) * kLumaSubScaleC), 0, kLumaSub - 1)]);

    if (p.hasTemporalDiff != 0) {
        const float3 pv = sampleYCC(partner, W, H, x, y);
        atomic_inc(&stats[H_YT + clamp((int)(fabs(pv.x - Y[4])  * kHistScaleY), 0, kHistBins - 1)]);
        atomic_inc(&stats[H_CTB + clamp((int)(fabs(pv.y - Cb[4]) * kHistScaleC), 0, kHistBins - 1)]);
        atomic_inc(&stats[H_CTR + clamp((int)(fabs(pv.z - Cr[4]) * kHistScaleC), 0, kHistBins - 1)]);
    }

    if ((get_global_id(0) & 1) == 0 && (get_global_id(1) & 1) == 0) {
        float bY[9], bCb[9], bCr[9];
        i = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx, ++i) {
                const float3 v = blockMeanYCC(curr, W, H, x + dx * 2, y + dy * 2);
                bY[i] = v.x; bCb[i] = v.y; bCr[i] = v.z;
            }
        const float lY  = 4.0f * bY[4]  - 2.0f * (bY[1] + bY[3] + bY[5] + bY[7])   + (bY[0] + bY[2] + bY[6] + bY[8]);
        const float lCb = 4.0f * bCb[4] - 2.0f * (bCb[1] + bCb[3] + bCb[5] + bCb[7]) + (bCb[0] + bCb[2] + bCb[6] + bCb[8]);
        const float lCr = 4.0f * bCr[4] - 2.0f * (bCr[1] + bCr[3] + bCr[5] + bCr[7]) + (bCr[0] + bCr[2] + bCr[6] + bCr[8]);
        if (lY != 0.0f || lCb != 0.0f || lCr != 0.0f) {  // v3.1 flat-sample skip
            atomic_inc(&stats[H_Y2 + clamp((int)(fabs(lY)  * kHistScaleY), 0, kHistBins - 1)]);
            atomic_inc(&stats[H_C2B + clamp((int)(fabs(lCb) * kHistScaleC), 0, kHistBins - 1)]);
            atomic_inc(&stats[H_C2R + clamp((int)(fabs(lCr) * kHistScaleC), 0, kHistBins - 1)]);
        }
    }
}

typedef struct QuantRes { float value; uint bin; } QuantRes;

inline QuantRes histQuantile(__global uint* stats, int base, int n, ulong total, float scale,
                             ulong num, ulong den)
{
    ulong cum = 0;
    const ulong target = (total * num + den - 1) / den;
    for (int b = 0; b < n; ++b) {
        cum += stats[base + b];
        if (cum >= target) {
            QuantRes r; r.value = ((float)b + 0.5f) / scale; r.bin = (uint)b;
            return r;
        }
    }
    QuantRes r; r.value = ((float)n - 0.5f) / scale; r.bin = n - 1;
    return r;
}

__kernel void FinalizeStatsKernel(NRParams p, __global uint* stats)
{
    if (get_global_id(0) != 0 || get_global_id(1) != 0)
        return;

    ulong totalF = 0, total2 = 0, totalT = 0;
    uint hmax = 1;
    for (int b = 0; b < kHistBins; ++b) {
        totalF += stats[H_YF + b];
        total2 += stats[H_Y2 + b];
        totalT += stats[H_YT + b];
        hmax = max(hmax, stats[H_YF + b]);
    }

    float sy = clamp(p.sigmaY, kSigmaMin, kSigmaMax);
    float scb = clamp(p.sigmaC, kSigmaMin, kSigmaMax);
    float scr = scb;
    float ty = sy, tcb = scb, tcr = scr;
    uint medBin = 0;
    float fineYv = sy, fineCv = scb, coarseYv = sy;   // v3.1: per-band (EQ scope)

    float gy[16], gc[16];
    for (int b = 0; b < 16; ++b) { gy[b] = 1.0f; gc[b] = 1.0f; }

    if (totalF >= 64) {
        const QuantRes mYf = histQuantile(stats, H_YF, kHistBins, totalF, kHistScaleY, 1, 2);
        medBin = mYf.bin;
        const float syFine = mYf.value * kMedianCal;
        const float scbFine = histQuantile(stats, H_CFB, kHistBins, totalF, kHistScaleC, 1, 2).value * kMedianCal;
        const float scrFine = histQuantile(stats, H_CFR, kHistBins, totalF, kHistScaleC, 1, 2).value * kMedianCal;
        const float syCoarse = 2.0f * histQuantile(stats, H_Y2, kHistBins, total2, kHistScaleY, 1, 2).value * kMedianCal;
        const float scbCoarse = 2.0f * histQuantile(stats, H_C2B, kHistBins, total2, kHistScaleC, 1, 2).value * kMedianCal;
        const float scrCoarse = 2.0f * histQuantile(stats, H_C2R, kHistBins, total2, kHistScaleC, 1, 2).value * kMedianCal;
        fineYv = syFine; fineCv = 0.5f * (scbFine + scrFine); coarseYv = syCoarse;

        const float lapSY = fmax(syFine, 0.9f * syCoarse);
        const float lapSCb = fmax(scbFine, 0.9f * scbCoarse);
        const float lapSCr = fmax(scrFine, 0.9f * scrCoarse);

        ty = lapSY;
        tcb = lapSCb;
        tcr = lapSCr;
        if (p.hasTemporalDiff != 0 && totalT >= 64) {
            const float medTY = histQuantile(stats, H_YT, kHistBins, totalT, kHistScaleY, 1, 2).value * kMedianCalT;
            const float q20TY = histQuantile(stats, H_YT, kHistBins, totalT, kHistScaleY, 1, 5).value * kQ20CalT;
            const float medTCb = histQuantile(stats, H_CTB, kHistBins, totalT, kHistScaleC, 1, 2).value * kMedianCalT;
            const float q20TCb = histQuantile(stats, H_CTB, kHistBins, totalT, kHistScaleC, 1, 5).value * kQ20CalT;
            const float medTCr = histQuantile(stats, H_CTR, kHistBins, totalT, kHistScaleC, 1, 2).value * kMedianCalT;
            const float q20TCr = histQuantile(stats, H_CTR, kHistBins, totalT, kHistScaleC, 1, 5).value * kQ20CalT;
            const float candY = (medTY <= 1.4f * q20TY) ? medTY : q20TY;
            const float candCb = (medTCb <= 1.4f * q20TCb) ? medTCb : q20TCb;
            const float candCr = (medTCr <= 1.4f * q20TCr) ? medTCr : q20TCr;
            if (candY > 0.0015f && candY <= 3.5f * lapSY) ty = candY;
            if (candCb > 0.0015f && candCb <= 3.5f * lapSCb) tcb = candCb;
            if (candCr > 0.0015f && candCr <= 3.5f * lapSCr) tcr = candCr;
        }

        const float adj = clamp(p.profileAdjust, 0.25f, 6.0f);
        sy = clamp(fmax(lapSY, 0.85f * ty) * adj, kSigmaMin, kSigmaMax);
        scb = clamp(fmax(lapSCb, 0.85f * tcb) * adj, kSigmaMin, kSigmaMax);
        scr = clamp(fmax(lapSCr, 0.85f * tcr) * adj, kSigmaMin, kSigmaMax);
        ty = clamp(ty * adj, kSigmaMin, kSigmaMax);
        tcb = clamp(tcb * adj, kSigmaMin, kSigmaMax);
        tcr = clamp(tcr * adj, kSigmaMin, kSigmaMax);

        const float q35RefY = histQuantile(stats, H_YF, kHistBins, totalF, kHistScaleY, 7, 20).value * kQ35Cal;
        // combined gain curve: reference is the mean of the per-channel quantiles
        const float q35RefC = 0.5f * (histQuantile(stats, H_CFB, kHistBins, totalF, kHistScaleC, 7, 20).value +
                                      histQuantile(stats, H_CFR, kHistBins, totalF, kHistScaleC, 7, 20).value) * kQ35Cal;
        for (int b = 0; b < kLumaBins; ++b) {
            ulong cy = 0, cc = 0;
            for (int s2 = 0; s2 < kLumaSub; ++s2) { cy += stats[L_Y + b * kLumaSub + s2]; cc += stats[L_C + b * kLumaSub + s2]; }
            if (cy >= 200 && q35RefY > 1e-6f) {
                const float sb = histQuantile(stats, L_Y + b * kLumaSub, kLumaSub, cy, kLumaSubScaleY, 7, 20).value * kQ35Cal;
                const float w = (float)cy / ((float)cy + 2000.0f);
                gy[b] = clamp(1.0f + w * (sb / q35RefY - 1.0f), 0.6f, 2.2f);
            }
            if (cc >= 200 && q35RefC > 1e-6f) {
                const float sb = histQuantile(stats, L_C + b * kLumaSub, kLumaSub, cc, kLumaSubScaleC, 7, 20).value * kQ35Cal;
                const float w = (float)cc / ((float)cc + 4000.0f);
                gc[b] = clamp(1.0f + w * (sb / q35RefC - 1.0f), 0.6f, 2.2f);
            }
        }
    }

    // v3: a locked profile overrides the measured sigmas and gains; the
    // histogram/medbin stay live so the HUD keeps showing the measurement.
    // v3.2: the lock stores the RAW measurement — Auto Profile Adjust is
    // applied here so the trim slider keeps working while locked.
    if (p.profileLocked != 0) {
        const float adjL = clamp(p.profileAdjust, 0.25f, 6.0f);
        sy = clamp(p.lockSY * adjL, kSigmaMin, kSigmaMax);
        scb = clamp(p.lockSC * adjL, kSigmaMin, kSigmaMax);
        scr = clamp(p.lockSCr * adjL, kSigmaMin, kSigmaMax);
        ty = clamp(p.lockTY * adjL, kSigmaMin, kSigmaMax);
        tcb = clamp(p.lockTC * adjL, kSigmaMin, kSigmaMax);
        tcr = clamp(p.lockTCr * adjL, kSigmaMin, kSigmaMax);
    }

    stats[S_SY] = as_uint(sy);
    stats[S_SCB] = as_uint(scb);
    stats[S_SCR] = as_uint(scr);
    stats[S_TY] = as_uint(ty);
    stats[S_TCB] = as_uint(tcb);
    stats[S_TCR] = as_uint(tcr);
    stats[S_MED] = medBin;
    stats[S_HMAX] = hmax;
    // v3.1: raw per-band estimates for the EQ scope (never lock-overridden)
    stats[S_FINEY] = as_uint(fineYv);
    stats[S_FINEC] = as_uint(fineCv);
    stats[S_CRSY]  = as_uint(coarseYv);
    for (int b = 0; b < kLumaBins; ++b) {
        const int b0 = clamp(b - 1, 0, kLumaBins - 1);
        const int b1 = clamp(b + 1, 0, kLumaBins - 1);
        stats[S_GY + b] = as_uint(p.profileLocked != 0 ? clamp(p.lockGainY[b], 0.6f, 2.2f)
                                  : 0.25f * gy[b0] + 0.5f * gy[b] + 0.25f * gy[b1]);
        stats[S_GC + b] = as_uint(p.profileLocked != 0 ? clamp(p.lockGainC[b], 0.6f, 2.2f)
                                  : 0.25f * gc[b0] + 0.5f * gc[b] + 0.25f * gc[b1]);
    }
}

// ---------------------------------------------------------------------------
// v3.5 P1 — per-neighbour exposure offset — see nr_core.h / MetalKernel.mm
// ---------------------------------------------------------------------------
__kernel void ExposureEstKernel(NRParams p, int W, int H,
                                __global const float4* f0, __global const float4* f1,
                                __global const float4* f2, __global const float4* f3,
                                __global const float4* f4, __global const float4* f5,
                                __global const float4* f6, __global uint* stats)
{
    const int x = 1 + get_global_id(0) * 4;
    const int y = 1 + get_global_id(1) * 4;
    if (x >= W - 1 || y >= H - 1)
        return;
    const int reach = (p.enableTemporal == 0) ? 0 : ((p.temporalFrames >= 7) ? 3 : (p.temporalFrames >= 5) ? 2 : 1);
    const float cyv = sampleYCC(f3, W, H, x, y).x;
    for (int k = 3 - reach; k <= 3 + reach; ++k) {
        if (k == 3)
            continue;
        __global const float4* nb = (k == 0) ? f0 : (k == 1) ? f1 : (k == 2) ? f2
                                  : (k == 4) ? f4 : (k == 5) ? f5 : f6;
        const float nyv = sampleYCC(nb, W, H, x, y).x;
        atomic_inc(&stats[H_EXP + ((k < 3) ? k : k - 1) * kExpBins +
            clamp((int)((nyv - cyv + 0.25f) * kExpScale), 0, kExpBins - 1)]);
    }
}

__kernel void FinalizeExposureKernel(__global uint* stats)
{
    if (get_global_id(0) != 0 || get_global_id(1) != 0)
        return;
    for (int s2 = 0; s2 < 6; ++s2) {
        uint etotal = 0;
        for (int b = 0; b < kExpBins; ++b)
            etotal += stats[H_EXP + s2 * kExpBins + b];
        float o = 0.0f;
        if (etotal >= 64) {
            uint cum = 0;
            const uint target = (etotal + 1) / 2;
            int mbin = kExpBins - 1;
            for (int b = 0; b < kExpBins; ++b) {
                cum += stats[H_EXP + s2 * kExpBins + b];
                if (cum >= target) { mbin = b; break; }
            }
            const float v = ((float)mbin + 0.5f) / kExpScale - 0.25f;
            if (fabs(v) >= kExpDead)
                o = v;
        }
        stats[S_EXPOFF + s2] = as_uint(o);
    }
}

inline void loadSigmasIn(NRParams p, __global const uint* stats,
                         float* sy, float* scb, float* scr,
                         float* ty, float* tcb, float* tcr)
{
    // v3.3 lock fast path: locked sigmas are a pure function of the params —
    // computed here with exactly FinalizeStats' arithmetic (bit-identical),
    // so the host can skip the NoiseEst/FinalizeStats dispatches when no
    // scope/analysis view is showing the live measurement.
    if (p.profileLocked != 0) {
        const float adjL = clamp(p.profileAdjust, 0.25f, 6.0f);
        *sy = clamp(p.lockSY * adjL, kSigmaMin, kSigmaMax);
        *scb = clamp(p.lockSC * adjL, kSigmaMin, kSigmaMax);
        *scr = clamp(p.lockSCr * adjL, kSigmaMin, kSigmaMax);
        *ty = clamp(p.lockTY * adjL, kSigmaMin, kSigmaMax);
        *tcb = clamp(p.lockTC * adjL, kSigmaMin, kSigmaMax);
        *tcr = clamp(p.lockTCr * adjL, kSigmaMin, kSigmaMax);
    } else if (p.profileSource != 2) {
        *sy = as_float(stats[S_SY]);
        *scb = as_float(stats[S_SCB]);
        *scr = as_float(stats[S_SCR]);
        *ty = as_float(stats[S_TY]);
        *tcb = as_float(stats[S_TCB]);
        *tcr = as_float(stats[S_TCR]);
    } else {
        *sy = *ty = clamp(p.sigmaY, kSigmaMin, kSigmaMax);
        *scb = *scr = *tcb = *tcr = clamp(p.sigmaC, kSigmaMin, kSigmaMax);
    }
}

__kernel void TemporalKernel(NRParams p, int W, int H,
                             __global const float4* f0, __global const float4* f1,
                             __global const float4* f2, __global const float4* f3,
                             __global const float4* f4, __global const float4* f5,
                             __global const float4* f6,
                             __global const uint* stats, __global float4* tmp)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H)
        return;

    float sigTYv, sigSCb, sigSCr, sigSYv, sigTCb, sigTCr;
    loadSigmasIn(p, stats, &sigSYv, &sigSCb, &sigSCr, &sigTYv, &sigTCb, &sigTCr);
    const float sigTC = 0.5f * (sigTCb + sigTCr);   // pair mean (zapper)
    const float mLow  = fmin(p.master, 1.0f);
    const float mHigh = fmax(p.master, 1.0f);
    // v3.1: sliders reach 125 — a matching neighbour may outweigh the centre
    const float tL = clamp(p.temporalLuma   * mLow, 0.0f, 1.25f);
    const float tC = clamp(p.temporalChroma * mLow, 0.0f, 1.25f);
    const float thrMul = 0.4f + 2.6f * clamp(p.motionThresh, 0.0f, 1.5f)
                       + 0.8f * (mHigh - 1.0f);
    // v3.3 B2: the stack grows to 7 frames (reach 3) for static heavy noise
    const int reach = (p.enableTemporal == 0) ? 0 : ((p.temporalFrames >= 7) ? 3 : (p.temporalFrames >= 5) ? 2 : 1);
    const float loY = kAbsDiffBias * sigTYv, hiY = loY + thrMul * sigTYv;
    // v3.3 B5: each chroma channel gets its own knee
    const float loCb = kAbsDiffBias * sigTCb, hiCb = loCb + thrMul * sigTCb;
    const float loCr = kAbsDiffBias * sigTCr, hiCr = loCr + thrMul * sigTCr;
    const float invSpanY = 1.0f / (hiY - loY);
    const float invSpanCb = 1.0f / (hiCb - loCb);
    const float invSpanCr = 1.0f / (hiCr - loCr);
    // v3 shift search engages only once the unshifted match is well into the
    // gate — high enough that pure noise almost never reaches it, which
    // also keeps GPU warps convergent on static footage (see nr_core.h).
    const int   track = (p.motionTracking != 0);
    const float searchThresh = loY + 0.75f * (hiY - loY);
    // v3.2 Ghost Guard — see nr_core.h for the signed-mean rationale
    const int   guard = (p.ghostGuard != 0);
    const float loS = kAbsDiffBias * sigTYv;
    const float invSpanS = 1.0f / (0.5f * thrMul * sigTYv);
    const int   zap = (reach >= 1) && (p.fireflyRemoval != 0) &&
                      (p.master > 0.0f) && (tL > 0.0f || tC > 0.0f);
    const float zapY = 6.0f * sigTYv;
    const float zapC = 6.0f * sigTC;

    float3 c9[9];
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i)
            c9[i] = sampleYCC(f3, W, H, x + dx, y + dy);

    // v3 firefly zapper — see nr_core.h for the three-test rationale
    if (zap) {
        float3 pv = sampleYCC(f2, W, H, x, y);
        float3 nv = sampleYCC(f4, W, H, x, y);
        pv.x -= as_float(stats[S_EXPOFF + 2]);   // v3.5 P1
        nv.x -= as_float(stats[S_EXPOFF + 3]);
        float lum9[9];
        for (int j = 0; j < 9; ++j) lum9[j] = c9[j].x;
        if (fabs(pv.x - nv.x) < 0.5f * zapY &&
            0.5f * (fabs(pv.y - nv.y) + fabs(pv.z - nv.z)) < 0.5f * zapC &&
            fabs(c9[4].x - med9(lum9)) > 0.5f * zapY) {
            const float mY  = med3(pv.x, c9[4].x, nv.x);
            const float mCb = med3(pv.y, c9[4].y, nv.y);
            const float mCr = med3(pv.z, c9[4].z, nv.z);
            if (fabs(c9[4].x - mY) > zapY ||
                0.5f * (fabs(c9[4].y - mCb) + fabs(c9[4].z - mCr)) > zapC) {
                c9[4] = (float3)(mY, mCb, mCr);
            }
        }
    }

    // v3.4: per-brightness gate calibration — every temporal threshold scales
    // with the centre pixel's gain-curve gain, exactly like the spatial stage
    // always did (see nr_core.h for the shadows/highlights rationale; the
    // 6-sigma zapper stays unscaled on purpose). Gains follow the same
    // locked/manual/live triple as the spatial pass: the lock fast path skips
    // FinalizeStats, so a locked profile's gains must come from the params.
    const int manualG = (p.profileSource == 2) && (p.profileLocked == 0);
    const int lockedG = (p.profileLocked != 0);
    const int gLb = clamp((int)(c9[4].x * kLumaBins), 0, kLumaBins - 1);
    const float gnY = lockedG ? clamp(p.lockGainY[gLb], 0.6f, 2.2f)
                    : manualG ? 1.0f : as_float(stats[S_GY + gLb]);
    const float gnC = lockedG ? clamp(p.lockGainC[gLb], 0.6f, 2.2f)
                    : manualG ? 1.0f : as_float(stats[S_GC + gLb]);
    const float invGnY = 1.0f / gnY;
    const float invGnC = 1.0f / gnC;

    float accY = c9[4].x, accCb = c9[4].y, accCr = c9[4].z;
    float sumWY = 1.0f, sumWY2 = 1.0f, sumWCb = 1.0f, sumWCr = 1.0f;

    // v3.3 B1: the centre's coarse block patch for the hierarchical search —
    // built lazily once per pixel, shared by all neighbours
    float cb9[9];
    bool haveCb9 = false;

    for (int k = 3 - reach; k <= 3 + reach; ++k) {
        if (k == 3)
            continue;
        __global const float4* f = (k == 0) ? f0 : (k == 1) ? f1 : (k == 2) ? f2
                                 : (k == 4) ? f4 : (k == 5) ? f5 : f6;

        const float oK = as_float(stats[S_EXPOFF + (k < 3 ? k : k - 1)]);   // v3.5 P1
        float dY, dCb, dCr, sdY;
        float3 fc;
        patchDiff(f, W, H, x, y, 0, 0, oK, c9, &dY, &dCb, &dCr, &sdY, &fc);

        // v3.3 B1 hierarchical shift search — see nr_core.h for the grid,
        // the margins and the drift-bias rationale
        float shiftTight = 1.0f;
        if (track && dY > searchThresh * gnY) {
            int wx = 0, wy = 0;
            // coarse level only when the unshifted match is FULLY outside
            // the knee (weight already zero — nothing to lose by hunting
            // far); static/drift noise crosses searchThresh routinely but
            // hiY rarely, so it never pays the 16-node sweep. See nr_core.h.
            if (dY > hiY * gnY) {
                if (!haveCb9) {
                    int i2 = 0;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx, ++i2)
                            cb9[i2] = blockMeanYCC(f3, W, H, x + dx * 2, y + dy * 2).x;
                    haveCb9 = true;
                }
                // coarse level: best step-4 node on block means
                float bestC = coarseDiff(f, W, H, x, y, kCoarseX[0], kCoarseY[0], oK, cb9);
                int bestOx = kCoarseX[0], bestOy = kCoarseY[0];
                for (int c = 1; c < 16; ++c) {
                    const float d = coarseDiff(f, W, H, x, y, kCoarseX[c], kCoarseY[c], oK, cb9);
                    if (d < bestC) { bestC = d; bestOx = kCoarseX[c]; bestOy = kCoarseY[c]; }
                }
                // the coarse winner must survive the real patch metric by 10%
                float dY2, dCb2, dCr2, sd2;
                float3 fc2;
                patchDiff(f, W, H, x, y, bestOx, bestOy, oK, c9, &dY2, &dCb2, &dCr2, &sd2, &fc2);
                if (dY2 < dY * 0.90f) {
                    dY = dY2; dCb = dCb2; dCr = dCr2; sdY = sd2; fc = fc2;
                    wx = bestOx; wy = bestOy;
                    shiftTight = 1.0f / 0.6f;
                }
            }
            // refine level: converging +/-1 walk around the winner
            for (int it = 0; it < 2; ++it) {
                int nwx = wx, nwy = wy;
                for (int c = 0; c < 8; ++c) {
                    const int tx = wx + kRefX[c], ty = wy + kRefY[c];
                    float dY2, dCb2, dCr2, sd2;
                    float3 fc2;
                    patchDiff(f, W, H, x, y, tx, ty, oK, c9, &dY2, &dCb2, &dCr2, &sd2, &fc2);
                    if (dY2 < dY * 0.99f) {
                        dY = dY2; dCb = dCb2; dCr = dCr2; sdY = sd2; fc = fc2;
                        nwx = tx; nwy = ty;
                        shiftTight = 1.0f / 0.6f;
                    }
                }
                if (nwx == wx && nwy == wy)
                    break;
                wx = nwx; wy = nwy;
            }
        }

        float gY = 1.0f - smooth01f((dY - loY * gnY) * invSpanY * invGnY * shiftTight);
        if (guard)
            gY *= 1.0f - smooth01f((fabs(sdY) - loS * gnY) * invSpanS * invGnY * shiftTight);
        // v3.3 B5: per-channel chroma gates (both slaved to the luma gate)
        const float gCb = 1.0f - smooth01f((dCb - loCb * gnC) * invSpanCb * invGnC * shiftTight);
        const float gCr = 1.0f - smooth01f((dCr - loCr * gnC) * invSpanCr * invGnC * shiftTight);
        const float wY = tL * gY;
        const float wCb = tC * gCb * gY;
        const float wCr = tC * gCr * gY;

        accY  += wY * fc.x;
        accCb += wCb * fc.y;
        accCr += wCr * fc.z;
        sumWY  += wY;
        sumWY2 += wY * wY;
        sumWCb += wCb;
        sumWCr += wCr;
    }

    tmp[y * W + x] = (float4)(accY / sumWY, accCb / sumWCb, accCr / sumWCr,
                              (sumWY * sumWY) / sumWY2);
}

__kernel void ResidualEstKernel(NRParams p, int W, int H,
                                __global const float4* tmp,
                                volatile __global uint* stats)
{
    const int x = 1 + get_global_id(0) * 2;
    const int y = 1 + get_global_id(1) * 2;
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
    if (lapY == 0.0f && lapCb == 0.0f && lapCr == 0.0f)  // v3.1 flat-sample skip
        return;
    atomic_inc(&stats[H_YR + clamp((int)(fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)]);
    atomic_inc(&stats[H_CRB + clamp((int)(fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)]);
    atomic_inc(&stats[H_CRR + clamp((int)(fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)]);
    const float effN = sampleTmp(tmp, W, H, x, y).w;
    // v3.3 B2: 64 bins (was 32) — see nr_core.h
    atomic_inc(&stats[H_EN + clamp((int)((effN - 1.0f) * 8.0f), 0, 63)]);

    // v3.2 coarse residual — see nr_core.h (even-aligned 2x2 blocks)
    if ((get_global_id(0) & 1) == 0 && (get_global_id(1) & 1) == 0) {
        float bY[9], bCb[9], bCr[9];
        i = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx, ++i) {
                const float3 v = blockMeanTmp(tmp, W, H, (x - 1) + dx * 2, (y - 1) + dy * 2);
                bY[i] = v.x; bCb[i] = v.y; bCr[i] = v.z;
            }
        const float lY  = 4.0f * bY[4]  - 2.0f * (bY[1] + bY[3] + bY[5] + bY[7])   + (bY[0] + bY[2] + bY[6] + bY[8]);
        const float lCb = 4.0f * bCb[4] - 2.0f * (bCb[1] + bCb[3] + bCb[5] + bCb[7]) + (bCb[0] + bCb[2] + bCb[6] + bCb[8]);
        const float lCr = 4.0f * bCr[4] - 2.0f * (bCr[1] + bCr[3] + bCr[5] + bCr[7]) + (bCr[0] + bCr[2] + bCr[6] + bCr[8]);
        if (lY != 0.0f || lCb != 0.0f || lCr != 0.0f) {
            atomic_inc(&stats[H_YR2 + clamp((int)(fabs(lY)  * kHistScaleY), 0, kHistBins - 1)]);
            atomic_inc(&stats[H_CR2B + clamp((int)(fabs(lCb) * kHistScaleC), 0, kHistBins - 1)]);
            atomic_inc(&stats[H_CR2R + clamp((int)(fabs(lCr) * kHistScaleC), 0, kHistBins - 1)]);
        }
    }
}

__kernel void FinalizeResidualKernel(NRParams p, __global uint* stats)
{
    if (get_global_id(0) != 0 || get_global_id(1) != 0)
        return;

    ulong total = 0, total2 = 0;
    for (int b = 0; b < kHistBins; ++b) {
        total += stats[H_YR + b];
        total2 += stats[H_YR2 + b];
    }

    // v3.3 lock fast path: under a lock the input-stat slots may not have
    // been written this frame — compute the locked values from the params
    // (bit-identical to what FinalizeStats writes when it does run).
    float sy = as_float(stats[S_SY]);
    float scb = as_float(stats[S_SCB]);
    float scr = as_float(stats[S_SCR]);
    if (p.profileLocked != 0) {
        const float adjL = clamp(p.profileAdjust, 0.25f, 6.0f);
        sy = clamp(p.lockSY * adjL, kSigmaMin, kSigmaMax);
        scb = clamp(p.lockSC * adjL, kSigmaMin, kSigmaMax);
        scr = clamp(p.lockSCr * adjL, kSigmaMin, kSigmaMax);
    }
    float ry = sy, rcb = scb, rcr = scr, enmed = 1.0f;

    if (total >= 64) {
        const float adj = clamp(p.profileAdjust, 0.25f, 6.0f);
        ry = histQuantile(stats, H_YR, kHistBins, total, kHistScaleY, 1, 2).value * kMedianCal * adj;
        rcb = histQuantile(stats, H_CRB, kHistBins, total, kHistScaleC, 1, 2).value * kMedianCal * adj;
        rcr = histQuantile(stats, H_CRR, kHistBins, total, kHistScaleC, 1, 2).value * kMedianCal * adj;
        // v3.2: two-scale residual — see nr_core.h
        if (total2 >= 64) {
            const float ryC = 2.0f * histQuantile(stats, H_YR2, kHistBins, total2, kHistScaleY, 1, 2).value * kMedianCal * adj;
            const float rcbC = 2.0f * histQuantile(stats, H_CR2B, kHistBins, total2, kHistScaleC, 1, 2).value * kMedianCal * adj;
            const float rcrC = 2.0f * histQuantile(stats, H_CR2R, kHistBins, total2, kHistScaleC, 1, 2).value * kMedianCal * adj;
            ry = fmax(ry, 0.9f * ryC);
            rcb = fmax(rcb, 0.9f * rcbC);
            rcr = fmax(rcr, 0.9f * rcrC);
        }
        enmed = 1.0f + histQuantile(stats, H_EN, 64, total, 8.0f, 1, 2).value;
        const float floorY = 0.5f * sy / sqrt(fmax(1.0f, enmed));
        const float floorCb = 0.5f * scb / sqrt(fmax(1.0f, enmed));
        const float floorCr = 0.5f * scr / sqrt(fmax(1.0f, enmed));
        ry = clamp(fmax(ry, floorY), kSigmaMin, sy > kSigmaMin ? sy : kSigmaMax);
        rcb = clamp(fmax(rcb, floorCb), kSigmaMin, scb > kSigmaMin ? scb : kSigmaMax);
        rcr = clamp(fmax(rcr, floorCr), kSigmaMin, scr > kSigmaMin ? scr : kSigmaMax);
    }

    stats[S_RY] = as_uint(ry);
    stats[S_RCB] = as_uint(rcb);
    stats[S_RCR] = as_uint(rcr);
    stats[S_ENMED] = as_uint(enmed);
}

// HUD v3 + scopes — see nr_core.h for the layout rationale
// glyph order: 0-9 . % A-Z + space - | =
__constant ulong kFont[43] = {
    0x3a33ae62eUL, 0x11842108eUL, 0x3a213221fUL, 0x3a213062eUL, 0x08ca97c42UL, 0x7e1e0862eUL,
    0x3a10f462eUL, 0x7c2222108UL, 0x3a317462eUL, 0x3a317842eUL, 0x00000018cUL, 0x632222263UL,
    0x3a31fc631UL, 0x7a31f463eUL, 0x3a308422eUL, 0x7a318c63eUL, 0x7e10f421fUL, 0x7e10f4210UL,
    0x3a30bc62fUL, 0x4631fc631UL, 0x38842108eUL, 0x1c4210a4cUL, 0x4654c5251UL, 0x42108421fUL,
    0x4775ac631UL, 0x47359c631UL, 0x3a318c62eUL, 0x7a31f4210UL, 0x3a318d64dUL, 0x7a31f5251UL,
    0x3e107043eUL, 0x7c8421084UL, 0x46318c62eUL, 0x46318c544UL, 0x4631ad771UL, 0x462a22a31UL,
    0x462a21084UL, 0x7c222221fUL, 0x0084f9080UL, 0x000000000UL, 0x000070000UL, 0x108421084UL,
    0x01f07c00UL,
};
#define G_DOT 10
#define G_PCT 11
#define G_A 12
#define G_B 13
#define G_C 14
#define G_D 15
#define G_E 16
#define G_F 17
#define G_G 18
#define G_H 19
#define G_I 20
#define G_J 21
#define G_K 22
#define G_L 23
#define G_M 24
#define G_N 25
#define G_O 26
#define G_P 27
#define G_Q 28
#define G_R 29
#define G_S 30
#define G_T 31
#define G_U 32
#define G_V 33
#define G_W 34
#define G_X 35
#define G_Y 36
#define G_Z 37
#define G_PLUS 38
#define G_SP 39
#define G_DASH 40
#define G_BAR 41
#define G_EQ 42

__constant int kLabIY[7]  = { G_I, G_N, G_P, G_U, G_T, G_SP, G_Y };
__constant int kLabIC[7]  = { G_I, G_N, G_P, G_U, G_T, G_SP, G_C };
__constant int kLabRY[10] = { G_R, G_E, G_S, G_I, G_D, G_U, G_A, G_L, G_SP, G_Y };
__constant int kLabRC[10] = { G_R, G_E, G_S, G_I, G_D, G_U, G_A, G_L, G_SP, G_C };
__constant int kLabAVGFR[10] = { G_A, G_V, G_G, G_SP, G_F, G_R, G_A, G_M, G_E, G_S };
__constant int kLabGAIN[4] = { G_G, G_A, G_I, G_N };
__constant int kLabDB[2]   = { G_D, G_B };
__constant int kLabLOCKED[14] = { G_P, G_R, G_O, G_F, G_I, G_L, G_E, G_SP,
                                  G_L, G_O, G_C, G_K, G_E, G_D };
__constant int kLabLIVE[14] = { G_M, G_E, G_A, G_S, G_U, G_R, G_I, G_N,
                                G_G, G_SP, G_L, G_I, G_V, G_E };
__constant int kLabTOFF[12] = { G_T, G_E, G_M, G_P, G_O, G_R, G_A, G_L,
                                G_SP, G_O, G_F, G_F };
__constant int kLabCURVE[19] = { G_N, G_O, G_I, G_S, G_E, G_SP, G_V, G_S,
                                 G_SP, G_B, G_R, G_I, G_G, G_H, G_T, G_N,
                                 G_E, G_S, G_S };
__constant int kLabHIST[31] = { G_N, G_O, G_I, G_S, G_E, G_SP, G_H, G_I,
                                G_S, G_T, G_O, G_G, G_R, G_A, G_M, G_SP,
                                G_DASH, G_SP, G_M, G_E, G_D, G_I, G_A, G_N,
                                G_SP, G_M, G_A, G_R, G_K, G_E, G_D };
__constant int kLabTITLE[8] = { G_N, G_O, G_I, G_S, G_E, G_SP, G_E, G_Q };
__constant int kLabOFF3[3]  = { G_O, G_F, G_F };
__constant int kLabFINE[4] = { G_F, G_I, G_N, G_E };
__constant int kLabMED[6]  = { G_M, G_E, G_D, G_I, G_U, G_M };
__constant int kLabCRS[6]  = { G_C, G_O, G_A, G_R, G_S, G_E };
__constant int kLabCOL[5]  = { G_C, G_O, G_L, G_O, G_R };
__constant int kLabPX1[3]  = { 1, G_P, G_X };
__constant int kLabPX38[5] = { 3, G_DASH, 8, G_P, G_X };
__constant int kLabPX16[5] = { 1, 6, G_P, G_X, G_PLUS };
__constant int kLabLEG[34] = { G_B, G_A, G_R, G_SP, G_EQ, G_SP, G_C, G_U,
                               G_T, G_SP, G_SP, G_A, G_M, G_B, G_E, G_R,
                               G_SP, G_EQ, G_SP, G_M, G_E, G_A, G_S, G_U,
                               G_R, G_E, G_D, G_SP, G_N, G_O, G_I, G_S,
                               G_E, G_SP };
__constant int kLabMO[33] = { G_M, G_O, G_T, G_I, G_O, G_N, G_SP, G_SP,
                              G_G, G_R, G_E, G_E, G_N, G_EQ, G_S, G_T,
                              G_A, G_C, G_K, G_E, G_D, G_SP, G_SP, G_R,
                              G_E, G_D, G_EQ, G_M, G_O, G_V, G_I, G_N,
                              G_G };
__constant float kDirX[8] = { 1, 0, -1, 0, 0.7071f, -0.7071f, -0.7071f, 0.7071f };
__constant float kDirY[8] = { 0, 1, 0, -1, 0.7071f, 0.7071f, -0.7071f, -0.7071f };

inline bool glyphPixel(int glyph, int gx, int gy)
{
    if (glyph < 0 || glyph > 42 || gx < 0 || gx >= 5 || gy < 0 || gy >= 7)
        return false;
    return (kFont[glyph] >> (34 - (gy * 5 + gx))) & 1UL;
}

// sc = integer text scale: 1 -> 6x7 px cells, 2 -> 12x14 px cells
inline bool textPixelC(__constant int* chars, int n, int tx, int ty, int lx, int ly, int sc)
{
    if (ly < ty || ly >= ty + 7 * sc || lx < tx || lx >= tx + n * 6 * sc)
        return false;
    const int gx = (lx - tx) / sc;
    const int ci = gx / 6;
    return glyphPixel(chars[ci], gx - ci * 6, (ly - ty) / sc);
}

inline bool textPixelP(const int* chars, int n, int tx, int ty, int lx, int ly, int sc)
{
    if (ly < ty || ly >= ty + 7 * sc || lx < tx || lx >= tx + n * 6 * sc)
        return false;
    const int gx = (lx - tx) / sc;
    const int ci = gx / 6;
    return glyphPixel(chars[ci], gx - ci * 6, (ly - ty) / sc);
}

inline void pctGlyphs(float pp, int* outg)
{
    const int v = clamp((int)(pp * 100.0f + 0.5f), 0, 9999);
    const int tens = (v / 1000) % 10;
    outg[0] = (tens == 0) ? G_SP : tens;
    outg[1] = (v / 100) % 10;
    outg[2] = G_DOT;
    outg[3] = (v / 10) % 10;
    outg[4] = v % 10;
    outg[5] = G_PCT;
}

inline void dec1Glyphs(float v, int* outg)
{
    const int t = clamp((int)(v * 10.0f + 0.5f), 0, 99);
    outg[0] = (t / 10) % 10;
    outg[1] = G_DOT;
    outg[2] = t % 10;
}

inline void pctIntGlyphs(float frac, int* outg)
{
    const int v = clamp((int)(frac * 100.0f + 0.5f), 0, 999);
    outg[0] = (v >= 100) ? (v / 100) % 10 : G_SP;
    outg[1] = (v >= 10) ? (v / 10) % 10 : G_SP;
    outg[2] = v % 10;
    outg[3] = G_PCT;
}

inline bool hudPixel(int x, int y, int W, int H,
                     float sy, float sc, float ry, float rc, float enmed,
                     uint medBin, uint hmax, __global const uint* stats,
                     int enableTemporal, int locked, float3* rgb)
{
    const int yd = H - 1 - y;   // OFX buffers are bottom-up; panel anchors top-left on screen
    const int s = max(1, H / 540);
    const int ox = 16 * s, oy = 16 * s;
    const int lw = 360, lh = 294;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    *rgb = (float3)(0.045f, 0.045f, 0.05f);
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        *rgb = (float3)(0.35f);
        return true;
    }

    const float sig[4] = { sy, sc, ry, rc };
    const int rowY[4] = { 10, 42, 74, 106 };

    for (int row = 0; row < 4; ++row) {
        const int ty0 = rowY[row];
        bool lit = false;
        if (row == 0) lit = textPixelC(kLabIY, 7,  10, ty0, lx, ly, 2);
        if (row == 1) lit = textPixelC(kLabIC, 7,  10, ty0, lx, ly, 2);
        if (row == 2) lit = textPixelC(kLabRY, 10, 10, ty0, lx, ly, 2);
        if (row == 3) lit = textPixelC(kLabRC, 10, 10, ty0, lx, ly, 2);
        if (!lit) {
            int vg[6];
            pctGlyphs(sig[row] * 100.0f, vg);
            lit = textPixelP(vg, 6, 278, ty0, lx, ly, 2);
        }
        if (lit) { *rgb = (float3)(1.0f); return true; }

        if (ly >= ty0 + 16 && ly < ty0 + 22 && lx >= 10 && lx < 350) {
            const float fill = clamp(sig[row] / 0.08f, 0.0f, 1.0f);
            const bool on = (lx - 10) < (int)(fill * 340.0f);
            const bool residRow = (row >= 2);
            if (on) *rgb = residRow ? (float3)(0.95f, 0.65f, 0.20f) : (float3)(0.20f, 0.65f, 0.95f);
            else    *rgb = (float3)(0.14f);
            return true;
        }
    }

    {
        const int ty0 = 138;
        bool lit = textPixelC(kLabAVGFR, 10, 10, ty0, lx, ly, 2);
        if (!lit) {
            int vg[3];
            dec1Glyphs(enableTemporal ? enmed : 1.0f, vg);
            lit = textPixelP(vg, 3, 138, ty0, lx, ly, 2);
        }
        if (!lit) lit = textPixelC(kLabGAIN, 4, 186, ty0, lx, ly, 2);
        if (!lit) {
            const float gainDb = clamp(20.0f * log10(fmax(sy, 1e-5f) / fmax(ry, 1e-5f)), 0.0f, 40.0f);
            int vg[4];
            vg[0] = G_PLUS;
            int d1[3];
            dec1Glyphs(gainDb, d1);
            vg[1] = d1[0]; vg[2] = d1[1]; vg[3] = d1[2];
            lit = textPixelP(vg, 4, 240, ty0, lx, ly, 2);
            if (!lit) lit = textPixelC(kLabDB, 2, 294, ty0, lx, ly, 2);
        }
        if (lit) { *rgb = (float3)(1.0f); return true; }
    }

    {
        const int ty0 = 160;
        if (locked) {
            if (textPixelC(kLabLOCKED, 14, 10, ty0, lx, ly, 2)) { *rgb = (float3)(0.95f, 0.65f, 0.20f); return true; }
        } else {
            if (textPixelC(kLabLIVE, 14, 10, ty0, lx, ly, 2)) { *rgb = (float3)(0.55f); return true; }
        }
        if (enableTemporal == 0 && textPixelC(kLabTOFF, 12, 190, ty0, lx, ly, 2)) {
            *rgb = (float3)(0.90f, 0.45f, 0.30f); return true;
        }
    }

    if (textPixelC(kLabCURVE, 19, 10, 182, lx, ly, 1)) { *rgb = (float3)(0.55f); return true; }
    if (lx >= 10 && lx < 346 && ly >= 194 && ly < 238) {
        const int bin = clamp((lx - 10) / 21, 0, kLumaBins - 1);
        const float gain = as_float(stats[S_GY + bin]);
        const float v = clamp((gain - 0.6f) / 1.6f, 0.0f, 1.0f);
        const bool bar = (237 - ly) < (int)(v * 43.0f + 0.5f);
        const bool ref = (237 - ly) == (int)((1.0f - 0.6f) / 1.6f * 43.0f + 0.5f);
        if (bar)      *rgb = (float3)(0.20f, 0.65f, 0.95f);
        else if (ref) *rgb = (float3)(0.42f);
        else          *rgb = (float3)(0.08f);
        return true;
    }

    if (textPixelC(kLabHIST, 31, 10, 244, lx, ly, 1)) { *rgb = (float3)(0.55f); return true; }
    if (lx >= 10 && lx < 346 && ly >= 252 && ly < 284) {
        const int bin = clamp((lx - 10) * kHistBins / 336, 0, kHistBins - 1);
        const float frac = (float)stats[H_YF + bin] / (float)max(hmax, 1u);
        const float hgt = 31.0f * sqrt(clamp(frac, 0.0f, 1.0f));
        const bool bar = (283 - ly) < (int)(hgt + 0.5f);
        if (bin == (int)medBin)  *rgb = (float3)(0.95f, 0.85f, 0.15f);
        else if (bar)            *rgb = (float3)(0.55f);
        else                     *rgb = (float3)(0.08f);
        return true;
    }

    return true;
}

// Noise EQ panel (top-right) — see nr_core.h
inline bool eqScopePixel(int x, int y, int W, int H,
                         float fineY, float fineC, float coarseY, float tyv, float tcv,
                         float eqFine, float eqMedium, float eqCoarse, float chromaBlotch,
                         int enableSpatial, float3* rgb)
{
    const int yd = H - 1 - y;
    const int s = max(1, H / 540);
    const int lw = 300, lh = 190;
    const int ox = W - 16 * s - lw * s, oy = 16 * s;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    *rgb = (float3)(0.045f, 0.045f, 0.05f);
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        *rgb = (float3)(0.35f);
        return true;
    }

    if (textPixelC(kLabTITLE, 8, 10, 8, lx, ly, 2)) { *rgb = (float3)(1.0f); return true; }
    if (enableSpatial == 0 && textPixelC(kLabOFF3, 3, 250, 8, lx, ly, 2)) {
        *rgb = (float3)(0.90f, 0.45f, 0.30f); return true;
    }
    if (textPixelC(kLabLEG, 34, 10, 172, lx, ly, 1)) { *rgb = (float3)(0.55f); return true; }

    const float fineM = fmax(fineY, coarseY);
    const float lowY = sqrt(fmax(0.0f, tyv * tyv - fineM * fineM));
    const float lowC = sqrt(fmax(0.0f, tcv * tcv - fineC * fineC));
    const float amt[4] = { clamp(eqFine, 0.0f, 3.0f) / 3.0f,
                           clamp(eqMedium, 0.0f, 1.5f) / 1.5f,
                           clamp(eqCoarse, 0.0f, 1.5f) / 1.5f,
                           clamp(chromaBlotch, 0.0f, 1.5f) / 1.5f };
    const float rawPct[4] = { clamp(eqFine, 0.0f, 3.0f),
                              clamp(eqMedium, 0.0f, 1.5f),
                              clamp(eqCoarse, 0.0f, 1.5f),
                              clamp(chromaBlotch, 0.0f, 1.5f) };
    const float meas[4] = { fineY, coarseY, lowY, lowC };

    for (int lane = 0; lane < 4; ++lane) {
        const int x0 = 10 + lane * 72;
        if (lx < x0 || lx >= x0 + 60)
            continue;

        {
            int vg[4];
            pctIntGlyphs(rawPct[lane], vg);
            if (textPixelP(vg, 4, x0 + 18, 32, lx, ly, 1)) { *rgb = (float3)(0.85f); return true; }
        }
        bool lit = false;
        if (lane == 0) lit = textPixelC(kLabFINE, 4, x0 + 18, 148, lx, ly, 1) ||
                             textPixelC(kLabPX1, 3, x0 + 21, 158, lx, ly, 1);
        if (lane == 1) lit = textPixelC(kLabMED, 6, x0 + 12, 148, lx, ly, 1) ||
                             textPixelC(kLabPX38, 5, x0 + 15, 158, lx, ly, 1);
        if (lane == 2) lit = textPixelC(kLabCRS, 6, x0 + 12, 148, lx, ly, 1) ||
                             textPixelC(kLabPX16, 5, x0 + 15, 158, lx, ly, 1);
        if (lane == 3) lit = textPixelC(kLabCOL, 5, x0 + 15, 148, lx, ly, 1) ||
                             textPixelC(kLabPX16, 5, x0 + 15, 158, lx, ly, 1);
        if (lit) { *rgb = (float3)(0.75f); return true; }

        if (ly >= 44 && ly < 144) {
            const int up = 143 - ly;
            const float nv = clamp(meas[lane] / 0.08f, 0.0f, 1.0f);
            const int markH = (int)(nv * 98.0f + 0.5f);
            const int barH = (int)(clamp(amt[lane], 0.0f, 1.0f) * 98.0f + 0.5f);
            if (up >= markH - 1 && up <= markH + 1 && meas[lane] > 1e-5f) {
                *rgb = (float3)(0.95f, 0.65f, 0.20f);
            } else if (up < barH) {
                *rgb = (up >= barH - 3) ? (float3)(0.90f) : (float3)(0.30f);
            } else {
                *rgb = (float3)(0.08f);
            }
            return true;
        }
        return true;
    }

    return true;
}

// Temporal-activity mini map (bottom-right) — see nr_core.h
inline bool motionScopePixel(int x, int y, int W, int H,
                             __global const float4* tmp, __global const float4* curr,
                             float3* rgb)
{
    const int yd = H - 1 - y;
    const int s = max(1, H / 540);
    const int mapW = 300;
    const int mapH = max(40, (mapW * H) / max(W, 1));
    const int lw = mapW + 2, lh = mapH + 18;
    const int ox = W - 16 * s - lw * s, oy = H - 16 * s - lh * s;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        *rgb = (float3)(0.35f);
        return true;
    }

    if (ly < 16) {
        *rgb = (float3)(0.045f, 0.045f, 0.05f);
        if (textPixelC(kLabMO, 33, 4, 5, lx, ly, 1)) *rgb = (float3)(0.85f);
        return true;
    }

    const int u = clamp(lx - 1, 0, mapW - 1);
    const int v = clamp(ly - 16, 0, mapH - 1);
    const int sx = clamp((u * W) / mapW, 0, W - 1);
    const int sdy = clamp((v * H) / mapH, 0, H - 1);
    const int sy2 = H - 1 - sdy;
    const float4 t = tmp[sy2 * W + sx];
    const float3 cin = rgb2ycc(curr[sy2 * W + sx].xyz);
    const float effN = fmax(1.0f, t.w);
    const float tt = clamp((effN - 1.0f) / 4.0f, 0.0f, 1.0f);
    const float3 heat = (float3)(0.90f + (0.10f - 0.90f) * tt,
                                 0.15f + (0.85f - 0.15f) * tt,
                                 0.10f + (0.20f - 0.10f) * tt);
    *rgb = cin.x * 0.45f + heat * 0.55f;
    return true;
}

// ---------------------------------------------------------------------------
// Stage 3b — v3.3 "Deep Clean": fine-NLM pre-pass at 0.6h over the temporal
// result into a second buffer; corrections clamped to noise size. The
// residual is re-measured on the output. See nr_core.h.
// ---------------------------------------------------------------------------
__kernel void DeepCleanKernel(NRParams p, int W, int H,
                              __global const float4* tmp,
                              __global const uint* stats,
                              __global float4* dst)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H)
        return;

    const int manual = (p.profileSource == 2) && (p.profileLocked == 0);
    const int locked = (p.profileLocked != 0);
    const float ryG = manual ? clamp(p.sigmaY, kSigmaMin, kSigmaMax) : as_float(stats[S_RY]);
    const float rcbG = manual ? clamp(p.sigmaC, kSigmaMin, kSigmaMax) : as_float(stats[S_RCB]);
    const float rcrG = manual ? clamp(p.sigmaC, kSigmaMin, kSigmaMax) : as_float(stats[S_RCR]);

    const int R = 2;
    const float4 tc = tmp[y * W + x];
    const int lb = clamp((int)(tc.x * kLumaBins), 0, kLumaBins - 1);
    const float gainYv = locked ? clamp(p.lockGainY[lb], 0.6f, 2.2f)
                       : manual ? 1.0f : as_float(stats[S_GY + lb]);
    const float gainCv = locked ? clamp(p.lockGainC[lb], 0.6f, 2.2f)
                       : manual ? 1.0f : as_float(stats[S_GC + lb]);
    const float sigY = clamp(ryG * gainYv, 1e-5f, 1.0f);
    // v3.3 B5: pooled-normalized chroma weight — see nr_core.h
    const float sigCb = clamp(rcbG * gainCv, 1e-5f, 1.0f);
    const float sigCr = clamp(rcrG * gainCv, 1e-5f, 1.0f);
    const float hY = 0.6f * kNlmHLuma * sigY;
    const float invHY2 = 1.0f / fmax(hY * hY, 1e-12f);
    const float invHC2 = 1.0f / fmax(0.36f * kNlmHChroma * kNlmHChroma, 1e-12f);
    const float invSCb2 = 1.0f / fmax(sigCb * sigCb, 1e-12f);
    const float invSCr2 = 1.0f / fmax(sigCr * sigCr, 1e-12f);
    const float biasY = 2.0f * sigY * sigY;
    const float biasCb = 2.0f * sigCb * sigCb;
    const float biasCr = 2.0f * sigCr * sigCr;

    float3 pPatch[9];
    {
        int i = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx, ++i)
                pPatch[i] = sampleTmp(tmp, W, H, x + dx, y + dy).xyz;
    }

    float accY = 0.0f, accCb = 0.0f, accCr = 0.0f;
    float sumWY = 0.0f, sumWC = 0.0f, wYmax = 0.0f, wCmax = 0.0f;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            if (dx == 0 && dy == 0)
                continue;
            const float3 ts = sampleTmp(tmp, W, H, x + dx, y + dy).xyz;
            float dY2 = 0.0f, dCb2 = 0.0f, dCr2 = 0.0f;
            int i = 0;
            for (int qy = -1; qy <= 1; ++qy) {
                for (int qx = -1; qx <= 1; ++qx, ++i) {
                    const float3 tq = sampleTmp(tmp, W, H, x + dx + qx, y + dy + qy).xyz;
                    const float3 e = pPatch[i] - tq;
                    dY2 += e.x * e.x;
                    dCb2 += e.y * e.y;
                    dCr2 += e.z * e.z;
                }
            }
            dY2 *= (1.0f / 9.0f);
            dCb2 *= (1.0f / 9.0f);
            dCr2 *= (1.0f / 9.0f);
            dY2 = fmax(0.0f, dY2 - biasY);
            const float dC2n = 0.5f * (fmax(0.0f, dCb2 - biasCb) * invSCb2 +
                                       fmax(0.0f, dCr2 - biasCr) * invSCr2);
            const float wY = exp(-dY2 * invHY2);
            const float wC = exp(-dC2n * invHC2) * exp(-dY2 * invHY2 * 0.25f);
            accY  += wY * ts.x;
            accCb += wC * ts.y;
            accCr += wC * ts.z;
            sumWY += wY;
            sumWC += wC;
            wYmax = fmax(wYmax, wY);
            wCmax = fmax(wCmax, wC);
        }
    }
    const float wYc = fmax(wYmax, 1e-4f);
    const float wCc = fmax(wCmax, 1e-4f);
    const float Yf  = (accY  + wYc * tc.x) / (sumWY + wYc);
    const float Cbf = (accCb + wCc * tc.y) / (sumWC + wCc);
    const float Crf = (accCr + wCc * tc.z) / (sumWC + wCc);
    dst[y * W + x] = (float4)(tc.x + clamp(Yf  - tc.x, -2.0f * sigY, 2.0f * sigY),
                              tc.y + clamp(Cbf - tc.y, -2.0f * sigCb, 2.0f * sigCb),
                              tc.z + clamp(Crf - tc.z, -2.0f * sigCr, 2.0f * sigCr),
                              tc.w);
}

// v3.3 A2: the fine-band NLM loop reads (2R+1)^2 x 10 samples per pixel with
// near-total overlap between neighbouring threads — each workgroup stages
// its tile of tmp (side + 2(R+1) square: R window + 1 px patch ring) into
// local memory as flat Y/Cb/Cr triples. Values are the same edge-clamped
// samples sampleTmp returns, so the math is untouched. The workgroup side
// comes from get_local_size (the host picks 16, falling back to 8 where the
// device caps the workgroup or local memory) — Metal/CUDA hardcode 16.
inline float3 tileAt(__local const float* tile, int tileW, int lx, int ly)
{
    const int i = (ly * tileW + lx) * 3;
    return (float3)(tile[i], tile[i + 1], tile[i + 2]);
}

// tmp is the working buffer (the Deep Clean output when that pass ran);
// tmpTrue is the TRUE temporal result for the After Temporal view and the
// motion scope. Without Deep Clean the host binds the same buffer to both.
__kernel void SpatialNLMKernel(NRParams p, int W, int H,
                               __global const float4* tmp, __global const float4* curr,
                               __global const uint* stats, __global float4* dst,
                               __global const float4* tmpTrue,
                               __local float* tile)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    // NOTE: no early out-of-bounds return here — every thread (in-frame or
    // not) must reach the cooperative tile load and its barrier below.

    const int manual = (p.profileSource == 2) && (p.profileLocked == 0);
    // v3.3 lock fast path: locked input sigmas/gains come straight from the
    // params (see loadSigmasIn); the residual pair still comes from the
    // stats buffer — the residual passes always run under a lock.
    const int locked = (p.profileLocked != 0);
    const float adjL = clamp(p.profileAdjust, 0.25f, 6.0f);
    const float sy = locked ? clamp(p.lockSY * adjL, kSigmaMin, kSigmaMax)
                   : manual ? clamp(p.sigmaY, kSigmaMin, kSigmaMax) : as_float(stats[S_SY]);
    // v3.3 B5: per-channel chroma pairs; the combined values below are the
    // pair means (bands, deband threshold, HUD)
    const float scbG = locked ? clamp(p.lockSC * adjL, kSigmaMin, kSigmaMax)
                    : manual ? clamp(p.sigmaC, kSigmaMin, kSigmaMax) : as_float(stats[S_SCB]);
    const float scrG = locked ? clamp(p.lockSCr * adjL, kSigmaMin, kSigmaMax)
                    : manual ? clamp(p.sigmaC, kSigmaMin, kSigmaMax) : as_float(stats[S_SCR]);
    const float sc = 0.5f * (scbG + scrG);
    const float tyG = locked ? clamp(p.lockTY * adjL, kSigmaMin, kSigmaMax)
                    : manual ? sy : as_float(stats[S_TY]);
    const float tcbG = locked ? clamp(p.lockTC * adjL, kSigmaMin, kSigmaMax)
                     : manual ? scbG : as_float(stats[S_TCB]);
    const float tcrG = locked ? clamp(p.lockTCr * adjL, kSigmaMin, kSigmaMax)
                     : manual ? scrG : as_float(stats[S_TCR]);
    const float tcG = 0.5f * (tcbG + tcrG);
    const float ryG = manual ? sy : as_float(stats[S_RY]);
    const float rcbG = manual ? scbG : as_float(stats[S_RCB]);
    const float rcrG = manual ? scrG : as_float(stats[S_RCR]);
    const float rcG = 0.5f * (rcbG + rcrG);
    const float enmed = manual ? 1.0f : as_float(stats[S_ENMED]);

    const float mLow  = fmin(p.master, 1.0f);
    const float mHigh = fmax(p.master, 1.0f);
    const float hBoost = pow(mHigh, 1.2f);

    const float sL = clamp(p.spatialLuma, 0.0f, 1.5f);
    const float sC = clamp(p.spatialChroma, 0.0f, 1.5f);
    // v3 Noise EQ: the fine slider scales the NLM band's blend (1 = v2.1);
    // v3.1: above 100% it also widens the similarity h
    const float eqF = clamp(p.eqFine, 0.0f, 3.0f);
    const float eqH = pow(fmax(1.0f, eqF), 0.8f);
    const float aY = (p.enableSpatial == 0) ? 0.0f : clamp(sL * mLow * eqF, 0.0f, 1.0f);
    const float aC = (p.enableSpatial == 0) ? 0.0f : clamp(sC * mLow * eqF, 0.0f, 1.0f);
    const float overL = (sL > 1.0f) ? 1.6f * pow(sL - 1.0f, 1.2f) : 0.0f;
    const float overC = (sC > 1.0f) ? 1.6f * pow(sC - 1.0f, 1.2f) : 0.0f;
    const float hMulY = (0.6f + 1.4f * pow(sL, 1.5f) + overL) * hBoost * eqH;
    const float hMulC = (0.6f + 1.4f * pow(sC, 1.5f) + overC) * hBoost * eqH;
    const float pd = clamp(p.preserveDetail, 0.0f, 1.0f);
    const int   R  = clamp(p.spatialRadius, 1, 10);
    const int   nlm = (p.spatialMode == 1);
    const int   runSpatial = (p.enableSpatial != 0) && (aY > 0.0f || aC > 0.0f);
    const float spatialSigma = fmax(1.0f, (float)R / 1.5f);
    const float invSpatial2 = 1.0f / (2.0f * spatialSigma * spatialSigma);
    // v3.1 Detail Rescue — see nr_core.h
    const float rescue = clamp(p.detailRescue, 0.0f, 1.0f);

    // v3.1: band sliders reach 150 — amount caps at 1, the extra drive
    // widens the similarity tolerances and the reach instead
    const float blotchRaw = clamp(p.chromaBlotch, 0.0f, 1.5f);
    const float medRaw    = clamp(p.eqMedium, 0.0f, 1.5f);
    const float coarseRaw = clamp(p.eqCoarse, 0.0f, 1.5f);
    const float blotch = (p.enableSpatial != 0) ? fmin(blotchRaw * mLow, 1.0f) : 0.0f;
    // v3 Noise EQ: medium band amount and coarse-band luma amount
    const float eqMed  = (p.enableSpatial != 0) ? fmin(medRaw * mLow, 1.0f) : 0.0f;
    const float coarseL = (p.enableSpatial != 0) ? fmin(coarseRaw * mLow, 1.0f) : 0.0f;
    const float medOver = fmax(1.0f, medRaw);
    const float crsOver = fmax(1.0f, fmax(blotchRaw, coarseRaw));
    const int   Rb = 2 + (int)(14.0f * fmax(blotchRaw, coarseRaw));
    const int   Rm = 3 + (int)(5.0f * medRaw);
    // Band tolerance: see nr_core.h
    const float bandRatioY = clamp(tyG / fmax(sy, 1e-6f), 1.0f, 3.0f);
    const float bandRatioC = clamp(tcG / fmax(sc, 1e-6f), 1.0f, 3.0f);

    const int refine = (p.enableRefine != 0) && (p.master > 0.0f);
    const float desat = refine ? clamp(p.shadowDesat, 0.0f, 1.0f) : 0.0f;
    const float desatRange = fmax(0.02f, p.desatRange);
    const float tex = refine ? clamp(p.lumaTexture, 0.0f, 1.0f) * mLow : 0.0f;
    // v3 deband thresholds — see nr_core.h
    const float debandAmt = refine ? clamp(p.deband, 0.0f, 1.0f) * mLow : 0.0f;
    const float dbThrY = fmax(0.010f, 1.5f * ryG);
    const float dbThrC = fmax(0.010f, 1.5f * rcG);
    const float grainAmt = refine ? clamp(p.grainAmount, 0.0f, 1.0f) * 0.06f : 0.0f;
    const float grainSize = clamp(p.grainSize, 0.5f, 6.0f);
    const float grainCh = clamp(p.grainChroma, 0.0f, 1.0f);
    const uint frame = (uint)p.frameIndex;
    // v3.2 global blend — see nr_core.h
    const float gBlend = clamp(p.globalBlend, 0.0f, 1.0f);

    // ---- v3.3 A2: cooperative tile stage (see tileAt above). runSpatial is
    // uniform across the grid, so the branch and its barrier are legal; the
    // out-of-bounds return must come only after the barrier. ----
    const int lsx = (int)get_local_size(0);
    const int lsy = (int)get_local_size(1);
    const int tileW = lsx + 2 * (R + 1);
    const int tileH = lsy + 2 * (R + 1);
    if (runSpatial) {
        const int tx0 = (int)get_group_id(0) * lsx - (R + 1);
        const int ty0 = (int)get_group_id(1) * lsy - (R + 1);
        for (int i = (int)get_local_id(1) * lsx + (int)get_local_id(0); i < tileW * tileH; i += lsx * lsy) {
            const int tyy = i / tileW, txx = i - tyy * tileW;
            const float4 v = tmp[clamp(ty0 + tyy, 0, H - 1) * W + clamp(tx0 + txx, 0, W - 1)];
            tile[i * 3 + 0] = v.x; tile[i * 3 + 1] = v.y; tile[i * 3 + 2] = v.z;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (x >= W || y >= H)
        return;
    const int ltx = (int)get_local_id(0) + R + 1;   // this pixel in tile coords
    const int lty = (int)get_local_id(1) + R + 1;

    const float4 tc = tmp[y * W + x];
    const int lb = clamp((int)(tc.x * kLumaBins), 0, kLumaBins - 1);
    const float gainYv = locked ? clamp(p.lockGainY[lb], 0.6f, 2.2f)
                       : manual ? 1.0f : as_float(stats[S_GY + lb]);
    const float gainCv = locked ? clamp(p.lockGainC[lb], 0.6f, 2.2f)
                       : manual ? 1.0f : as_float(stats[S_GC + lb]);
    const float sigY = clamp(ryG * gainYv, 1e-5f, 1.0f);
    // v3.3 B5 — see nr_core.h: fine band per channel, bands on the mean
    const float sigCb = clamp(rcbG * gainCv, 1e-5f, 1.0f);
    const float sigCr = clamp(rcrG * gainCv, 1e-5f, 1.0f);
    const float sigC = 0.5f * (sigCb + sigCr);

    float Yo = tc.x, Cbo = tc.y, Cro = tc.z;

    if (runSpatial) {
        float3 pPatch[9];
        {
            int i = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx, ++i)
                    pPatch[i] = tileAt(tile, tileW, ltx + dx, lty + dy);
        }

        float mean = 0.0f, m2 = 0.0f;
        for (int i = 0; i < 9; ++i) { mean += pPatch[i].x; m2 += pPatch[i].x * pPatch[i].x; }
        mean *= (1.0f / 9.0f);
        const float var = fmax(0.0f, m2 * (1.0f / 9.0f) - mean * mean);
        const float edginess = clamp(sqrt(fmax(var - sigY * sigY, 0.0f)) / (3.0f * sigY), 0.0f, 1.0f);

        const float hY = kNlmHLuma   * sigY * hMulY * (1.0f - pd * 0.85f * edginess);
        // v3.3 B5: pooled-normalized chroma weight — see nr_core.h
        const float mC = hMulC * (1.0f - pd * 0.50f * edginess);
        const float invHY2 = 1.0f / fmax(hY * hY, 1e-12f);
        const float invHC2 = 1.0f / fmax(kNlmHChroma * kNlmHChroma * mC * mC, 1e-12f);
        const float invSCb2 = 1.0f / fmax(sigCb * sigCb, 1e-12f);
        const float invSCr2 = 1.0f / fmax(sigCr * sigCr, 1e-12f);
        const float biasY = 2.0f * sigY * sigY;
        const float biasCb = 2.0f * sigCb * sigCb;
        const float biasCr = 2.0f * sigCr * sigCr;

        float accY = 0.0f, accCb = 0.0f, accCr = 0.0f;
        float sumWY = 0.0f, sumWC = 0.0f, wYmax = 0.0f, wCmax = 0.0f;

        for (int dy = -R; dy <= R; ++dy) {
            for (int dx = -R; dx <= R; ++dx) {
                if (dx == 0 && dy == 0)
                    continue;
                const float3 ts = tileAt(tile, tileW, ltx + dx, lty + dy);

                float dY2, dCb2, dCr2;
                if (nlm) {
                    dY2 = 0.0f; dCb2 = 0.0f; dCr2 = 0.0f;
                    int i = 0;
                    for (int qy = -1; qy <= 1; ++qy) {
                        for (int qx = -1; qx <= 1; ++qx, ++i) {
                            const float3 tq = tileAt(tile, tileW, ltx + dx + qx, lty + dy + qy);
                            const float3 e = pPatch[i] - tq;
                            dY2 += e.x * e.x;
                            dCb2 += e.y * e.y;
                            dCr2 += e.z * e.z;
                        }
                    }
                    dY2 *= (1.0f / 9.0f);
                    dCb2 *= (1.0f / 9.0f);
                    dCr2 *= (1.0f / 9.0f);
                } else {
                    const float3 e = tc.xyz - ts;
                    dY2 = e.x * e.x;
                    dCb2 = e.y * e.y;
                    dCr2 = e.z * e.z;
                }

                dY2 = fmax(0.0f, dY2 - biasY);
                const float dC2n = 0.5f * (fmax(0.0f, dCb2 - biasCb) * invSCb2 +
                                           fmax(0.0f, dCr2 - biasCr) * invSCr2);

                float wY = exp(-dY2 * invHY2);
                float wC = exp(-dC2n * invHC2) * exp(-dY2 * invHY2 * 0.25f);
                if (!nlm) {
                    const float fall = exp(-(float)(dx * dx + dy * dy) * invSpatial2);
                    wY *= fall;
                    wC *= fall;
                }

                accY  += wY * ts.x;
                accCb += wC * ts.y;
                accCr += wC * ts.z;
                sumWY += wY;
                sumWC += wC;
                wYmax = fmax(wYmax, wY);
                wCmax = fmax(wCmax, wC);
            }
        }

        const float wYc = fmax(wYmax, 1e-4f);
        const float wCc = fmax(wCmax, 1e-4f);
        const float Yf  = (accY  + wYc * tc.x) / (sumWY + wYc);
        const float Cbf = (accCb + wCc * tc.y) / (sumWC + wCc);
        const float Crf = (accCr + wCc * tc.z) / (sumWC + wCc);

        // v3.1 Detail Rescue — see nr_core.h for the coring rationale
        if (rescue > 0.0f) {
            const float kY = sigY * (2.0f + 6.0f * (1.0f - rescue));
            const float kCb = sigCb * (3.0f + 9.0f * (1.0f - rescue));
            const float kCr = sigCr * (3.0f + 9.0f * (1.0f - rescue));
            Yo  = tc.x - aY * clamp(tc.x - Yf,  -kY, kY);
            Cbo = tc.y - aC * clamp(tc.y - Cbf, -kCb, kCb);
            Cro = tc.z - aC * clamp(tc.z - Crf, -kCr, kCr);
        } else {
            Yo  = tc.x + aY * (Yf  - tc.x);
            Cbo = tc.y + aC * (Cbf - tc.y);
            Cro = tc.z + aC * (Crf - tc.z);
        }
    }

    // v3 medium band — see nr_core.h for the block-mean-domain rationale
    if (eqMed > 0.0f) {
        const float mScale = 2.6f * sigY * bandRatioY * hBoost * medOver;
        const float myDen = 1.0f / fmax(mScale, 1e-6f);
        const float mcDen = 1.0f / fmax(3.0f * sigC * bandRatioC * hBoost * medOver, 1e-6f);
        const float3 b0 = blockMeanTmp(tmp, W, H, x, y);
        float accMY = b0.x, accMB = b0.y, accMR = b0.z, sumWm = 1.0f;
        for (int d = 0; d < 8; ++d) {
            for (int ri = 1; ri <= 2; ++ri) {
                const float rr = Rm * ((float)ri / 2.0f);
                const float3 bm = blockMeanTmp(tmp, W, H,
                                               x + (int)(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                                               y + (int)(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f)));
                const float eY = (bm.x - b0.x) * myDen;
                const float eC = (0.5f * (fabs(bm.y - b0.y) + fabs(bm.z - b0.z))) * mcDen;
                const float w = exp(-(eY * eY + eC * eC));
                accMY += w * bm.x;
                accMB += w * bm.y;
                accMR += w * bm.z;
                sumWm += w;
            }
        }
        const float lim = 2.5f * mScale;
        Yo  += eqMed * clamp(accMY / sumWm - b0.x, -lim, lim);
        Cbo += eqMed * (accMB / sumWm - b0.y);
        Cro += eqMed * (accMR / sumWm - b0.z);
    }

    // coarse band — chroma path is the v2.1 blotch pass unchanged; the v3
    // luma component works on 4x4 block means (see nr_core.h)
    if (blotch > 0.0f || coarseL > 0.0f) {
        const float gyDen = 1.0f / fmax(2.0f * sigY * hBoost * crsOver, 1e-6f);
        const float gcDen = 1.0f / fmax(3.0f * sigC * hBoost * crsOver, 1e-6f);
        const float cScale = 2.2f * sigY * bandRatioY * hBoost * crsOver;
        const float glDen = 1.0f / fmax(cScale, 1e-6f);
        const int RbL = 2 + (int)(30.0f * coarseRaw);
        float3 c0 = (float3)(0.0f);
        if (coarseL > 0.0f)
            c0 = blockMean4Tmp(tmp, W, H, x, y);
        float accB = tc.y, accR = tc.z, sumW = 1.0f;
        float accL = c0.x, sumWL = 1.0f;
        for (int d = 0; d < 8; ++d) {
            for (int ri = 1; ri <= 3; ++ri) {
                const float rr = Rb * ((float)ri / 3.0f);
                const int sx = x + (int)(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f));
                const int sy2 = y + (int)(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f));
                const float3 ts = sampleTmp(tmp, W, H, sx, sy2).xyz;
                const float eY = (ts.x - tc.x) * gyDen;
                const float eC = (0.5f * (fabs(ts.y - tc.y) + fabs(ts.z - tc.z))) * gcDen;
                const float w = exp(-(eY * eY + eC * eC));
                accB += w * ts.y;
                accR += w * ts.z;
                sumW += w;
                if (coarseL > 0.0f) {
                    const float rrL = RbL * ((float)ri / 3.0f);
                    const int lx2 = x + (int)(kDirX[d] * rrL + (kDirX[d] > 0 ? 0.5f : -0.5f));
                    const int ly2 = y + (int)(kDirY[d] * rrL + (kDirY[d] > 0 ? 0.5f : -0.5f));
                    const float3 b4 = blockMean4Tmp(tmp, W, H, lx2, ly2);
                    const float eL = (b4.x - c0.x) * glDen;
                    const float eLC = (0.5f * (fabs(b4.y - c0.y) + fabs(b4.z - c0.z))) * gcDen;
                    const float wL = exp(-(eL * eL + eLC * eLC));
                    accL += wL * b4.x;
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
            Yo += coarseL * clamp(accL / sumWL - c0.x, -lim, lim);
        }
    }

    const float4 c = curr[y * W + x];
    const float3 cin = rgb2ycc(c.xyz);

    float Yr = Yo, Cbr = Cbo, Crr = Cro;
    if (refine) {
        const float sat = 1.0f - desat * (1.0f - smooth01f(Yr / desatRange));
        Cbr *= sat;
        Crr *= sat;
        Yr += tex * (cin.x - Yr);
        // v3 deband — see nr_core.h for the agreement-confidence rationale
        if (debandAmt > 0.0f) {
            const float dyDen = 1.0f / dbThrY;
            const float dcDen = 1.0f / dbThrC;
            const float3 b0 = blockMeanTmp(tmp, W, H, x, y);
            float accDY = b0.x, accDB = b0.y, accDR = b0.z, sumWd = 1.0f;
            for (int d = 0; d < 8; ++d) {
                for (int ri = 1; ri <= 3; ++ri) {
                    const float rr = 16.0f * ((float)ri / 3.0f);
                    const float3 bm = blockMeanTmp(tmp, W, H,
                                                   x + (int)(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                                                   y + (int)(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f)));
                    const float eY = (bm.x - b0.x) * dyDen;
                    const float eC = (0.5f * (fabs(bm.y - b0.y) + fabs(bm.z - b0.z))) * dcDen;
                    const float w = exp(-(eY * eY + eC * eC));
                    accDY += w * bm.x;
                    accDB += w * bm.y;
                    accDR += w * bm.z;
                    sumWd += w;
                }
            }
            const float agree = (sumWd - 1.0f) * (1.0f / 24.0f);
            const float conf = agree * agree;
            Yr  += debandAmt * conf * clamp(accDY / sumWd - Yr,  -dbThrY, dbThrY);
            Cbr += debandAmt * conf * clamp(accDB / sumWd - Cbr, -dbThrC, dbThrC);
            Crr += debandAmt * conf * clamp(accDR / sumWd - Crr, -dbThrC, dbThrC);
            const float dith = 0.7f * debandAmt / 255.0f;
            Yr += dith * 0.5f * (hashNoise((uint)x, (uint)y, frame, 3u) +
                                 hashNoise((uint)x, (uint)y, frame + 977u, 3u));
        }
        if (grainAmt > 0.0f) {
            const float yc = clamp(Yr, 0.0f, 1.0f);
            const float resp = 0.25f + 0.75f * (4.0f * yc * (1.0f - yc));
            const float gn = valueNoise((float)x, (float)y, grainSize, frame, 0u);
            Yr += grainAmt * resp * gn;
            if (grainCh > 0.0f) {
                Cbr += grainAmt * grainCh * 0.6f * resp * valueNoise((float)x, (float)y, grainSize, frame, 1u);
                Crr += grainAmt * grainCh * 0.6f * resp * valueNoise((float)x, (float)y, grainSize, frame, 2u);
            }
        }
    }

    // v3.2 global blend — result only; Noise Removed stays full-strength
    if (gBlend < 1.0f) {
        Yr  = cin.x + gBlend * (Yr  - cin.x);
        Cbr = cin.y + gBlend * (Cbr - cin.y);
        Crr = cin.z + gBlend * (Crr - cin.z);
    }

    float3 rgb = ycc2rgb((float3)(Yr, Cbr, Crr));
    const float3 dn = ycc2rgb((float3)(Yo, Cbo, Cro));
    float4 o = (float4)(rgb, c.w);

    if (p.viewMode == 1) {
        if (x < W / 2) o.xyz = c.xyz;
        if (abs(x - W / 2) <= 1) o.xyz = (float3)(1.0f);
    } else if (p.viewMode == 2) {
        o.xyz = c.xyz;
    } else if (p.viewMode == 3) {
        // the TRUE temporal result, even when Deep Clean rewrote tmp
        o.xyz = ycc2rgb(tmpTrue[y * W + x].xyz);
    } else if (p.viewMode == 4) {
        // noise removed: noise-adaptive gain + soft knee — see nr_core.h
        const float nrGain = 0.08f / clamp(sy, 0.004f, 0.08f);
        const float3 d = (c.xyz - dn) * nrGain;
        o.xyz = 0.5f + 0.5f * d / (0.5f + fabs(d));
    } else if (p.viewMode == 5) {
        // noise analysis: the result; the measurement scope is drawn by the
        // overlay pass below
        o.xyz = rgb;
    } else if (p.viewMode == 6) {
        const float effN = fmax(1.0f, tc.w);
        const float t = clamp((effN - 1.0f) / 4.0f, 0.0f, 1.0f);
        const float3 heat = (float3)(0.90f + (0.10f - 0.90f) * t,
                                     0.15f + (0.85f - 0.15f) * t,
                                     0.10f + (0.20f - 0.10f) * t);
        o.xyz = cin.x * 0.45f + heat * 0.55f;
    } else if (p.viewMode == 7) {
        float mean = 0.0f, m2 = 0.0f;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
                const float v = sampleTmp(tmp, W, H, x + dx, y + dy).x;
                mean += v; m2 += v * v;
            }
        mean *= (1.0f / 25.0f);
        const float var = fmax(0.0f, m2 * (1.0f / 25.0f) - mean * mean);
        const float sigNoise = fmax(sy * gainYv, 1e-5f);
        const float sigSignal = sqrt(fmax(var - sigNoise * sigNoise, 0.0f));
        const float snrDb = 20.0f * log10(fmax(sigSignal, 1e-6f) / sigNoise);
        const float t = clamp((snrDb + 6.0f) / 36.0f, 0.0f, 1.0f);
        float3 heat;
        if (t < 0.5f) {
            const float u = t * 2.0f;
            heat = (float3)(0.85f + (0.95f - 0.85f) * u,
                            0.10f + (0.70f - 0.10f) * u,
                            0.75f + (0.15f - 0.75f) * u);
        } else {
            const float u = (t - 0.5f) * 2.0f;
            heat = (float3)(0.95f + (0.10f - 0.95f) * u,
                            0.70f + (0.85f - 0.70f) * u,
                            0.15f + (0.20f - 0.15f) * u);
        }
        o.xyz = cin.x * 0.35f + heat * 0.65f;
    } else if (p.viewMode == 8) {
        // noise matte: normalized noise dominance in RGB+alpha
        float mean = 0.0f, m2 = 0.0f;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
                const float v = sampleTmp(tmp, W, H, x + dx, y + dy).x;
                mean += v; m2 += v * v;
            }
        mean *= (1.0f / 25.0f);
        const float var = fmax(0.0f, m2 * (1.0f / 25.0f) - mean * mean);
        const float sigNoise = fmax(sy * gainYv, 1e-5f);
        const float sigSignal = sqrt(fmax(var - sigNoise * sigNoise, 0.0f));
        const float snrDb = 20.0f * log10(fmax(sigSignal, 1e-6f) / sigNoise);
        const float m = 1.0f - clamp((snrDb + 6.0f) / 36.0f, 0.0f, 1.0f);
        o = (float4)(m, m, m, m);
    }

    // ---- v3.1 scope overlays: drawn over ANY view, never into alpha ----
    {
        const int wantHud = (p.scopeMeasure != 0) || (p.viewMode == 5);
        const int wantEq  = (p.scopeEq != 0);
        const int wantMo  = (p.scopeMotion != 0);
        if (wantHud || wantEq || wantMo) {
            float3 sco = o.xyz;
            int drew = 0;
            if (wantEq) {
                const float fineY = manual ? sy : as_float(stats[S_FINEY]);
                const float fineC = manual ? sc : as_float(stats[S_FINEC]);
                const float crsY  = manual ? sy : as_float(stats[S_CRSY]);
                if (eqScopePixel(x, y, W, H, fineY, fineC, crsY, tyG, tcG,
                                 p.eqFine, p.eqMedium, p.eqCoarse, p.chromaBlotch,
                                 p.enableSpatial, &sco))
                    drew = 1;
            }
            if (wantMo && motionScopePixel(x, y, W, H, tmpTrue, curr, &sco))
                drew = 1;
            if (wantHud) {
                if (hudPixel(x, y, W, H, sy, sc, ryG, rcG, enmed,
                             stats[S_MED], stats[S_HMAX], stats, p.enableTemporal,
                             p.profileLocked, &sco)) {
                    drew = 1;
                } else if (p.profileSource == 1) {
                    const float rHalf = 0.5f * p.regionSize * (float)min(W, H);
                    const float cx = p.regionCX * W, cyy = p.regionCY * H;
                    const float ax = fabs((float)x - cx), ay = fabs((float)y - cyy);
                    const int onEdge = (ax <= rHalf && ay <= rHalf) &&
                                       (ax >= rHalf - 2.0f || ay >= rHalf - 2.0f);
                    if (onEdge) { sco = (float3)(1.0f, 1.0f, 0.1f); drew = 1; }
                }
            }
            if (drew) o.xyz = sco;
        }
    }

    dst[y * W + x] = o;
}

)CLC";

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

namespace {

void CheckError(cl_int p_Error, const char* p_Msg)
{
    if (p_Error != CL_SUCCESS)
        fprintf(stderr, "OpenNR: %s [%d]\n", p_Msg, p_Error);
}

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

struct QueueResources
{
    cl_kernel est = nullptr;
    cl_kernel fin = nullptr;
    cl_kernel temp = nullptr;
    cl_kernel est2 = nullptr;
    cl_kernel fin2 = nullptr;
    cl_kernel deep = nullptr;
    cl_kernel expE = nullptr;   // v3.5 P1
    cl_kernel expF = nullptr;
    cl_kernel nlm = nullptr;
    cl_mem tmp = nullptr;
    cl_mem tmp2 = nullptr;   // v3.3 Deep Clean output (lazy — only when used)
    cl_mem stats = nullptr;
    int w = 0, h = 0;
    size_t nlmWG = 16;   // v3.3 A2: NLM workgroup side (16, or 8 on capped devices)
};

} // namespace

void RunOpenCLNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                 const float* const p_Srcs[7], float* p_Dst)
{
    cl_int error;
    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    static std::map<void*, QueueResources> s_resources;
    static Locker s_locker;

    cl_context clContext = NULL;
    error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
    CheckError(error, "Unable to get the context");

    QueueResources res;
    s_locker.Lock();
    {
        QueueResources& r = s_resources[p_CmdQ];
        if (!r.est)
        {
            cl_program program = clCreateProgramWithSource(clContext, 1, &kKernelSource, NULL, &error);
            CheckError(error, "Unable to create program");

            error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (error != CL_SUCCESS)
            {
                CheckError(error, "Unable to build program");
                cl_device_id deviceId = NULL;
                clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
                char log[16384] = { 0 };
                clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
                fprintf(stderr, "OpenNR OpenCL build log:\n%s\n", log);
                s_locker.Unlock();
                return;
            }

            r.est  = clCreateKernel(program, "NoiseEstKernel", &error);
            r.fin  = clCreateKernel(program, "FinalizeStatsKernel", &error);
            r.temp = clCreateKernel(program, "TemporalKernel", &error);
            r.est2 = clCreateKernel(program, "ResidualEstKernel", &error);
            r.fin2 = clCreateKernel(program, "FinalizeResidualKernel", &error);
            r.deep = clCreateKernel(program, "DeepCleanKernel", &error);
            r.expE = clCreateKernel(program, "ExposureEstKernel", &error);
            r.expF = clCreateKernel(program, "FinalizeExposureKernel", &error);
            r.nlm  = clCreateKernel(program, "SpatialNLMKernel", &error);
            CheckError(error, "Unable to create kernels");

            // v3.3 A2: the tiled NLM kernel wants 16x16 workgroups; fall
            // back to 8x8 where the device caps the kernel's workgroup size
            // or has a small local memory (tile at wg 16, R 10 is 17.3 KB).
            r.nlmWG = 16;
            cl_device_id devId = NULL;
            clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &devId, NULL);
            if (devId && r.nlm)
            {
                size_t kwg = 0;
                cl_ulong localMem = 0;
                clGetKernelWorkGroupInfo(r.nlm, devId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kwg, NULL);
                clGetDeviceInfo(devId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMem, NULL);
                if (kwg < 256 || localMem < 20480)
                    r.nlmWG = 8;
            }
        }

        if (!r.tmp || r.w != p_Width || r.h != p_Height)
        {
            if (r.tmp)   clReleaseMemObject(r.tmp);
            if (r.stats) clReleaseMemObject(r.stats);
            if (r.tmp2)  { clReleaseMemObject(r.tmp2); r.tmp2 = nullptr; }   // re-created lazily
            r.tmp = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                                   static_cast<size_t>(p_Width) * p_Height * 4 * sizeof(float), NULL, &error);
            r.stats = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                                     NR_STATS_UINTS * sizeof(cl_uint), NULL, &error);
            CheckError(error, "Unable to create buffers");
            r.w = p_Width;
            r.h = p_Height;
        }
        // v3.3 Deep Clean writes a second frame-sized buffer; allocate it
        // only once the pass is actually used
        if ((p_Params.deepClean != 0) && (p_Params.enableSpatial != 0) &&
            (p_Params.master > 0.0f) && !r.tmp2)
        {
            r.tmp2 = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                                    static_cast<size_t>(p_Width) * p_Height * 4 * sizeof(float), NULL, &error);
            CheckError(error, "Unable to create deep-clean buffer");
        }
        res = r;
    }
    s_locker.Unlock();

    if (!res.est || !res.tmp)
        return;

    NRParams params = p_Params;
    const float* partner = p_Srcs[3];
    if (p_Srcs[2] != p_Srcs[3])      partner = p_Srcs[2];
    else if (p_Srcs[4] != p_Srcs[3]) partner = p_Srcs[4];
    params.hasTemporalDiff = (partner != p_Srcs[3]) ? 1 : 0;

    int W = p_Width, H = p_Height;
    // v3.3 lock fast path: a locked profile still runs input estimation when
    // a scope/analysis view is showing the live measurement; otherwise the
    // estimation is output-inert — the kernels compute the locked values
    // straight from the params — and the NoiseEst/FinalizeStats dispatches
    // are skipped entirely. The residual passes always run under a lock (the
    // residual depends on what the temporal stage removed from THIS frame).
    const bool wantLiveStats = (params.scopeMeasure != 0) || (params.scopeEq != 0) ||
                               (params.viewMode == 5);
    const bool residualLive = (params.profileSource != 2) || (params.profileLocked != 0);
    const bool inputLive = residualLive && ((params.profileLocked == 0) || wantLiveStats);
    size_t global[2]  = { static_cast<size_t>(W), static_cast<size_t>(H) };
    size_t globalH[2] = { static_cast<size_t>((W + 1) / 2), static_cast<size_t>((H + 1) / 2) };
    size_t one[2] = { 1, 1 };

    if (residualLive)
    {
        const cl_uint zero = 0;
        error = clEnqueueFillBuffer(cmdQ, res.stats, &zero, sizeof(cl_uint), 0,
                                    NR_STATS_UINTS * sizeof(cl_uint), 0, NULL, NULL);
        CheckError(error, "Unable to zero stats");
    }
    if (inputLive)
    {
        int c = 0;
        error  = clSetKernelArg(res.est, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.est, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.est, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.est, c++, sizeof(cl_mem), &p_Srcs[3]);
        error |= clSetKernelArg(res.est, c++, sizeof(cl_mem), &partner);
        error |= clSetKernelArg(res.est, c++, sizeof(cl_mem), &res.stats);
        CheckError(error, "est args");
        error = clEnqueueNDRangeKernel(cmdQ, res.est, 2, NULL, globalH, NULL, 0, NULL, NULL);
        CheckError(error, "est enqueue");

        c = 0;
        error  = clSetKernelArg(res.fin, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.fin, c++, sizeof(cl_mem), &res.stats);
        CheckError(error, "fin args");
        error = clEnqueueNDRangeKernel(cmdQ, res.fin, 2, NULL, one, NULL, 0, NULL, NULL);
        CheckError(error, "fin enqueue");
    }

    // v3.5 P1: per-neighbour exposure offsets — see nr_core.h. The exposure
    // block zeroes whenever the temporal stack runs (manual profiles skip
    // every other stats write).
    const int reachHost = (params.enableTemporal == 0) ? 0
                        : ((params.temporalFrames >= 7) ? 3 : (params.temporalFrames >= 5) ? 2 : 1);
    if (reachHost >= 1)
    {
        if (!residualLive)
        {
            const cl_uint zero = 0;
            error = clEnqueueFillBuffer(cmdQ, res.stats, &zero, sizeof(cl_uint),
                                        NR_STATS_HIST_EXP * sizeof(cl_uint),
                                        (NR_STATS_UINTS - NR_STATS_HIST_EXP) * sizeof(cl_uint),
                                        0, NULL, NULL);
            CheckError(error, "Unable to zero exposure stats");
        }
        size_t globalQ[2] = { static_cast<size_t>((W + 3) / 4), static_cast<size_t>((H + 3) / 4) };
        {
            int c = 0;
            error  = clSetKernelArg(res.expE, c++, sizeof(NRParams), &params);
            error |= clSetKernelArg(res.expE, c++, sizeof(int), &W);
            error |= clSetKernelArg(res.expE, c++, sizeof(int), &H);
            for (int i = 0; i < 7; ++i)
                error |= clSetKernelArg(res.expE, c++, sizeof(cl_mem), &p_Srcs[i]);
            error |= clSetKernelArg(res.expE, c++, sizeof(cl_mem), &res.stats);
            CheckError(error, "expE args");
            error = clEnqueueNDRangeKernel(cmdQ, res.expE, 2, NULL, globalQ, NULL, 0, NULL, NULL);
            CheckError(error, "expE enqueue");
        }
        int c = 0;
        error = clSetKernelArg(res.expF, c++, sizeof(cl_mem), &res.stats);
        CheckError(error, "expF args");
        error = clEnqueueNDRangeKernel(cmdQ, res.expF, 2, NULL, one, NULL, 0, NULL, NULL);
        CheckError(error, "expF enqueue");
    }

    {
        int c = 0;
        error  = clSetKernelArg(res.temp, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.temp, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.temp, c++, sizeof(int), &H);
        for (int i = 0; i < 7; ++i)
            error |= clSetKernelArg(res.temp, c++, sizeof(cl_mem), &p_Srcs[i]);
        error |= clSetKernelArg(res.temp, c++, sizeof(cl_mem), &res.stats);
        error |= clSetKernelArg(res.temp, c++, sizeof(cl_mem), &res.tmp);
        CheckError(error, "temporal args");
        error = clEnqueueNDRangeKernel(cmdQ, res.temp, 2, NULL, global, NULL, 0, NULL, NULL);
        CheckError(error, "temporal enqueue");
    }

    if (residualLive)
    {
        int c = 0;
        error  = clSetKernelArg(res.est2, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.est2, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.est2, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.est2, c++, sizeof(cl_mem), &res.tmp);
        error |= clSetKernelArg(res.est2, c++, sizeof(cl_mem), &res.stats);
        CheckError(error, "est2 args");
        error = clEnqueueNDRangeKernel(cmdQ, res.est2, 2, NULL, globalH, NULL, 0, NULL, NULL);
        CheckError(error, "est2 enqueue");

        c = 0;
        error  = clSetKernelArg(res.fin2, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.fin2, c++, sizeof(cl_mem), &res.stats);
        CheckError(error, "fin2 args");
        error = clEnqueueNDRangeKernel(cmdQ, res.fin2, 2, NULL, one, NULL, 0, NULL, NULL);
        CheckError(error, "fin2 enqueue");
    }

    // v3.3 Deep Clean: fine pre-pass on the PASS-1 residual, then re-zero
    // the residual histogram slots (NOT the sigma/gain slots between them)
    // and re-measure on the cleaned buffer (see nr_core.h).
    const bool deep = (params.deepClean != 0) && (params.enableSpatial != 0) &&
                      (params.master > 0.0f) && res.tmp2;
    cl_mem working = deep ? res.tmp2 : res.tmp;
    if (deep)
    {
        int c = 0;
        error  = clSetKernelArg(res.deep, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.deep, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.deep, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.deep, c++, sizeof(cl_mem), &res.tmp);
        error |= clSetKernelArg(res.deep, c++, sizeof(cl_mem), &res.stats);
        error |= clSetKernelArg(res.deep, c++, sizeof(cl_mem), &res.tmp2);
        CheckError(error, "deep args");
        error = clEnqueueNDRangeKernel(cmdQ, res.deep, 2, NULL, global, NULL, 0, NULL, NULL);
        CheckError(error, "deep enqueue");

        if (residualLive)
        {
            // v3.3 B5 layout: ALL residual histograms are contiguous before
            // the sigma slots — one range zeroes them all
            const cl_uint zero = 0;
            error = clEnqueueFillBuffer(cmdQ, res.stats, &zero, sizeof(cl_uint),
                                        NR_STATS_HIST_YR * sizeof(cl_uint),
                                        (NR_STATS_SIGMA_SY - NR_STATS_HIST_YR) * sizeof(cl_uint),
                                        0, NULL, NULL);
            CheckError(error, "Unable to re-zero residual stats");

            int c2 = 0;
            error  = clSetKernelArg(res.est2, c2++, sizeof(NRParams), &params);
            error |= clSetKernelArg(res.est2, c2++, sizeof(int), &W);
            error |= clSetKernelArg(res.est2, c2++, sizeof(int), &H);
            error |= clSetKernelArg(res.est2, c2++, sizeof(cl_mem), &res.tmp2);
            error |= clSetKernelArg(res.est2, c2++, sizeof(cl_mem), &res.stats);
            CheckError(error, "est2b args");
            error = clEnqueueNDRangeKernel(cmdQ, res.est2, 2, NULL, globalH, NULL, 0, NULL, NULL);
            CheckError(error, "est2b enqueue");

            c2 = 0;
            error  = clSetKernelArg(res.fin2, c2++, sizeof(NRParams), &params);
            error |= clSetKernelArg(res.fin2, c2++, sizeof(cl_mem), &res.stats);
            CheckError(error, "fin2b args");
            error = clEnqueueNDRangeKernel(cmdQ, res.fin2, 2, NULL, one, NULL, 0, NULL, NULL);
            CheckError(error, "fin2b enqueue");
        }
    }

    {
        int c = 0;
        error  = clSetKernelArg(res.nlm, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.nlm, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.nlm, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.nlm, c++, sizeof(cl_mem), &working);
        error |= clSetKernelArg(res.nlm, c++, sizeof(cl_mem), &p_Srcs[3]);
        error |= clSetKernelArg(res.nlm, c++, sizeof(cl_mem), &res.stats);
        error |= clSetKernelArg(res.nlm, c++, sizeof(cl_mem), &p_Dst);
        error |= clSetKernelArg(res.nlm, c++, sizeof(cl_mem), &res.tmp);   // the TRUE temporal result
        // v3.3 A2: local-memory tile for the fine-band loop — sized for the
        // actual radius and the chosen workgroup (same formula as the kernel)
        const int Rsp = params.spatialRadius < 1 ? 1 : (params.spatialRadius > 10 ? 10 : params.spatialRadius);
        const size_t wg = res.nlmWG;
        const size_t tileW = wg + 2 * (static_cast<size_t>(Rsp) + 1);
        error |= clSetKernelArg(res.nlm, c++, 3 * sizeof(float) * tileW * tileW, NULL);
        CheckError(error, "nlm args");
        // the tile ties threads to workgroups, so the workgroup size must be
        // explicit and the global size rounded up (the kernel guards overrun)
        size_t localNlm[2]  = { wg, wg };
        size_t globalNlm[2] = { (static_cast<size_t>(W) + wg - 1) / wg * wg,
                                (static_cast<size_t>(H) + wg - 1) / wg * wg };
        error = clEnqueueNDRangeKernel(cmdQ, res.nlm, 2, NULL, globalNlm, localNlm, 0, NULL, NULL);
        CheckError(error, "nlm enqueue");
    }
}
