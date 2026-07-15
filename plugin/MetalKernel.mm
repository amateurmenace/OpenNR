// OpenNR — Metal implementation of the denoising pipeline (macOS), v2.
// Line-by-line port of plugin/nr_core.h; keep the two in sync.

#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>
#include <cstdio>

#include "NRParams.h"

static const char* kKernelSource = R"MSL(

#include <metal_stdlib>
using namespace metal;

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

constant float kMedianCal   = 0.247100f;   // 1 / (6 * 0.674490)
constant float kQ35Cal      = 0.367299f;   // 1 / (6 * 0.453762)
constant float kMedianCalT  = 1.048360f;   // 1 / (0.674490 * sqrt(2))
constant float kQ20CalT     = 2.791278f;   // 1 / (0.253347 * sqrt(2))
constant float kAbsDiffBias = 1.128379f;
constant float kNlmHLuma    = 1.15f;
constant float kNlmHChroma  = 2.20f;
constant int   kHistBins    = 256;
constant float kHistScaleY  = 512.0f;
constant float kHistScaleC  = 1024.0f;
constant int   kLumaBins    = 16;
constant int   kLumaSub     = 64;
constant float kLumaSubScaleY = 128.0f;
constant float kLumaSubScaleC = 256.0f;
constant float kSigmaMin    = 1e-4f;
constant float kSigmaMax    = 0.25f;
// v3.5 P1 exposure match — see nr_core.h
constant int   kExpBins  = 128;
constant float kExpScale = 256.0f;
constant float kExpDead  = 0.006f;

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

inline float smooth01(float t)
{
    t = clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float3 rgb2ycc(float3 c)
{
    const float y = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    return float3(y, (c.z - y) / 1.8556f, (c.x - y) / 1.5748f);
}

inline float3 ycc2rgb(float3 v)
{
    const float r = v.x + 1.5748f * v.z;
    const float b = v.x + 1.8556f * v.y;
    const float g = (v.x - 0.2126f * r - 0.0722f * b) / 0.7152f;
    return float3(r, g, b);
}

inline float3 sampleYCC(const device float4* img, int W, int H, int x, int y)
{
    x = clamp(x, 0, W - 1);
    y = clamp(y, 0, H - 1);
    return rgb2ycc(img[y * W + x].xyz);
}

inline float3 blockMeanYCC(const device float4* img, int W, int H, int x, int y)
{
    float3 acc = float3(0.0f);
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
            acc += sampleYCC(img, W, H, x + dx, y + dy);
    return acc * 0.25f;
}

inline float4 sampleTmp(const device float4* tmp, int W, int H, int x, int y)
{
    x = clamp(x, 0, W - 1);
    y = clamp(y, 0, H - 1);
    return tmp[y * W + x];
}

inline float med3(float a, float b, float c)
{
    return max(min(a, b), min(max(a, b), c));
}

inline void sort2(thread float& a, thread float& b)
{
    const float t = min(a, b);
    b = max(a, b);
    a = t;
}

// median of 9 via the standard 19-exchange network (Smith 1996)
inline float med9(thread const float* v)
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
inline float3 blockMeanTmp(const device float4* tmp, int W, int H, int x, int y)
{
    float3 acc = float3(0.0f);
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
            acc += sampleTmp(tmp, W, H, x + dx, y + dy).xyz;
    return acc * 0.25f;
}

// 4x4 block mean of tmp, centred-ish on (x,y)
inline float3 blockMean4Tmp(const device float4* tmp, int W, int H, int x, int y)
{
    float3 acc = float3(0.0f);
    for (int dy = -1; dy < 3; ++dy)
        for (int dx = -1; dx < 3; ++dx)
            acc += sampleTmp(tmp, W, H, x + dx, y + dy).xyz;
    return acc * 0.0625f;
}

// v3.3 B1 hierarchical shift search offsets — see nr_core.h for the grid
// design, margins and the drift-bias rationale
constant int kCoarseX[16] = { 4, -4, 0,  0, 4, -4,  4, -4, 8, -8, 0,  0, 8, -8,  8, -8 };
constant int kCoarseY[16] = { 0,  0, 4, -4, 4,  4, -4, -4, 0,  0, 8, -8, 8,  8, -8, -8 };
constant int kRefX[8] = { 1, -1, 0,  0, 1, -1,  1, -1 };
constant int kRefY[8] = { 0,  0, 1, -1, 1,  1, -1, -1 };

// Mean |3x3 patch difference| of neighbour frame f shifted by (ox,oy) against
// the current frame's patch; also returns the SIGNED mean luma difference
// (v3.2 Ghost Guard) and the shifted centre sample.
// expOff (v3.5 P1) — see nr_core.h: the neighbour's global exposure offset
inline void patchDiff(const device float4* f, int W, int H, int x, int y,
                      int ox, int oy, float expOff, thread const float3* c9,
                      thread float& dY, thread float& dCb, thread float& dCr,
                      thread float& sdY, thread float3& fc)
{
    dY = 0.0f; dCb = 0.0f; dCr = 0.0f; sdY = 0.0f; fc = float3(0.0f);
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx, ++i) {
            float3 v = sampleYCC(f, W, H, x + ox + dx, y + oy + dy);
            v.x -= expOff;
            if (i == 4) fc = v;
            dY += fabs(v.x - c9[i].x);
            sdY += (v.x - c9[i].x);
            // v3.3 B5: per-channel chroma diffs — see nr_core.h
            dCb += fabs(v.y - c9[i].y);
            dCr += fabs(v.z - c9[i].z);
        }
    }
    dY *= (1.0f / 9.0f);
    dCb *= (1.0f / 9.0f);
    dCr *= (1.0f / 9.0f);
    sdY *= (1.0f / 9.0f);
}

// v3.3 B1: coarse matching score for the hierarchical shift search — mean
// |2x2-block-mean luma difference| of a 3x3 stride-2 block patch (6x6 px
// support); see nr_core.h.
inline float coarseDiff(const device float4* f, int W, int H, int x, int y,
                        int ox, int oy, float expOff, thread const float* cb9)
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
    return (float(h & 0xFFFFFFu) / 16777215.0f) * 2.0f - 1.0f;
}

inline float valueNoise(float x, float y, float size, uint f, uint ch)
{
    const float gx = x / size, gy = y / size;
    const int ix = int(floor(gx)), iy = int(floor(gy));
    float fx = gx - ix, fy = gy - iy;
    fx = fx * fx * fx * (fx * (fx * 6.0f - 15.0f) + 10.0f);
    fy = fy * fy * fy * (fy * (fy * 6.0f - 15.0f) + 10.0f);
    const float n00 = hashNoise(uint(ix),     uint(iy),     f, ch);
    const float n10 = hashNoise(uint(ix + 1), uint(iy),     f, ch);
    const float n01 = hashNoise(uint(ix),     uint(iy + 1), f, ch);
    const float n11 = hashNoise(uint(ix + 1), uint(iy + 1), f, ch);
    return (n00 + (n10 - n00) * fx) + ((n01 + (n11 - n01) * fx) - (n00 + (n10 - n00) * fx)) * fy;
}

// ---------------------------------------------------------------------------
// Stage 1a — input histograms
// ---------------------------------------------------------------------------
kernel void NoiseEstKernel(constant NRParams& p  [[buffer(0)]],
                           constant int& W       [[buffer(1)]],
                           constant int& H       [[buffer(2)]],
                           const device float4* curr [[buffer(3)]],
                           const device float4* partner [[buffer(4)]],
                           device atomic_uint* stats [[buffer(5)]],
                           uint2 id [[thread_position_in_grid]])
{
    const int x = 1 + int(id.x) * 2;
    const int y = 1 + int(id.y) * 2;
    if (x >= W - 1 || y >= H - 1)
        return;

    if (p.profileSource == 1) {
        const float rHalf = 0.5f * p.regionSize * float(min(W, H));
        const float cx = p.regionCX * W, cy = p.regionCY * H;
        const int x0 = clamp(int(cx - rHalf), 1, W - 1);
        const int x1 = clamp(int(cx + rHalf), 1, W - 1);
        const int y0 = clamp(int(cy - rHalf), 1, H - 1);
        const int y1 = clamp(int(cy + rHalf), 1, H - 1);
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

    // v3.1: exactly-flat samples (letterbox bars, crushed blacks, synthetic
    // graphics) carry no noise evidence — skip the sample entirely
    if (lapY == 0.0f && lapCb == 0.0f && lapCr == 0.0f)
        return;

    atomic_fetch_add_explicit(&stats[H_YF + clamp(int(fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[H_CFB + clamp(int(fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[H_CFR + clamp(int(fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);

    const int lb = clamp(int(Y[4] * kLumaBins), 0, kLumaBins - 1);
    atomic_fetch_add_explicit(&stats[L_Y + lb * kLumaSub + clamp(int(fabs(lapY)  * kLumaSubScaleY), 0, kLumaSub - 1)], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[L_C + lb * kLumaSub + clamp(int(fabs(lapCb) * kLumaSubScaleC), 0, kLumaSub - 1)], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[L_C + lb * kLumaSub + clamp(int(fabs(lapCr) * kLumaSubScaleC), 0, kLumaSub - 1)], 1u, memory_order_relaxed);

    if (p.hasTemporalDiff != 0) {
        const float3 pv = sampleYCC(partner, W, H, x, y);
        atomic_fetch_add_explicit(&stats[H_YT + clamp(int(fabs(pv.x - Y[4])  * kHistScaleY), 0, kHistBins - 1)], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[H_CTB + clamp(int(fabs(pv.y - Cb[4]) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[H_CTR + clamp(int(fabs(pv.z - Cr[4]) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);
    }

    if ((id.x & 1) == 0 && (id.y & 1) == 0) {
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
            atomic_fetch_add_explicit(&stats[H_Y2 + clamp(int(fabs(lY)  * kHistScaleY), 0, kHistBins - 1)], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[H_C2B + clamp(int(fabs(lCb) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[H_C2R + clamp(int(fabs(lCr) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 1b — quantiles -> input sigmas + brightness gains (single thread)
// ---------------------------------------------------------------------------
struct QuantRes { float value; uint bin; };

static QuantRes histQuantile(device uint* stats, int base, int n, ulong total, float scale,
                             ulong num, ulong den)
{
    ulong cum = 0;
    const ulong target = (total * num + den - 1) / den;
    for (int b = 0; b < n; ++b) {
        cum += stats[base + b];
        if (cum >= target)
            return { (float(b) + 0.5f) / scale, uint(b) };
    }
    return { (float(n) - 0.5f) / scale, uint(n - 1) };
}

kernel void FinalizeStatsKernel(constant NRParams& p [[buffer(0)]],
                                device uint* stats   [[buffer(1)]],
                                uint tid [[thread_position_in_grid]])
{
    if (tid != 0)
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

        const float lapSY = max(syFine, 0.9f * syCoarse);
        const float lapSCb = max(scbFine, 0.9f * scbCoarse);
        const float lapSCr = max(scrFine, 0.9f * scrCoarse);

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
        sy = clamp(max(lapSY, 0.85f * ty) * adj, kSigmaMin, kSigmaMax);
        scb = clamp(max(lapSCb, 0.85f * tcb) * adj, kSigmaMin, kSigmaMax);
        scr = clamp(max(lapSCr, 0.85f * tcr) * adj, kSigmaMin, kSigmaMax);
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
                const float w = float(cy) / (float(cy) + 2000.0f);
                gy[b] = clamp(1.0f + w * (sb / q35RefY - 1.0f), 0.6f, 2.2f);
            }
            if (cc >= 200 && q35RefC > 1e-6f) {
                const float sb = histQuantile(stats, L_C + b * kLumaSub, kLumaSub, cc, kLumaSubScaleC, 7, 20).value * kQ35Cal;
                const float w = float(cc) / (float(cc) + 4000.0f);
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

    stats[S_SY] = as_type<uint>(sy);
    stats[S_SCB] = as_type<uint>(scb);
    stats[S_SCR] = as_type<uint>(scr);
    stats[S_TY] = as_type<uint>(ty);
    stats[S_TCB] = as_type<uint>(tcb);
    stats[S_TCR] = as_type<uint>(tcr);
    stats[S_MED] = medBin;
    stats[S_HMAX] = hmax;
    // v3.1: raw per-band estimates for the EQ scope (never lock-overridden —
    // the scope shows what is MEASURED; the lock governs what is USED)
    stats[S_FINEY] = as_type<uint>(fineYv);
    stats[S_FINEC] = as_type<uint>(fineCv);
    stats[S_CRSY]  = as_type<uint>(coarseYv);
    for (int b = 0; b < kLumaBins; ++b) {
        const int b0 = clamp(b - 1, 0, kLumaBins - 1);
        const int b1 = clamp(b + 1, 0, kLumaBins - 1);
        stats[S_GY + b] = as_type<uint>(p.profileLocked != 0 ? clamp(p.lockGainY[b], 0.6f, 2.2f)
                                        : 0.25f * gy[b0] + 0.5f * gy[b] + 0.25f * gy[b1]);
        stats[S_GC + b] = as_type<uint>(p.profileLocked != 0 ? clamp(p.lockGainC[b], 0.6f, 2.2f)
                                        : 0.25f * gc[b0] + 0.5f * gc[b] + 0.25f * gc[b1]);
    }
}

// ---------------------------------------------------------------------------
// Stage 2 — motion-adaptive temporal merge (hard-knee gate)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// v3.5 P1 — per-neighbour exposure offset: histogram of signed luma diffs on
// a stride-4 grid (one dispatch per neighbour), then a single-thread median
// with the deadzone. See nr_core.h.
// ---------------------------------------------------------------------------
kernel void ExposureEstKernel(constant NRParams& p [[buffer(0)]],
                              constant int& W [[buffer(1)]],
                              constant int& H [[buffer(2)]],
                              const device float4* f0 [[buffer(3)]],
                              const device float4* f1 [[buffer(4)]],
                              const device float4* f2 [[buffer(5)]],
                              const device float4* f3 [[buffer(6)]],
                              const device float4* f4 [[buffer(7)]],
                              const device float4* f5 [[buffer(8)]],
                              const device float4* f6 [[buffer(9)]],
                              device atomic_uint* stats [[buffer(10)]],
                              uint2 id [[thread_position_in_grid]])
{
    const int x = 1 + int(id.x) * 4;
    const int y = 1 + int(id.y) * 4;
    if (x >= W - 1 || y >= H - 1)
        return;
    const int reach = (p.enableTemporal == 0) ? 0 : ((p.temporalFrames >= 7) ? 3 : (p.temporalFrames >= 5) ? 2 : 1);
    const device float4* frames[7] = { f0, f1, f2, f3, f4, f5, f6 };
    const float cyv = sampleYCC(f3, W, H, x, y).x;
    for (int k = 3 - reach; k <= 3 + reach; ++k) {
        if (k == 3)
            continue;
        const float nyv = sampleYCC(frames[k], W, H, x, y).x;
        atomic_fetch_add_explicit(&stats[H_EXP + (k < 3 ? k : k - 1) * kExpBins +
            clamp(int((nyv - cyv + 0.25f) * kExpScale), 0, kExpBins - 1)], 1u, memory_order_relaxed);
    }
}

kernel void FinalizeExposureKernel(device uint* stats [[buffer(0)]],
                                   uint tid [[thread_position_in_grid]])
{
    if (tid != 0)
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
            const float v = (float(mbin) + 0.5f) / kExpScale - 0.25f;
            if (fabs(v) >= kExpDead)
                o = v;
        }
        stats[S_EXPOFF + s2] = as_type<uint>(o);
    }
}

inline void loadSigmasIn(constant NRParams& p, const device uint* stats,
                         thread float& sy, thread float& scb, thread float& scr,
                         thread float& ty, thread float& tcb, thread float& tcr)
{
    // v3.3 lock fast path: locked sigmas are a pure function of the params —
    // computed here with exactly FinalizeStats' arithmetic (bit-identical),
    // so the host can skip the NoiseEst/FinalizeStats dispatches when no
    // scope/analysis view is showing the live measurement.
    if (p.profileLocked != 0) {
        const float adjL = clamp(p.profileAdjust, 0.25f, 6.0f);
        sy = clamp(p.lockSY * adjL, kSigmaMin, kSigmaMax);
        scb = clamp(p.lockSC * adjL, kSigmaMin, kSigmaMax);
        scr = clamp(p.lockSCr * adjL, kSigmaMin, kSigmaMax);
        ty = clamp(p.lockTY * adjL, kSigmaMin, kSigmaMax);
        tcb = clamp(p.lockTC * adjL, kSigmaMin, kSigmaMax);
        tcr = clamp(p.lockTCr * adjL, kSigmaMin, kSigmaMax);
    } else if (p.profileSource != 2) {
        sy = as_type<float>(stats[S_SY]);
        scb = as_type<float>(stats[S_SCB]);
        scr = as_type<float>(stats[S_SCR]);
        ty = as_type<float>(stats[S_TY]);
        tcb = as_type<float>(stats[S_TCB]);
        tcr = as_type<float>(stats[S_TCR]);
    } else {
        sy = ty = clamp(p.sigmaY, kSigmaMin, kSigmaMax);
        scb = scr = tcb = tcr = clamp(p.sigmaC, kSigmaMin, kSigmaMax);
    }
}

kernel void TemporalKernel(constant NRParams& p [[buffer(0)]],
                           constant int& W [[buffer(1)]],
                           constant int& H [[buffer(2)]],
                           const device float4* f0 [[buffer(3)]],
                           const device float4* f1 [[buffer(4)]],
                           const device float4* f2 [[buffer(5)]],
                           const device float4* f3 [[buffer(6)]],
                           const device float4* f4 [[buffer(7)]],
                           const device float4* f5 [[buffer(8)]],
                           const device float4* f6 [[buffer(9)]],
                           const device uint* stats [[buffer(10)]],
                           device float4* tmp [[buffer(11)]],
                           uint2 id [[thread_position_in_grid]])
{
    const int x = int(id.x), y = int(id.y);
    if (x >= W || y >= H)
        return;

    float sigSY, sigSCb, sigSCr, sigTY, sigTCb, sigTCr;
    loadSigmasIn(p, stats, sigSY, sigSCb, sigSCr, sigTY, sigTCb, sigTCr);
    const float sigTC = 0.5f * (sigTCb + sigTCr);   // pair mean (zapper)
    const float mLow  = min(p.master, 1.0f);
    const float mHigh = max(p.master, 1.0f);
    // v3.1: sliders reach 125 — a matching neighbour may outweigh the centre
    const float tL = clamp(p.temporalLuma   * mLow, 0.0f, 1.25f);
    const float tC = clamp(p.temporalChroma * mLow, 0.0f, 1.25f);
    const float thrMul = 0.4f + 2.6f * clamp(p.motionThresh, 0.0f, 1.5f)
                       + 0.8f * (mHigh - 1.0f);
    // v3.3 B2: the stack grows to 7 frames (reach 3) for static heavy noise
    const int reach = (p.enableTemporal == 0) ? 0 : ((p.temporalFrames >= 7) ? 3 : (p.temporalFrames >= 5) ? 2 : 1);
    const float loY = kAbsDiffBias * sigTY, hiY = loY + thrMul * sigTY;
    // v3.3 B5: each chroma channel gets its own knee
    const float loCb = kAbsDiffBias * sigTCb, hiCb = loCb + thrMul * sigTCb;
    const float loCr = kAbsDiffBias * sigTCr, hiCr = loCr + thrMul * sigTCr;
    const float invSpanY = 1.0f / (hiY - loY);
    const float invSpanCb = 1.0f / (hiCb - loCb);
    const float invSpanCr = 1.0f / (hiCr - loCr);
    // v3 shift search engages only once the unshifted match is well into the
    // gate — high enough that pure noise almost never reaches it, which
    // also keeps GPU warps convergent on static footage (see nr_core.h).
    const bool  track = (p.motionTracking != 0);
    const float searchThresh = loY + 0.75f * (hiY - loY);
    // v3.2 Ghost Guard — see nr_core.h for the signed-mean rationale
    const bool  guard = (p.ghostGuard != 0);
    const float loS = kAbsDiffBias * sigTY;
    const float invSpanS = 1.0f / (0.5f * thrMul * sigTY);
    const bool  zap = (reach >= 1) && (p.fireflyRemoval != 0) &&
                      (p.master > 0.0f) && (tL > 0.0f || tC > 0.0f);
    const float zapY = 6.0f * sigTY;
    const float zapC = 6.0f * sigTC;

    const device float4* frames[7] = { f0, f1, f2, f3, f4, f5, f6 };

    float3 c9[9];
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i)
            c9[i] = sampleYCC(f3, W, H, x + dx, y + dy);

    // v3 firefly zapper — see nr_core.h for the three-test rationale
    if (zap) {
        float3 pv = sampleYCC(f2, W, H, x, y);
        float3 nv = sampleYCC(f4, W, H, x, y);
        pv.x -= as_type<float>(stats[S_EXPOFF + 2]);   // v3.5 P1: cross-frame
        nv.x -= as_type<float>(stats[S_EXPOFF + 3]);   // tests exposure-matched
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
                c9[4] = float3(mY, mCb, mCr);
            }
        }
    }

    // v3.4: per-brightness gate calibration — every temporal threshold scales
    // with the centre pixel's gain-curve gain, exactly like the spatial stage
    // always did (see nr_core.h for the shadows/highlights rationale; the
    // 6-sigma zapper stays unscaled on purpose). Gains follow the same
    // locked/manual/live triple as the spatial pass: the lock fast path skips
    // FinalizeStats, so a locked profile's gains must come from the params.
    const bool manualG = (p.profileSource == 2) && (p.profileLocked == 0);
    const bool lockedG = (p.profileLocked != 0);
    const int gLb = clamp(int(c9[4].x * kLumaBins), 0, kLumaBins - 1);
    const float gnY = lockedG ? clamp(p.lockGainY[gLb], 0.6f, 2.2f)
                    : manualG ? 1.0f : as_type<float>(stats[S_GY + gLb]);
    const float gnC = lockedG ? clamp(p.lockGainC[gLb], 0.6f, 2.2f)
                    : manualG ? 1.0f : as_type<float>(stats[S_GC + gLb]);
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
        const device float4* f = frames[k];

        const float oK = as_type<float>(stats[S_EXPOFF + (k < 3 ? k : k - 1)]);   // v3.5 P1
        float dY, dCb, dCr, sdY;
        float3 fc;
        patchDiff(f, W, H, x, y, 0, 0, oK, c9, dY, dCb, dCr, sdY, fc);

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
                patchDiff(f, W, H, x, y, bestOx, bestOy, oK, c9, dY2, dCb2, dCr2, sd2, fc2);
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
                    patchDiff(f, W, H, x, y, tx, ty, oK, c9, dY2, dCb2, dCr2, sd2, fc2);
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

        float gY = 1.0f - smooth01((dY - loY * gnY) * invSpanY * invGnY * shiftTight);
        if (guard)
            gY *= 1.0f - smooth01((fabs(sdY) - loS * gnY) * invSpanS * invGnY * shiftTight);
        // v3.3 B5: per-channel chroma gates (both slaved to the luma gate)
        const float gCb = 1.0f - smooth01((dCb - loCb * gnC) * invSpanCb * invGnC * shiftTight);
        const float gCr = 1.0f - smooth01((dCr - loCr * gnC) * invSpanCr * invGnC * shiftTight);
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

    tmp[y * W + x] = float4(accY / sumWY, accCb / sumWCb, accCr / sumWCr,
                            (sumWY * sumWY) / sumWY2);
}

// ---------------------------------------------------------------------------
// Stage 3 — residual measurement on the temporal result
// ---------------------------------------------------------------------------
kernel void ResidualEstKernel(constant NRParams& p [[buffer(0)]],
                              constant int& W [[buffer(1)]],
                              constant int& H [[buffer(2)]],
                              const device float4* tmp [[buffer(3)]],
                              device atomic_uint* stats [[buffer(4)]],
                              uint2 id [[thread_position_in_grid]])
{
    const int x = 1 + int(id.x) * 2;
    const int y = 1 + int(id.y) * 2;
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
    atomic_fetch_add_explicit(&stats[H_YR + clamp(int(fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[H_CRB + clamp(int(fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[H_CRR + clamp(int(fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);
    const float effN = sampleTmp(tmp, W, H, x, y).w;
    // v3.3 B2: 64 bins (was 32) — see nr_core.h
    atomic_fetch_add_explicit(&stats[H_EN + clamp(int((effN - 1.0f) * 8.0f), 0, 63)], 1u, memory_order_relaxed);

    // v3.2 coarse residual — see nr_core.h (even-aligned 2x2 blocks)
    if ((id.x & 1) == 0 && (id.y & 1) == 0) {
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
            atomic_fetch_add_explicit(&stats[H_YR2 + clamp(int(fabs(lY)  * kHistScaleY), 0, kHistBins - 1)], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[H_CR2B + clamp(int(fabs(lCb) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[H_CR2R + clamp(int(fabs(lCr) * kHistScaleC), 0, kHistBins - 1)], 1u, memory_order_relaxed);
        }
    }
}

kernel void FinalizeResidualKernel(constant NRParams& p [[buffer(0)]],
                                   device uint* stats   [[buffer(1)]],
                                   uint tid [[thread_position_in_grid]])
{
    if (tid != 0)
        return;

    ulong total = 0, total2 = 0;
    for (int b = 0; b < kHistBins; ++b) {
        total += stats[H_YR + b];
        total2 += stats[H_YR2 + b];
    }

    // v3.3 lock fast path: under a lock the input-stat slots may not have
    // been written this frame — compute the locked values from the params
    // (bit-identical to what FinalizeStats writes when it does run).
    float sy = as_type<float>(stats[S_SY]);
    float scb = as_type<float>(stats[S_SCB]);
    float scr = as_type<float>(stats[S_SCR]);
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
            ry = max(ry, 0.9f * ryC);
            rcb = max(rcb, 0.9f * rcbC);
            rcr = max(rcr, 0.9f * rcrC);
        }
        enmed = 1.0f + histQuantile(stats, H_EN, 64, total, 8.0f, 1, 2).value;
        const float floorY = 0.5f * sy / sqrt(max(1.0f, enmed));
        const float floorCb = 0.5f * scb / sqrt(max(1.0f, enmed));
        const float floorCr = 0.5f * scr / sqrt(max(1.0f, enmed));
        ry = clamp(max(ry, floorY), kSigmaMin, sy > kSigmaMin ? sy : kSigmaMax);
        rcb = clamp(max(rcb, floorCb), kSigmaMin, scb > kSigmaMin ? scb : kSigmaMax);
        rcr = clamp(max(rcr, floorCr), kSigmaMin, scr > kSigmaMin ? scr : kSigmaMax);
    }

    stats[S_RY] = as_type<uint>(ry);
    stats[S_RCB] = as_type<uint>(rcb);
    stats[S_RCR] = as_type<uint>(rcr);
    stats[S_ENMED] = as_type<uint>(enmed);
}

// ---------------------------------------------------------------------------
// Stage 3b — v3.3 "Deep Clean": fine-NLM pre-pass at 0.6h over the temporal
// result into a second buffer; corrections clamped to noise size. The
// residual is re-measured on the output. See nr_core.h.
// ---------------------------------------------------------------------------
kernel void DeepCleanKernel(constant NRParams& p [[buffer(0)]],
                            constant int& W [[buffer(1)]],
                            constant int& H [[buffer(2)]],
                            const device float4* tmp [[buffer(3)]],
                            const device uint* stats [[buffer(4)]],
                            device float4* dst [[buffer(5)]],
                            uint2 id [[thread_position_in_grid]])
{
    const int x = int(id.x), y = int(id.y);
    if (x >= W || y >= H)
        return;

    const bool manual = (p.profileSource == 2) && (p.profileLocked == 0);
    const bool locked = (p.profileLocked != 0);
    const float ryG = manual ? clamp(p.sigmaY, kSigmaMin, kSigmaMax) : as_type<float>(stats[S_RY]);
    const float rcbG = manual ? clamp(p.sigmaC, kSigmaMin, kSigmaMax) : as_type<float>(stats[S_RCB]);
    const float rcrG = manual ? clamp(p.sigmaC, kSigmaMin, kSigmaMax) : as_type<float>(stats[S_RCR]);

    const int R = 2;
    const float4 tc = tmp[y * W + x];
    const int lb = clamp(int(tc.x * kLumaBins), 0, kLumaBins - 1);
    const float gainYv = locked ? clamp(p.lockGainY[lb], 0.6f, 2.2f)
                       : manual ? 1.0f : as_type<float>(stats[S_GY + lb]);
    const float gainCv = locked ? clamp(p.lockGainC[lb], 0.6f, 2.2f)
                       : manual ? 1.0f : as_type<float>(stats[S_GC + lb]);
    const float sigY = clamp(ryG * gainYv, 1e-5f, 1.0f);
    // v3.3 B5: pooled-normalized chroma weight — see nr_core.h
    const float sigCb = clamp(rcbG * gainCv, 1e-5f, 1.0f);
    const float sigCr = clamp(rcrG * gainCv, 1e-5f, 1.0f);
    const float hY = 0.6f * kNlmHLuma * sigY;
    const float invHY2 = 1.0f / max(hY * hY, 1e-12f);
    const float invHC2 = 1.0f / max(0.36f * kNlmHChroma * kNlmHChroma, 1e-12f);
    const float invSCb2 = 1.0f / max(sigCb * sigCb, 1e-12f);
    const float invSCr2 = 1.0f / max(sigCr * sigCr, 1e-12f);
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
            dY2 = max(0.0f, dY2 - biasY);
            const float dC2n = 0.5f * (max(0.0f, dCb2 - biasCb) * invSCb2 +
                                       max(0.0f, dCr2 - biasCr) * invSCr2);
            const float wY = exp(-dY2 * invHY2);
            const float wC = exp(-dC2n * invHC2) * exp(-dY2 * invHY2 * 0.25f);
            accY  += wY * ts.x;
            accCb += wC * ts.y;
            accCr += wC * ts.z;
            sumWY += wY;
            sumWC += wC;
            wYmax = max(wYmax, wY);
            wCmax = max(wCmax, wC);
        }
    }
    const float wYc = max(wYmax, 1e-4f);
    const float wCc = max(wCmax, 1e-4f);
    const float Yf  = (accY  + wYc * tc.x) / (sumWY + wYc);
    const float Cbf = (accCb + wCc * tc.y) / (sumWC + wCc);
    const float Crf = (accCr + wCc * tc.z) / (sumWC + wCc);
    dst[y * W + x] = float4(tc.x + clamp(Yf  - tc.x, -2.0f * sigY, 2.0f * sigY),
                            tc.y + clamp(Cbf - tc.y, -2.0f * sigCb, 2.0f * sigCb),
                            tc.z + clamp(Crf - tc.z, -2.0f * sigCr, 2.0f * sigCr),
                            tc.w);
}

// ---------------------------------------------------------------------------
// HUD v3 + scopes — see nr_core.h for the layout rationale
// ---------------------------------------------------------------------------
// glyph order: 0-9 . % A-Z + space - | =
constant ulong kFont[43] = {
    0x3a33ae62eULL, 0x11842108eULL, 0x3a213221fULL, 0x3a213062eULL, 0x08ca97c42ULL, 0x7e1e0862eULL,
    0x3a10f462eULL, 0x7c2222108ULL, 0x3a317462eULL, 0x3a317842eULL, 0x00000018cULL, 0x632222263ULL,
    0x3a31fc631ULL, 0x7a31f463eULL, 0x3a308422eULL, 0x7a318c63eULL, 0x7e10f421fULL, 0x7e10f4210ULL,
    0x3a30bc62fULL, 0x4631fc631ULL, 0x38842108eULL, 0x1c4210a4cULL, 0x4654c5251ULL, 0x42108421fULL,
    0x4775ac631ULL, 0x47359c631ULL, 0x3a318c62eULL, 0x7a31f4210ULL, 0x3a318d64dULL, 0x7a31f5251ULL,
    0x3e107043eULL, 0x7c8421084ULL, 0x46318c62eULL, 0x46318c544ULL, 0x4631ad771ULL, 0x462a22a31ULL,
    0x462a21084ULL, 0x7c222221fULL, 0x0084f9080ULL, 0x000000000ULL, 0x000070000ULL, 0x108421084ULL,
    0x01f07c00ULL,
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

constant int kLabIY[7]  = { G_I, G_N, G_P, G_U, G_T, G_SP, G_Y };
constant int kLabIC[7]  = { G_I, G_N, G_P, G_U, G_T, G_SP, G_C };
constant int kLabRY[10] = { G_R, G_E, G_S, G_I, G_D, G_U, G_A, G_L, G_SP, G_Y };
constant int kLabRC[10] = { G_R, G_E, G_S, G_I, G_D, G_U, G_A, G_L, G_SP, G_C };
constant int kLabAVGFR[10] = { G_A, G_V, G_G, G_SP, G_F, G_R, G_A, G_M, G_E, G_S };
constant int kLabGAIN[4] = { G_G, G_A, G_I, G_N };
constant int kLabDB[2]   = { G_D, G_B };
constant int kLabLOCKED[14] = { G_P, G_R, G_O, G_F, G_I, G_L, G_E, G_SP,
                                G_L, G_O, G_C, G_K, G_E, G_D };
constant int kLabLIVE[14] = { G_M, G_E, G_A, G_S, G_U, G_R, G_I, G_N,
                              G_G, G_SP, G_L, G_I, G_V, G_E };
constant int kLabTOFF[12] = { G_T, G_E, G_M, G_P, G_O, G_R, G_A, G_L,
                              G_SP, G_O, G_F, G_F };
constant int kLabCURVE[19] = { G_N, G_O, G_I, G_S, G_E, G_SP, G_V, G_S,
                               G_SP, G_B, G_R, G_I, G_G, G_H, G_T, G_N,
                               G_E, G_S, G_S };
constant int kLabHIST[31] = { G_N, G_O, G_I, G_S, G_E, G_SP, G_H, G_I,
                              G_S, G_T, G_O, G_G, G_R, G_A, G_M, G_SP,
                              G_DASH, G_SP, G_M, G_E, G_D, G_I, G_A, G_N,
                              G_SP, G_M, G_A, G_R, G_K, G_E, G_D };
constant int kLabTITLE[8] = { G_N, G_O, G_I, G_S, G_E, G_SP, G_E, G_Q };
constant int kLabOFF3[3]  = { G_O, G_F, G_F };
constant int kLabFINE[4] = { G_F, G_I, G_N, G_E };
constant int kLabMED[6]  = { G_M, G_E, G_D, G_I, G_U, G_M };
constant int kLabCRS[6]  = { G_C, G_O, G_A, G_R, G_S, G_E };
constant int kLabCOL[5]  = { G_C, G_O, G_L, G_O, G_R };
constant int kLabPX1[3]  = { 1, G_P, G_X };
constant int kLabPX38[5] = { 3, G_DASH, 8, G_P, G_X };
constant int kLabPX16[5] = { 1, 6, G_P, G_X, G_PLUS };
constant int kLabLEG[34] = { G_B, G_A, G_R, G_SP, G_EQ, G_SP, G_C, G_U,
                             G_T, G_SP, G_SP, G_A, G_M, G_B, G_E, G_R,
                             G_SP, G_EQ, G_SP, G_M, G_E, G_A, G_S, G_U,
                             G_R, G_E, G_D, G_SP, G_N, G_O, G_I, G_S,
                             G_E, G_SP };
constant int kLabMO[33] = { G_M, G_O, G_T, G_I, G_O, G_N, G_SP, G_SP,
                            G_G, G_R, G_E, G_E, G_N, G_EQ, G_S, G_T,
                            G_A, G_C, G_K, G_E, G_D, G_SP, G_SP, G_R,
                            G_E, G_D, G_EQ, G_M, G_O, G_V, G_I, G_N,
                            G_G };

inline bool glyphPixel(int glyph, int gx, int gy)
{
    if (glyph < 0 || glyph > 42 || gx < 0 || gx >= 5 || gy < 0 || gy >= 7)
        return false;
    return (kFont[glyph] >> (34 - (gy * 5 + gx))) & 1ULL;
}

// sc = integer text scale: 1 -> 6x7 px cells, 2 -> 12x14 px cells
inline bool textPixel(constant int* chars, int n, int tx, int ty, int lx, int ly, int sc)
{
    if (ly < ty || ly >= ty + 7 * sc || lx < tx || lx >= tx + n * 6 * sc)
        return false;
    const int gx = (lx - tx) / sc;
    const int ci = gx / 6;
    return glyphPixel(chars[ci], gx - ci * 6, (ly - ty) / sc);
}

inline bool textPixelT(thread const int* chars, int n, int tx, int ty, int lx, int ly, int sc)
{
    if (ly < ty || ly >= ty + 7 * sc || lx < tx || lx >= tx + n * 6 * sc)
        return false;
    const int gx = (lx - tx) / sc;
    const int ci = gx / 6;
    return glyphPixel(chars[ci], gx - ci * 6, (ly - ty) / sc);
}

inline void pctGlyphs(float pp, thread int* outg)
{
    const int v = clamp(int(pp * 100.0f + 0.5f), 0, 9999);
    const int tens = (v / 1000) % 10;
    outg[0] = (tens == 0) ? G_SP : tens;
    outg[1] = (v / 100) % 10;
    outg[2] = G_DOT;
    outg[3] = (v / 10) % 10;
    outg[4] = v % 10;
    outg[5] = G_PCT;
}

inline void dec1Glyphs(float v, thread int* outg)
{
    const int t = clamp(int(v * 10.0f + 0.5f), 0, 99);
    outg[0] = (t / 10) % 10;
    outg[1] = G_DOT;
    outg[2] = t % 10;
}

inline void pctIntGlyphs(float frac, thread int* outg)
{
    const int v = clamp(int(frac * 100.0f + 0.5f), 0, 999);
    outg[0] = (v >= 100) ? (v / 100) % 10 : G_SP;
    outg[1] = (v >= 10) ? (v / 10) % 10 : G_SP;
    outg[2] = v % 10;
    outg[3] = G_PCT;
}

inline bool hudPixel(int x, int y, int W, int H,
                     float sy, float sc, float ry, float rc, float enmed,
                     uint medBin, uint hmax, const device uint* stats,
                     int enableTemporal, int locked, thread float3& rgb)
{
    const int yd = H - 1 - y;   // OFX buffers are bottom-up; panel anchors top-left on screen
    const int s = max(1, H / 540);
    const int ox = 16 * s, oy = 16 * s;
    const int lw = 360, lh = 294;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    rgb = float3(0.045f, 0.045f, 0.05f);
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        rgb = float3(0.35f);
        return true;
    }

    const float sig[4] = { sy, sc, ry, rc };
    const int rowY[4] = { 10, 42, 74, 106 };

    for (int row = 0; row < 4; ++row) {
        const int ty0 = rowY[row];
        bool lit = false;
        if (row == 0) lit = textPixel(kLabIY, 7,  10, ty0, lx, ly, 2);
        if (row == 1) lit = textPixel(kLabIC, 7,  10, ty0, lx, ly, 2);
        if (row == 2) lit = textPixel(kLabRY, 10, 10, ty0, lx, ly, 2);
        if (row == 3) lit = textPixel(kLabRC, 10, 10, ty0, lx, ly, 2);
        if (!lit) {
            int vg[6];
            pctGlyphs(sig[row] * 100.0f, vg);
            lit = textPixelT(vg, 6, 278, ty0, lx, ly, 2);
        }
        if (lit) { rgb = float3(1.0f); return true; }

        if (ly >= ty0 + 16 && ly < ty0 + 22 && lx >= 10 && lx < 350) {
            const float fill = clamp(sig[row] / 0.08f, 0.0f, 1.0f);
            const bool on = (lx - 10) < int(fill * 340.0f);
            const bool residRow = (row >= 2);
            if (on) rgb = residRow ? float3(0.95f, 0.65f, 0.20f) : float3(0.20f, 0.65f, 0.95f);
            else    rgb = float3(0.14f);
            return true;
        }
    }

    {
        const int ty0 = 138;
        bool lit = textPixel(kLabAVGFR, 10, 10, ty0, lx, ly, 2);
        if (!lit) {
            int vg[3];
            dec1Glyphs(enableTemporal ? enmed : 1.0f, vg);
            lit = textPixelT(vg, 3, 138, ty0, lx, ly, 2);
        }
        if (!lit) lit = textPixel(kLabGAIN, 4, 186, ty0, lx, ly, 2);
        if (!lit) {
            const float gainDb = clamp(20.0f * log10(max(sy, 1e-5f) / max(ry, 1e-5f)), 0.0f, 40.0f);
            int vg[4];
            vg[0] = G_PLUS;
            int d1[3];
            dec1Glyphs(gainDb, d1);
            vg[1] = d1[0]; vg[2] = d1[1]; vg[3] = d1[2];
            lit = textPixelT(vg, 4, 240, ty0, lx, ly, 2);
            if (!lit) lit = textPixel(kLabDB, 2, 294, ty0, lx, ly, 2);
        }
        if (lit) { rgb = float3(1.0f); return true; }
    }

    {
        const int ty0 = 160;
        if (locked) {
            if (textPixel(kLabLOCKED, 14, 10, ty0, lx, ly, 2)) { rgb = float3(0.95f, 0.65f, 0.20f); return true; }
        } else {
            if (textPixel(kLabLIVE, 14, 10, ty0, lx, ly, 2)) { rgb = float3(0.55f); return true; }
        }
        if (enableTemporal == 0 && textPixel(kLabTOFF, 12, 190, ty0, lx, ly, 2)) {
            rgb = float3(0.90f, 0.45f, 0.30f); return true;
        }
    }

    if (textPixel(kLabCURVE, 19, 10, 182, lx, ly, 1)) { rgb = float3(0.55f); return true; }
    if (lx >= 10 && lx < 346 && ly >= 194 && ly < 238) {
        const int bin = clamp((lx - 10) / 21, 0, kLumaBins - 1);
        const float gain = as_type<float>(stats[S_GY + bin]);
        const float v = clamp((gain - 0.6f) / 1.6f, 0.0f, 1.0f);
        const bool bar = (237 - ly) < int(v * 43.0f + 0.5f);
        const bool ref = (237 - ly) == int((1.0f - 0.6f) / 1.6f * 43.0f + 0.5f);
        if (bar)      rgb = float3(0.20f, 0.65f, 0.95f);
        else if (ref) rgb = float3(0.42f);
        else          rgb = float3(0.08f);
        return true;
    }

    if (textPixel(kLabHIST, 31, 10, 244, lx, ly, 1)) { rgb = float3(0.55f); return true; }
    if (lx >= 10 && lx < 346 && ly >= 252 && ly < 284) {
        const int bin = clamp((lx - 10) * kHistBins / 336, 0, kHistBins - 1);
        const float frac = float(stats[H_YF + bin]) / float(max(hmax, 1u));
        const float hgt = 31.0f * sqrt(clamp(frac, 0.0f, 1.0f));
        const bool bar = (283 - ly) < int(hgt + 0.5f);
        if (bin == int(medBin))  rgb = float3(0.95f, 0.85f, 0.15f);
        else if (bar)            rgb = float3(0.55f);
        else                     rgb = float3(0.08f);
        return true;
    }

    return true;
}

// Noise EQ panel (top-right) — see nr_core.h
inline bool eqScopePixel(int x, int y, int W, int H,
                         float fineY, float fineC, float coarseY, float ty, float tc,
                         float eqFine, float eqMedium, float eqCoarse, float chromaBlotch,
                         int enableSpatial, thread float3& rgb)
{
    const int yd = H - 1 - y;
    const int s = max(1, H / 540);
    const int lw = 300, lh = 190;
    const int ox = W - 16 * s - lw * s, oy = 16 * s;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    rgb = float3(0.045f, 0.045f, 0.05f);
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        rgb = float3(0.35f);
        return true;
    }

    if (textPixel(kLabTITLE, 8, 10, 8, lx, ly, 2)) { rgb = float3(1.0f); return true; }
    if (enableSpatial == 0 && textPixel(kLabOFF3, 3, 250, 8, lx, ly, 2)) {
        rgb = float3(0.90f, 0.45f, 0.30f); return true;
    }
    if (textPixel(kLabLEG, 34, 10, 172, lx, ly, 1)) { rgb = float3(0.55f); return true; }

    const float fineM = max(fineY, coarseY);
    const float lowY = sqrt(max(0.0f, ty * ty - fineM * fineM));
    const float lowC = sqrt(max(0.0f, tc * tc - fineC * fineC));
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
            if (textPixelT(vg, 4, x0 + 18, 32, lx, ly, 1)) { rgb = float3(0.85f); return true; }
        }
        bool lit = false;
        if (lane == 0) lit = textPixel(kLabFINE, 4, x0 + 18, 148, lx, ly, 1) ||
                             textPixel(kLabPX1, 3, x0 + 21, 158, lx, ly, 1);
        if (lane == 1) lit = textPixel(kLabMED, 6, x0 + 12, 148, lx, ly, 1) ||
                             textPixel(kLabPX38, 5, x0 + 15, 158, lx, ly, 1);
        if (lane == 2) lit = textPixel(kLabCRS, 6, x0 + 12, 148, lx, ly, 1) ||
                             textPixel(kLabPX16, 5, x0 + 15, 158, lx, ly, 1);
        if (lane == 3) lit = textPixel(kLabCOL, 5, x0 + 15, 148, lx, ly, 1) ||
                             textPixel(kLabPX16, 5, x0 + 15, 158, lx, ly, 1);
        if (lit) { rgb = float3(0.75f); return true; }

        if (ly >= 44 && ly < 144) {
            const int up = 143 - ly;
            const float nv = clamp(meas[lane] / 0.08f, 0.0f, 1.0f);
            const int markH = int(nv * 98.0f + 0.5f);
            const int barH = int(clamp(amt[lane], 0.0f, 1.0f) * 98.0f + 0.5f);
            if (up >= markH - 1 && up <= markH + 1 && meas[lane] > 1e-5f) {
                rgb = float3(0.95f, 0.65f, 0.20f);
            } else if (up < barH) {
                rgb = (up >= barH - 3) ? float3(0.90f) : float3(0.30f);
            } else {
                rgb = float3(0.08f);
            }
            return true;
        }
        return true;
    }

    return true;
}

// Temporal-activity mini map (bottom-right) — see nr_core.h
inline bool motionScopePixel(int x, int y, int W, int H,
                             const device float4* tmp, const device float4* curr,
                             thread float3& rgb)
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
        rgb = float3(0.35f);
        return true;
    }

    if (ly < 16) {
        rgb = float3(0.045f, 0.045f, 0.05f);
        if (textPixel(kLabMO, 33, 4, 5, lx, ly, 1)) rgb = float3(0.85f);
        return true;
    }

    const int u = clamp(lx - 1, 0, mapW - 1);
    const int v = clamp(ly - 16, 0, mapH - 1);
    const int sx = clamp((u * W) / mapW, 0, W - 1);
    const int sdy = clamp((v * H) / mapH, 0, H - 1);
    const int sy2 = H - 1 - sdy;
    const float4 t = tmp[sy2 * W + sx];
    const float3 cin = rgb2ycc(curr[sy2 * W + sx].xyz);
    const float effN = max(1.0f, t.w);
    const float tt = clamp((effN - 1.0f) / 4.0f, 0.0f, 1.0f);
    const float3 heat = float3(0.90f + (0.10f - 0.90f) * tt,
                               0.15f + (0.85f - 0.15f) * tt,
                               0.10f + (0.20f - 0.10f) * tt);
    rgb = cin.x * 0.45f + heat * 0.55f;
    return true;
}

// ---------------------------------------------------------------------------
// Stage 4+5+6 — spatial + blotch + refine + output
// ---------------------------------------------------------------------------
constant float kDirX[8] = { 1, 0, -1, 0, 0.7071f, -0.7071f, -0.7071f, 0.7071f };
constant float kDirY[8] = { 0, 1, 0, -1, 0.7071f, 0.7071f, -0.7071f, -0.7071f };

// v3.3 A2: the fine-band NLM loop reads (2R+1)^2 x 10 samples per pixel with
// near-total overlap between neighbouring threads — each 16x16 threadgroup
// cooperatively stages its tile of tmp (16 + 2(R+1) square: R window + 1 px
// patch ring) into threadgroup memory as flat Y/Cb/Cr triples. Values are
// the same edge-clamped samples sampleTmp returns, so the math is untouched.
// Max footprint at R=10: 38*38*3 floats = 17.3 KB (< the 32 KB minimum).
inline float3 tileAt(threadgroup const float* tile, int tileW, int lx, int ly)
{
    const int i = (ly * tileW + lx) * 3;
    return float3(tile[i], tile[i + 1], tile[i + 2]);
}

// tmp is the working buffer (the Deep Clean output when that pass ran);
// tmpTrue is the TRUE temporal result for the After Temporal view and the
// motion scope. Without Deep Clean the host binds the same buffer to both.
kernel void SpatialNLMKernel(constant NRParams& p [[buffer(0)]],
                             constant int& W [[buffer(1)]],
                             constant int& H [[buffer(2)]],
                             const device float4* tmp [[buffer(3)]],
                             const device float4* curr [[buffer(4)]],
                             const device uint* stats [[buffer(5)]],
                             device float4* dst [[buffer(6)]],
                             const device float4* tmpTrue [[buffer(7)]],
                             threadgroup float* tile [[threadgroup(0)]],
                             uint2 id [[thread_position_in_grid]],
                             uint2 lid [[thread_position_in_threadgroup]],
                             uint2 gid [[threadgroup_position_in_grid]])
{
    const int x = int(id.x), y = int(id.y);
    // NOTE: no early out-of-bounds return here — every thread (in-frame or
    // not) must reach the cooperative tile load and its barrier below.

    const bool manual = (p.profileSource == 2) && (p.profileLocked == 0);
    // v3.3 lock fast path: locked input sigmas/gains come straight from the
    // params (see loadSigmasIn); the residual pair still comes from the
    // stats buffer — the residual passes always run under a lock.
    const bool locked = (p.profileLocked != 0);
    const float adjL = clamp(p.profileAdjust, 0.25f, 6.0f);
    const float sy = locked ? clamp(p.lockSY * adjL, kSigmaMin, kSigmaMax)
                   : manual ? clamp(p.sigmaY, kSigmaMin, kSigmaMax) : as_type<float>(stats[S_SY]);
    // v3.3 B5: per-channel chroma pairs; the combined values below are the
    // pair means (bands, deband threshold, HUD)
    const float scbG = locked ? clamp(p.lockSC * adjL, kSigmaMin, kSigmaMax)
                    : manual ? clamp(p.sigmaC, kSigmaMin, kSigmaMax) : as_type<float>(stats[S_SCB]);
    const float scrG = locked ? clamp(p.lockSCr * adjL, kSigmaMin, kSigmaMax)
                    : manual ? clamp(p.sigmaC, kSigmaMin, kSigmaMax) : as_type<float>(stats[S_SCR]);
    const float sc = 0.5f * (scbG + scrG);
    const float tyG = locked ? clamp(p.lockTY * adjL, kSigmaMin, kSigmaMax)
                    : manual ? sy : as_type<float>(stats[S_TY]);
    const float tcbG = locked ? clamp(p.lockTC * adjL, kSigmaMin, kSigmaMax)
                     : manual ? scbG : as_type<float>(stats[S_TCB]);
    const float tcrG = locked ? clamp(p.lockTCr * adjL, kSigmaMin, kSigmaMax)
                     : manual ? scrG : as_type<float>(stats[S_TCR]);
    const float tcG = 0.5f * (tcbG + tcrG);
    const float ryG = manual ? sy : as_type<float>(stats[S_RY]);
    const float rcbG = manual ? scbG : as_type<float>(stats[S_RCB]);
    const float rcrG = manual ? scrG : as_type<float>(stats[S_RCR]);
    const float rcG = 0.5f * (rcbG + rcrG);
    const float enmed = manual ? 1.0f : as_type<float>(stats[S_ENMED]);

    const float mLow  = min(p.master, 1.0f);
    const float mHigh = max(p.master, 1.0f);
    const float hBoost = pow(mHigh, 1.2f);

    const float sL = clamp(p.spatialLuma, 0.0f, 1.5f);
    const float sC = clamp(p.spatialChroma, 0.0f, 1.5f);
    // v3 Noise EQ: the fine slider scales the NLM band's blend (1 = v2.1);
    // v3.1: above 100% it also widens the similarity h
    const float eqF = clamp(p.eqFine, 0.0f, 3.0f);
    const float eqH = pow(max(1.0f, eqF), 0.8f);
    const float aY = (p.enableSpatial == 0) ? 0.0f : clamp(sL * mLow * eqF, 0.0f, 1.0f);
    const float aC = (p.enableSpatial == 0) ? 0.0f : clamp(sC * mLow * eqF, 0.0f, 1.0f);
    const float overL = (sL > 1.0f) ? 1.6f * pow(sL - 1.0f, 1.2f) : 0.0f;
    const float overC = (sC > 1.0f) ? 1.6f * pow(sC - 1.0f, 1.2f) : 0.0f;
    const float hMulY = (0.6f + 1.4f * pow(sL, 1.5f) + overL) * hBoost * eqH;
    const float hMulC = (0.6f + 1.4f * pow(sC, 1.5f) + overC) * hBoost * eqH;
    const float pd = clamp(p.preserveDetail, 0.0f, 1.0f);
    const int   R  = clamp(p.spatialRadius, 1, 10);
    const bool  nlm = (p.spatialMode == 1);
    const bool  runSpatial = (p.enableSpatial != 0) && (aY > 0.0f || aC > 0.0f);
    const float spatialSigma = max(1.0f, float(R) / 1.5f);
    const float invSpatial2 = 1.0f / (2.0f * spatialSigma * spatialSigma);
    // v3.1 Detail Rescue — see nr_core.h
    const float rescue = clamp(p.detailRescue, 0.0f, 1.0f);

    // v3.1: band sliders reach 150 — amount caps at 1, the extra drive
    // widens the similarity tolerances and the reach instead
    const float blotchRaw = clamp(p.chromaBlotch, 0.0f, 1.5f);
    const float medRaw    = clamp(p.eqMedium, 0.0f, 1.5f);
    const float coarseRaw = clamp(p.eqCoarse, 0.0f, 1.5f);
    const float blotch = (p.enableSpatial != 0) ? min(blotchRaw * mLow, 1.0f) : 0.0f;
    // v3 Noise EQ: medium band amount and coarse-band luma amount
    const float eqMed  = (p.enableSpatial != 0) ? min(medRaw * mLow, 1.0f) : 0.0f;
    const float coarseL = (p.enableSpatial != 0) ? min(coarseRaw * mLow, 1.0f) : 0.0f;
    const float medOver = max(1.0f, medRaw);
    const float crsOver = max(1.0f, max(blotchRaw, coarseRaw));
    const int   Rb = 2 + int(14.0f * max(blotchRaw, coarseRaw));
    const int   Rm = 3 + int(5.0f * medRaw);
    // Band tolerance: see nr_core.h
    const float bandRatioY = clamp(tyG / max(sy, 1e-6f), 1.0f, 3.0f);
    const float bandRatioC = clamp(tcG / max(sc, 1e-6f), 1.0f, 3.0f);

    const bool refine = (p.enableRefine != 0) && (p.master > 0.0f);
    const float desat = refine ? clamp(p.shadowDesat, 0.0f, 1.0f) : 0.0f;
    const float desatRange = max(0.02f, p.desatRange);
    const float tex = refine ? clamp(p.lumaTexture, 0.0f, 1.0f) * mLow : 0.0f;
    // v3 deband thresholds — see nr_core.h
    const float debandAmt = refine ? clamp(p.deband, 0.0f, 1.0f) * mLow : 0.0f;
    const float dbThrY = max(0.010f, 1.5f * ryG);
    const float dbThrC = max(0.010f, 1.5f * rcG);
    const float grainAmt = refine ? clamp(p.grainAmount, 0.0f, 1.0f) * 0.06f : 0.0f;
    const float grainSize = clamp(p.grainSize, 0.5f, 6.0f);
    const float grainCh = clamp(p.grainChroma, 0.0f, 1.0f);
    const uint frame = uint(p.frameIndex);
    // v3.2 global blend — see nr_core.h
    const float gBlend = clamp(p.globalBlend, 0.0f, 1.0f);

    // ---- v3.3 A2: cooperative tile stage (see tileAt above). runSpatial is
    // uniform across the grid, so the branch and its barrier are legal; the
    // out-of-bounds return must come only after the barrier. ----
    const int tileW = 16 + 2 * (R + 1);
    if (runSpatial) {
        const int tx0 = int(gid.x) * 16 - (R + 1);
        const int ty0 = int(gid.y) * 16 - (R + 1);
        for (int i = int(lid.y) * 16 + int(lid.x); i < tileW * tileW; i += 256) {
            const int tyy = i / tileW, txx = i - tyy * tileW;
            const float4 v = tmp[clamp(ty0 + tyy, 0, H - 1) * W + clamp(tx0 + txx, 0, W - 1)];
            tile[i * 3 + 0] = v.x; tile[i * 3 + 1] = v.y; tile[i * 3 + 2] = v.z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (x >= W || y >= H)
        return;
    const int ltx = int(lid.x) + R + 1;   // this pixel in tile coords
    const int lty = int(lid.y) + R + 1;

    const float4 tc = tmp[y * W + x];
    const int lb = clamp(int(tc.x * kLumaBins), 0, kLumaBins - 1);
    const float gainYv = locked ? clamp(p.lockGainY[lb], 0.6f, 2.2f)
                       : manual ? 1.0f : as_type<float>(stats[S_GY + lb]);
    const float gainCv = locked ? clamp(p.lockGainC[lb], 0.6f, 2.2f)
                       : manual ? 1.0f : as_type<float>(stats[S_GC + lb]);
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
        const float var = max(0.0f, m2 * (1.0f / 9.0f) - mean * mean);
        const float edginess = clamp(sqrt(max(var - sigY * sigY, 0.0f)) / (3.0f * sigY), 0.0f, 1.0f);

        const float hY = kNlmHLuma   * sigY * hMulY * (1.0f - pd * 0.85f * edginess);
        // v3.3 B5: pooled-normalized chroma weight — see nr_core.h
        const float mC = hMulC * (1.0f - pd * 0.50f * edginess);
        const float invHY2 = 1.0f / max(hY * hY, 1e-12f);
        const float invHC2 = 1.0f / max(kNlmHChroma * kNlmHChroma * mC * mC, 1e-12f);
        const float invSCb2 = 1.0f / max(sigCb * sigCb, 1e-12f);
        const float invSCr2 = 1.0f / max(sigCr * sigCr, 1e-12f);
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

                dY2 = max(0.0f, dY2 - biasY);
                const float dC2n = 0.5f * (max(0.0f, dCb2 - biasCb) * invSCb2 +
                                           max(0.0f, dCr2 - biasCr) * invSCr2);

                float wY = exp(-dY2 * invHY2);
                float wC = exp(-dC2n * invHC2) * exp(-dY2 * invHY2 * 0.25f);
                if (!nlm) {
                    const float fall = exp(-float(dx * dx + dy * dy) * invSpatial2);
                    wY *= fall;
                    wC *= fall;
                }

                accY  += wY * ts.x;
                accCb += wC * ts.y;
                accCr += wC * ts.z;
                sumWY += wY;
                sumWC += wC;
                wYmax = max(wYmax, wY);
                wCmax = max(wCmax, wC);
            }
        }

        const float wYc = max(wYmax, 1e-4f);
        const float wCc = max(wCmax, 1e-4f);
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
        const float myDen = 1.0f / max(mScale, 1e-6f);
        const float mcDen = 1.0f / max(3.0f * sigC * bandRatioC * hBoost * medOver, 1e-6f);
        const float3 b0 = blockMeanTmp(tmp, W, H, x, y);
        float accMY = b0.x, accMB = b0.y, accMR = b0.z, sumWm = 1.0f;
        for (int d = 0; d < 8; ++d) {
            for (int ri = 1; ri <= 2; ++ri) {
                const float rr = Rm * (float(ri) / 2.0f);
                const float3 bm = blockMeanTmp(tmp, W, H,
                                               x + int(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                                               y + int(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f)));
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
        const float gyDen = 1.0f / max(2.0f * sigY * hBoost * crsOver, 1e-6f);
        const float gcDen = 1.0f / max(3.0f * sigC * hBoost * crsOver, 1e-6f);
        const float cScale = 2.2f * sigY * bandRatioY * hBoost * crsOver;
        const float glDen = 1.0f / max(cScale, 1e-6f);
        const int RbL = 2 + int(30.0f * coarseRaw);
        float3 c0 = float3(0.0f);
        if (coarseL > 0.0f)
            c0 = blockMean4Tmp(tmp, W, H, x, y);
        float accB = tc.y, accR = tc.z, sumW = 1.0f;
        float accL = c0.x, sumWL = 1.0f;
        for (int d = 0; d < 8; ++d) {
            for (int ri = 1; ri <= 3; ++ri) {
                const float rr = Rb * (float(ri) / 3.0f);
                const int sx = x + int(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f));
                const int sy2 = y + int(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f));
                const float3 ts = sampleTmp(tmp, W, H, sx, sy2).xyz;
                const float eY = (ts.x - tc.x) * gyDen;
                const float eC = (0.5f * (fabs(ts.y - tc.y) + fabs(ts.z - tc.z))) * gcDen;
                const float w = exp(-(eY * eY + eC * eC));
                accB += w * ts.y;
                accR += w * ts.z;
                sumW += w;
                if (coarseL > 0.0f) {
                    const float rrL = RbL * (float(ri) / 3.0f);
                    const int lx2 = x + int(kDirX[d] * rrL + (kDirX[d] > 0 ? 0.5f : -0.5f));
                    const int ly2 = y + int(kDirY[d] * rrL + (kDirY[d] > 0 ? 0.5f : -0.5f));
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
        const float sat = 1.0f - desat * (1.0f - smooth01(Yr / desatRange));
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
                    const float rr = 16.0f * (float(ri) / 3.0f);
                    const float3 bm = blockMeanTmp(tmp, W, H,
                                                   x + int(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                                                   y + int(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f)));
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
            Yr += dith * 0.5f * (hashNoise(uint(x), uint(y), frame, 3u) +
                                 hashNoise(uint(x), uint(y), frame + 977u, 3u));
        }
        if (grainAmt > 0.0f) {
            const float yc = clamp(Yr, 0.0f, 1.0f);
            const float resp = 0.25f + 0.75f * (4.0f * yc * (1.0f - yc));
            const float gn = valueNoise(float(x), float(y), grainSize, frame, 0u);
            Yr += grainAmt * resp * gn;
            if (grainCh > 0.0f) {
                Cbr += grainAmt * grainCh * 0.6f * resp * valueNoise(float(x), float(y), grainSize, frame, 1u);
                Crr += grainAmt * grainCh * 0.6f * resp * valueNoise(float(x), float(y), grainSize, frame, 2u);
            }
        }
    }

    // v3.2 global blend — result only; Noise Removed stays full-strength
    if (gBlend < 1.0f) {
        Yr  = cin.x + gBlend * (Yr  - cin.x);
        Cbr = cin.y + gBlend * (Cbr - cin.y);
        Crr = cin.z + gBlend * (Crr - cin.z);
    }

    float3 rgb = ycc2rgb(float3(Yr, Cbr, Crr));
    const float3 dn = ycc2rgb(float3(Yo, Cbo, Cro));
    float4 o = float4(rgb, c.w);

    if (p.viewMode == 1) {
        if (x < W / 2) o.xyz = c.xyz;
        if (abs(x - W / 2) <= 1) o.xyz = float3(1.0f);
    } else if (p.viewMode == 2) {
        o.xyz = c.xyz;
    } else if (p.viewMode == 3) {
        // the TRUE temporal result, even when Deep Clean rewrote tmp
        o.xyz = ycc2rgb(tmpTrue[y * W + x].xyz);
    } else if (p.viewMode == 4) {
        // noise removed: noise-adaptive gain + soft knee — see nr_core.h
        const float nrGain = 0.08f / clamp(sy, 0.004f, 0.08f);
        const float3 d = (c.xyz - dn) * nrGain;
        o.xyz = 0.5f + 0.5f * d / (0.5f + abs(d));
    } else if (p.viewMode == 5) {
        // noise analysis: the result; the measurement scope is drawn by the
        // overlay pass below
        o.xyz = rgb;
    } else if (p.viewMode == 6) {
        const float effN = max(1.0f, tc.w);
        const float t = clamp((effN - 1.0f) / 4.0f, 0.0f, 1.0f);
        const float3 heat = float3(0.90f + (0.10f - 0.90f) * t,
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
        const float var = max(0.0f, m2 * (1.0f / 25.0f) - mean * mean);
        const float sigNoise = max(sy * gainYv, 1e-5f);
        const float sigSignal = sqrt(max(var - sigNoise * sigNoise, 0.0f));
        const float snrDb = 20.0f * log10(max(sigSignal, 1e-6f) / sigNoise);
        const float t = clamp((snrDb + 6.0f) / 36.0f, 0.0f, 1.0f);
        float3 heat;
        if (t < 0.5f) {
            const float u = t * 2.0f;
            heat = float3(0.85f + (0.95f - 0.85f) * u,
                          0.10f + (0.70f - 0.10f) * u,
                          0.75f + (0.15f - 0.75f) * u);
        } else {
            const float u = (t - 0.5f) * 2.0f;
            heat = float3(0.95f + (0.10f - 0.95f) * u,
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
        const float var = max(0.0f, m2 * (1.0f / 25.0f) - mean * mean);
        const float sigNoise = max(sy * gainYv, 1e-5f);
        const float sigSignal = sqrt(max(var - sigNoise * sigNoise, 0.0f));
        const float snrDb = 20.0f * log10(max(sigSignal, 1e-6f) / sigNoise);
        const float m = 1.0f - clamp((snrDb + 6.0f) / 36.0f, 0.0f, 1.0f);
        o = float4(m, m, m, m);
    }

    // ---- v3.1 scope overlays: drawn over ANY view, never into alpha ----
    {
        const bool wantHud = (p.scopeMeasure != 0) || (p.viewMode == 5);
        const bool wantEq  = (p.scopeEq != 0);
        const bool wantMo  = (p.scopeMotion != 0);
        if (wantHud || wantEq || wantMo) {
            float3 sco = o.xyz;
            bool drew = false;
            if (wantEq) {
                const float fineY = manual ? sy : as_type<float>(stats[S_FINEY]);
                const float fineC = manual ? sc : as_type<float>(stats[S_FINEC]);
                const float crsY  = manual ? sy : as_type<float>(stats[S_CRSY]);
                if (eqScopePixel(x, y, W, H, fineY, fineC, crsY, tyG, tcG,
                                 p.eqFine, p.eqMedium, p.eqCoarse, p.chromaBlotch,
                                 p.enableSpatial, sco))
                    drew = true;
            }
            if (wantMo && motionScopePixel(x, y, W, H, tmpTrue, curr, sco))
                drew = true;
            if (wantHud) {
                if (hudPixel(x, y, W, H, sy, sc, ryG, rcG, enmed,
                             stats[S_MED], stats[S_HMAX], stats, p.enableTemporal,
                             p.profileLocked, sco)) {
                    drew = true;
                } else if (p.profileSource == 1) {
                    const float rHalf = 0.5f * p.regionSize * float(min(W, H));
                    const float cx = p.regionCX * W, cyy = p.regionCY * H;
                    const float ax = fabs(float(x) - cx), ay = fabs(float(y) - cyy);
                    const bool onEdge = (ax <= rHalf && ay <= rHalf) &&
                                        (ax >= rHalf - 2.0f || ay >= rHalf - 2.0f);
                    if (onEdge) { sco = float3(1.0f, 1.0f, 0.1f); drew = true; }
                }
            }
            if (drew) o.xyz = sco;
        }
    }

    dst[y * W + x] = o;
}

)MSL";

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct QueueResources
{
    id<MTLComputePipelineState> est  = nil;
    id<MTLComputePipelineState> fin  = nil;
    id<MTLComputePipelineState> temp = nil;
    id<MTLComputePipelineState> est2 = nil;
    id<MTLComputePipelineState> fin2 = nil;
    id<MTLComputePipelineState> deep = nil;
    id<MTLComputePipelineState> expE = nil;   // v3.5 P1
    id<MTLComputePipelineState> expF = nil;
    id<MTLComputePipelineState> nlm  = nil;
    id<MTLBuffer> tmp   = nil;
    id<MTLBuffer> tmp2  = nil;   // v3.3 Deep Clean output (lazy — only when used)
    id<MTLBuffer> stats = nil;
    int w = 0, h = 0;
};

static std::mutex s_mutex;
static std::unordered_map<void*, QueueResources> s_resources;

static id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, id<MTLLibrary> lib, NSString* name)
{
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "OpenNR: missing Metal kernel %s\n", name.UTF8String);
        return nil;
    }
    id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!ps)
        fprintf(stderr, "OpenNR: pipeline error %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String);
    return ps;
}

void RunMetalNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                const float* const p_Srcs[7], float* p_Dst)
{
    id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
    id<MTLDevice> device = queue.device;

    QueueResources res;
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        QueueResources& r = s_resources[p_CmdQ];

        if (r.est == nil) {
            NSError* err = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
#if defined(MAC_OS_VERSION_15_0) && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeFast;
            } else
#endif
            {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }
            id<MTLLibrary> lib = [device newLibraryWithSource:@(kKernelSource) options:options error:&err];
            if (!lib) {
                fprintf(stderr, "OpenNR: Metal compile failed: %s\n", err.localizedDescription.UTF8String);
                return;
            }
            r.est  = makePipeline(device, lib, @"NoiseEstKernel");
            r.fin  = makePipeline(device, lib, @"FinalizeStatsKernel");
            r.temp = makePipeline(device, lib, @"TemporalKernel");
            r.est2 = makePipeline(device, lib, @"ResidualEstKernel");
            r.fin2 = makePipeline(device, lib, @"FinalizeResidualKernel");
            r.deep = makePipeline(device, lib, @"DeepCleanKernel");
            r.expE = makePipeline(device, lib, @"ExposureEstKernel");
            r.expF = makePipeline(device, lib, @"FinalizeExposureKernel");
            r.nlm  = makePipeline(device, lib, @"SpatialNLMKernel");
            if (!r.est || !r.fin || !r.temp || !r.est2 || !r.fin2 || !r.deep ||
                !r.expE || !r.expF || !r.nlm)
                return;
        }

        if (r.stats == nil)
            r.stats = [device newBufferWithLength:(NR_STATS_UINTS * sizeof(uint32_t))
                                          options:MTLResourceStorageModePrivate];

        if (r.tmp == nil || r.w != p_Width || r.h != p_Height) {
            r.tmp = [device newBufferWithLength:(static_cast<size_t>(p_Width) * p_Height * 4 * sizeof(float))
                                        options:MTLResourceStorageModePrivate];
            r.tmp2 = nil;   // re-created lazily at the new size when needed
            r.w = p_Width;
            r.h = p_Height;
        }
        // v3.3 Deep Clean writes a second frame-sized buffer; allocate it
        // only once the pass is actually used
        const bool wantDeep = (p_Params.deepClean != 0) && (p_Params.enableSpatial != 0) &&
                              (p_Params.master > 0.0f);
        if (wantDeep && r.tmp2 == nil)
            r.tmp2 = [device newBufferWithLength:(static_cast<size_t>(p_Width) * p_Height * 4 * sizeof(float))
                                         options:MTLResourceStorageModePrivate];
        res = r;
    }

    NRParams params = p_Params;
    const float* partnerPtr = p_Srcs[3];
    if (p_Srcs[2] != p_Srcs[3])      partnerPtr = p_Srcs[2];
    else if (p_Srcs[4] != p_Srcs[3]) partnerPtr = p_Srcs[4];
    params.hasTemporalDiff = (partnerPtr != p_Srcs[3]) ? 1 : 0;

    id<MTLBuffer> src[7];
    for (int i = 0; i < 7; ++i)
        src[i] = reinterpret_cast<id<MTLBuffer> >(const_cast<float*>(p_Srcs[i]));
    id<MTLBuffer> partner = reinterpret_cast<id<MTLBuffer> >(const_cast<float*>(partnerPtr));
    id<MTLBuffer> dst = reinterpret_cast<id<MTLBuffer> >(p_Dst);

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    cmdBuf.label = @"OpenNR";

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

    // v3.5 P1: the exposure histograms accumulate per frame and live past
    // the full-buffer zero's gate, so their range zeroes whenever the
    // temporal stack runs (manual profiles skip everything else)
    const int reachHost = (params.enableTemporal == 0) ? 0
                        : ((params.temporalFrames >= 7) ? 3 : (params.temporalFrames >= 5) ? 2 : 1);
    if (residualLive || reachHost >= 1) {
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        if (residualLive)
            [blit fillBuffer:res.stats range:NSMakeRange(0, NR_STATS_UINTS * sizeof(uint32_t)) value:0];
        else
            [blit fillBuffer:res.stats
                       range:NSMakeRange(NR_STATS_HIST_EXP * sizeof(uint32_t),
                                         (NR_STATS_UINTS - NR_STATS_HIST_EXP) * sizeof(uint32_t))
                       value:0];
        [blit endEncoding];
    }

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    const MTLSize tg = MTLSizeMake(16, 16, 1);
    int W = p_Width, H = p_Height;
    const MTLSize gridFull = MTLSizeMake((p_Width + tg.width - 1) / tg.width,
                                         (p_Height + tg.height - 1) / tg.height, 1);
    const MTLSize gridHalf = MTLSizeMake((p_Width / 2 + tg.width - 1) / tg.width,
                                         (p_Height / 2 + tg.height - 1) / tg.height, 1);

    if (inputLive) {
        [enc setComputePipelineState:res.est];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        [enc setBuffer:src[3] offset:0 atIndex:3];
        [enc setBuffer:partner offset:0 atIndex:4];
        [enc setBuffer:res.stats offset:0 atIndex:5];
        [enc dispatchThreadgroups:gridHalf threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [enc setComputePipelineState:res.fin];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBuffer:res.stats offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // v3.5 P1: per-neighbour exposure offsets (one small dispatch each,
    // then the single-thread medians) — the temporal kernel reads them
    if (reachHost >= 1) {
        const MTLSize gridQ = MTLSizeMake((p_Width / 4 + tg.width) / tg.width,
                                          (p_Height / 4 + tg.height) / tg.height, 1);
        [enc setComputePipelineState:res.expE];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        for (int i = 0; i < 7; ++i)
            [enc setBuffer:src[i] offset:0 atIndex:(3 + i)];
        [enc setBuffer:res.stats offset:0 atIndex:10];
        [enc dispatchThreadgroups:gridQ threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:res.expF];
        [enc setBuffer:res.stats offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    {
        [enc setComputePipelineState:res.temp];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        for (int i = 0; i < 7; ++i)
            [enc setBuffer:src[i] offset:0 atIndex:(3 + i)];
        [enc setBuffer:res.stats offset:0 atIndex:10];
        [enc setBuffer:res.tmp offset:0 atIndex:11];
        [enc dispatchThreadgroups:gridFull threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    if (residualLive) {
        [enc setComputePipelineState:res.est2];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        [enc setBuffer:res.tmp offset:0 atIndex:3];
        [enc setBuffer:res.stats offset:0 atIndex:4];
        [enc dispatchThreadgroups:gridHalf threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [enc setComputePipelineState:res.fin2];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBuffer:res.stats offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    // v3.3 Deep Clean: fine pre-pass on the PASS-1 residual, then re-zero
    // the residual histogram slots and re-measure on the cleaned buffer so
    // the main pass adapts to what is actually left (see nr_core.h).
    const bool deep = (params.deepClean != 0) && (params.enableSpatial != 0) &&
                      (params.master > 0.0f) && (res.tmp2 != nil);
    id<MTLBuffer> working = deep ? res.tmp2 : res.tmp;

    if (deep) {
        [enc setComputePipelineState:res.deep];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        [enc setBuffer:res.tmp offset:0 atIndex:3];
        [enc setBuffer:res.stats offset:0 atIndex:4];
        [enc setBuffer:res.tmp2 offset:0 atIndex:5];
        [enc dispatchThreadgroups:gridFull threadsPerThreadgroup:tg];
    }

    if (deep && residualLive) {
        // the residual slots hold pass-1 counts — blit encoders cannot
        // interleave with compute, so close, zero the two histogram ranges
        // (NOT the sigma/gain slots between them), and reopen
        [enc endEncoding];
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        // v3.3 B5 layout: ALL residual histograms are contiguous before the
        // sigma slots — one range zeroes them all
        [blit fillBuffer:res.stats
                   range:NSMakeRange(NR_STATS_HIST_YR * sizeof(uint32_t),
                                     (NR_STATS_SIGMA_SY - NR_STATS_HIST_YR) * sizeof(uint32_t))
                   value:0];
        [blit endEncoding];
        enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:res.est2];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        [enc setBuffer:res.tmp2 offset:0 atIndex:3];
        [enc setBuffer:res.stats offset:0 atIndex:4];
        [enc dispatchThreadgroups:gridHalf threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [enc setComputePipelineState:res.fin2];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBuffer:res.stats offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    } else if (deep) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    {
        [enc setComputePipelineState:res.nlm];
        [enc setBytes:&params length:sizeof(NRParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        [enc setBuffer:working offset:0 atIndex:3];
        [enc setBuffer:src[3] offset:0 atIndex:4];
        [enc setBuffer:res.stats offset:0 atIndex:5];
        [enc setBuffer:dst offset:0 atIndex:6];
        [enc setBuffer:res.tmp offset:0 atIndex:7];   // the TRUE temporal result
        // v3.3 A2: threadgroup tile for the fine-band loop — sized for the
        // actual radius (same formula as the kernel; 17.3 KB at R=10 max)
        const int Rsp = params.spatialRadius < 1 ? 1 : (params.spatialRadius > 10 ? 10 : params.spatialRadius);
        const int tileW = 16 + 2 * (Rsp + 1);
        [enc setThreadgroupMemoryLength:(3 * sizeof(float) * tileW * tileW) atIndex:0];
        [enc dispatchThreadgroups:gridFull threadsPerThreadgroup:tg];
    }

    [enc endEncoding];
    [cmdBuf commit];
}
