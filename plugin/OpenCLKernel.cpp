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

#define H_YF    0
#define H_CF    256
#define H_Y2    512
#define H_C2    768
#define H_YT    1024
#define H_CT    1280
#define L_Y     1536
#define L_C     2560
#define H_YR    3584
#define H_CR    3840
#define H_EN    4096
#define S_SY    4128
#define S_SC    4129
#define S_TY    4130
#define S_TC    4131
#define S_RY    4132
#define S_RC    4133
#define S_MED   4134
#define S_HMAX  4135
#define S_GY    4136
#define S_GC    4152
#define S_ENMED 4168

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

    atomic_inc(&stats[H_YF + clamp((int)(fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)]);
    atomic_inc(&stats[H_CF + clamp((int)(fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)]);
    atomic_inc(&stats[H_CF + clamp((int)(fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)]);

    const int lb = clamp((int)(Y[4] * kLumaBins), 0, kLumaBins - 1);
    atomic_inc(&stats[L_Y + lb * kLumaSub + clamp((int)(fabs(lapY)  * kLumaSubScaleY), 0, kLumaSub - 1)]);
    atomic_inc(&stats[L_C + lb * kLumaSub + clamp((int)(fabs(lapCb) * kLumaSubScaleC), 0, kLumaSub - 1)]);
    atomic_inc(&stats[L_C + lb * kLumaSub + clamp((int)(fabs(lapCr) * kLumaSubScaleC), 0, kLumaSub - 1)]);

    if (p.hasTemporalDiff != 0) {
        const float3 pv = sampleYCC(partner, W, H, x, y);
        atomic_inc(&stats[H_YT + clamp((int)(fabs(pv.x - Y[4])  * kHistScaleY), 0, kHistBins - 1)]);
        atomic_inc(&stats[H_CT + clamp((int)(fabs(pv.y - Cb[4]) * kHistScaleC), 0, kHistBins - 1)]);
        atomic_inc(&stats[H_CT + clamp((int)(fabs(pv.z - Cr[4]) * kHistScaleC), 0, kHistBins - 1)]);
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
        atomic_inc(&stats[H_Y2 + clamp((int)(fabs(lY)  * kHistScaleY), 0, kHistBins - 1)]);
        atomic_inc(&stats[H_C2 + clamp((int)(fabs(lCb) * kHistScaleC), 0, kHistBins - 1)]);
        atomic_inc(&stats[H_C2 + clamp((int)(fabs(lCr) * kHistScaleC), 0, kHistBins - 1)]);
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
    float sc = clamp(p.sigmaC, kSigmaMin, kSigmaMax);
    float ty = sy, tc = sc;
    uint medBin = 0;

    float gy[16], gc[16];
    for (int b = 0; b < 16; ++b) { gy[b] = 1.0f; gc[b] = 1.0f; }

    if (totalF >= 64) {
        const QuantRes mYf = histQuantile(stats, H_YF, kHistBins, totalF, kHistScaleY, 1, 2);
        medBin = mYf.bin;
        const float syFine = mYf.value * kMedianCal;
        const float scFine = histQuantile(stats, H_CF, kHistBins, totalF * 2, kHistScaleC, 1, 2).value * kMedianCal;
        const float syCoarse = 2.0f * histQuantile(stats, H_Y2, kHistBins, total2, kHistScaleY, 1, 2).value * kMedianCal;
        const float scCoarse = 2.0f * histQuantile(stats, H_C2, kHistBins, total2 * 2, kHistScaleC, 1, 2).value * kMedianCal;

        const float lapSY = fmax(syFine, 0.9f * syCoarse);
        const float lapSC = fmax(scFine, 0.9f * scCoarse);

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

        const float adj = clamp(p.profileAdjust, 0.25f, 4.0f);
        sy = clamp(fmax(lapSY, 0.85f * ty) * adj, kSigmaMin, kSigmaMax);
        sc = clamp(fmax(lapSC, 0.85f * tc) * adj, kSigmaMin, kSigmaMax);
        ty = clamp(ty * adj, kSigmaMin, kSigmaMax);
        tc = clamp(tc * adj, kSigmaMin, kSigmaMax);

        const float q35RefY = histQuantile(stats, H_YF, kHistBins, totalF, kHistScaleY, 7, 20).value * kQ35Cal;
        const float q35RefC = histQuantile(stats, H_CF, kHistBins, totalF * 2, kHistScaleC, 7, 20).value * kQ35Cal;
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

    stats[S_SY] = as_uint(sy);
    stats[S_SC] = as_uint(sc);
    stats[S_TY] = as_uint(ty);
    stats[S_TC] = as_uint(tc);
    stats[S_MED] = medBin;
    stats[S_HMAX] = hmax;
    for (int b = 0; b < kLumaBins; ++b) {
        const int b0 = clamp(b - 1, 0, kLumaBins - 1);
        const int b1 = clamp(b + 1, 0, kLumaBins - 1);
        stats[S_GY + b] = as_uint(0.25f * gy[b0] + 0.5f * gy[b] + 0.25f * gy[b1]);
        stats[S_GC + b] = as_uint(0.25f * gc[b0] + 0.5f * gc[b] + 0.25f * gc[b1]);
    }
}

inline float4 loadSigmasIn(NRParams p, __global const uint* stats)
{
    if (p.profileSource != 2)
        return (float4)(as_float(stats[S_SY]), as_float(stats[S_SC]),
                        as_float(stats[S_TY]), as_float(stats[S_TC]));
    const float sy = clamp(p.sigmaY, kSigmaMin, kSigmaMax);
    const float sc = clamp(p.sigmaC, kSigmaMin, kSigmaMax);
    return (float4)(sy, sc, sy, sc);
}

__kernel void TemporalKernel(NRParams p, int W, int H,
                             __global const float4* f0, __global const float4* f1,
                             __global const float4* f2, __global const float4* f3,
                             __global const float4* f4,
                             __global const uint* stats, __global float4* tmp)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H)
        return;

    const float4 sig = loadSigmasIn(p, stats);
    const float mLow  = fmin(p.master, 1.0f);
    const float mHigh = fmax(p.master, 1.0f);
    const float tL = clamp(p.temporalLuma   * mLow, 0.0f, 1.0f);
    const float tC = clamp(p.temporalChroma * mLow, 0.0f, 1.0f);
    const float thrMul = 0.4f + 2.6f * clamp(p.motionThresh, 0.0f, 1.0f)
                       + 0.8f * (mHigh - 1.0f);
    const int reach = (p.enableTemporal == 0) ? 0 : ((p.temporalFrames >= 5) ? 2 : 1);
    const float loY = kAbsDiffBias * sig.z, hiY = loY + thrMul * sig.z;
    const float loC = kAbsDiffBias * sig.w, hiC = loC + thrMul * sig.w;
    const float invSpanY = 1.0f / (hiY - loY);
    const float invSpanC = 1.0f / (hiC - loC);

    float3 c9[9];
    int i = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx, ++i)
            c9[i] = sampleYCC(f2, W, H, x + dx, y + dy);

    float accY = c9[4].x, accCb = c9[4].y, accCr = c9[4].z;
    float sumWY = 1.0f, sumWY2 = 1.0f, sumWC = 1.0f;

    for (int k = 2 - reach; k <= 2 + reach; ++k) {
        if (k == 2)
            continue;
        __global const float4* f = (k == 0) ? f0 : ((k == 1) ? f1 : ((k == 3) ? f3 : f4));

        float dY = 0.0f, dC = 0.0f;
        float3 fc = (float3)(0.0f);
        i = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx, ++i) {
                const float3 v = sampleYCC(f, W, H, x + dx, y + dy);
                if (i == 4) fc = v;
                dY += fabs(v.x - c9[i].x);
                dC += 0.5f * (fabs(v.y - c9[i].y) + fabs(v.z - c9[i].z));
            }
        }
        dY *= (1.0f / 9.0f);
        dC *= (1.0f / 9.0f);

        const float gY = 1.0f - smooth01f((dY - loY) * invSpanY);
        const float gC = 1.0f - smooth01f((dC - loC) * invSpanC);
        const float wY = tL * gY;
        const float wC = tC * gC * gY;

        accY  += wY * fc.x;
        accCb += wC * fc.y;
        accCr += wC * fc.z;
        sumWY  += wY;
        sumWY2 += wY * wY;
        sumWC  += wC;
    }

    tmp[y * W + x] = (float4)(accY / sumWY, accCb / sumWC, accCr / sumWC,
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
    atomic_inc(&stats[H_YR + clamp((int)(fabs(lapY)  * kHistScaleY), 0, kHistBins - 1)]);
    atomic_inc(&stats[H_CR + clamp((int)(fabs(lapCb) * kHistScaleC), 0, kHistBins - 1)]);
    atomic_inc(&stats[H_CR + clamp((int)(fabs(lapCr) * kHistScaleC), 0, kHistBins - 1)]);
    const float effN = sampleTmp(tmp, W, H, x, y).w;
    atomic_inc(&stats[H_EN + clamp((int)((effN - 1.0f) * 8.0f), 0, 31)]);
}

__kernel void FinalizeResidualKernel(NRParams p, __global uint* stats)
{
    if (get_global_id(0) != 0 || get_global_id(1) != 0)
        return;

    ulong total = 0;
    for (int b = 0; b < kHistBins; ++b)
        total += stats[H_YR + b];

    const float sy = as_float(stats[S_SY]);
    const float sc = as_float(stats[S_SC]);
    float ry = sy, rc = sc, enmed = 1.0f;

    if (total >= 64) {
        const float adj = clamp(p.profileAdjust, 0.25f, 4.0f);
        ry = histQuantile(stats, H_YR, kHistBins, total, kHistScaleY, 1, 2).value * kMedianCal * adj;
        rc = histQuantile(stats, H_CR, kHistBins, total * 2, kHistScaleC, 1, 2).value * kMedianCal * adj;
        enmed = 1.0f + histQuantile(stats, H_EN, 32, total, 8.0f, 1, 2).value;
        const float floorY = 0.5f * sy / sqrt(fmax(1.0f, enmed));
        const float floorC = 0.5f * sc / sqrt(fmax(1.0f, enmed));
        ry = clamp(fmax(ry, floorY), kSigmaMin, sy > kSigmaMin ? sy : kSigmaMax);
        rc = clamp(fmax(rc, floorC), kSigmaMin, sc > kSigmaMin ? sc : kSigmaMax);
    }

    stats[S_RY] = as_uint(ry);
    stats[S_RC] = as_uint(rc);
    stats[S_ENMED] = as_uint(enmed);
}

// glyph order: 0-9 . % A C E F I L M O P R S T U Y B D G N + space
__constant ulong kFont[32] = {
    0x3a33ae62eUL, 0x11842108eUL, 0x3a213221fUL, 0x3a213062eUL, 0x08ca97c42UL, 0x7e1e0862eUL,
    0x3a10f462eUL, 0x7c2222108UL, 0x3a317462eUL, 0x3a317842eUL, 0x00000018cUL, 0x632222263UL,
    0x3a31fc631UL, 0x3a308422eUL, 0x7e10f421fUL, 0x7e10f4210UL, 0x38842108eUL, 0x42108421fUL,
    0x4775ac631UL, 0x3a318c62eUL, 0x7a31f4210UL, 0x7a31f5251UL, 0x3e107043eUL, 0x7c8421084UL,
    0x46318c62eUL, 0x462a21084UL, 0x7a31f463eUL, 0x7a318c63eUL, 0x3a30bc62fUL, 0x47359c631UL,
    0x0084f9080UL, 0x000000000UL,
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

__constant int kLabIY[4]  = { G_I, G_N, G_SP, G_Y };
__constant int kLabIC[4]  = { G_I, G_N, G_SP, G_C };
__constant int kLabRY[10] = { G_R, G_E, G_S, G_I, G_D, G_U, G_A, G_L, G_SP, G_Y };
__constant int kLabRC[10] = { G_R, G_E, G_S, G_I, G_D, G_U, G_A, G_L, G_SP, G_C };
__constant int kLabOFF[3] = { G_O, G_F, G_F };
__constant int kLabEFFN[5] = { G_E, G_F, G_F, G_SP, G_N };
__constant int kLabGAIN[4] = { G_G, G_A, G_I, G_N };
__constant int kLabDB[2]   = { G_D, G_B };
__constant float kDirX[8] = { 1, 0, -1, 0, 0.7071f, -0.7071f, -0.7071f, 0.7071f };
__constant float kDirY[8] = { 0, 1, 0, -1, 0.7071f, 0.7071f, -0.7071f, -0.7071f };

inline bool glyphPixel(int glyph, int gx, int gy)
{
    if (glyph < 0 || glyph > 31 || gx < 0 || gx >= 5 || gy < 0 || gy >= 7)
        return false;
    return (kFont[glyph] >> (34 - (gy * 5 + gx))) & 1UL;
}

inline bool textPixelC(__constant int* chars, int n, int tx, int ty, int lx, int ly)
{
    if (ly < ty || ly >= ty + 7 || lx < tx || lx >= tx + n * 6)
        return false;
    const int ci = (lx - tx) / 6;
    return glyphPixel(chars[ci], (lx - tx) - ci * 6, ly - ty);
}

inline bool textPixelP(const int* chars, int n, int tx, int ty, int lx, int ly)
{
    if (ly < ty || ly >= ty + 7 || lx < tx || lx >= tx + n * 6)
        return false;
    const int ci = (lx - tx) / 6;
    return glyphPixel(chars[ci], (lx - tx) - ci * 6, ly - ty);
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

inline bool hudPixel(int x, int y, int W, int H,
                     float sy, float sc, float ry, float rc, float enmed,
                     uint medBin, uint hmax, __global const uint* stats,
                     int enableTemporal, float3* rgb)
{
    const int yd = H - 1 - y;   // OFX buffers are bottom-up; panel anchors top-left on screen
    const int s = max(1, H / 540);
    const int ox = 16 * s, oy = 16 * s;
    const int lw = 320, lh = 198;
    if (x < ox || yd < oy || x >= ox + lw * s || yd >= oy + lh * s)
        return false;
    const int lx = (x - ox) / s;
    const int ly = (yd - oy) / s;

    *rgb = *rgb * 0.20f + (float3)(0.015f, 0.015f, 0.02f);
    if (lx == 0 || ly == 0 || lx == lw - 1 || ly == lh - 1) {
        *rgb = (float3)(0.35f);
        return true;
    }

    const float sig[4] = { sy, sc, ry, rc };
    const int rowY[4] = { 6, 28, 50, 72 };

    for (int row = 0; row < 4; ++row) {
        const int ty0 = rowY[row];
        bool lit = false;
        if (row == 0) lit = textPixelC(kLabIY, 4,  8, ty0, lx, ly);
        if (row == 1) lit = textPixelC(kLabIC, 4,  8, ty0, lx, ly);
        if (row == 2) lit = textPixelC(kLabRY, 10, 8, ty0, lx, ly);
        if (row == 3) lit = textPixelC(kLabRC, 10, 8, ty0, lx, ly);
        if (!lit) {
            int vg[6];
            pctGlyphs(sig[row] * 100.0f, vg);
            lit = textPixelP(vg, 6, 252, ty0, lx, ly);
        }
        if (lit) { *rgb = (float3)(1.0f); return true; }

        if (ly >= ty0 + 9 && ly < ty0 + 13 && lx >= 8 && lx < 288) {
            const float fill = clamp(sig[row] / 0.08f, 0.0f, 1.0f);
            const bool on = (lx - 8) < (int)(fill * 280.0f);
            const bool residRow = (row >= 2);
            if (on) *rgb = residRow ? (float3)(0.95f, 0.65f, 0.20f) : (float3)(0.20f, 0.65f, 0.95f);
            else    *rgb = (float3)(0.16f);
            return true;
        }
    }

    {
        const int ty0 = 94;
        bool lit = textPixelC(kLabEFFN, 5, 8, ty0, lx, ly);
        if (!lit) {
            int vg[3];
            dec1Glyphs(enableTemporal ? enmed : 1.0f, vg);
            lit = textPixelP(vg, 3, 44, ty0, lx, ly);
        }
        if (!lit) lit = textPixelC(kLabGAIN, 4, 120, ty0, lx, ly);
        if (!lit) {
            const float gainDb = clamp(20.0f * log10(fmax(sy, 1e-5f) / fmax(ry, 1e-5f)), 0.0f, 40.0f);
            int vg[4];
            vg[0] = G_PLUS;
            int d1[3];
            dec1Glyphs(gainDb, d1);
            vg[1] = d1[0]; vg[2] = d1[1]; vg[3] = d1[2];
            lit = textPixelP(vg, 4, 150, ty0, lx, ly);
            if (!lit) lit = textPixelC(kLabDB, 2, 178, ty0, lx, ly);
        }
        if (lit) { *rgb = (float3)(1.0f); return true; }
        if (enableTemporal == 0 && textPixelC(kLabOFF, 3, 220, ty0, lx, ly)) { *rgb = (float3)(0.7f); return true; }
    }

    if (lx >= 8 && lx < 264 && ly >= 108 && ly < 148) {
        const int bin = clamp((lx - 8) / 16, 0, kLumaBins - 1);
        const float gain = as_float(stats[S_GY + bin]);
        const float v = clamp((gain - 0.6f) / 1.6f, 0.0f, 1.0f);
        const bool bar = (147 - ly) < (int)(v * 39.0f + 0.5f);
        const bool ref = (147 - ly) == (int)((1.0f - 0.6f) / 1.6f * 39.0f + 0.5f);
        if (bar)      *rgb = (float3)(0.20f, 0.65f, 0.95f);
        else if (ref) *rgb = (float3)(0.32f);
        else          *rgb = (float3)(0.10f);
        return true;
    }

    if (lx >= 8 && lx < 268 && ly >= 156 && ly < 192) {
        const int bin = clamp((lx - 8) * kHistBins / 260, 0, kHistBins - 1);
        const float hgt = 35.0f * (float)stats[H_YF + bin] / (float)max(hmax, 1u);
        const bool bar = (191 - ly) < (int)(hgt + 0.5f);
        if (bin == (int)medBin)  *rgb = (float3)(0.95f, 0.85f, 0.15f);
        else if (bar)            *rgb = (float3)(0.55f);
        else                     *rgb = (float3)(0.10f);
        return true;
    }

    return true;
}

__kernel void SpatialNLMKernel(NRParams p, int W, int H,
                               __global const float4* tmp, __global const float4* curr,
                               __global const uint* stats, __global float4* dst)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H)
        return;

    const int manual = (p.profileSource == 2);
    const float sy = manual ? clamp(p.sigmaY, kSigmaMin, kSigmaMax) : as_float(stats[S_SY]);
    const float sc = manual ? clamp(p.sigmaC, kSigmaMin, kSigmaMax) : as_float(stats[S_SC]);
    const float ryG = manual ? sy : as_float(stats[S_RY]);
    const float rcG = manual ? sc : as_float(stats[S_RC]);
    const float enmed = manual ? 1.0f : as_float(stats[S_ENMED]);

    const float mLow  = fmin(p.master, 1.0f);
    const float mHigh = fmax(p.master, 1.0f);
    const float hBoost = pow(mHigh, 1.2f);

    const float sL = clamp(p.spatialLuma, 0.0f, 1.0f);
    const float sC = clamp(p.spatialChroma, 0.0f, 1.0f);
    const float aY = (p.enableSpatial == 0) ? 0.0f : clamp(sL * mLow, 0.0f, 1.0f);
    const float aC = (p.enableSpatial == 0) ? 0.0f : clamp(sC * mLow, 0.0f, 1.0f);
    const float hMulY = (0.6f + 1.4f * pow(sL, 1.5f)) * hBoost;
    const float hMulC = (0.6f + 1.4f * pow(sC, 1.5f)) * hBoost;
    const float pd = clamp(p.preserveDetail, 0.0f, 1.0f);
    const int   R  = clamp(p.spatialRadius, 1, 8);
    const int   nlm = (p.spatialMode == 1);
    const int   runSpatial = (p.enableSpatial != 0) && (aY > 0.0f || aC > 0.0f);
    const float spatialSigma = fmax(1.0f, (float)R / 1.5f);
    const float invSpatial2 = 1.0f / (2.0f * spatialSigma * spatialSigma);

    const float blotch = (p.enableSpatial != 0) ? clamp(p.chromaBlotch, 0.0f, 1.0f) * mLow : 0.0f;
    const int   Rb = 2 + (int)(14.0f * clamp(p.chromaBlotch, 0.0f, 1.0f));

    const int refine = (p.enableRefine != 0) && (p.master > 0.0f);
    const float desat = refine ? clamp(p.shadowDesat, 0.0f, 1.0f) : 0.0f;
    const float desatRange = fmax(0.02f, p.desatRange);
    const float tex = refine ? clamp(p.lumaTexture, 0.0f, 1.0f) * mLow : 0.0f;
    const float grainAmt = refine ? clamp(p.grainAmount, 0.0f, 1.0f) * 0.06f : 0.0f;
    const float grainSize = clamp(p.grainSize, 0.5f, 6.0f);
    const float grainCh = clamp(p.grainChroma, 0.0f, 1.0f);
    const uint frame = (uint)p.frameIndex;

    const float4 tc = tmp[y * W + x];
    const int lb = clamp((int)(tc.x * kLumaBins), 0, kLumaBins - 1);
    const float gainYv = manual ? 1.0f : as_float(stats[S_GY + lb]);
    const float gainCv = manual ? 1.0f : as_float(stats[S_GC + lb]);
    const float sigY = clamp(ryG * gainYv, 1e-5f, 1.0f);
    const float sigC = clamp(rcG * gainCv, 1e-5f, 1.0f);

    float Yo = tc.x, Cbo = tc.y, Cro = tc.z;

    if (runSpatial) {
        float3 pPatch[9];
        {
            int i = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx, ++i)
                    pPatch[i] = sampleTmp(tmp, W, H, x + dx, y + dy).xyz;
        }

        float mean = 0.0f, m2 = 0.0f;
        for (int i = 0; i < 9; ++i) { mean += pPatch[i].x; m2 += pPatch[i].x * pPatch[i].x; }
        mean *= (1.0f / 9.0f);
        const float var = fmax(0.0f, m2 * (1.0f / 9.0f) - mean * mean);
        const float edginess = clamp(sqrt(fmax(var - sigY * sigY, 0.0f)) / (3.0f * sigY), 0.0f, 1.0f);

        const float hY = kNlmHLuma   * sigY * hMulY * (1.0f - pd * 0.85f * edginess);
        const float hC = kNlmHChroma * sigC * hMulC * (1.0f - pd * 0.50f * edginess);
        const float invHY2 = 1.0f / fmax(hY * hY, 1e-12f);
        const float invHC2 = 1.0f / fmax(hC * hC, 1e-12f);
        const float biasY = 2.0f * sigY * sigY;
        const float biasC = 2.0f * sigC * sigC;

        float accY = 0.0f, accCb = 0.0f, accCr = 0.0f;
        float sumWY = 0.0f, sumWC = 0.0f, wYmax = 0.0f, wCmax = 0.0f;

        for (int dy = -R; dy <= R; ++dy) {
            for (int dx = -R; dx <= R; ++dx) {
                if (dx == 0 && dy == 0)
                    continue;
                const float3 ts = sampleTmp(tmp, W, H, x + dx, y + dy).xyz;

                float dY2, dC2;
                if (nlm) {
                    dY2 = 0.0f; dC2 = 0.0f;
                    int i = 0;
                    for (int qy = -1; qy <= 1; ++qy) {
                        for (int qx = -1; qx <= 1; ++qx, ++i) {
                            const float3 tq = sampleTmp(tmp, W, H, x + dx + qx, y + dy + qy).xyz;
                            const float3 e = pPatch[i] - tq;
                            dY2 += e.x * e.x;
                            dC2 += 0.5f * (e.y * e.y + e.z * e.z);
                        }
                    }
                    dY2 *= (1.0f / 9.0f);
                    dC2 *= (1.0f / 9.0f);
                } else {
                    const float3 e = tc.xyz - ts;
                    dY2 = e.x * e.x;
                    dC2 = 0.5f * (e.y * e.y + e.z * e.z);
                }

                dY2 = fmax(0.0f, dY2 - biasY);
                dC2 = fmax(0.0f, dC2 - biasC);

                float wY = exp(-dY2 * invHY2);
                float wC = exp(-dC2 * invHC2) * exp(-dY2 * invHY2 * 0.25f);
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

        Yo  = tc.x + aY * (Yf  - tc.x);
        Cbo = tc.y + aC * (Cbf - tc.y);
        Cro = tc.z + aC * (Crf - tc.z);
    }

    if (blotch > 0.0f) {
        const float gyDen = 1.0f / fmax(2.0f * sigY * hBoost, 1e-6f);
        const float gcDen = 1.0f / fmax(3.0f * sigC * hBoost, 1e-6f);
        float accB = tc.y, accR = tc.z, sumW = 1.0f;
        for (int d = 0; d < 8; ++d) {
            for (int ri = 1; ri <= 3; ++ri) {
                const float rr = Rb * ((float)ri / 3.0f);
                const float3 ts = sampleTmp(tmp, W, H,
                                            x + (int)(kDirX[d] * rr + (kDirX[d] > 0 ? 0.5f : -0.5f)),
                                            y + (int)(kDirY[d] * rr + (kDirY[d] > 0 ? 0.5f : -0.5f))).xyz;
                const float eY = (ts.x - tc.x) * gyDen;
                const float eC = (0.5f * (fabs(ts.y - tc.y) + fabs(ts.z - tc.z))) * gcDen;
                const float w = exp(-(eY * eY + eC * eC));
                accB += w * ts.y;
                accR += w * ts.z;
                sumW += w;
            }
        }
        Cbo += blotch * (accB / sumW - Cbo);
        Cro += blotch * (accR / sumW - Cro);
    }

    const float4 c = curr[y * W + x];
    const float3 cin = rgb2ycc(c.xyz);

    float Yr = Yo, Cbr = Cbo, Crr = Cro;
    if (refine) {
        const float sat = 1.0f - desat * (1.0f - smooth01f(Yr / desatRange));
        Cbr *= sat;
        Crr *= sat;
        Yr += tex * (cin.x - Yr);
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

    float3 rgb = ycc2rgb((float3)(Yr, Cbr, Crr));
    const float3 dn = ycc2rgb((float3)(Yo, Cbo, Cro));
    float4 o = (float4)(rgb, c.w);

    if (p.viewMode == 1) {
        if (x < W / 2) o.xyz = c.xyz;
        if (abs(x - W / 2) <= 1) o.xyz = (float3)(1.0f);
    } else if (p.viewMode == 2) {
        o.xyz = c.xyz;
    } else if (p.viewMode == 3) {
        o.xyz = ycc2rgb(tc.xyz);
    } else if (p.viewMode == 4) {
        o.xyz = 0.5f + (c.xyz - dn) * 4.0f;
    } else if (p.viewMode == 5) {
        float3 hud = rgb;
        if (!hudPixel(x, y, W, H, sy, sc, ryG, rcG, enmed,
                      stats[S_MED], stats[S_HMAX], stats, p.enableTemporal, &hud)) {
            if (p.profileSource == 1) {
                const float rHalf = 0.5f * p.regionSize * (float)min(W, H);
                const float cx = p.regionCX * W, cyy = p.regionCY * H;
                const float ax = fabs((float)x - cx), ay = fabs((float)y - cyy);
                const int onEdge = (ax <= rHalf && ay <= rHalf) &&
                                   (ax >= rHalf - 2.0f || ay >= rHalf - 2.0f);
                if (onEdge) hud = (float3)(1.0f, 1.0f, 0.1f);
            }
        }
        o.xyz = hud;
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
    cl_kernel nlm = nullptr;
    cl_mem tmp = nullptr;
    cl_mem stats = nullptr;
    int w = 0, h = 0;
};

} // namespace

void RunOpenCLNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                 const float* const p_Srcs[5], float* p_Dst)
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
            r.nlm  = clCreateKernel(program, "SpatialNLMKernel", &error);
            CheckError(error, "Unable to create kernels");
        }

        if (!r.tmp || r.w != p_Width || r.h != p_Height)
        {
            if (r.tmp)   clReleaseMemObject(r.tmp);
            if (r.stats) clReleaseMemObject(r.stats);
            r.tmp = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                                   static_cast<size_t>(p_Width) * p_Height * 4 * sizeof(float), NULL, &error);
            r.stats = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                                     NR_STATS_UINTS * sizeof(cl_uint), NULL, &error);
            CheckError(error, "Unable to create buffers");
            r.w = p_Width;
            r.h = p_Height;
        }
        res = r;
    }
    s_locker.Unlock();

    if (!res.est || !res.tmp)
        return;

    NRParams params = p_Params;
    const float* partner = p_Srcs[2];
    if (p_Srcs[1] != p_Srcs[2])      partner = p_Srcs[1];
    else if (p_Srcs[3] != p_Srcs[2]) partner = p_Srcs[3];
    params.hasTemporalDiff = (partner != p_Srcs[2]) ? 1 : 0;

    int W = p_Width, H = p_Height;
    const bool autoProfile = (params.profileSource != 2);
    size_t global[2]  = { static_cast<size_t>(W), static_cast<size_t>(H) };
    size_t globalH[2] = { static_cast<size_t>((W + 1) / 2), static_cast<size_t>((H + 1) / 2) };
    size_t one[2] = { 1, 1 };

    if (autoProfile)
    {
        const cl_uint zero = 0;
        error = clEnqueueFillBuffer(cmdQ, res.stats, &zero, sizeof(cl_uint), 0,
                                    NR_STATS_UINTS * sizeof(cl_uint), 0, NULL, NULL);
        CheckError(error, "Unable to zero stats");

        int c = 0;
        error  = clSetKernelArg(res.est, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.est, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.est, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.est, c++, sizeof(cl_mem), &p_Srcs[2]);
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

    {
        int c = 0;
        error  = clSetKernelArg(res.temp, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.temp, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.temp, c++, sizeof(int), &H);
        for (int i = 0; i < 5; ++i)
            error |= clSetKernelArg(res.temp, c++, sizeof(cl_mem), &p_Srcs[i]);
        error |= clSetKernelArg(res.temp, c++, sizeof(cl_mem), &res.stats);
        error |= clSetKernelArg(res.temp, c++, sizeof(cl_mem), &res.tmp);
        CheckError(error, "temporal args");
        error = clEnqueueNDRangeKernel(cmdQ, res.temp, 2, NULL, global, NULL, 0, NULL, NULL);
        CheckError(error, "temporal enqueue");
    }

    if (autoProfile)
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

    {
        int c = 0;
        error  = clSetKernelArg(res.nlm, c++, sizeof(NRParams), &params);
        error |= clSetKernelArg(res.nlm, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.nlm, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.nlm, c++, sizeof(cl_mem), &res.tmp);
        error |= clSetKernelArg(res.nlm, c++, sizeof(cl_mem), &p_Srcs[2]);
        error |= clSetKernelArg(res.nlm, c++, sizeof(cl_mem), &res.stats);
        error |= clSetKernelArg(res.nlm, c++, sizeof(cl_mem), &p_Dst);
        CheckError(error, "nlm args");
        error = clEnqueueNDRangeKernel(cmdQ, res.nlm, 2, NULL, global, NULL, 0, NULL, NULL);
        CheckError(error, "nlm enqueue");
    }
}
