// Speak — Metal implementation of the film-reconstruction pipeline (macOS).
// Line-by-line port of plugin/speak_core.h; keep the two in sync. Verified by
// test/test_speak_metal.mm against the CPU reference (parity ~2e-5 mean).

#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>
#include <cstdio>

#include "SpeakParams.h"

static const char* kSpeakKernelSource = R"MSL(

#include <metal_stdlib>
using namespace metal;

// ---- struct layout: IDENTICAL to SpeakParams.h (all 4-byte fields) ----
typedef struct SpeakProfile
{
    float negDmin[3];   float negDmax[3];   float negGamma[3];
    float negToe[3];    float negShoulder[3]; float negSpeed[3];
    float printerLights[3]; float printerMaster;
    float prnDmin[3];   float prnDmax[3];   float prnGamma[3];
    float prnToe[3];    float prnShoulder[3]; float prnSpeed[3];
    float dyeCouple[9]; float subSat[3];    float subSatKnee[3];
    float splitShadow[3]; float splitHigh[3]; float splitPivot; float splitBalance;
    float halAmount;    float halRadius;   float halThresh;
    float systemGamma;  int residualLUT;    int profileVersion; int _pad0;
} SpeakProfile;

typedef struct SpeakParams
{
    int inputColorSpace; int outputMode; int grainRef; float strength;
    int frameIndex; int viewMode;
    int enableTone; int enableDye; int enableSplit; int enableOptics;
    int scopeHD; int scopeDensity; int scopeVector;
    SpeakProfile profile;
} SpeakParams;

constant float kLog10_2 = 0.301029996f;
constant float k18Gray  = 0.18f;
constant float kPrinterPt = 0.025f;
constant float kLinTiny = 1e-8f;
constant float kKneeMin = 0.05f;
constant float kDI_A = 0.0075f;
constant float kDI_B = 7.0f;
constant float kDI_C = 0.07329248f;
constant float kDI_M = 10.44426855f;
constant float kDI_LIN_CUT = 0.00262409f;
constant float kDI_LOG_CUT = 0.02740668f;

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float pow10f(float x) { return exp2(x * 3.32192809f); }

inline float softplusf(float z)
{
    float az = z < 0.0f ? -z : z;
    return (z > 0.0f ? z : 0.0f) + log(1.0f + exp(-az));
}

inline float diDecode(float v)
{
    return (v <= kDI_LOG_CUT) ? (v / kDI_M) : (exp2(v / kDI_C - kDI_B) - kDI_A);
}
inline float diEncode(float L)
{
    return (L <= kDI_LIN_CUT) ? (L * kDI_M) : ((log2(L + kDI_A) + kDI_B) * kDI_C);
}
inline float acesCctDecode(float v)
{
    if (v <= 0.155251141552511f) return (v - 0.0729055341958355f) / 10.5402377416545f;
    return exp2(v * 17.52f - 9.72f);
}
inline float acesCctEncode(float L)
{
    if (L <= 0.0078125f) return 10.5402377416545f * L + 0.0729055341958355f;
    return (log2(L) + 9.72f) / 17.52f;
}
inline float decodeToLinear(int cs, float v)
{
    if (cs == 0) return diDecode(v);           // DWG/DI
    if (cs == 1) return v <= 0.0f ? 0.0f : pow(v, 2.4f); // Rec.709 g2.4
    if (cs == 3) return acesCctDecode(v);      // ACEScct
    return v;                                   // DWG-linear / linear
}
inline float encodeFromLinear(int cs, float L)
{
    if (cs == 0) return diEncode(L);
    if (cs == 1) return L <= 0.0f ? 0.0f : pow(L, 1.0f / 2.4f);
    if (cs == 3) return acesCctEncode(L);
    return L;
}

constant float kDWG_to_XYZ[9] = {
    0.70062239f, 0.14877482f, 0.10105872f,
    0.27411851f, 0.87363190f,-0.14775041f,
   -0.09896291f,-0.13789533f, 1.32591599f };
constant float kXYZ_to_Rec709[9] = {
    3.24045420f,-1.53713850f,-0.49853140f,
   -0.96926600f, 1.87601080f, 0.04155600f,
    0.05564340f,-0.20402590f, 1.05722520f };
inline void mul3(constant float* m, float r, float g, float b,
                 thread float& oR, thread float& oG, thread float& oB)
{
    oR = m[0] * r + m[1] * g + m[2] * b;
    oG = m[3] * r + m[4] * g + m[5] * b;
    oB = m[6] * r + m[7] * g + m[8] * b;
}
inline void gamutToRec709Lin(int cs, float r, float g, float b,
                             thread float& oR, thread float& oG, thread float& oB)
{
    if (cs == 0 || cs == 2) {
        float X, Y, Z;
        mul3(kDWG_to_XYZ, r, g, b, X, Y, Z);
        mul3(kXYZ_to_Rec709, X, Y, Z, oR, oG, oB);
    } else {
        oR = r; oG = g; oB = b;
    }
}

inline float hdCurve(float logH, float Dmin, float Dmax, float gamma,
                     float toe, float shoulder, float speed)
{
    float t = toe < kKneeMin ? kKneeMin : toe;
    float s = shoulder < kKneeMin ? kKneeMin : shoulder;
    float d1 = Dmin + (gamma / t) * softplusf(t * (logH - speed));
    return Dmax - (1.0f / s) * softplusf(s * (Dmax - d1));
}
inline float chainDensity(float stops, int ch, constant SpeakProfile& p)
{
    float logH = stops * kLog10_2;
    float Dneg = hdCurve(logH, p.negDmin[ch], p.negDmax[ch], p.negGamma[ch],
                         p.negToe[ch], p.negShoulder[ch], p.negSpeed[ch]);
    float printerOff = (p.printerMaster + p.printerLights[ch]) * kPrinterPt;
    float logHprn = -Dneg + printerOff;
    return hdCurve(logHprn, p.prnDmin[ch], p.prnDmax[ch], p.prnGamma[ch],
                   p.prnToe[ch], p.prnShoulder[ch], p.prnSpeed[ch]);
}
inline float toneChannel(float lin, int ch, constant SpeakProfile& p)
{
    float stops = log2((lin < kLinTiny ? kLinTiny : lin) / k18Gray);
    float Dprn = chainDensity(stops, ch, p);
    float Dref = chainDensity(0.0f, ch, p);
    return k18Gray * pow10f(-(Dprn - Dref));
}
#define SPEAK_EXP_BINS       128
#define SPEAK_STATS_HIST_EXP 0
#define SPEAK_STATS_HIST_MAX 128
#define SPEAK_WF_COLS        128
#define SPEAK_WF_ROWS        96
#define SPEAK_WF_DMAX        3.0f
#define SPEAK_STATS_WF       129
#define SPEAK_STATS_WF_MAX   (129 + 36864)
#define SPEAK_HAL_MAXLEV     14
#define SPEAK_HAL_MINDIM     8
#define SPEAK_HAL_SIGMA_MIN  0.05f

inline int wfColOf(int x, int W)
{
    int c = x * SPEAK_WF_COLS / (W > 0 ? W : 1);
    return c < 0 ? 0 : (c >= SPEAK_WF_COLS ? SPEAK_WF_COLS - 1 : c);
}
inline int wfRowOf(float D)
{
    int r = int(D / SPEAK_WF_DMAX * SPEAK_WF_ROWS);
    return r < 0 ? 0 : (r >= SPEAK_WF_ROWS ? SPEAK_WF_ROWS - 1 : r);
}
inline int wfIdx(int ch, int col, int row)
{
    return SPEAK_STATS_WF + ch * (SPEAK_WF_COLS * SPEAK_WF_ROWS) + col * SPEAK_WF_ROWS + row;
}

inline int expBinOf(float stops)
{
    int b = int((stops + 6.0f) / 12.0f * SPEAK_EXP_BINS);
    return b < 0 ? 0 : (b >= SPEAK_EXP_BINS ? SPEAK_EXP_BINS - 1 : b);
}
inline float pixelStops(int cs, float r, float g, float b)
{
    float m = (decodeToLinear(cs, r) + decodeToLinear(cs, g) + decodeToLinear(cs, b)) * (1.0f / 3.0f);
    return log2((m < kLinTiny ? kLinTiny : m) / k18Gray);
}

inline float density10(float lin)
{
    return -log2(lin < 1e-6f ? 1e-6f : lin) * kLog10_2;
}
inline float softCapKnee(float d, float cap)
{
    if (cap <= 0.0f) return d;
    return cap - (1.0f / 8.0f) * softplusf(8.0f * (cap - d));
}
inline void subtractiveColor(float r, float g, float b, constant SpeakProfile& p,
                             thread float& oR, thread float& oG, thread float& oB)
{
    float DR = density10(r), DG = density10(g), DB = density10(b);
    float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    float devR = DR - Dbar, devG = DG - Dbar, devB = DB - Dbar;
    float cR = (1.0f + p.subSat[0]) * devR - (p.dyeCouple[1] * devG + p.dyeCouple[2] * devB);
    float cG = (1.0f + p.subSat[1]) * devG - (p.dyeCouple[3] * devR + p.dyeCouple[5] * devB);
    float cB = (1.0f + p.subSat[2]) * devB - (p.dyeCouple[6] * devR + p.dyeCouple[7] * devG);
    float DpR = softCapKnee(Dbar + cR, p.subSatKnee[0]);
    float DpG = softCapKnee(Dbar + cG, p.subSatKnee[1]);
    float DpB = softCapKnee(Dbar + cB, p.subSatKnee[2]);
    oR = pow10f(-DpR); oG = pow10f(-DpG); oB = pow10f(-DpB);
}
inline bool dyeActive(constant SpeakProfile& p)
{
    return p.subSat[0] != 0.0f || p.subSat[1] != 0.0f || p.subSat[2] != 0.0f ||
           p.dyeCouple[1] != 0.0f || p.dyeCouple[2] != 0.0f || p.dyeCouple[3] != 0.0f ||
           p.dyeCouple[5] != 0.0f || p.dyeCouple[6] != 0.0f || p.dyeCouple[7] != 0.0f;
}

inline float smooth01(float t) { t = clampf(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
inline void splitWeights(float Dbar, constant SpeakProfile& p, thread float& wShadow, thread float& wHigh)
{
    float grayD  = 0.744727f;
    float pivotD = grayD - p.splitPivot * kLog10_2;
    float halfW  = 0.25f + 1.5f * clampf(p.splitBalance, 0.0f, 1.0f);
    float x = (Dbar - pivotD) / halfW;
    wShadow = smooth01(x);
    wHigh   = smooth01(-x);
}
inline void splitTone(float r, float g, float b, constant SpeakProfile& p,
                      thread float& oR, thread float& oG, thread float& oB)
{
    float DR = density10(r), DG = density10(g), DB = density10(b);
    float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    float wS, wH;
    splitWeights(Dbar, p, wS, wH);
    oR = pow10f(-(DR + wS * p.splitShadow[0] + wH * p.splitHigh[0]));
    oG = pow10f(-(DG + wS * p.splitShadow[1] + wH * p.splitHigh[1]));
    oB = pow10f(-(DB + wS * p.splitShadow[2] + wH * p.splitHigh[2]));
}
inline bool splitActive(constant SpeakProfile& p)
{
    return p.splitShadow[0] != 0.0f || p.splitShadow[1] != 0.0f || p.splitShadow[2] != 0.0f ||
           p.splitHigh[0] != 0.0f || p.splitHigh[1] != 0.0f || p.splitHigh[2] != 0.0f;
}

// ---- HALATION (Phase 4) — see speak_core.h for the physics and the gates ----
constant float kHalWeight[3] = { 1.0f, 0.30f, 0.10f };

inline float halExcess(float lin, float thresh)
{
    float l = lin < 0.0f ? 0.0f : lin;
    return l > thresh ? (l - thresh) : 0.0f;
}
inline float halAmountOf(constant SpeakParams& pr)
{
    return (pr.enableOptics != 0) ? pr.profile.halAmount : 0.0f;
}
inline float halSigmaPx(int H, constant SpeakParams& pr)
{
    float s = pr.profile.halRadius * 0.01f * float(H);
    return s < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : s;
}
inline bool halActive(constant SpeakParams& pr)
{
    return (pr.enableTone != 0) && (pr.strength > 0.0f) && (halAmountOf(pr) > 0.0f);
}

// ---- pyramid geometry (identical arithmetic in all four backends) ----
inline int halLevelCount(int W, int H)
{
    int n = 1, w = W, h = H;
    while (n < SPEAK_HAL_MAXLEV && w > SPEAK_HAL_MINDIM && h > SPEAK_HAL_MINDIM) {
        w = (w + 1) / 2; h = (h + 1) / 2; n++;
    }
    return n;
}
inline void halLevelInfo(int W, int H, int L, thread int& lw, thread int& lh, thread int& off)
{
    int w = W, h = H, o = 0;
    for (int i = 0; i < L; ++i) { o += w * h; w = (w + 1) / 2; h = (h + 1) / 2; }
    lw = w; lh = h; off = o;
}

constant float kHalSigmaC   = 0.645497f;
constant float kHalCoreFall  = 3.0f;
constant float kHalSkirtFall = 1.0f;
inline float halLevelWeight(int L, float sigmaTarget)
{
    float s = sigmaTarget < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : sigmaTarget;
    float Lt = log2(s / kHalSigmaC);
    float d = float(L) - Lt;
    return (d <= 0.0f) ? exp2(kHalCoreFall * d) : exp2(-kHalSkirtFall * d);
}

inline float halFetch(device const float* arena, int off, int lw, int lh, int x, int y, int c)
{
    int xx = x < 0 ? 0 : (x >= lw ? lw - 1 : x);
    int yy = y < 0 ? 0 : (y >= lh ? lh - 1 : y);
    return arena[(off + yy * lw + xx) * 3 + c];
}

constant float kHalDec[4] = { 0.125f, 0.375f, 0.375f, 0.125f };
inline void halDecimatePixel(device const float* arena, int sOff, int sW, int sH,
                             int dx, int dy, thread float* out)
{
    for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i)
                acc += kHalDec[i] * kHalDec[j] *
                       halFetch(arena, sOff, sW, sH, 2 * dx - 1 + i, 2 * dy - 1 + j, c);
        out[c] = acc;
    }
}

// Cubic B-spline read of one level onto a (W,H) grid. B-spline, NOT bilinear and
// NOT Catmull-Rom: it is C2 (bilinear is only C0, and its derivative
// discontinuity at every texel boundary is what made the skirt read as hard
// rectangular BLOCKS), and its weights are non-negative and sum to 1 (so no
// undershoot below zero, and energy is preserved).
inline void halBSpline(float t, thread float* w)
{
    float t2 = t * t, t3 = t2 * t;
    w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) * (1.0f / 6.0f);
    w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) * (1.0f / 6.0f);
    w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) * (1.0f / 6.0f);
    w[3] = t3 * (1.0f / 6.0f);
}
inline float halSampleLevel(device const float* arena, int off, int lw, int lh,
                            int W, int H, int x, int y, int c)
{
    float fx = (float(x) + 0.5f) * float(lw) / float(W) - 0.5f;
    float fy = (float(y) + 0.5f) * float(lh) / float(H) - 0.5f;
    int x0 = int(floor(fx)), y0 = int(floor(fy));
    float wx[4], wy[4];
    halBSpline(fx - float(x0), wx);
    halBSpline(fy - float(y0), wy);
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        float row = 0.0f;
        for (int i = 0; i < 4; ++i)
            row += wx[i] * halFetch(arena, off, lw, lh, x0 - 1 + i, y0 - 1 + j, c);
        acc += wy[j] * row;
    }
    return acc;
}

// The mixture's total weight — the normalizer that makes the pyramid
// energy-preserving. Same loop as speak_core.h on every backend.
inline float halWeightSum(int nLev, float sigmaTarget)
{
    float wsum = 0.0f;
    for (int L = 0; L < nLev; ++L) wsum += halLevelWeight(L, sigmaTarget);
    return wsum;
}

// COARSE-TO-FINE ACCUMULATE — one pixel of level L (halAccumPixel in
// speak_core.h):  acc_L = w_L * level_L + upsample_2x(acc_{L+1}), running
// L = nLev-1 down to 0, in place in the arena. acc_0 is then the whole
// (unnormalized) mixture at full resolution.
//
// Going one octave at a time is what fixes the blocky skirt: each step
// interpolates only between ADJACENT samples and is then re-filtered by every
// step below it. It is also strictly CHEAPER — total work sum_L (level L's
// pixels) = 4/3 N, against nLev*N for the old full-res reads.
//
// IN-PLACE SAFETY: a thread reads its OWN (x,y) at level L plus a neighbourhood
// at level L+1 — a DISJOINT arena region, finished by the previous dispatch —
// and writes only its own (x,y) at level L. No thread reads another thread's
// level-L pixel, so in place is safe and no atomics are involved.
//
// The level geometry (both levels' dims + offsets, L and nLev) is passed IN from
// the host rather than recomputed from L in-kernel, exactly as SpeakDecimateKernel
// does and for the same reason: this kernel binds `arena` WRITABLE, and the
// back-to-back halLevelInfo calls were observed to misbehave under that binding.
// The host computes them with the same halLevelInfo, so the two agree by
// construction. `sigmaTarget` stays in-kernel so it cannot drift from the
// normalize pass.
struct HalAccum { int lw, lh, off, cw, ch, coff, L, nLev; };
inline void halAccumPixel(device const float* arena, constant HalAccum& lv,
                          float sigmaTarget, int x, int y, thread float* out)
{
    float wl = halLevelWeight(lv.L, sigmaTarget);
    if (lv.L >= lv.nLev - 1) {                 // the coarsest level: nothing above it
        for (int c = 0; c < 3; ++c) out[c] = wl * halFetch(arena, lv.off, lv.lw, lv.lh, x, y, c);
        return;
    }
    // the coarser level is B-spline-upsampled ONTO LEVEL L's grid: the dst dims
    // handed to halSampleLevel are (lw, lh), NOT (W, H).
    for (int c = 0; c < 3; ++c)
        out[c] = wl * halFetch(arena, lv.off, lv.lw, lv.lh, x, y, c)
               + halSampleLevel(arena, lv.coff, lv.cw, lv.ch, lv.lw, lv.lh, x, y, c);
}

inline void lookLinear(float r, float g, float b,
                       float scatR, float scatG, float scatB,
                       constant SpeakParams& pr,
                       thread float& oR, thread float& oG, thread float& oB)
{
    int cs = pr.inputColorSpace;
    float lr = decodeToLinear(cs, r);
    float lg = decodeToLinear(cs, g);
    float lb = decodeToLinear(cs, b);
    float mr = lr, mg = lg, mb = lb;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        float s = clampf(pr.strength, 0.0f, 1.0f);
        float a = halAmountOf(pr);
        // RE-EXPOSURE: the scattered light goes in the CURVE'S ARGUMENT. The dry
        // side of the mix stays lr, not the re-exposed value — see speak_core.h.
        float er = lr, eg = lg, eb = lb;
        if (a > 0.0f) {
            er = lr + a * kHalWeight[0] * scatR;
            eg = lg + a * kHalWeight[1] * scatG;
            eb = lb + a * kHalWeight[2] * scatB;
        }
        mr = lerpf(lr, toneChannel(er, 0, pr.profile), s);
        mg = lerpf(lg, toneChannel(eg, 1, pr.profile), s);
        mb = lerpf(lb, toneChannel(eb, 2, pr.profile), s);
    }
    if ((pr.enableDye != 0) && dyeActive(pr.profile)) subtractiveColor(mr, mg, mb, pr.profile, mr, mg, mb);
    if ((pr.enableSplit != 0) && splitActive(pr.profile)) splitTone(mr, mg, mb, pr.profile, mr, mg, mb);
    oR = mr; oG = mg; oB = mb;
}

inline float scopeYStops(float inStops, int ch, constant SpeakParams& pr)
{
    float lin = k18Gray * exp2(inStops);
    float outLin = lin;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        float s = clampf(pr.strength, 0.0f, 1.0f);
        outLin = lerpf(lin, toneChannel(lin, ch, pr.profile), s);
    }
    return log2((outLin < kLinTiny ? kLinTiny : outLin) / k18Gray);
}

inline bool hdScopePixel(int x, int y, int W, int H, constant SpeakParams& pr,
                         device const uint* stats,
                         thread float& outR, thread float& outG, thread float& outB)
{
    if (pr.scopeHD == 0) return false;
    int sc = (H / 540) > 1 ? (H / 540) : 1;
    int panelW = 220 * sc, panelH = 150 * sc;
    int margin = 12 * sc;
    int px0 = margin, py0 = margin;
    int yd = H - 1 - y;
    int lx = x - px0, ly = yd - py0;
    if (lx < 0 || ly < 0 || lx >= panelW || ly >= panelH) return false;

    int pad = 6 * sc;
    int plotW = panelW - 2 * pad, plotH = panelH - 2 * pad;
    int gx = lx - pad, gy = ly - pad;

    outR = 0.06f; outG = 0.06f; outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        outR = 0.30f; outG = 0.30f; outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    float rowStops = 6.0f - 12.0f * (float(gy) / (plotH - 1));
    int gcol0 = int((0.0f + 6.0f) / 12.0f * (plotW - 1) + 0.5f);
    int grow0 = int((6.0f - 0.0f) / 12.0f * (plotH - 1) + 0.5f);
    if (gx == gcol0 || gy == grow0) { outR = 0.24f; outG = 0.24f; outB = 0.24f; return true; }
    if ((gx % (plotW / 6)) == 0 || (gy % (plotH / 6)) == 0) { outR = 0.13f; outG = 0.13f; outB = 0.13f; }

    uint hmax = stats[SPEAK_STATS_HIST_MAX];
    if (hmax > 0u) {
        int hb = expBinOf(-6.0f + 12.0f * (float(gx) / (plotW - 1)));
        float f = float(stats[SPEAK_STATS_HIST_EXP + hb]) / float(hmax);
        int barH = int(sqrt(f) * (plotH * 0.45f) + 0.5f);
        if (gy >= plotH - barH) { outR = 0.16f; outG = 0.19f; outB = 0.24f; }
    }

    int chR[3]; chR[0]=1; chR[1]=0; chR[2]=0;
    int chG[3]; chG[0]=0; chG[1]=1; chG[2]=0;
    int chB[3]; chB[0]=0; chB[1]=0; chB[2]=1;
    for (int ch = 0; ch < 3; ++ch) {
        float inS  = -6.0f + 12.0f * (float(gx)     / (plotW - 1));
        float inS2 = -6.0f + 12.0f * (float(gx + 1) / (plotW - 1));
        float y0 = scopeYStops(inS,  ch, pr);
        float y1 = scopeYStops(inS2, ch, pr);
        if (y0 > y1) { float tt = y0; y0 = y1; y1 = tt; }
        float lo = y1 < y0 ? y1 : y0, hi = y1 > y0 ? y1 : y0;
        if (rowStops <= hi + 0.09f && rowStops >= lo - 0.09f) {
            outR = 0.10f + 0.85f * chR[ch];
            outG = 0.10f + 0.85f * chG[ch];
            outB = 0.10f + 0.85f * chB[ch];
            return true;
        }
    }
    if (gy >= plotH - 5 * sc && gy < plotH - 1 * sc) {
        int sw = gx / (6 * sc);
        if (gx % (6 * sc) < 4 * sc) {
            if (sw == 0) { outR = 0.95f; outG = 0.10f; outB = 0.10f; return true; }
            if (sw == 1) { outR = 0.10f; outG = 0.95f; outB = 0.10f; return true; }
            if (sw == 2) { outR = 0.10f; outG = 0.10f; outB = 0.95f; return true; }
        }
    }
    return true;
}

inline bool densityScopePixel(int x, int y, int W, int H, constant SpeakParams& pr,
                              device const uint* stats,
                              thread float& outR, thread float& outG, thread float& outB)
{
    if (pr.scopeDensity == 0) return false;

    int sc = (H / 540) > 1 ? (H / 540) : 1;
    int panelW = 220 * sc, panelH = 150 * sc;
    int margin = 12 * sc;
    int px0 = W - margin - panelW, py0 = margin;
    int yd = H - 1 - y;
    int lx = x - px0, ly = yd - py0;
    if (lx < 0 || ly < 0 || lx >= panelW || ly >= panelH) return false;

    int pad = 6 * sc;
    int plotW = panelW - 2 * pad, plotH = panelH - 2 * pad;
    int gx = lx - pad, gy = ly - pad;

    outR = 0.06f; outG = 0.06f; outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        outR = 0.30f; outG = 0.30f; outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    int rowGray = int(0.744727f / SPEAK_WF_DMAX * (plotH - 1) + 0.5f);
    if (gy == 0) { outR = 0.35f; outG = 0.35f; outB = 0.35f; return true; }
    if (gy == rowGray) { outR = 0.30f; outG = 0.26f; outB = 0.12f; return true; }

    int chW = plotW / 3;
    int ch = gx / (chW > 0 ? chW : 1);
    if (ch > 2) ch = 2;
    if (gx - ch * chW == 0 && ch > 0) { outR = 0.16f; outG = 0.16f; outB = 0.16f; return true; }

    uint wmax = stats[SPEAK_STATS_WF_MAX];
    if (wmax > 0u) {
        int within = gx - ch * chW;
        int wcol = within * SPEAK_WF_COLS / (chW > 0 ? chW : 1);
        int wrow = gy * SPEAK_WF_ROWS / plotH;
        uint c = stats[wfIdx(ch, wcol < SPEAK_WF_COLS ? wcol : SPEAK_WF_COLS - 1,
                             wrow < SPEAK_WF_ROWS ? wrow : SPEAK_WF_ROWS - 1)];
        if (c > 0u) {
            float inten = sqrt(float(c) / float(wmax));
            float v = 0.12f + 0.88f * (inten > 1.0f ? 1.0f : inten);
            outR = (ch == 0) ? v : 0.05f;
            outG = (ch == 1) ? v : 0.05f;
            outB = (ch == 2) ? v : 0.05f;
        }
    }
    return true;
}

inline void deliverInput(constant SpeakParams& pr, float r, float g, float b,
                         thread float& oR, thread float& oG, thread float& oB)
{
    if (pr.outputMode == 1) {
        int cs = pr.inputColorSpace;
        float lr = decodeToLinear(cs, r);
        float lg = decodeToLinear(cs, g);
        float lb = decodeToLinear(cs, b);
        float rr, rg, rb;
        gamutToRec709Lin(cs, lr, lg, lb, rr, rg, rb);
        rr = rr < 0.0f ? 0.0f : rr;
        rg = rg < 0.0f ? 0.0f : rg;
        rb = rb < 0.0f ? 0.0f : rb;
        oR = encodeFromLinear(1, rr);
        oG = encodeFromLinear(1, rg);
        oB = encodeFromLinear(1, rb);
    } else {
        oR = r; oG = g; oB = b;
    }
}

inline void processPixel(float r, float g, float b,
                         float scatR, float scatG, float scatB,
                         int x, int y, int W, int H,
                         constant SpeakParams& pr, device const uint* stats,
                         thread float& outR, thread float& outG, thread float& outB)
{
    int cs = pr.inputColorSpace;
    bool toneOn = (pr.enableTone != 0) && (pr.strength > 0.0f);
    bool dyeOn = (pr.enableDye != 0) && dyeActive(pr.profile);
    bool splitOn = (pr.enableSplit != 0) && splitActive(pr.profile);
    bool bake = (pr.outputMode == 1);
    if (!toneOn && !dyeOn && !splitOn && !bake) {
        outR = r; outG = g; outB = b;
    } else {
        float mr, mg, mb;
        lookLinear(r, g, b, scatR, scatG, scatB, pr, mr, mg, mb);
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, mr, mg, mb, rr, rg, rb);
            rr = rr < 0.0f ? 0.0f : rr;
            rg = rg < 0.0f ? 0.0f : rg;
            rb = rb < 0.0f ? 0.0f : rb;
            outR = encodeFromLinear(1, rr);
            outG = encodeFromLinear(1, rg);
            outB = encodeFromLinear(1, rb);
        } else {
            outR = encodeFromLinear(cs, mr);
            outG = encodeFromLinear(cs, mg);
            outB = encodeFromLinear(cs, mb);
        }
    }
    if (pr.viewMode == 2 || (pr.viewMode == 1 && x < W / 2))
        deliverInput(pr, r, g, b, outR, outG, outB);

    // Isolated-scatter view: the ACTUAL injected re-exposure a * w_c * S_c,
    // delivered through the SAME output transform as the picture. Deliberately
    // NOT auto-normalized — see speak_core.h.
    if (pr.viewMode == 3) {
        float a = halAmountOf(pr);
        float on = ((pr.enableTone != 0) && (pr.strength > 0.0f)) ? 1.0f : 0.0f;
        float sr = on * a * kHalWeight[0] * scatR;
        float sg = on * a * kHalWeight[1] * scatG;
        float sb = on * a * kHalWeight[2] * scatB;
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, sr, sg, sb, rr, rg, rb);
            sr = rr < 0.0f ? 0.0f : rr;
            sg = rg < 0.0f ? 0.0f : rg;
            sb = rb < 0.0f ? 0.0f : rb;
            outR = encodeFromLinear(1, sr);
            outG = encodeFromLinear(1, sg);
            outB = encodeFromLinear(1, sb);
        } else {
            outR = encodeFromLinear(cs, sr);
            outG = encodeFromLinear(cs, sg);
            outB = encodeFromLinear(cs, sb);
        }
    }

    float sr, sg, sb;
    if (hdScopePixel(x, y, W, H, pr, stats, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
    if (densityScopePixel(x, y, W, H, pr, stats, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
}

// The scatter pyramid, mirroring buildHalScatter in speak_core.h. Four passes:
//   excess (full res, level 0)
//   -> decimate  (one dispatch per level, L = 1..nLev-1, fine to coarse)
//   -> accum     (one dispatch per level, L = nLev-1..0, coarse to fine, in place)
//   -> normalize (full res)
// Atomics-free by construction: the decimation is order-independent and the
// accumulate is per-pixel with a disjoint read of the level above.

// Level 0 of the arena: the per-channel scene-linear highlight excess.
// THRESHOLD BEFORE DECIMATION — mean(max(0, l-t)) != max(0, mean(l)-t).
kernel void SpeakExcessKernel(constant SpeakParams& p [[buffer(0)]],
                              constant int& W [[buffer(1)]],
                              constant int& H [[buffer(2)]],
                              device const float* src [[buffer(3)]],
                              device float* arena [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= W || (int)gid.y >= H) return;
    int x = int(gid.x), y = int(gid.y);
    int cs = p.inputColorSpace;
    float th = p.profile.halThresh;
    int i = (y * W + x) * 4;
    int o = (y * W + x) * 3;
    arena[o + 0] = halExcess(decodeToLinear(cs, src[i + 0]), th);
    arena[o + 1] = halExcess(decodeToLinear(cs, src[i + 1]), th);
    arena[o + 2] = halExcess(decodeToLinear(cs, src[i + 2]), th);
}

// One octave: level L-1 -> level L. Dispatched once per level, on level L's grid.
//
// The source/dest geometry is passed IN rather than recomputed from L in-kernel.
// That mirrors buildHalScatter, which likewise hoists halLevelInfo out of the
// pixel loop, and it keeps the only writable-arena kernel free of the two
// back-to-back halLevelInfo calls whose results a writable `arena` binding was
// observed to disturb (levels above the threadgroup size came out reading the
// wrong octave; the read-only debug twin of this kernel computed them
// correctly). Host and kernel therefore agree by construction.
struct HalLevel { int sw, sh, so, dw, dh, doff; };
kernel void SpeakDecimateKernel(constant HalLevel& lv [[buffer(0)]],
                                device float* arena [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= lv.dw || (int)gid.y >= lv.dh) return;
    int x = int(gid.x), y = int(gid.y);
    float v[3];
    halDecimatePixel(arena, lv.so, lv.sw, lv.sh, x, y, v);
    int o = (lv.doff + y * lv.dw + x) * 3;
    arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
}

// One octave of the coarse-to-fine accumulate, IN PLACE in the arena.
// Dispatched once per level, from nLev-1 down to 0, on THAT LEVEL's grid, with a
// barrier between levels: level L reads level L+1, written by the previous
// dispatch. Mirrors buildHalScatter's accumulate loop.
kernel void SpeakAccumKernel(constant SpeakParams& p [[buffer(0)]],
                             constant int& H [[buffer(1)]],
                             constant HalAccum& lv [[buffer(2)]],
                             device float* arena [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= lv.lw || (int)gid.y >= lv.lh) return;
    int x = int(gid.x), y = int(gid.y);
    float sig = halSigmaPx(H, p);
    float v[3];
    halAccumPixel(arena, lv, sig, x, y, v);
    int o = (lv.off + y * lv.lw + x) * 3;
    arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
}

// Normalize level 0 of the accumulated arena into the scatter plane:
// sum(w) = 1 => energy preserved. `inv` is recomputed here rather than passed in
// from the host so it uses the very same in-kernel halLevelWeight the accumulate
// used — a host-side sum through a different exp2/log2 would drift the whole
// scatter field by a scale parity would then charge to the pyramid.
kernel void SpeakNormalizeKernel(constant SpeakParams& p [[buffer(0)]],
                                 constant int& W [[buffer(1)]],
                                 constant int& H [[buffer(2)]],
                                 device const float* arena [[buffer(3)]],
                                 device float* scat [[buffer(4)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= W || (int)gid.y >= H) return;
    int x = int(gid.x), y = int(gid.y);
    int nLev = halLevelCount(W, H);
    float sig = halSigmaPx(H, p);
    float inv = 1.0f / halWeightSum(nLev, sig);
    int o = (y * W + x) * 3;
    scat[o + 0] = arena[o + 0] * inv;
    scat[o + 1] = arena[o + 1] * inv;
    scat[o + 2] = arena[o + 2] * inv;
}

// Scope measurement pass: bin the frame on a stride-2 grid. Integer atomics are
// order-independent, so the counts are identical on every backend.
//
// `scat` is only DEREFERENCED when the host actually built it (same condition as
// speakFrame's `hal`); when halation is skipped a small placeholder buffer is
// bound, so the guard is load-bearing, not just a mirror of the core's null test.
kernel void SpeakStatsKernel(constant SpeakParams& p [[buffer(0)]],
                             constant int& W [[buffer(1)]],
                             constant int& H [[buffer(2)]],
                             device const float* src [[buffer(3)]],
                             device atomic_uint* stats [[buffer(4)]],
                             device const float* scat [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]])
{
    int x = int(gid.x) * 2, y = int(gid.y) * 2;
    if (x >= W || y >= H) return;
    int i = (y * W + x) * 4;
    int j = (y * W + x) * 3;
    if (p.scopeHD != 0) {
        int bin = expBinOf(pixelStops(p.inputColorSpace, src[i + 0], src[i + 1], src[i + 2]));
        atomic_fetch_add_explicit(&stats[SPEAK_STATS_HIST_EXP + bin], 1u, memory_order_relaxed);
    }
    if (p.scopeDensity != 0) {
        bool hal = halActive(p) || (p.viewMode == 3);
        float sR = 0.0f, sG = 0.0f, sB = 0.0f;
        if (hal) { sR = scat[j + 0]; sG = scat[j + 1]; sB = scat[j + 2]; }
        float mr, mg, mb;
        lookLinear(src[i + 0], src[i + 1], src[i + 2], sR, sG, sB, p, mr, mg, mb);
        int col = wfColOf(x, W);
        atomic_fetch_add_explicit(&stats[wfIdx(0, col, wfRowOf(density10(mr)))], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[wfIdx(1, col, wfRowOf(density10(mg)))], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[wfIdx(2, col, wfRowOf(density10(mb)))], 1u, memory_order_relaxed);
    }
}

kernel void SpeakStatsMaxKernel(device uint* stats [[buffer(0)]])
{
    uint mx = 0u;
    for (int b = 0; b < SPEAK_EXP_BINS; ++b)
        if (stats[SPEAK_STATS_HIST_EXP + b] > mx) mx = stats[SPEAK_STATS_HIST_EXP + b];
    stats[SPEAK_STATS_HIST_MAX] = mx;
    uint wmx = 0u;
    for (int k = 0; k < SPEAK_WF_COLS * SPEAK_WF_ROWS * 3; ++k)
        if (stats[SPEAK_STATS_WF + k] > wmx) wmx = stats[SPEAK_STATS_WF + k];
    stats[SPEAK_STATS_WF_MAX] = wmx;
}

kernel void SpeakKernel(constant SpeakParams& p [[buffer(0)]],
                        constant int& W [[buffer(1)]],
                        constant int& H [[buffer(2)]],
                        device const float* src [[buffer(3)]],
                        device float* dst [[buffer(4)]],
                        device const uint* stats [[buffer(5)]],
                        device const float* scat [[buffer(6)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= W || (int)gid.y >= H) return;
    int x = int(gid.x), y = int(gid.y);
    int i = (y * W + x) * 4;
    int j = (y * W + x) * 3;
    bool hal = halActive(p) || (p.viewMode == 3);
    float sR = 0.0f, sG = 0.0f, sB = 0.0f;
    if (hal) { sR = scat[j + 0]; sG = scat[j + 1]; sB = scat[j + 2]; }
    float oR, oG, oB;
    processPixel(src[i + 0], src[i + 1], src[i + 2], sR, sG, sB, x, y, W, H, p, stats, oR, oG, oB);
    dst[i + 0] = oR; dst[i + 1] = oG; dst[i + 2] = oB; dst[i + 3] = src[i + 3];
}
)MSL";

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// ---- halation host mirrors of speak_core.h (kept textually parallel) ----
// Per-level geometry handed to SpeakDecimateKernel / SpeakAccumKernel (layout
// must match the HalLevel / HalAccum structs declared in the MSL source above;
// all fields 4 bytes).
struct HalLevel { int sw, sh, so, dw, dh, doff; };
struct HalAccum { int lw, lh, off, cw, ch, coff, L, nLev; };

static int halLevelCount(int W, int H)
{
    int n = 1, w = W, h = H;
    while (n < SPEAK_HAL_MAXLEV && w > SPEAK_HAL_MINDIM && h > SPEAK_HAL_MINDIM) {
        w = (w + 1) / 2; h = (h + 1) / 2; n++;
    }
    return n;
}
static void halLevelInfo(int W, int H, int L, int& lw, int& lh, int& off)
{
    int w = W, h = H, o = 0;
    for (int i = 0; i < L; ++i) { o += w * h; w = (w + 1) / 2; h = (h + 1) / 2; }
    lw = w; lh = h; off = o;
}
static int halArenaPixels(int W, int H)
{
    int lw, lh, off;
    const int n = halLevelCount(W, H);
    halLevelInfo(W, H, n, lw, lh, off);   // offset just past the last level
    return off;
}
static float halAmountOf(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) ? pr.profile.halAmount : 0.0f;
}
static bool halActive(const SpeakParams& pr)
{
    return (pr.enableTone != 0) && (pr.strength > 0.0f) && (halAmountOf(pr) > 0.0f);
}

struct SpeakRes {
    id<MTLComputePipelineState> main = nil;
    id<MTLComputePipelineState> stats = nil;
    id<MTLComputePipelineState> statsMax = nil;
    id<MTLComputePipelineState> excess = nil;
    id<MTLComputePipelineState> decimate = nil;
    id<MTLComputePipelineState> accum = nil;
    id<MTLComputePipelineState> normalize = nil;
    id<MTLBuffer> statsBuf = nil;
    // The scatter buffers are SIZE-DEPENDENT (unlike statsBuf, whose layout is
    // fixed): the host hands us proxy and full-res frames through the SAME queue,
    // so the allocated length must be tracked and grown or the first
    // proxy->full-res switch is a buffer overrun.
    id<MTLBuffer> arenaBuf = nil;
    size_t arenaFloats = 0;
    id<MTLBuffer> scatBuf = nil;
    size_t scatFloats = 0;
    // Bound in place of scatBuf when the whole chain is skipped, so the stats and
    // main kernels always get a VALID binding (a null binding crashes). Its
    // contents are never read: the kernels guard the load on the same condition
    // the host uses to skip.
    id<MTLBuffer> nullBuf = nil;
};
static std::mutex s_speakMutex;
static std::unordered_map<void*, SpeakRes> s_speakPipe;

void RunMetalSpeak(void* p_CmdQ, int p_Width, int p_Height,
                   const SpeakParams& p_Params, const float* p_Src, float* p_Dst)
{
    id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
    id<MTLDevice> device = queue.device;

    SpeakRes res;
    {
        std::lock_guard<std::mutex> lock(s_speakMutex);
        SpeakRes& r = s_speakPipe[p_CmdQ];
        if (r.main == nil) {
            NSError* err = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
#if defined(MAC_OS_VERSION_15_0) && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0
            if (@available(macOS 15.0, *)) { options.mathMode = MTLMathModeFast; } else
#endif
            {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }
            id<MTLLibrary> lib = [device newLibraryWithSource:@(kSpeakKernelSource) options:options error:&err];
            if (!lib) {
                fprintf(stderr, "Speak: Metal compile failed: %s\n", err.localizedDescription.UTF8String);
                return;
            }
            id<MTLFunction> fn = [lib newFunctionWithName:@"SpeakKernel"];
            r.main = [device newComputePipelineStateWithFunction:fn error:&err];
            id<MTLFunction> fs = [lib newFunctionWithName:@"SpeakStatsKernel"];
            r.stats = [device newComputePipelineStateWithFunction:fs error:&err];
            id<MTLFunction> fm = [lib newFunctionWithName:@"SpeakStatsMaxKernel"];
            r.statsMax = [device newComputePipelineStateWithFunction:fm error:&err];
            id<MTLFunction> fe = [lib newFunctionWithName:@"SpeakExcessKernel"];
            r.excess = [device newComputePipelineStateWithFunction:fe error:&err];
            id<MTLFunction> fd = [lib newFunctionWithName:@"SpeakDecimateKernel"];
            r.decimate = [device newComputePipelineStateWithFunction:fd error:&err];
            id<MTLFunction> fa = [lib newFunctionWithName:@"SpeakAccumKernel"];
            r.accum = [device newComputePipelineStateWithFunction:fa error:&err];
            id<MTLFunction> fnm = [lib newFunctionWithName:@"SpeakNormalizeKernel"];
            r.normalize = [device newComputePipelineStateWithFunction:fnm error:&err];
            if (!r.main || !r.stats || !r.statsMax ||
                !r.excess || !r.decimate || !r.accum || !r.normalize) {
                fprintf(stderr, "Speak: pipeline failed\n"); return;
            }
        }
        if (r.statsBuf == nil)
            r.statsBuf = [device newBufferWithLength:(SPEAK_STATS_UINTS * sizeof(uint32_t))
                                             options:MTLResourceStorageModePrivate];
        if (r.nullBuf == nil)
            r.nullBuf = [device newBufferWithLength:(4 * sizeof(float))
                                            options:MTLResourceStorageModePrivate];
        // Skip the whole scatter chain when halation is inactive (mirrors
        // speakFrame's `hal`): the identity path stays bit-exact AND free.
        const bool wantHalAlloc = halActive(p_Params) || (p_Params.viewMode == SPEAK_VIEW_SCATTER);
        if (wantHalAlloc) {
            const size_t needArena = static_cast<size_t>(halArenaPixels(p_Width, p_Height)) * 3;
            const size_t needScat = static_cast<size_t>(p_Width) * p_Height * 3;
            if (r.arenaBuf == nil || r.arenaFloats < needArena) {
                r.arenaBuf = [device newBufferWithLength:(needArena * sizeof(float))
                                                 options:MTLResourceStorageModePrivate];
                r.arenaFloats = needArena;
            }
            if (r.scatBuf == nil || r.scatFloats < needScat) {
                r.scatBuf = [device newBufferWithLength:(needScat * sizeof(float))
                                                options:MTLResourceStorageModePrivate];
                r.scatFloats = needScat;
            }
            if (!r.arenaBuf || !r.scatBuf) { fprintf(stderr, "Speak: scatter alloc failed\n"); return; }
        }
        res = r;
    }

    SpeakParams params = p_Params;
    int W = p_Width, H = p_Height;
    id<MTLBuffer> src = reinterpret_cast<id<MTLBuffer> >(const_cast<float*>(p_Src));
    id<MTLBuffer> dst = reinterpret_cast<id<MTLBuffer> >(p_Dst);

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    cmdBuf.label = @"Speak";

    // Measure the frame only when a scope is actually showing it.
    const bool wantStats = (params.scopeHD != 0) || (params.scopeDensity != 0);
    {
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit fillBuffer:res.statsBuf range:NSMakeRange(0, SPEAK_STATS_UINTS * sizeof(uint32_t)) value:0];
        [blit endEncoding];
    }

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    const MTLSize tg = MTLSizeMake(16, 16, 1);

    // The scatter pyramid must exist BEFORE the stats pass — the density parade
    // measures the HALATED result (a scatter-blind scope is a bug parity cannot
    // catch, because all four backends would agree on the same wrong parade).
    const bool wantHal = halActive(params) || (params.viewMode == SPEAK_VIEW_SCATTER);
    id<MTLBuffer> scatBind = wantHal ? res.scatBuf : res.nullBuf;
    if (wantHal) {
        const MTLSize full = MTLSizeMake((p_Width + 15) / 16, (p_Height + 15) / 16, 1);
        [enc setComputePipelineState:res.excess];
        [enc setBytes:&params length:sizeof(SpeakParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        [enc setBuffer:src offset:0 atIndex:3];
        [enc setBuffer:res.arenaBuf offset:0 atIndex:4];
        [enc dispatchThreadgroups:full threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        const int nLev = halLevelCount(W, H);
        for (int L = 1; L < nLev; ++L) {
            HalLevel lv;
            halLevelInfo(W, H, L - 1, lv.sw, lv.sh, lv.so);
            halLevelInfo(W, H, L,     lv.dw, lv.dh, lv.doff);
            [enc setComputePipelineState:res.decimate];
            [enc setBytes:&lv length:sizeof(HalLevel) atIndex:0];
            [enc setBuffer:res.arenaBuf offset:0 atIndex:1];
            // the grid is THIS level's dims
            [enc dispatchThreadgroups:MTLSizeMake((lv.dw + 15) / 16, (lv.dh + 15) / 16, 1)
                threadsPerThreadgroup:tg];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];   // level L-1 -> L
        }

        // Coarse-to-fine accumulate, in place, one octave at a time — one
        // dispatch per level on THAT LEVEL's grid. The barrier between levels is
        // REQUIRED: level L reads the neighbourhood of level L+1 that the
        // previous dispatch just wrote.
        for (int L = nLev - 1; L >= 0; --L) {
            HalAccum lv;
            halLevelInfo(W, H, L, lv.lw, lv.lh, lv.off);
            if (L < nLev - 1) halLevelInfo(W, H, L + 1, lv.cw, lv.ch, lv.coff);
            else              { lv.cw = 0; lv.ch = 0; lv.coff = 0; }   // unread: no level above
            lv.L = L; lv.nLev = nLev;
            [enc setComputePipelineState:res.accum];
            [enc setBytes:&params length:sizeof(SpeakParams) atIndex:0];
            [enc setBytes:&H length:sizeof(int) atIndex:1];
            [enc setBytes:&lv length:sizeof(HalAccum) atIndex:2];
            [enc setBuffer:res.arenaBuf offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((lv.lw + 15) / 16, (lv.lh + 15) / 16, 1)
                threadsPerThreadgroup:tg];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];   // level L+1 -> L
        }

        [enc setComputePipelineState:res.normalize];
        [enc setBytes:&params length:sizeof(SpeakParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        [enc setBuffer:res.arenaBuf offset:0 atIndex:3];
        [enc setBuffer:res.scatBuf offset:0 atIndex:4];
        [enc dispatchThreadgroups:full threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    if (wantStats) {
        [enc setComputePipelineState:res.stats];
        [enc setBytes:&params length:sizeof(SpeakParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        [enc setBuffer:src offset:0 atIndex:3];
        [enc setBuffer:res.statsBuf offset:0 atIndex:4];
        [enc setBuffer:scatBind offset:0 atIndex:5];
        // ceil(W/2) sample threads (the kernel indexes x = gid*2). This form
        // already over-covered and was never short, but state the intent the
        // same way the other two backends now do — they WERE short by one column
        // at W = 1 (mod 32).
        const MTLSize gh = MTLSizeMake(((p_Width + 1) / 2 + 15) / 16,
                                       ((p_Height + 1) / 2 + 15) / 16, 1);
        [enc dispatchThreadgroups:gh threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [enc setComputePipelineState:res.statsMax];
        [enc setBuffer:res.statsBuf offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    [enc setComputePipelineState:res.main];
    [enc setBytes:&params length:sizeof(SpeakParams) atIndex:0];
    [enc setBytes:&W length:sizeof(int) atIndex:1];
    [enc setBytes:&H length:sizeof(int) atIndex:2];
    [enc setBuffer:src offset:0 atIndex:3];
    [enc setBuffer:dst offset:0 atIndex:4];
    [enc setBuffer:res.statsBuf offset:0 atIndex:5];
    [enc setBuffer:scatBind offset:0 atIndex:6];
    const MTLSize grid = MTLSizeMake((p_Width + 15) / 16, (p_Height + 15) / 16, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmdBuf commit];
}
