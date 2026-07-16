// Speak — OpenCL implementation of the film-reconstruction pipeline.
// Line-by-line port of plugin/speak_core.h; keep the two in sync. This is the
// render path Resolve uses on NVIDIA/AMD/Intel when CUDA isn't advertised, and
// the primary path on Windows/Linux.

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

#include "SpeakParams.h"

static const char* kSpeakKernelSource = R"CLC(

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

#define kLog10_2 0.301029996f
#define k18Gray  0.18f
#define kPrinterPt 0.025f
#define kLinTiny 1e-8f
#define kKneeMin 0.05f
#define kDI_A 0.0075f
#define kDI_B 7.0f
#define kDI_C 0.07329248f
#define kDI_M 10.44426855f
#define kDI_LIN_CUT 0.00262409f
#define kDI_LOG_CUT 0.02740668f

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
    if (cs == 0) return diDecode(v);
    if (cs == 1) return v <= 0.0f ? 0.0f : pow(v, 2.4f);
    if (cs == 3) return acesCctDecode(v);
    return v;
}
inline float encodeFromLinear(int cs, float L)
{
    if (cs == 0) return diEncode(L);
    if (cs == 1) return L <= 0.0f ? 0.0f : pow(L, 1.0f / 2.4f);
    if (cs == 3) return acesCctEncode(L);
    return L;
}

__constant float kDWG_to_XYZ[9] = {
    0.70062239f, 0.14877482f, 0.10105872f,
    0.27411851f, 0.87363190f,-0.14775041f,
   -0.09896291f,-0.13789533f, 1.32591599f };
__constant float kXYZ_to_Rec709[9] = {
    3.24045420f,-1.53713850f,-0.49853140f,
   -0.96926600f, 1.87601080f, 0.04155600f,
    0.05564340f,-0.20402590f, 1.05722520f };
inline void mul3(__constant float* m, float r, float g, float b,
                 float* oR, float* oG, float* oB)
{
    *oR = m[0] * r + m[1] * g + m[2] * b;
    *oG = m[3] * r + m[4] * g + m[5] * b;
    *oB = m[6] * r + m[7] * g + m[8] * b;
}
inline void gamutToRec709Lin(int cs, float r, float g, float b,
                             float* oR, float* oG, float* oB)
{
    if (cs == 0 || cs == 2) {
        float X, Y, Z;
        mul3(kDWG_to_XYZ, r, g, b, &X, &Y, &Z);
        mul3(kXYZ_to_Rec709, X, Y, Z, oR, oG, oB);
    } else {
        *oR = r; *oG = g; *oB = b;
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
inline float chainDensity(float stops, int ch, const SpeakProfile* p)
{
    float logH = stops * kLog10_2;
    float Dneg = hdCurve(logH, p->negDmin[ch], p->negDmax[ch], p->negGamma[ch],
                         p->negToe[ch], p->negShoulder[ch], p->negSpeed[ch]);
    float printerOff = (p->printerMaster + p->printerLights[ch]) * kPrinterPt;
    float logHprn = -Dneg + printerOff;
    return hdCurve(logHprn, p->prnDmin[ch], p->prnDmax[ch], p->prnGamma[ch],
                   p->prnToe[ch], p->prnShoulder[ch], p->prnSpeed[ch]);
}
inline float toneChannel(float lin, int ch, const SpeakProfile* p)
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

inline int wfColOf(int x, int W)
{
    int c = x * SPEAK_WF_COLS / (W > 0 ? W : 1);
    return c < 0 ? 0 : (c >= SPEAK_WF_COLS ? SPEAK_WF_COLS - 1 : c);
}
inline int wfRowOf(float D)
{
    int r = (int)(D / SPEAK_WF_DMAX * SPEAK_WF_ROWS);
    return r < 0 ? 0 : (r >= SPEAK_WF_ROWS ? SPEAK_WF_ROWS - 1 : r);
}
inline int wfIdx(int ch, int col, int row)
{
    return SPEAK_STATS_WF + ch * (SPEAK_WF_COLS * SPEAK_WF_ROWS) + col * SPEAK_WF_ROWS + row;
}

inline int expBinOf(float stops)
{
    int b = (int)((stops + 6.0f) / 12.0f * SPEAK_EXP_BINS);
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
inline void subtractiveColor(float r, float g, float b, const SpeakProfile* p,
                             float* oR, float* oG, float* oB)
{
    float DR = density10(r), DG = density10(g), DB = density10(b);
    float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    float devR = DR - Dbar, devG = DG - Dbar, devB = DB - Dbar;
    float cR = (1.0f + p->subSat[0]) * devR - (p->dyeCouple[1] * devG + p->dyeCouple[2] * devB);
    float cG = (1.0f + p->subSat[1]) * devG - (p->dyeCouple[3] * devR + p->dyeCouple[5] * devB);
    float cB = (1.0f + p->subSat[2]) * devB - (p->dyeCouple[6] * devR + p->dyeCouple[7] * devG);
    float DpR = softCapKnee(Dbar + cR, p->subSatKnee[0]);
    float DpG = softCapKnee(Dbar + cG, p->subSatKnee[1]);
    float DpB = softCapKnee(Dbar + cB, p->subSatKnee[2]);
    *oR = pow10f(-DpR); *oG = pow10f(-DpG); *oB = pow10f(-DpB);
}
inline bool dyeActive(const SpeakProfile* p)
{
    return p->subSat[0] != 0.0f || p->subSat[1] != 0.0f || p->subSat[2] != 0.0f ||
           p->dyeCouple[1] != 0.0f || p->dyeCouple[2] != 0.0f || p->dyeCouple[3] != 0.0f ||
           p->dyeCouple[5] != 0.0f || p->dyeCouple[6] != 0.0f || p->dyeCouple[7] != 0.0f;
}

inline float smooth01(float t) { t = clampf(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
inline void splitWeights(float Dbar, const SpeakProfile* p, float* wShadow, float* wHigh)
{
    float grayD  = 0.744727f;
    float pivotD = grayD - p->splitPivot * kLog10_2;
    float halfW  = 0.25f + 1.5f * clampf(p->splitBalance, 0.0f, 1.0f);
    float x = (Dbar - pivotD) / halfW;
    *wShadow = smooth01(x);
    *wHigh   = smooth01(-x);
}
inline void splitTone(float r, float g, float b, const SpeakProfile* p,
                      float* oR, float* oG, float* oB)
{
    float DR = density10(r), DG = density10(g), DB = density10(b);
    float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    float wS, wH;
    splitWeights(Dbar, p, &wS, &wH);
    *oR = pow10f(-(DR + wS * p->splitShadow[0] + wH * p->splitHigh[0]));
    *oG = pow10f(-(DG + wS * p->splitShadow[1] + wH * p->splitHigh[1]));
    *oB = pow10f(-(DB + wS * p->splitShadow[2] + wH * p->splitHigh[2]));
}
inline bool splitActive(const SpeakProfile* p)
{
    return p->splitShadow[0] != 0.0f || p->splitShadow[1] != 0.0f || p->splitShadow[2] != 0.0f ||
           p->splitHigh[0] != 0.0f || p->splitHigh[1] != 0.0f || p->splitHigh[2] != 0.0f;
}

// ---------------------------------------------------------------------------
// HALATION (Phase 4) — the scatter pyramid. Port of speak_core.h's halation
// block; see there for the physics and for why the injection is a re-exposure
// ahead of the negative rather than an end-chain overlay. Atomics-free by
// construction (box decimation is order-independent), which also keeps it clear
// of Apple's OpenCL global-int32-atomics miscompile.
// ---------------------------------------------------------------------------
#define SPEAK_HAL_MAXLEV     14
#define SPEAK_HAL_MINDIM     8
#define SPEAK_HAL_SIGMA_MIN  0.05f

__constant float kHalWeight[3] = { 1.0f, 0.30f, 0.10f };

inline float halExcess(float lin, float thresh)
{
    float l = lin < 0.0f ? 0.0f : lin;
    return l > thresh ? (l - thresh) : 0.0f;
}
inline float halAmountOf(const SpeakParams* pr)
{
    return (pr->enableOptics != 0) ? pr->profile.halAmount : 0.0f;
}
inline float halSigmaPx(int H, const SpeakParams* pr)
{
    float s = pr->profile.halRadius * 0.01f * (float)H;
    return s < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : s;
}
inline bool halActive(const SpeakParams* pr)
{
    return (pr->enableTone != 0) && (pr->strength > 0.0f) && (halAmountOf(pr) > 0.0f);
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
inline void halLevelInfo(int W, int H, int L, int* lw, int* lh, int* off)
{
    int w = W, h = H, o = 0;
    for (int i = 0; i < L; ++i) { o += w * h; w = (w + 1) / 2; h = (h + 1) / 2; }
    *lw = w; *lh = h; *off = o;
}

#define kHalSigmaC   0.645497f
#define kHalCoreFall  3.0f
#define kHalSkirtFall 1.0f
inline float halLevelWeight(int L, float sigmaTarget)
{
    float s = sigmaTarget < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : sigmaTarget;
    float Lt = log2(s / kHalSigmaC);
    float d = (float)L - Lt;
    return (d <= 0.0f) ? exp2(kHalCoreFall * d) : exp2(-kHalSkirtFall * d);
}

// ---- pyramid taps (per-pixel, so the GPU kernels are textual ports) ----
inline float halFetch(__global const float* arena, int off, int lw, int lh, int x, int y, int c)
{
    int xx = x < 0 ? 0 : (x >= lw ? lw - 1 : x);
    int yy = y < 0 ? 0 : (y >= lh ? lh - 1 : y);
    return arena[((size_t)off + (size_t)yy * lw + xx) * 3 + c];
}

__constant float kHalDec[4] = { 0.125f, 0.375f, 0.375f, 0.125f };
inline void halDecimatePixel(__global const float* arena, int sOff, int sW, int sH,
                             int dx, int dy, float* out)
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

// Cubic B-spline read of one level onto a (W,H) grid.
//
// B-spline, NOT bilinear and NOT Catmull-Rom, for two reasons that both matter
// here: it is C2 (bilinear is only C0, and its derivative discontinuity at every
// texel boundary is what made the skirt read as hard rectangular BLOCKS), and
// its weights are NON-NEGATIVE and sum to 1 (Catmull-Rom's overshoot would
// undershoot a scatter field below zero and break energy conservation). It is
// approximating rather than interpolating — it blurs slightly — which for a
// light-scatter field is exactly what is wanted.
inline void halBSpline(float t, float* w)
{
    float t2 = t * t, t3 = t2 * t;
    w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) * (1.0f / 6.0f);
    w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) * (1.0f / 6.0f);
    w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) * (1.0f / 6.0f);
    w[3] = t3 * (1.0f / 6.0f);
}
inline float halSampleLevel(__global const float* arena, int off, int lw, int lh,
                            int W, int H, int x, int y, int c)
{
    float fx = ((float)x + 0.5f) * (float)lw / (float)W - 0.5f;
    float fy = ((float)y + 0.5f) * (float)lh / (float)H - 0.5f;
    int x0 = (int)floor(fx), y0 = (int)floor(fy);
    float wx[4], wy[4];
    halBSpline(fx - (float)x0, wx);
    halBSpline(fy - (float)y0, wy);
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
// energy-preserving (each level is mean-preserving, so a convex combination of
// them carries exactly the source's energy). Same loop on every backend.
inline float halWeightSum(int nLev, float sigmaTarget)
{
    float wsum = 0.0f;
    for (int L = 0; L < nLev; ++L) wsum += halLevelWeight(L, sigmaTarget);
    return wsum;
}

// COARSE-TO-FINE ACCUMULATE — one pixel of level L.
//     acc_L = w_L * level_L + upsample_2x(acc_{L+1})
// running L = nLev-1 down to 0, in place in the arena. acc_0 is then the whole
// (unnormalized) mixture at full resolution.
//
// WHY NOT SAMPLE EVERY LEVEL DIRECTLY AT FULL RES (what this replaced): reading
// a coarse level at full res interpolates between texels 2^L px apart, and
// bilinear is only C0 — the derivative discontinuity at every texel boundary
// reads as VISIBLE RECTANGULAR BLOCKS, worst on exactly the coarse levels that
// carry the skirt. It shipped past every numeric gate (energy, ladder, tail,
// resolution) because none of them measured isotropy. Going one octave at a time
// fixes it because each step interpolates only between ADJACENT samples and is
// then re-filtered by every step below it. It is also strictly CHEAPER: the
// total work is sum_L (level L's pixels) = 4/3 N, against nLev*N.
//
// IN-PLACE IS SAFE: a thread reads its OWN (x,y) at level L and a neighbourhood
// at level L+1 (a disjoint arena region), and writes only its own (x,y) at
// level L. Ordering BETWEEN levels is the only requirement (see the host's
// per-level barriers).
//
// The level geometry arrives as ARGUMENTS rather than being derived here from L,
// for the same reason SpeakDecimateKernel takes it — see that kernel's header:
// Apple's OpenCL optimizer miscompiles in-kernel halLevelInfo feeding halFetch's
// clamped tap addressing. The host computes it with its own textual mirror of
// halLevelInfo, so the body below stays a line-for-line port of the core's.
inline void halAccumPixel(__global float* arena, int L, int nLev, float sigmaTarget,
                          int lw, int lh, int off, int cw, int ch, int coff,
                          int x, int y, float* out)
{
    float wl = halLevelWeight(L, sigmaTarget);
    if (L >= nLev - 1) {                       // the coarsest level: nothing above it
        for (int c = 0; c < 3; ++c) out[c] = wl * halFetch(arena, off, lw, lh, x, y, c);
        return;
    }
    for (int c = 0; c < 3; ++c)
        out[c] = wl * halFetch(arena, off, lw, lh, x, y, c)
               + halSampleLevel(arena, coff, cw, ch, lw, lh, x, y, c);
}

inline void lookLinear(float r, float g, float b,
                       float scatR, float scatG, float scatB,
                       const SpeakParams* pr,
                       float* oR, float* oG, float* oB)
{
    int cs = pr->inputColorSpace;
    float lr = decodeToLinear(cs, r);
    float lg = decodeToLinear(cs, g);
    float lb = decodeToLinear(cs, b);
    float mr = lr, mg = lg, mb = lb;
    if ((pr->enableTone != 0) && (pr->strength > 0.0f)) {
        float s = clampf(pr->strength, 0.0f, 1.0f);
        float a = halAmountOf(pr);
        // RE-EXPOSURE: the scattered light adds to the scene light ENTERING the
        // negative, so it goes in the CURVE'S ARGUMENT. The dry side of the mix
        // stays lr, not the re-exposed value — otherwise strength 0 would leave
        // the raw scatter added to linear with no curve downstream, which is
        // precisely the end-chain overlay the arm rejected (and would break the
        // identity gate G3).
        float er = lr, eg = lg, eb = lb;
        if (a > 0.0f) {
            er = lr + a * kHalWeight[0] * scatR;
            eg = lg + a * kHalWeight[1] * scatG;
            eb = lb + a * kHalWeight[2] * scatB;
        }
        mr = lerpf(lr, toneChannel(er, 0, &pr->profile), s);
        mg = lerpf(lg, toneChannel(eg, 1, &pr->profile), s);
        mb = lerpf(lb, toneChannel(eb, 2, &pr->profile), s);
    }
    if ((pr->enableDye != 0) && dyeActive(&pr->profile)) subtractiveColor(mr, mg, mb, &pr->profile, &mr, &mg, &mb);
    if ((pr->enableSplit != 0) && splitActive(&pr->profile)) splitTone(mr, mg, mb, &pr->profile, &mr, &mg, &mb);
    *oR = mr; *oG = mg; *oB = mb;
}

inline float scopeYStops(float inStops, int ch, const SpeakParams* pr)
{
    float lin = k18Gray * exp2(inStops);
    float outLin = lin;
    if ((pr->enableTone != 0) && (pr->strength > 0.0f)) {
        float s = clampf(pr->strength, 0.0f, 1.0f);
        outLin = lerpf(lin, toneChannel(lin, ch, &pr->profile), s);
    }
    return log2((outLin < kLinTiny ? kLinTiny : outLin) / k18Gray);
}

inline bool hdScopePixel(int x, int y, int W, int H, const SpeakParams* pr,
                         __global const uint* stats,
                         float* outR, float* outG, float* outB)
{
    if (pr->scopeHD == 0) return false;
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

    *outR = 0.06f; *outG = 0.06f; *outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        *outR = 0.30f; *outG = 0.30f; *outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    float rowStops = 6.0f - 12.0f * ((float)gy / (plotH - 1));
    int gcol0 = (int)((0.0f + 6.0f) / 12.0f * (plotW - 1) + 0.5f);
    int grow0 = (int)((6.0f - 0.0f) / 12.0f * (plotH - 1) + 0.5f);
    if (gx == gcol0 || gy == grow0) { *outR = 0.24f; *outG = 0.24f; *outB = 0.24f; return true; }
    if ((gx % (plotW / 6)) == 0 || (gy % (plotH / 6)) == 0) { *outR = 0.13f; *outG = 0.13f; *outB = 0.13f; }

    uint hmax = stats[SPEAK_STATS_HIST_MAX];
    if (hmax > 0u) {
        int hb = expBinOf(-6.0f + 12.0f * ((float)gx / (plotW - 1)));
        float f = (float)(stats[SPEAK_STATS_HIST_EXP + hb]) / (float)hmax;
        int barH = (int)(sqrt(f) * (plotH * 0.45f) + 0.5f);
        if (gy >= plotH - barH) { *outR = 0.16f; *outG = 0.19f; *outB = 0.24f; }
    }

    int chR[3]; chR[0]=1; chR[1]=0; chR[2]=0;
    int chG[3]; chG[0]=0; chG[1]=1; chG[2]=0;
    int chB[3]; chB[0]=0; chB[1]=0; chB[2]=1;
    for (int ch = 0; ch < 3; ++ch) {
        float inS  = -6.0f + 12.0f * ((float)gx       / (plotW - 1));
        float inS2 = -6.0f + 12.0f * ((float)(gx + 1) / (plotW - 1));
        float y0 = scopeYStops(inS,  ch, pr);
        float y1 = scopeYStops(inS2, ch, pr);
        if (y0 > y1) { float tt = y0; y0 = y1; y1 = tt; }
        float lo = y1 < y0 ? y1 : y0, hi = y1 > y0 ? y1 : y0;
        if (rowStops <= hi + 0.09f && rowStops >= lo - 0.09f) {
            *outR = 0.10f + 0.85f * chR[ch];
            *outG = 0.10f + 0.85f * chG[ch];
            *outB = 0.10f + 0.85f * chB[ch];
            return true;
        }
    }
    if (gy >= plotH - 5 * sc && gy < plotH - 1 * sc) {
        int sw = gx / (6 * sc);
        if (gx % (6 * sc) < 4 * sc) {
            if (sw == 0) { *outR = 0.95f; *outG = 0.10f; *outB = 0.10f; return true; }
            if (sw == 1) { *outR = 0.10f; *outG = 0.95f; *outB = 0.10f; return true; }
            if (sw == 2) { *outR = 0.10f; *outG = 0.10f; *outB = 0.95f; return true; }
        }
    }
    return true;
}

inline bool densityScopePixel(int x, int y, int W, int H, const SpeakParams* pr,
                              __global const uint* stats,
                              float* outR, float* outG, float* outB)
{
    if (pr->scopeDensity == 0) return false;

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

    *outR = 0.06f; *outG = 0.06f; *outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        *outR = 0.30f; *outG = 0.30f; *outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    int rowGray = (int)(0.744727f / SPEAK_WF_DMAX * (plotH - 1) + 0.5f);
    if (gy == 0) { *outR = 0.35f; *outG = 0.35f; *outB = 0.35f; return true; }
    if (gy == rowGray) { *outR = 0.30f; *outG = 0.26f; *outB = 0.12f; return true; }

    int chW = plotW / 3;
    int ch = gx / (chW > 0 ? chW : 1);
    if (ch > 2) ch = 2;
    if (gx - ch * chW == 0 && ch > 0) { *outR = 0.16f; *outG = 0.16f; *outB = 0.16f; return true; }

    uint wmax = stats[SPEAK_STATS_WF_MAX];
    if (wmax > 0u) {
        int within = gx - ch * chW;
        int wcol = within * SPEAK_WF_COLS / (chW > 0 ? chW : 1);
        int wrow = gy * SPEAK_WF_ROWS / plotH;
        uint c = stats[wfIdx(ch, wcol < SPEAK_WF_COLS ? wcol : SPEAK_WF_COLS - 1,
                             wrow < SPEAK_WF_ROWS ? wrow : SPEAK_WF_ROWS - 1)];
        if (c > 0u) {
            float inten = sqrt((float)c / (float)wmax);
            float v = 0.12f + 0.88f * (inten > 1.0f ? 1.0f : inten);
            *outR = (ch == 0) ? v : 0.05f;
            *outG = (ch == 1) ? v : 0.05f;
            *outB = (ch == 2) ? v : 0.05f;
        }
    }
    return true;
}

inline void deliverInput(const SpeakParams* pr, float r, float g, float b,
                         float* oR, float* oG, float* oB)
{
    if (pr->outputMode == 1) {
        int cs = pr->inputColorSpace;
        float lr = decodeToLinear(cs, r);
        float lg = decodeToLinear(cs, g);
        float lb = decodeToLinear(cs, b);
        float rr, rg, rb;
        gamutToRec709Lin(cs, lr, lg, lb, &rr, &rg, &rb);
        rr = rr < 0.0f ? 0.0f : rr;
        rg = rg < 0.0f ? 0.0f : rg;
        rb = rb < 0.0f ? 0.0f : rb;
        *oR = encodeFromLinear(1, rr);
        *oG = encodeFromLinear(1, rg);
        *oB = encodeFromLinear(1, rb);
    } else {
        *oR = r; *oG = g; *oB = b;
    }
}

inline void processPixel(float r, float g, float b,
                         float scatR, float scatG, float scatB,
                         int x, int y, int W, int H,
                         const SpeakParams* pr, __global const uint* stats,
                         float* outR, float* outG, float* outB)
{
    int cs = pr->inputColorSpace;
    bool toneOn = (pr->enableTone != 0) && (pr->strength > 0.0f);
    bool dyeOn = (pr->enableDye != 0) && dyeActive(&pr->profile);
    bool splitOn = (pr->enableSplit != 0) && splitActive(&pr->profile);
    bool bake = (pr->outputMode == 1);
    if (!toneOn && !dyeOn && !splitOn && !bake) {
        *outR = r; *outG = g; *outB = b;
    } else {
        float mr, mg, mb;
        lookLinear(r, g, b, scatR, scatG, scatB, pr, &mr, &mg, &mb);
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, mr, mg, mb, &rr, &rg, &rb);
            rr = rr < 0.0f ? 0.0f : rr;
            rg = rg < 0.0f ? 0.0f : rg;
            rb = rb < 0.0f ? 0.0f : rb;
            *outR = encodeFromLinear(1, rr);
            *outG = encodeFromLinear(1, rg);
            *outB = encodeFromLinear(1, rb);
        } else {
            *outR = encodeFromLinear(cs, mr);
            *outG = encodeFromLinear(cs, mg);
            *outB = encodeFromLinear(cs, mb);
        }
    }
    if (pr->viewMode == 2 || (pr->viewMode == 1 && x < W / 2))
        deliverInput(pr, r, g, b, outR, outG, outB);

    // Isolated-scatter view: the ACTUAL injected re-exposure a * w_c * S_c,
    // delivered through the SAME output transform as the picture. Deliberately
    // NOT auto-normalized — see speak_core.h.
    if (pr->viewMode == 3) {
        float a = halAmountOf(pr);
        float on = ((pr->enableTone != 0) && (pr->strength > 0.0f)) ? 1.0f : 0.0f;
        float sr = on * a * kHalWeight[0] * scatR;
        float sg = on * a * kHalWeight[1] * scatG;
        float sb = on * a * kHalWeight[2] * scatB;
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, sr, sg, sb, &rr, &rg, &rb);
            sr = rr < 0.0f ? 0.0f : rr;
            sg = rg < 0.0f ? 0.0f : rg;
            sb = rb < 0.0f ? 0.0f : rb;
            *outR = encodeFromLinear(1, sr);
            *outG = encodeFromLinear(1, sg);
            *outB = encodeFromLinear(1, sb);
        } else {
            *outR = encodeFromLinear(cs, sr);
            *outG = encodeFromLinear(cs, sg);
            *outB = encodeFromLinear(cs, sb);
        }
    }

    float sr, sg, sb;
    if (hdScopePixel(x, y, W, H, pr, stats, &sr, &sg, &sb)) { *outR = sr; *outG = sg; *outB = sb; }
    if (densityScopePixel(x, y, W, H, pr, stats, &sr, &sg, &sb)) { *outR = sr; *outG = sg; *outB = sb; }
}

// ---------------------------------------------------------------------------
// The scatter pyramid, mirroring buildHalScatter in speak_core.h. Four passes:
//   excess (full res, level 0) -> decimate (once per level, fine to coarse) ->
//   accum (once per level, COARSE TO FINE, in place) -> normalize (full res).
// The host skips them all when halation is inactive, and the readers below gate
// on the SAME condition, so nothing ever touches the placeholder binding that
// stands in for the buffers in that case.
// ---------------------------------------------------------------------------

// Level 0: the per-channel scene-linear highlight excess. THRESHOLD BEFORE
// DECIMATION — mean(max(0, l-t)) != max(0, mean(l)-t).
__kernel void SpeakExcessKernel(SpeakParams p, int W, int H,
                                __global const float* src, __global float* arena)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int cs = p.inputColorSpace;
    float th = p.profile.halThresh;
    size_t i = ((size_t)y * W + x) * 4;
    size_t o = ((size_t)y * W + x) * 3;
    arena[o + 0] = halExcess(decodeToLinear(cs, src[i + 0]), th);
    arena[o + 1] = halExcess(decodeToLinear(cs, src[i + 1]), th);
    arena[o + 2] = halExcess(decodeToLinear(cs, src[i + 2]), th);
}

// One dispatch per level L = 1..nLev-1: build level L from level L-1. Reads and
// writes disjoint regions of the same arena, so the only ordering requirement is
// between dispatches (see the host's barriers).
//
// The level geometry arrives as ARGUMENTS rather than being derived here from L.
// The host computes it with its own textual mirror of halLevelInfo, so this
// kernel body stays a line-for-line port of buildHalScatter's inner loop — the
// host loop above it now carries the two halLevelInfo(L-1)/halLevelInfo(L) calls
// that the core's loop body has.
//
// This is NOT a style choice. Apple's OpenCL optimizer MISCOMPILES the in-kernel
// form: with `halLevelInfo(W,H,L-1,&sw,&sh,&so)` feeding halFetch's clamped tap
// addressing, levels >= 3 come out as ~0.5 * source[x + dw] instead of the 4x4
// binomial mean (levels 0-2 are unaffected, which is why a small frame looks
// fine). Verified: -cl-opt-disable fixes it; passing the SOURCE geometry (sw,sh,
// so) as args fixes it; passing only the DEST geometry does not. This is the
// same class of defect as the global-int32-atomics miscompile that already
// forces the stats pass to be skipped on this runtime. The arithmetic is
// unchanged — it just happens on the host now.
__kernel void SpeakDecimateKernel(int sw, int sh, int so, int dw, int dh, int doff,
                                  __global float* arena)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= dw || y >= dh) return;
    float v[3];
    halDecimatePixel(arena, so, sw, sh, x, y, v);
    size_t o = ((size_t)doff + (size_t)y * dw + x) * 3;
    arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
}

// One dispatch per level L = nLev-1 down to 0, COARSE TO FINE, with the grid set
// to LEVEL L's dims: acc_L = w_L * level_L + upsample_2x(acc_{L+1}), written IN
// PLACE. Level L reads level L+1, which the PREVIOUS dispatch wrote, so the
// host's barrier between levels is required. Geometry comes in as args (see
// halAccumPixel's header). `cw/ch/coff` are level L+1's and are unread at
// L == nLev-1.
__kernel void SpeakAccumKernel(SpeakParams p, int H, int L, int nLev,
                               int lw, int lh, int off, int cw, int ch, int coff,
                               __global float* arena)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= lw || y >= lh) return;
    float sig = halSigmaPx(H, &p);
    float v[3];
    halAccumPixel(arena, L, nLev, sig, lw, lh, off, cw, ch, coff, x, y, v);
    size_t o = ((size_t)off + (size_t)y * lw + x) * 3;
    arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
}

// Normalize level 0 into the scatter plane: sum(w) = 1 => energy preserved.
__kernel void SpeakNormalizeKernel(SpeakParams p, int W, int H,
                                   __global const float* arena, __global float* scat)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int nLev = halLevelCount(W, H);
    float sig = halSigmaPx(H, &p);
    float inv = 1.0f / halWeightSum(nLev, sig);
    size_t k = ((size_t)y * W + x) * 3;
    scat[k + 0] = arena[k + 0] * inv;
    scat[k + 1] = arena[k + 1] * inv;
    scat[k + 2] = arena[k + 2] * inv;
}

// Scope measurement pass: bin the frame on a stride-2 grid. Integer atomics are
// order-independent, so the counts are identical on every backend.
//
// It reads the SCATTER plane: the density parade must measure the HALATED
// result. A scatter-blind scope is a bug parity CANNOT catch — every backend
// would agree on the same wrong parade (see the lookLinear header in the core).
__kernel void SpeakStatsKernel(SpeakParams p, int W, int H,
                               __global const float* src, __global const float* scat,
                               volatile __global uint* stats)
{
    int x = get_global_id(0) * 2, y = get_global_id(1) * 2;
    if (x >= W || y >= H) return;
    int i = (y * W + x) * 4;
    int j = (y * W + x) * 3;
    if (p.scopeHD != 0) {
        int bin = expBinOf(pixelStops(p.inputColorSpace, src[i + 0], src[i + 1], src[i + 2]));
        atomic_inc(&stats[SPEAK_STATS_HIST_EXP + bin]);
    }
    if (p.scopeDensity != 0) {
        // Mirrors the core's `scat ? scat[j] : 0.0f`: the host builds the plane
        // under exactly this condition, so a false `hal` means the binding is a
        // placeholder and must not be dereferenced.
        bool hal = halActive(&p) || (p.viewMode == 3);
        float sR = 0.0f, sG = 0.0f, sB = 0.0f;
        if (hal) { sR = scat[j + 0]; sG = scat[j + 1]; sB = scat[j + 2]; }
        float mr, mg, mb;
        lookLinear(src[i + 0], src[i + 1], src[i + 2], sR, sG, sB, &p, &mr, &mg, &mb);
        int col = wfColOf(x, W);
        atomic_inc(&stats[wfIdx(0, col, wfRowOf(density10(mr)))]);
        atomic_inc(&stats[wfIdx(1, col, wfRowOf(density10(mg)))]);
        atomic_inc(&stats[wfIdx(2, col, wfRowOf(density10(mb)))]);
    }
}

__kernel void SpeakStatsMaxKernel(__global uint* stats)
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

__kernel void SpeakKernel(SpeakParams p, int W, int H,
                          __global const float* src, __global const float* scat,
                          __global float* dst, __global const uint* stats)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int i = (y * W + x) * 4;
    int j = (y * W + x) * 3;
    // Mirrors speakFrame's `sc ? sc[j] : 0.0f` — same gate as the host's, so the
    // placeholder binding used when halation is off is never dereferenced.
    bool hal = halActive(&p) || (p.viewMode == 3);
    float sR = 0.0f, sG = 0.0f, sB = 0.0f;
    if (hal) { sR = scat[j + 0]; sG = scat[j + 1]; sB = scat[j + 2]; }
    float oR, oG, oB;
    processPixel(src[i + 0], src[i + 1], src[i + 2], sR, sG, sB,
                 x, y, W, H, &p, stats, &oR, &oG, &oB);
    dst[i + 0] = oR; dst[i + 1] = oG; dst[i + 2] = oB; dst[i + 3] = src[i + 3];
}
)CLC";

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
namespace {

void SpeakCheck(cl_int e, const char* m) { if (e != CL_SUCCESS) fprintf(stderr, "Speak: %s [%d]\n", m, e); }

class SpeakLocker
{
public:
#ifdef _WIN64
    SpeakLocker() { InitializeCriticalSection(&m); }
    ~SpeakLocker() { DeleteCriticalSection(&m); }
    void Lock() { EnterCriticalSection(&m); }
    void Unlock() { LeaveCriticalSection(&m); }
    CRITICAL_SECTION m;
#else
    SpeakLocker() { pthread_mutex_init(&m, NULL); }
    ~SpeakLocker() { pthread_mutex_destroy(&m); }
    void Lock() { pthread_mutex_lock(&m); }
    void Unlock() { pthread_mutex_unlock(&m); }
    pthread_mutex_t m;
#endif
};

// Host-side mirrors of speak_core.h's pyramid geometry — the dispatch sizes and
// the buffer sizes MUST be the same arithmetic the kernels use, so these are
// textual ports of halLevelCount/halLevelInfo/halArenaPixels/halAmountOf/
// halActive, exactly like the CLC copies above.
int halLevelCount(int W, int H)
{
    int n = 1, w = W, h = H;
    while (n < SPEAK_HAL_MAXLEV && w > SPEAK_HAL_MINDIM && h > SPEAK_HAL_MINDIM) {
        w = (w + 1) / 2; h = (h + 1) / 2; n++;
    }
    return n;
}
void halLevelInfo(int W, int H, int L, int& lw, int& lh, int& off)
{
    int w = W, h = H, o = 0;
    for (int i = 0; i < L; ++i) { o += w * h; w = (w + 1) / 2; h = (h + 1) / 2; }
    lw = w; lh = h; off = o;
}
int halArenaPixels(int W, int H)
{
    int lw, lh, off;
    const int n = halLevelCount(W, H);
    halLevelInfo(W, H, n, lw, lh, off);   // offset just past the last level
    return off;
}
float halAmountOf(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) ? pr.profile.halAmount : 0.0f;
}
bool halActive(const SpeakParams& pr)
{
    return (pr.enableTone != 0) && (pr.strength > 0.0f) && (halAmountOf(pr) > 0.0f);
}

struct SpeakRes {
    cl_kernel k = NULL;         // main
    cl_kernel kStats = NULL;    // scope measurement pass
    cl_kernel kMax = NULL;      // bin-max finalize
    cl_kernel kExcess = NULL;   // scatter pyramid: level 0 highlight excess
    cl_kernel kDecimate = NULL; // scatter pyramid: level L from level L-1
    cl_kernel kAccum = NULL;    // scatter pyramid: coarse-to-fine accumulate, in place
    cl_kernel kNorm = NULL;     // scatter pyramid: normalize level 0 into the plane
    cl_mem stats = NULL;
    // Unlike `stats` (fixed size, allocated once), these two are SIZE-DEPENDENT:
    // the OFX host hands us any image size and it changes between renders (proxy
    // vs full res, different clips). Cache the allocated size and grow when it is
    // no longer enough — otherwise the first proxy->full-res switch overruns.
    // Growth-only: an oversized buffer is harmless (every offset is recomputed
    // from W,H on both sides), and it avoids reallocating on every size wobble.
    cl_mem arena = NULL; size_t arenaBytes = 0;
    cl_mem scat = NULL;  size_t scatBytes = 0;
};

} // namespace

void RunOpenCLSpeak(void* p_CmdQ, int p_Width, int p_Height,
                    const SpeakParams& p_Params, const float* p_Src, float* p_Dst)
{
    cl_int error;
    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    static std::map<void*, SpeakRes> s_res;
    static SpeakLocker s_lock;

    cl_context clContext = NULL;
    error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
    SpeakCheck(error, "get context");

    SpeakParams params = p_Params;
    int W = p_Width, H = p_Height;

    // Mirror speak_core.h's speakFrame gate. When halation is inactive the whole
    // three-pass scatter chain is skipped and the buffers stay at their minimum
    // placeholder size — the identity path stays bit-exact AND free. The kernels
    // gate their scatter reads on this same condition, so the placeholder
    // binding is never dereferenced (a NULL binding would crash, so we always
    // bind something valid).
    const bool wantHal = halActive(params) || params.viewMode == SPEAK_VIEW_SCATTER;
    const size_t arenaNeed = wantHal
        ? static_cast<size_t>(halArenaPixels(W, H)) * 3 * sizeof(cl_float) : sizeof(cl_float);
    const size_t scatNeed = wantHal
        ? static_cast<size_t>(W) * H * 3 * sizeof(cl_float) : sizeof(cl_float);

    SpeakRes res;
    s_lock.Lock();
    {
        SpeakRes& r = s_res[p_CmdQ];
        if (!r.k) {
            cl_program program = clCreateProgramWithSource(clContext, 1, &kSpeakKernelSource, NULL, &error);
            SpeakCheck(error, "create program");
            error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (error != CL_SUCCESS) {
                cl_device_id dev = NULL;
                clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &dev, NULL);
                char log[16384] = { 0 };
                clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
                fprintf(stderr, "Speak OpenCL build log:\n%s\n", log);
                s_lock.Unlock();
                return;
            }
            r.k = clCreateKernel(program, "SpeakKernel", &error);
            SpeakCheck(error, "create kernel");
            r.kStats = clCreateKernel(program, "SpeakStatsKernel", &error);
            SpeakCheck(error, "create stats kernel");
            r.kMax = clCreateKernel(program, "SpeakStatsMaxKernel", &error);
            SpeakCheck(error, "create stats-max kernel");
            r.kExcess = clCreateKernel(program, "SpeakExcessKernel", &error);
            SpeakCheck(error, "create excess kernel");
            r.kDecimate = clCreateKernel(program, "SpeakDecimateKernel", &error);
            SpeakCheck(error, "create decimate kernel");
            r.kAccum = clCreateKernel(program, "SpeakAccumKernel", &error);
            SpeakCheck(error, "create accum kernel");
            r.kNorm = clCreateKernel(program, "SpeakNormalizeKernel", &error);
            SpeakCheck(error, "create normalize kernel");
        }
        if (!r.stats)
            r.stats = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                                     SPEAK_STATS_UINTS * sizeof(cl_uint), NULL, &error);
        if (!r.arena || r.arenaBytes < arenaNeed) {
            if (r.arena) clReleaseMemObject(r.arena);
            r.arena = clCreateBuffer(clContext, CL_MEM_READ_WRITE, arenaNeed, NULL, &error);
            SpeakCheck(error, "create arena buffer");
            r.arenaBytes = (error == CL_SUCCESS) ? arenaNeed : 0;
        }
        if (!r.scat || r.scatBytes < scatNeed) {
            if (r.scat) clReleaseMemObject(r.scat);
            r.scat = clCreateBuffer(clContext, CL_MEM_READ_WRITE, scatNeed, NULL, &error);
            SpeakCheck(error, "create scatter buffer");
            r.scatBytes = (error == CL_SUCCESS) ? scatNeed : 0;
        }
        res = r;
    }
    s_lock.Unlock();

    cl_mem src = reinterpret_cast<cl_mem>(const_cast<float*>(p_Src));
    cl_mem dst = reinterpret_cast<cl_mem>(p_Dst);

    const size_t local[2] = { 16, 16 };

    // ---- the scatter pyramid ----
    //   excess -> decimate(L=1..nLev-1) -> accum(L=nLev-1..0) -> normalize
    // Every pass depends on the previous one, and the queue is NOT ours (Resolve
    // hands it in), so we cannot assume it was created in-order. A barrier costs
    // nothing on an in-order queue and is what makes an out-of-order one correct.
    if (wantHal) {
        const int nLev = halLevelCount(W, H);
        int c = 0;
        error  = clSetKernelArg(res.kExcess, c++, sizeof(SpeakParams), &params);
        error |= clSetKernelArg(res.kExcess, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.kExcess, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.kExcess, c++, sizeof(cl_mem), &src);
        error |= clSetKernelArg(res.kExcess, c++, sizeof(cl_mem), &res.arena);
        SpeakCheck(error, "set excess args");
        const size_t globalF[2] = { static_cast<size_t>((W + 15) / 16) * 16,
                                    static_cast<size_t>((H + 15) / 16) * 16 };
        error = clEnqueueNDRangeKernel(cmdQ, res.kExcess, 2, NULL, globalF, local, 0, NULL, NULL);
        SpeakCheck(error, "enqueue excess");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);

        for (int L = 1; L < nLev; ++L) {
            int sw, sh, so, dw, dh, doff;
            halLevelInfo(W, H, L - 1, sw, sh, so);
            halLevelInfo(W, H, L,     dw, dh, doff);
            int c2 = 0;
            error  = clSetKernelArg(res.kDecimate, c2++, sizeof(int), &sw);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &sh);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &so);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &dw);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &dh);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(int), &doff);
            error |= clSetKernelArg(res.kDecimate, c2++, sizeof(cl_mem), &res.arena);
            SpeakCheck(error, "set decimate args");
            const size_t globalL[2] = { static_cast<size_t>((dw + 15) / 16) * 16,
                                        static_cast<size_t>((dh + 15) / 16) * 16 };
            error = clEnqueueNDRangeKernel(cmdQ, res.kDecimate, 2, NULL, globalL, local, 0, NULL, NULL);
            SpeakCheck(error, "enqueue decimate");
            clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
        }

        // Coarse-to-fine accumulate, in place, one octave at a time — the same
        // loop buildHalScatter runs, with level L's pixel loop become the grid.
        // The barrier after each level is REQUIRED, not defensive: level L reads
        // level L+1, which the previous dispatch wrote.
        for (int L = nLev - 1; L >= 0; --L) {
            int lw, lh, off, cw, ch, coff;
            halLevelInfo(W, H, L,     lw, lh, off);
            halLevelInfo(W, H, L + 1, cw, ch, coff);
            int c2 = 0;
            error  = clSetKernelArg(res.kAccum, c2++, sizeof(SpeakParams), &params);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &H);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &L);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &nLev);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &lw);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &lh);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &off);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &cw);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &ch);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(int), &coff);
            error |= clSetKernelArg(res.kAccum, c2++, sizeof(cl_mem), &res.arena);
            SpeakCheck(error, "set accum args");
            const size_t globalL[2] = { static_cast<size_t>((lw + 15) / 16) * 16,
                                        static_cast<size_t>((lh + 15) / 16) * 16 };
            error = clEnqueueNDRangeKernel(cmdQ, res.kAccum, 2, NULL, globalL, local, 0, NULL, NULL);
            SpeakCheck(error, "enqueue accum");
            clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
        }

        c = 0;
        error  = clSetKernelArg(res.kNorm, c++, sizeof(SpeakParams), &params);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(cl_mem), &res.arena);
        error |= clSetKernelArg(res.kNorm, c++, sizeof(cl_mem), &res.scat);
        SpeakCheck(error, "set normalize args");
        error = clEnqueueNDRangeKernel(cmdQ, res.kNorm, 2, NULL, globalF, local, 0, NULL, NULL);
        SpeakCheck(error, "enqueue normalize");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
    }

    // Zero the stats, then measure the frame only when a scope is showing it.
    // This runs AFTER the scatter pass because the density parade measures it.
    const cl_uint zero = 0;
    clEnqueueFillBuffer(cmdQ, res.stats, &zero, sizeof(cl_uint), 0,
                        SPEAK_STATS_UINTS * sizeof(cl_uint), 0, NULL, NULL);
    clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
    if (params.scopeHD != 0 || params.scopeDensity != 0) {
        int c = 0;
        error  = clSetKernelArg(res.kStats, c++, sizeof(SpeakParams), &params);
        error |= clSetKernelArg(res.kStats, c++, sizeof(int), &W);
        error |= clSetKernelArg(res.kStats, c++, sizeof(int), &H);
        error |= clSetKernelArg(res.kStats, c++, sizeof(cl_mem), &src);
        error |= clSetKernelArg(res.kStats, c++, sizeof(cl_mem), &res.scat);
        error |= clSetKernelArg(res.kStats, c++, sizeof(cl_mem), &res.stats);
        SpeakCheck(error, "set stats args");
        const size_t localH[2]  = { 16, 16 };
        // ceil(W/2) sample threads, not floor(W/2): the kernel indexes x = gid*2
        // and the CPU reference loops x = 0,2,..<W, so an odd W needs (W+1)/2
        // columns. With floor(W/2) this fell exactly one sample column short
        // whenever floor(W/2) landed on a multiple of the 16-wide group — i.e.
        // W = 1 (mod 32), which includes 1921 and 3841. Scope-only, but it made
        // the parade and histogram silently miss the frame's last sample column
        // at those sizes. (Metal's dispatch over-covers and was never affected.)
        const size_t globalH[2] = { static_cast<size_t>(((p_Width  + 1) / 2 + 15) / 16) * 16,
                                    static_cast<size_t>(((p_Height + 1) / 2 + 15) / 16) * 16 };
        error = clEnqueueNDRangeKernel(cmdQ, res.kStats, 2, NULL, globalH, localH, 0, NULL, NULL);
        SpeakCheck(error, "enqueue stats");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);

        error = clSetKernelArg(res.kMax, 0, sizeof(cl_mem), &res.stats);
        const size_t one[2] = { 1, 1 };
        error |= clEnqueueNDRangeKernel(cmdQ, res.kMax, 2, NULL, one, NULL, 0, NULL, NULL);
        SpeakCheck(error, "enqueue stats-max");
        clEnqueueBarrierWithWaitList(cmdQ, 0, NULL, NULL);
    }

    int c = 0;
    error  = clSetKernelArg(res.k, c++, sizeof(SpeakParams), &params);
    error |= clSetKernelArg(res.k, c++, sizeof(int), &W);
    error |= clSetKernelArg(res.k, c++, sizeof(int), &H);
    error |= clSetKernelArg(res.k, c++, sizeof(cl_mem), &src);
    error |= clSetKernelArg(res.k, c++, sizeof(cl_mem), &res.scat);
    error |= clSetKernelArg(res.k, c++, sizeof(cl_mem), &dst);
    error |= clSetKernelArg(res.k, c++, sizeof(cl_mem), &res.stats);
    SpeakCheck(error, "set args");

    const size_t global[2] = { static_cast<size_t>((p_Width + 15) / 16) * 16,
                               static_cast<size_t>((p_Height + 15) / 16) * 16 };
    error = clEnqueueNDRangeKernel(cmdQ, res.k, 2, NULL, global, local, 0, NULL, NULL);
    SpeakCheck(error, "enqueue");
}
