// Speak — CUDA implementation of the film-reconstruction pipeline (NVIDIA).
// Line-by-line port of plugin/speak_core.h; keep the two in sync. Like Hush's
// CudaKernel.cu this cannot be RUN on the dev Mac — it is a faithful textual
// port, syntax/type-checked via test/check_cuda_syntax.sh and treated as
// unverified until it has run on real NVIDIA hardware (CUDA stays OFF in the
// Windows build; Resolve renders Speak via OpenCL there).

#include <cuda_runtime.h>

#include <cstdio>
#include <map>

#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include "SpeakParams.h"

namespace {

#define kLog10_2   0.301029996f
#define k18Gray    0.18f
#define kPrinterPt 0.025f
#define kLinTiny   1e-8f
#define kKneeMin   0.05f
#define kDI_A      0.0075f
#define kDI_B      7.0f
#define kDI_C      0.07329248f
#define kDI_M      10.44426855f
#define kDI_LIN_CUT 0.00262409f
#define kDI_LOG_CUT 0.02740668f

__device__ inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline float pow10f(float x) { return exp2f(x * 3.32192809f); }

__device__ inline float softplusf(float z)
{
    float az = z < 0.0f ? -z : z;
    return (z > 0.0f ? z : 0.0f) + logf(1.0f + expf(-az));
}

__device__ inline float diDecode(float v)
{
    return (v <= kDI_LOG_CUT) ? (v / kDI_M) : (exp2f(v / kDI_C - kDI_B) - kDI_A);
}
__device__ inline float diEncode(float L)
{
    return (L <= kDI_LIN_CUT) ? (L * kDI_M) : ((log2f(L + kDI_A) + kDI_B) * kDI_C);
}
__device__ inline float acesCctDecode(float v)
{
    if (v <= 0.155251141552511f) return (v - 0.0729055341958355f) / 10.5402377416545f;
    return exp2f(v * 17.52f - 9.72f);
}
__device__ inline float acesCctEncode(float L)
{
    if (L <= 0.0078125f) return 10.5402377416545f * L + 0.0729055341958355f;
    return (log2f(L) + 9.72f) / 17.52f;
}
__device__ inline float decodeToLinear(int cs, float v)
{
    if (cs == 0) return diDecode(v);
    if (cs == 1) return v <= 0.0f ? 0.0f : powf(v, 2.4f);
    if (cs == 3) return acesCctDecode(v);
    return v;
}
__device__ inline float encodeFromLinear(int cs, float L)
{
    if (cs == 0) return diEncode(L);
    if (cs == 1) return L <= 0.0f ? 0.0f : powf(L, 1.0f / 2.4f);
    if (cs == 3) return acesCctEncode(L);
    return L;
}

__device__ const float kDWG_to_XYZ[9] = {
    0.70062239f, 0.14877482f, 0.10105872f,
    0.27411851f, 0.87363190f,-0.14775041f,
   -0.09896291f,-0.13789533f, 1.32591599f };
__device__ const float kXYZ_to_Rec709[9] = {
    3.24045420f,-1.53713850f,-0.49853140f,
   -0.96926600f, 1.87601080f, 0.04155600f,
    0.05564340f,-0.20402590f, 1.05722520f };
__device__ inline void mul3(const float* m, float r, float g, float b,
                            float& oR, float& oG, float& oB)
{
    oR = m[0] * r + m[1] * g + m[2] * b;
    oG = m[3] * r + m[4] * g + m[5] * b;
    oB = m[6] * r + m[7] * g + m[8] * b;
}
__device__ inline void gamutToRec709Lin(int cs, float r, float g, float b,
                                        float& oR, float& oG, float& oB)
{
    if (cs == 0 || cs == 2) {
        float X, Y, Z;
        mul3(kDWG_to_XYZ, r, g, b, X, Y, Z);
        mul3(kXYZ_to_Rec709, X, Y, Z, oR, oG, oB);
    } else {
        oR = r; oG = g; oB = b;
    }
}

__device__ inline float hdCurve(float logH, float Dmin, float Dmax, float gamma,
                                float toe, float shoulder, float speed)
{
    float t = toe < kKneeMin ? kKneeMin : toe;
    float s = shoulder < kKneeMin ? kKneeMin : shoulder;
    float d1 = Dmin + (gamma / t) * softplusf(t * (logH - speed));
    return Dmax - (1.0f / s) * softplusf(s * (Dmax - d1));
}
__device__ inline float chainDensity(float stops, int ch, const SpeakProfile& p)
{
    float logH = stops * kLog10_2;
    float Dneg = hdCurve(logH, p.negDmin[ch], p.negDmax[ch], p.negGamma[ch],
                         p.negToe[ch], p.negShoulder[ch], p.negSpeed[ch]);
    float printerOff = (p.printerMaster + p.printerLights[ch]) * kPrinterPt;
    float logHprn = -Dneg + printerOff;
    return hdCurve(logHprn, p.prnDmin[ch], p.prnDmax[ch], p.prnGamma[ch],
                   p.prnToe[ch], p.prnShoulder[ch], p.prnSpeed[ch]);
}
__device__ inline float toneChannel(float lin, int ch, const SpeakProfile& p)
{
    float stops = log2f((lin < kLinTiny ? kLinTiny : lin) / k18Gray);
    float Dprn = chainDensity(stops, ch, p);
    float Dref = chainDensity(0.0f, ch, p);
    return k18Gray * pow10f(-(Dprn - Dref));
}
__device__ inline int wfColOf(int x, int W)
{
    int c = x * SPEAK_WF_COLS / (W > 0 ? W : 1);
    return c < 0 ? 0 : (c >= SPEAK_WF_COLS ? SPEAK_WF_COLS - 1 : c);
}
__device__ inline int wfRowOf(float D)
{
    int r = (int)(D / SPEAK_WF_DMAX * SPEAK_WF_ROWS);
    return r < 0 ? 0 : (r >= SPEAK_WF_ROWS ? SPEAK_WF_ROWS - 1 : r);
}
__device__ inline int wfIdx(int ch, int col, int row)
{
    return SPEAK_STATS_WF + ch * (SPEAK_WF_COLS * SPEAK_WF_ROWS) + col * SPEAK_WF_ROWS + row;
}

__device__ inline int expBinOf(float stops)
{
    int b = (int)((stops + 6.0f) / 12.0f * SPEAK_EXP_BINS);
    return b < 0 ? 0 : (b >= SPEAK_EXP_BINS ? SPEAK_EXP_BINS - 1 : b);
}
__device__ inline float pixelStops(int cs, float r, float g, float b)
{
    float m = (decodeToLinear(cs, r) + decodeToLinear(cs, g) + decodeToLinear(cs, b)) * (1.0f / 3.0f);
    return log2f((m < kLinTiny ? kLinTiny : m) / k18Gray);
}

__device__ inline float density10(float lin)
{
    return -log2f(lin < 1e-6f ? 1e-6f : lin) * kLog10_2;
}
__device__ inline float softCapKnee(float d, float cap)
{
    if (cap <= 0.0f) return d;
    return cap - (1.0f / 8.0f) * softplusf(8.0f * (cap - d));
}
__device__ inline void subtractiveColor(float r, float g, float b, const SpeakProfile& p,
                                        float& oR, float& oG, float& oB)
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
__device__ inline bool dyeActive(const SpeakProfile& p)
{
    return p.subSat[0] != 0.0f || p.subSat[1] != 0.0f || p.subSat[2] != 0.0f ||
           p.dyeCouple[1] != 0.0f || p.dyeCouple[2] != 0.0f || p.dyeCouple[3] != 0.0f ||
           p.dyeCouple[5] != 0.0f || p.dyeCouple[6] != 0.0f || p.dyeCouple[7] != 0.0f;
}

__device__ inline float smooth01(float t) { t = clampf(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
__device__ inline void splitWeights(float Dbar, const SpeakProfile& p, float& wShadow, float& wHigh)
{
    float grayD  = 0.744727f;
    float pivotD = grayD - p.splitPivot * kLog10_2;
    float halfW  = 0.25f + 1.5f * clampf(p.splitBalance, 0.0f, 1.0f);
    float x = (Dbar - pivotD) / halfW;
    wShadow = smooth01(x);
    wHigh   = smooth01(-x);
}
__device__ inline void splitTone(float r, float g, float b, const SpeakProfile& p,
                                 float& oR, float& oG, float& oB)
{
    float DR = density10(r), DG = density10(g), DB = density10(b);
    float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    float wS, wH;
    splitWeights(Dbar, p, wS, wH);
    oR = pow10f(-(DR + wS * p.splitShadow[0] + wH * p.splitHigh[0]));
    oG = pow10f(-(DG + wS * p.splitShadow[1] + wH * p.splitHigh[1]));
    oB = pow10f(-(DB + wS * p.splitShadow[2] + wH * p.splitHigh[2]));
}
__device__ inline bool splitActive(const SpeakProfile& p)
{
    return p.splitShadow[0] != 0.0f || p.splitShadow[1] != 0.0f || p.splitShadow[2] != 0.0f ||
           p.splitHigh[0] != 0.0f || p.splitHigh[1] != 0.0f || p.splitHigh[2] != 0.0f;
}

// ---------------------------------------------------------------------------
// HALATION (Phase 4) — port of speak_core.h's halation block. See that file for
// the physics and for why the injection point is the NEGATIVE'S EXPOSURE and not
// the negative->print gap. Atomics-free by construction (box decimation is
// order-independent) — the pyramid must never depend on them.
//
// Functions the HOST also needs (buffer sizing + per-level dispatch geometry)
// are __host__ __device__; the rest stay __device__.
// ---------------------------------------------------------------------------
#define kHalSigmaC   0.645497f   // sqrt(0.25 + 1/6): sigma_L / 2^L
#define kHalCoreFall 3.0f        // octaves tighter than target: fast cut
#define kHalSkirtFall 1.0f       // octaves broader: halve per octave

__device__ const float kHalWeight[3] = { 1.0f, 0.30f, 0.10f };
__device__ const float kHalDec[4] = { 0.125f, 0.375f, 0.375f, 0.125f };

__device__ inline float halExcess(float lin, float thresh)
{
    float l = lin < 0.0f ? 0.0f : lin;
    return l > thresh ? (l - thresh) : 0.0f;
}
__host__ __device__ inline float halAmountOf(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) ? pr.profile.halAmount : 0.0f;
}
__device__ inline float halSigmaPx(int H, const SpeakParams& pr)
{
    float s = pr.profile.halRadius * 0.01f * (float)H;
    return s < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : s;
}
__host__ __device__ inline bool halActive(const SpeakParams& pr)
{
    return (pr.enableTone != 0) && (pr.strength > 0.0f) && (halAmountOf(pr) > 0.0f);
}

// ---- pyramid geometry (identical arithmetic in all four backends) ----
__host__ __device__ inline int halLevelCount(int W, int H)
{
    int n = 1, w = W, h = H;
    while (n < SPEAK_HAL_MAXLEV && w > SPEAK_HAL_MINDIM && h > SPEAK_HAL_MINDIM) {
        w = (w + 1) / 2; h = (h + 1) / 2; n++;
    }
    return n;
}
__host__ __device__ inline void halLevelInfo(int W, int H, int L, int& lw, int& lh, int& off)
{
    int w = W, h = H, o = 0;
    for (int i = 0; i < L; ++i) { o += w * h; w = (w + 1) / 2; h = (h + 1) / 2; }
    lw = w; lh = h; off = o;
}
__host__ __device__ inline int halArenaPixels(int W, int H)
{
    int lw, lh, off;
    int n = halLevelCount(W, H);
    halLevelInfo(W, H, n, lw, lh, off);
    return off;
}

// Level L's effective full-res sigma (decimation + upsample terms — see the
// core). Used by the host/tests only; kept here for textual parallelism.
__device__ inline float halLevelSigma(int L)
{
    float q = exp2f(2.0f * (float)L);
    float up = (L > 0) ? (q * (1.0f / 6.0f)) : 0.0f;
    return sqrtf(0.25f * (q - 1.0f) + up);
}

__device__ inline float halLevelWeight(int L, float sigmaTarget)
{
    float s = sigmaTarget < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : sigmaTarget;
    float Lt = log2f(s / kHalSigmaC);
    float d = (float)L - Lt;
    return (d <= 0.0f) ? exp2f(kHalCoreFall * d) : exp2f(-kHalSkirtFall * d);
}

// ---- pyramid taps ----
__device__ inline float halFetch(const float* arena, int off, int lw, int lh, int x, int y, int c)
{
    int xx = x < 0 ? 0 : (x >= lw ? lw - 1 : x);
    int yy = y < 0 ? 0 : (y >= lh ? lh - 1 : y);
    return arena[((size_t)off + (size_t)yy * lw + xx) * 3 + c];
}

__device__ inline void halDecimatePixel(const float* arena, int sOff, int sW, int sH,
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

// Cubic B-spline read of one level onto a (W,H) grid — B-spline, NOT bilinear
// and NOT Catmull-Rom. See the core: bilinear is only C0 and its derivative
// discontinuity at every texel boundary is what made the skirt read as hard
// rectangular BLOCKS; Catmull-Rom's overshoot would push a scatter field below
// zero. The B-spline weights are C2, non-negative and sum to 1.
__device__ inline void halBSpline(float t, float* w)
{
    float t2 = t * t, t3 = t2 * t;
    w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) * (1.0f / 6.0f);
    w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) * (1.0f / 6.0f);
    w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) * (1.0f / 6.0f);
    w[3] = t3 * (1.0f / 6.0f);
}
__device__ inline float halSampleLevel(const float* arena, int off, int lw, int lh,
                                       int W, int H, int x, int y, int c)
{
    float fx = ((float)x + 0.5f) * (float)lw / (float)W - 0.5f;
    float fy = ((float)y + 0.5f) * (float)lh / (float)H - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);   // floorf: fx/fy go negative at the edge
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
// energy-preserving. Same loop, same halLevelWeight, on every backend.
__device__ inline float halWeightSum(int nLev, float sigmaTarget)
{
    float wsum = 0.0f;
    for (int L = 0; L < nLev; ++L) wsum += halLevelWeight(L, sigmaTarget);
    return wsum;
}

// COARSE-TO-FINE ACCUMULATE — one pixel of level L (replaces halScatterAt,
// which read EVERY level directly at full res with a bilinear tap):
//     acc_L = w_L * level_L + upsample_2x(acc_{L+1})
// running L = nLev-1 down to 0, in place in the arena. acc_0 is then the whole
// (unnormalized) mixture at full resolution.
//
// Note the dst dims handed to halSampleLevel are (lw, lh) — LEVEL L's grid, NOT
// (W, H): the coarser level is upsampled ONE OCTAVE onto level L, so each step
// interpolates only between ADJACENT samples and is then re-filtered by every
// step below it. Measured: worst-case angular variation 3.72 -> 0.13. It is also
// strictly CHEAPER — sum_L (level L's pixels) = 4/3 N, against nLev*N.
//
// IN-PLACE SAFETY: a thread reads its OWN (x,y) at level L plus a neighbourhood
// at level L+1 — a DISJOINT arena region, finished by the previous dispatch —
// and writes only its own (x,y) at level L. No thread reads another thread's
// level-L pixel, so in place is safe and no atomics are involved.
//
// The level geometry (both levels' dims + offsets, L and nLev) is passed IN from
// the host rather than recomputed from L in-kernel — same as SpeakMetalKernel and
// SpeakOpenCLKernel, and for the reason recorded there: the accumulate holds the
// arena WRITABLE, and the back-to-back halLevelInfo calls were observed to
// misbehave under that. The host computes them with the SAME halLevelInfo, which
// is deterministic integer arithmetic, so the values agree with the core's by
// construction and parity is untouched. `sigmaTarget` stays in-kernel so it
// cannot drift from the normalize pass.
__device__ inline void halAccumPixel(float* arena, int L, int nLev, float sigmaTarget,
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

// `scat*` are the energy-normalized, blurred, PER-CHANNEL scene-linear highlight
// excesses at this pixel (0 when halation is off). Deliberately NOT defaulted —
// see the core: a scatter-blind density parade is an L3 bug parity CANNOT catch.
__device__ inline void lookLinear(float r, float g, float b,
                                  float scatR, float scatG, float scatB,
                                  const SpeakParams& pr,
                                  float& oR, float& oG, float& oB)
{
    int cs = pr.inputColorSpace;
    float lr = decodeToLinear(cs, r);
    float lg = decodeToLinear(cs, g);
    float lb = decodeToLinear(cs, b);
    float mr = lr, mg = lg, mb = lb;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        float s = clampf(pr.strength, 0.0f, 1.0f);
        float a = halAmountOf(pr);
        // RE-EXPOSURE: the scattered light adds to the scene light ENTERING the
        // negative, so it goes in the CURVE'S ARGUMENT. The dry side of the mix
        // stays lr, NOT the re-exposed value — see the core.
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

__device__ inline float scopeYStops(float inStops, int ch, const SpeakParams& pr)
{
    float lin = k18Gray * exp2f(inStops);
    float outLin = lin;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        float s = clampf(pr.strength, 0.0f, 1.0f);
        outLin = lerpf(lin, toneChannel(lin, ch, pr.profile), s);
    }
    return log2f((outLin < kLinTiny ? kLinTiny : outLin) / k18Gray);
}

__device__ inline bool hdScopePixel(int x, int y, int W, int H, const SpeakParams& pr,
                                    const unsigned int* stats,
                                    float& outR, float& outG, float& outB)
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

    float rowStops = 6.0f - 12.0f * ((float)gy / (plotH - 1));
    int gcol0 = (int)((0.0f + 6.0f) / 12.0f * (plotW - 1) + 0.5f);
    int grow0 = (int)((6.0f - 0.0f) / 12.0f * (plotH - 1) + 0.5f);
    if (gx == gcol0 || gy == grow0) { outR = 0.24f; outG = 0.24f; outB = 0.24f; return true; }
    if ((gx % (plotW / 6)) == 0 || (gy % (plotH / 6)) == 0) { outR = 0.13f; outG = 0.13f; outB = 0.13f; }

    unsigned int hmax = stats[SPEAK_STATS_HIST_MAX];
    if (hmax > 0u) {
        int hb = expBinOf(-6.0f + 12.0f * ((float)gx / (plotW - 1)));
        float f = (float)(stats[SPEAK_STATS_HIST_EXP + hb]) / (float)hmax;
        int barH = (int)(sqrtf(f) * (plotH * 0.45f) + 0.5f);
        if (gy >= plotH - barH) { outR = 0.16f; outG = 0.19f; outB = 0.24f; }
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

__device__ inline bool densityScopePixel(int x, int y, int W, int H, const SpeakParams& pr,
                                         const unsigned int* stats,
                                         float& outR, float& outG, float& outB)
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

    int rowGray = (int)(0.744727f / SPEAK_WF_DMAX * (plotH - 1) + 0.5f);
    if (gy == 0) { outR = 0.35f; outG = 0.35f; outB = 0.35f; return true; }
    if (gy == rowGray) { outR = 0.30f; outG = 0.26f; outB = 0.12f; return true; }

    int chW = plotW / 3;
    int ch = gx / (chW > 0 ? chW : 1);
    if (ch > 2) ch = 2;
    if (gx - ch * chW == 0 && ch > 0) { outR = 0.16f; outG = 0.16f; outB = 0.16f; return true; }

    unsigned int wmax = stats[SPEAK_STATS_WF_MAX];
    if (wmax > 0u) {
        int within = gx - ch * chW;
        int wcol = within * SPEAK_WF_COLS / (chW > 0 ? chW : 1);
        int wrow = gy * SPEAK_WF_ROWS / plotH;
        unsigned int c = stats[wfIdx(ch, wcol < SPEAK_WF_COLS ? wcol : SPEAK_WF_COLS - 1,
                                     wrow < SPEAK_WF_ROWS ? wrow : SPEAK_WF_ROWS - 1)];
        if (c > 0u) {
            float inten = sqrtf((float)c / (float)wmax);
            float v = 0.12f + 0.88f * (inten > 1.0f ? 1.0f : inten);
            outR = (ch == 0) ? v : 0.05f;
            outG = (ch == 1) ? v : 0.05f;
            outB = (ch == 2) ? v : 0.05f;
        }
    }
    return true;
}

__device__ inline void deliverInput(const SpeakParams& pr, float r, float g, float b,
                                    float& oR, float& oG, float& oB)
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

__device__ inline void processPixel(float r, float g, float b,
                                    float scatR, float scatG, float scatB,
                                    int x, int y, int W, int H,
                                    const SpeakParams& pr, const unsigned int* stats,
                                    float& outR, float& outG, float& outB)
{
    int cs = pr.inputColorSpace;
    bool toneOn = (pr.enableTone != 0) && (pr.strength > 0.0f);
    bool dyeOn = (pr.enableDye != 0) && dyeActive(pr.profile);
    bool splitOn = (pr.enableSplit != 0) && splitActive(pr.profile);
    bool bake = (pr.outputMode == 1);
    if (!toneOn && !dyeOn && !splitOn && !bake) {
        // Halation cannot reach here: halActive() requires toneOn.
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
    // NOT auto-normalized — see the core.
    if (pr.viewMode == SPEAK_VIEW_SCATTER) {
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

// ---------------------------------------------------------------------------
// The scatter pyramid — a textual port of speak_core.h's buildHalScatter, split
// across FOUR kernels (excess -> decimate -> accum -> normalize) because a GPU
// cannot carry the level barriers inside one kernel: decimate and accum are each
// dispatched ONCE PER LEVEL. They all run on the SAME stream as the stats/main
// kernels, and STREAM ORDER IS THE BARRIER: each launch completes before the next
// begins, so the level-L decimation always reads a finished level L-1, the
// level-L accumulate always reads a finished level L+1, and the stats pass always
// reads a finished scatter plane. No explicit sync, and no atomics anywhere in
// this chain (decimation and accumulation are both order-independent).
// ---------------------------------------------------------------------------

// Level 0 of the arena: the per-channel scene-linear highlight excess.
// THRESHOLD BEFORE DECIMATION — mean(max(0, l-t)) != max(0, mean(l)-t).
__global__ void SpeakExcessKernel(SpeakParams p, int W, int H,
                                  const float4* src, float* arena)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int cs = p.inputColorSpace;
    float th = p.profile.halThresh;
    float4 s = src[y * W + x];
    size_t o = ((size_t)y * W + x) * 3;   // level 0's arena offset is 0
    arena[o + 0] = halExcess(decodeToLinear(cs, s.x), th);
    arena[o + 1] = halExcess(decodeToLinear(cs, s.y), th);
    arena[o + 2] = halExcess(decodeToLinear(cs, s.z), th);
}

// One dispatch PER LEVEL L = 1..nLev-1: [1,3,3,1]/8 decimation of level L-1 into
// level L. The read and write regions are disjoint slices of the same arena.
__global__ void SpeakDecimateKernel(int W, int H, int L, float* arena)
{
    int sw, sh, so, dw, dh, doff;
    halLevelInfo(W, H, L - 1, sw, sh, so);
    halLevelInfo(W, H, L,     dw, dh, doff);
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    float v[3];
    halDecimatePixel(arena, so, sw, sh, x, y, v);
    size_t o = ((size_t)doff + (size_t)y * dw + x) * 3;
    arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
}

// One dispatch PER LEVEL L = nLev-1 down to 0: the coarse-to-fine accumulate,
// IN PLACE in the arena, with the grid = LEVEL L's dims. All the level geometry
// comes from the host's halLevelInfo — see halAccumPixel.
//
// In-place is safe: see halAccumPixel. Each thread reads its own (x,y) at level
// L plus a neighbourhood at level L+1 (a disjoint arena region) and writes only
// its own (x,y) at level L.
//
// The ORDER is load-bearing — level L reads level L+1 as rewritten by the
// PREVIOUS dispatch — and stream ordering supplies that barrier for free: each
// launch on the stream completes before the next begins, so no explicit
// inter-level sync is needed.
__global__ void SpeakAccumKernel(SpeakParams p, int H, int L, int nLev,
                                 int lw, int lh, int off, int cw, int ch, int coff,
                                 float* arena)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= lw || y >= lh) return;
    float sig = halSigmaPx(H, p);
    float v[3];
    halAccumPixel(arena, L, nLev, sig, lw, lh, off, cw, ch, coff, x, y, v);
    size_t o = ((size_t)off + (size_t)y * lw + x) * 3;
    arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
}

// Full res: normalize level 0 of the accumulated arena into the scatter plane.
// sum(w) = 1 => energy preserved. halWeightSum runs in-kernel against the same
// __device__ halLevelWeight the accumulate used, so there is no host/device math
// path that could drift from the reference.
__global__ void SpeakNormalizeKernel(SpeakParams p, int W, int H, int nLev,
                                     const float* arena, float* scat)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float sig = halSigmaPx(H, p);
    float inv = 1.0f / halWeightSum(nLev, sig);
    size_t o = ((size_t)y * W + x) * 3;   // level 0's arena offset is 0
    scat[o + 0] = arena[o + 0] * inv;
    scat[o + 1] = arena[o + 1] * inv;
    scat[o + 2] = arena[o + 2] * inv;
}

// Scope measurement pass: bin the frame on a stride-2 grid. Integer atomics are
// order-independent, so the counts are identical on every backend.
// `scat` is the W*H*3 scatter field, or null when halation is off — the density
// parade MUST see it, or it draws the halo darker than the pixels became.
__global__ void SpeakStatsKernel(SpeakParams p, int W, int H,
                                 const float4* src, const float* scat, unsigned int* stats)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (x >= W || y >= H) return;
    float4 s = src[y * W + x];
    size_t j = ((size_t)y * W + x) * 3;
    if (p.scopeHD != 0) {
        int bin = expBinOf(pixelStops(p.inputColorSpace, s.x, s.y, s.z));
        atomicAdd(&stats[SPEAK_STATS_HIST_EXP + bin], 1u);
    }
    if (p.scopeDensity != 0) {
        float mr, mg, mb;
        lookLinear(s.x, s.y, s.z,
                   scat ? scat[j + 0] : 0.0f, scat ? scat[j + 1] : 0.0f,
                   scat ? scat[j + 2] : 0.0f, p, mr, mg, mb);
        int col = wfColOf(x, W);
        atomicAdd(&stats[wfIdx(0, col, wfRowOf(density10(mr)))], 1u);
        atomicAdd(&stats[wfIdx(1, col, wfRowOf(density10(mg)))], 1u);
        atomicAdd(&stats[wfIdx(2, col, wfRowOf(density10(mb)))], 1u);
    }
}

__global__ void SpeakStatsMaxKernel(unsigned int* stats)
{
    unsigned int mx = 0u;
    for (int b = 0; b < SPEAK_EXP_BINS; ++b)
        if (stats[SPEAK_STATS_HIST_EXP + b] > mx) mx = stats[SPEAK_STATS_HIST_EXP + b];
    stats[SPEAK_STATS_HIST_MAX] = mx;
    unsigned int wmx = 0u;
    for (int k = 0; k < SPEAK_WF_COLS * SPEAK_WF_ROWS * 3; ++k)
        if (stats[SPEAK_STATS_WF + k] > wmx) wmx = stats[SPEAK_STATS_WF + k];
    stats[SPEAK_STATS_WF_MAX] = wmx;
}

__global__ void SpeakKernel(SpeakParams p, int W, int H,
                            const float4* src, const float* scat, float4* dst,
                            const unsigned int* stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
    size_t j = ((size_t)y * W + x) * 3;
    float4 s = src[i];
    float oR, oG, oB;
    processPixel(s.x, s.y, s.z,
                 scat ? scat[j + 0] : 0.0f, scat ? scat[j + 1] : 0.0f,
                 scat ? scat[j + 2] : 0.0f,
                 x, y, W, H, p, stats, oR, oG, oB);
    float4 o; o.x = oR; o.y = oG; o.z = oB; o.w = s.w;
    dst[i] = o;
}

// Host side ------------------------------------------------------------------
// The per-stream resource cache is shared mutable state: Resolve can render on
// several streams concurrently, and std::map insertion is not thread-safe, so
// every touch of s_res is under this lock (same class, same style as Hush's
// CudaKernel.cu).
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

struct SpeakCudaRes
{
    unsigned int* stats = nullptr;   // fixed size — allocated once
    // The scatter buffers are SIZE-DEPENDENT, unlike stats: the host hands us a
    // different frame size for proxy vs full res (and for every clip), so the
    // allocated size is stored alongside the pointer and the pair is rebuilt
    // whenever it changes. Getting this wrong is a buffer overrun on the first
    // proxy->full-res switch. Lazy: a render with halation off never allocates.
    float* arena = nullptr;          // halArenaPixels(W,H) * 3 floats
    float* scat = nullptr;           // W * H * 3 floats
    int halW = 0, halH = 0;          // frame size arena/scat were allocated for
};

} // namespace

void RunCudaSpeak(void* p_Stream, int p_Width, int p_Height,
                  const SpeakParams& p_Params, const float* p_Src, float* p_Dst)
{
    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);

    // Mirrors speakFrame: build the scatter only when halation is live, so the
    // identity path stays bit-exact AND free (no allocation, no dispatches).
    const bool hal = halActive(p_Params) || (p_Params.viewMode == SPEAK_VIEW_SCATTER);
    const int nLev = hal ? halLevelCount(p_Width, p_Height) : 0;
    const int arenaPix = hal ? halArenaPixels(p_Width, p_Height) : 0;

    // One resource set per stream, allocated lazily (mirrors Hush's per-stream
    // resource cache, lock included).
    static std::map<void*, SpeakCudaRes> s_res;
    static Locker s_locker;

    SpeakCudaRes res;
    s_locker.Lock();
    {
        SpeakCudaRes& r = s_res[p_Stream];
        if (!r.stats)
            cudaMalloc(&r.stats, SPEAK_STATS_UINTS * sizeof(unsigned int));
        // Re-allocate the scatter pair whenever the frame size changes — the
        // size check lives here (not at first use) because a size change while
        // halation was off must still be caught the next time it comes on.
        if (hal && (!r.arena || !r.scat || r.halW != p_Width || r.halH != p_Height)) {
            if (r.arena) { cudaFree(r.arena); r.arena = nullptr; }
            if (r.scat)  { cudaFree(r.scat);  r.scat  = nullptr; }
            cudaMalloc(&r.arena, static_cast<size_t>(arenaPix) * 3 * sizeof(float));
            cudaMalloc(&r.scat, static_cast<size_t>(p_Width) * p_Height * 3 * sizeof(float));
            r.halW = p_Width;
            r.halH = p_Height;
        }
        res = r;
    }
    s_locker.Unlock();

    // Null when halation is off, exactly as speakFrame passes nullptr — the
    // kernels guard the read the same way computeStats/processPixel do. (CUDA
    // takes a null pointer argument happily; it is simply never dereferenced.)
    const float* sc = hal ? res.scat : nullptr;

    dim3 block(16, 16, 1);
    dim3 grid((p_Width + block.x - 1) / block.x, (p_Height + block.y - 1) / block.y, 1);

    // Dispatch order, mirroring buildHalScatter's three stages:
    //   excess -> decimate(L=1..nLev-1) -> accum(L=nLev-1..0) -> normalize
    //          -> stats -> main.
    // The accumulate runs COARSE TO FINE, one dispatch per level, each reading
    // the level above it as rewritten by the previous dispatch. Those inter-level
    // barriers are FREE: everything here is on one stream, and stream order means
    // each launch completes before the next begins. The scatter must exist BEFORE
    // stats, because the density parade measures the halated result.
    if (hal) {
        SpeakExcessKernel<<<grid, block, 0, stream>>>(p_Params, p_Width, p_Height,
                                                      reinterpret_cast<const float4*>(p_Src),
                                                      res.arena);
        for (int L = 1; L < nLev; ++L) {
            int lw, lh, off;
            halLevelInfo(p_Width, p_Height, L, lw, lh, off);
            dim3 gridL((lw + block.x - 1) / block.x, (lh + block.y - 1) / block.y, 1);
            SpeakDecimateKernel<<<gridL, block, 0, stream>>>(p_Width, p_Height, L, res.arena);
        }
        // The level geometry is computed HOST-side by the same halLevelInfo the
        // kernels use and passed in — the accumulate holds the arena writable,
        // and this is the pattern that has worked for that (see halAccumPixel).
        // Level L+1's geometry is computed unconditionally; at L = nLev-1 the
        // kernel takes the coarsest-level branch and simply never reads it.
        for (int L = nLev - 1; L >= 0; --L) {
            int lw, lh, off, cw, ch, coff;
            halLevelInfo(p_Width, p_Height, L,     lw, lh, off);
            halLevelInfo(p_Width, p_Height, L + 1, cw, ch, coff);
            dim3 gridL((lw + block.x - 1) / block.x, (lh + block.y - 1) / block.y, 1);
            SpeakAccumKernel<<<gridL, block, 0, stream>>>(p_Params, p_Height, L, nLev,
                                                          lw, lh, off, cw, ch, coff, res.arena);
        }
        SpeakNormalizeKernel<<<grid, block, 0, stream>>>(p_Params, p_Width, p_Height, nLev,
                                                         res.arena, res.scat);
    }

    cudaMemsetAsync(res.stats, 0, SPEAK_STATS_UINTS * sizeof(unsigned int), stream);
    if (p_Params.scopeHD != 0 || p_Params.scopeDensity != 0) {
        // ceil(W/2) sample threads, not floor(W/2) — see SpeakOpenCLKernel.cpp:
        // the kernel indexes x = gid*2, so an odd W needs (W+1)/2 columns, and
        // floor(W/2) fell one column short at W = 1 (mod 32) (e.g. 1921, 3841).
        dim3 gridH(((p_Width + 1) / 2 + block.x - 1) / block.x,
                   ((p_Height + 1) / 2 + block.y - 1) / block.y, 1);
        SpeakStatsKernel<<<gridH, block, 0, stream>>>(p_Params, p_Width, p_Height,
                                                      reinterpret_cast<const float4*>(p_Src),
                                                      sc, res.stats);
        SpeakStatsMaxKernel<<<1, 1, 0, stream>>>(res.stats);
    }

    SpeakKernel<<<grid, block, 0, stream>>>(p_Params, p_Width, p_Height,
                                            reinterpret_cast<const float4*>(p_Src), sc,
                                            reinterpret_cast<float4*>(p_Dst), res.stats);
}
