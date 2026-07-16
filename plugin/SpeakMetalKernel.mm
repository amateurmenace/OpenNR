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

inline void lookLinear(float r, float g, float b, constant SpeakParams& pr,
                       thread float& oR, thread float& oG, thread float& oB)
{
    int cs = pr.inputColorSpace;
    float lr = decodeToLinear(cs, r);
    float lg = decodeToLinear(cs, g);
    float lb = decodeToLinear(cs, b);
    float mr = lr, mg = lg, mb = lb;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        float s = clampf(pr.strength, 0.0f, 1.0f);
        mr = lerpf(lr, toneChannel(lr, 0, pr.profile), s);
        mg = lerpf(lg, toneChannel(lg, 1, pr.profile), s);
        mb = lerpf(lb, toneChannel(lb, 2, pr.profile), s);
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

inline void processPixel(float r, float g, float b, int x, int y, int W, int H,
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
        lookLinear(r, g, b, pr, mr, mg, mb);
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

    float sr, sg, sb;
    if (hdScopePixel(x, y, W, H, pr, stats, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
    if (densityScopePixel(x, y, W, H, pr, stats, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
}

// Scope measurement pass: bin the frame on a stride-2 grid. Integer atomics are
// order-independent, so the counts are identical on every backend.
kernel void SpeakStatsKernel(constant SpeakParams& p [[buffer(0)]],
                             constant int& W [[buffer(1)]],
                             constant int& H [[buffer(2)]],
                             device const float* src [[buffer(3)]],
                             device atomic_uint* stats [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]])
{
    int x = int(gid.x) * 2, y = int(gid.y) * 2;
    if (x >= W || y >= H) return;
    int i = (y * W + x) * 4;
    if (p.scopeHD != 0) {
        int bin = expBinOf(pixelStops(p.inputColorSpace, src[i + 0], src[i + 1], src[i + 2]));
        atomic_fetch_add_explicit(&stats[SPEAK_STATS_HIST_EXP + bin], 1u, memory_order_relaxed);
    }
    if (p.scopeDensity != 0) {
        float mr, mg, mb;
        lookLinear(src[i + 0], src[i + 1], src[i + 2], p, mr, mg, mb);
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
                        uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= W || (int)gid.y >= H) return;
    int x = int(gid.x), y = int(gid.y);
    int i = (y * W + x) * 4;
    float oR, oG, oB;
    processPixel(src[i + 0], src[i + 1], src[i + 2], x, y, W, H, p, stats, oR, oG, oB);
    dst[i + 0] = oR; dst[i + 1] = oG; dst[i + 2] = oB; dst[i + 3] = src[i + 3];
}
)MSL";

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
struct SpeakRes {
    id<MTLComputePipelineState> main = nil;
    id<MTLComputePipelineState> stats = nil;
    id<MTLComputePipelineState> statsMax = nil;
    id<MTLBuffer> statsBuf = nil;
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
            if (!r.main || !r.stats || !r.statsMax) { fprintf(stderr, "Speak: pipeline failed\n"); return; }
        }
        if (r.statsBuf == nil)
            r.statsBuf = [device newBufferWithLength:(SPEAK_STATS_UINTS * sizeof(uint32_t))
                                             options:MTLResourceStorageModePrivate];
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
    if (wantStats) {
        [enc setComputePipelineState:res.stats];
        [enc setBytes:&params length:sizeof(SpeakParams) atIndex:0];
        [enc setBytes:&W length:sizeof(int) atIndex:1];
        [enc setBytes:&H length:sizeof(int) atIndex:2];
        [enc setBuffer:src offset:0 atIndex:3];
        [enc setBuffer:res.statsBuf offset:0 atIndex:4];
        const MTLSize gh = MTLSizeMake((p_Width / 2 + 16) / 16, (p_Height / 2 + 16) / 16, 1);
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
    const MTLSize grid = MTLSizeMake((p_Width + 15) / 16, (p_Height + 15) / 16, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmdBuf commit];
}
