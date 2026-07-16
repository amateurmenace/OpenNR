// Speak — film-reconstruction core (the CPU reference and single source of
// truth for the algorithm). The three GPU kernels (SpeakMetalKernel.mm,
// SpeakCudaKernel.cu, SpeakOpenCLKernel.cpp) are line-by-line ports of this
// file; ANY change to the math must be applied to all four and verified with
// test/test_speak_metal (parity ~2e-5 mean). Keep constants, curve formulas
// and loop order textually parallel — same discipline as Hush's nr_core.h.
//
// Phase 1 — the density spine:
//   color-manage in  ->  Log-Exposure Spine (log2, 18% gray = 0)
//                    ->  per-channel Negative H&D characteristic curves
//                    ->  Printer Lights (per-channel logE offset, in the gap)
//                    ->  per-channel Print H&D characteristic curves
//                    ->  positive transmittance  ->  color-manage out
//   + a live, deterministic H&D curve scope rendered INTO the image.
//
// Every look is gated behind `strength`: at strength 0 the node is a bit-exact
// pass-through (identity), and for a gray-balanced profile a neutral input maps
// to a neutral output exactly, by construction (see processPixel).
//
// MIT License.

#ifndef OPENNR_SPEAK_CORE_H
#define OPENNR_SPEAK_CORE_H

#include <cmath>
#include <cstdint>

#include "SpeakParams.h"

namespace speakcore {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const float kLog10_2    = 0.301029996f; // log10(2): stops -> log10 exposure
static const float k18Gray     = 0.18f;        // scene-linear middle gray datum
static const float kPrinterPt  = 0.025f;       // 1 printer point = 0.025 log10 E
static const float kLinTiny    = 1e-8f;        // floor before log2 (avoids -inf)
static const float kKneeMin    = 0.05f;        // min toe/shoulder sharpness

// DaVinci Intermediate transfer (verified against the colour-science reference
// and Blackmagic's DWG/DI white paper). Encode/decode are exact inverses, so
// the working round-trip is lossless by construction.
static const float kDI_A       = 0.0075f;
static const float kDI_B       = 7.0f;
static const float kDI_C       = 0.07329248f;
static const float kDI_M       = 10.44426855f;
static const float kDI_LIN_CUT = 0.00262409f;
static const float kDI_LOG_CUT = 0.02740668f;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float pow10f(float x) { return std::exp2(x * 3.32192809f); } // 10^x via exp2

// Numerically stable softplus: log(1 + e^z) = max(z,0) + log(1 + e^-|z|).
// Written with primitives every backend has (no log1p) so the four ports are
// textually identical. Monotone increasing, C-infinity; the H&D building block.
static inline float softplusf(float z)
{
    const float az = z < 0.0f ? -z : z;
    return (z > 0.0f ? z : 0.0f) + std::log(1.0f + std::exp(-az));
}

// ---------------------------------------------------------------------------
// Color management — per-channel transfer only (gamut is preserved: the tone
// spine operates per channel, so no gamut matrix is needed here). The declared
// working space is DaVinci Wide Gamut / Intermediate.
// ---------------------------------------------------------------------------
static inline float diDecode(float v)   // DI -> scene-linear
{
    return (v <= kDI_LOG_CUT) ? (v / kDI_M)
                              : (std::exp2(v / kDI_C - kDI_B) - kDI_A);
}
static inline float diEncode(float L)   // scene-linear -> DI
{
    return (L <= kDI_LIN_CUT) ? (L * kDI_M)
                              : ((std::log2(L + kDI_A) + kDI_B) * kDI_C);
}

// ACEScct (used only when the user declares an ACES timeline).
static inline float acesCctDecode(float v)
{
    if (v <= 0.155251141552511f) return (v - 0.0729055341958355f) / 10.5402377416545f;
    return std::exp2(v * 17.52f - 9.72f);
}
static inline float acesCctEncode(float L)
{
    if (L <= 0.0078125f) return 10.5402377416545f * L + 0.0729055341958355f;
    return (std::log2(L) + 9.72f) / 17.52f;
}

static inline float decodeToLinear(int cs, float v)
{
    switch (cs) {
        case SPEAK_CS_DWG_INTERMEDIATE: return diDecode(v);
        case SPEAK_CS_REC709_G24:       return v <= 0.0f ? 0.0f : std::pow(v, 2.4f);
        case SPEAK_CS_ACESCCT:          return acesCctDecode(v);
        case SPEAK_CS_DWG_LINEAR:
        case SPEAK_CS_LINEAR:
        default:                        return v;
    }
}
static inline float encodeFromLinear(int cs, float L)
{
    switch (cs) {
        case SPEAK_CS_DWG_INTERMEDIATE: return diEncode(L);
        case SPEAK_CS_REC709_G24:       return L <= 0.0f ? 0.0f : std::pow(L, 1.0f / 2.4f);
        case SPEAK_CS_ACESCCT:          return acesCctEncode(L);
        case SPEAK_CS_DWG_LINEAR:
        case SPEAK_CS_LINEAR:
        default:                        return L;
    }
}

// ---------------------------------------------------------------------------
// Gamut colorimetry (for the Bake-to-Rec.709 output mode). DaVinci Wide Gamut
// -> CIE XYZ (D65) is the published white-paper / colour-science matrix; XYZ ->
// linear Rec.709 is the standard D65 matrix. Both share D65 so a neutral maps
// to a neutral (the tiny residual is the published matrix's own rounding). The
// inverse matrices used by the round-trip CST gate live in the test.
// ---------------------------------------------------------------------------
static const float kDWG_to_XYZ[9] = {
    0.70062239f, 0.14877482f, 0.10105872f,
    0.27411851f, 0.87363190f,-0.14775041f,
   -0.09896291f,-0.13789533f, 1.32591599f };
static const float kXYZ_to_Rec709[9] = {
    3.24045420f,-1.53713850f,-0.49853140f,
   -0.96926600f, 1.87601080f, 0.04155600f,
    0.05564340f,-0.20402590f, 1.05722520f };

static inline void mul3(const float* m, float r, float g, float b,
                        float& oR, float& oG, float& oB)
{
    oR = m[0] * r + m[1] * g + m[2] * b;
    oG = m[3] * r + m[4] * g + m[5] * b;
    oB = m[6] * r + m[7] * g + m[8] * b;
}

// Working-space linear RGB -> linear Rec.709. Bake targets the DaVinci Wide
// Gamut working space (the documented use); Rec.709 in is a gamut-identity, and
// other declared spaces get a transfer-only bake (gamut left as-is) — stated in
// the UI hint so the mode never claims a conversion it does not perform.
static inline void gamutToRec709Lin(int cs, float r, float g, float b,
                                    float& oR, float& oG, float& oB)
{
    if (cs == SPEAK_CS_DWG_INTERMEDIATE || cs == SPEAK_CS_DWG_LINEAR) {
        float X, Y, Z;
        mul3(kDWG_to_XYZ, r, g, b, X, Y, Z);
        mul3(kXYZ_to_Rec709, X, Y, Z, oR, oG, oB);
    } else {
        oR = r; oG = g; oB = b;
    }
}

// ---------------------------------------------------------------------------
// The closed-form Hurter-Driffield characteristic curve  D(logH).
//
// Two stable softplus segments give independently tunable toe and shoulder
// while staying strictly monotone increasing for all inputs:
//   d1 = Dmin + (gamma/toe) * softplus( toe * (logH - speed) )   -- toe -> straight
//   D  = Dmax - (1/shoulder) * softplus( shoulder * (Dmax - d1) ) -- straight -> shoulder
// As toe,shoulder -> large this approaches a hard-clipped straight line of
// slope gamma between Dmin and Dmax; as they shrink the knees lengthen. The
// params ARE the plotted curve — no LUT, no interpolation drift.
// ---------------------------------------------------------------------------
static inline float hdCurve(float logH, float Dmin, float Dmax, float gamma,
                            float toe, float shoulder, float speed)
{
    const float t = toe      < kKneeMin ? kKneeMin : toe;
    const float s = shoulder < kKneeMin ? kKneeMin : shoulder;
    const float d1 = Dmin + (gamma / t) * softplusf(t * (logH - speed));
    return Dmax - (1.0f / s) * softplusf(s * (Dmax - d1));
}

// The full negative -> printer-light -> print cascade for one channel, in
// density. `stops` is the scene log2-exposure relative to 18% gray (the
// canonical Log-Exposure Spine datum). Returns the PRINT density D_prn.
static inline float chainDensity(float stops, int ch, const SpeakProfile& p)
{
    const float logH = stops * kLog10_2;
    // Negative characteristic curve.
    const float Dneg = hdCurve(logH, p.negDmin[ch], p.negDmax[ch], p.negGamma[ch],
                               p.negToe[ch], p.negShoulder[ch], p.negSpeed[ch]);
    // Print exposure = light through the negative (transmittance 10^-Dneg, i.e.
    // log10 exposure -Dneg) plus the printer-light timing offset, in the gap.
    const float printerOff = (p.printerMaster + p.printerLights[ch]) * kPrinterPt;
    const float logHprn = -Dneg + printerOff;
    // Print characteristic curve.
    return hdCurve(logHprn, p.prnDmin[ch], p.prnDmax[ch], p.prnGamma[ch],
                   p.prnToe[ch], p.prnShoulder[ch], p.prnSpeed[ch]);
}

// One channel of the density spine: scene-linear in -> scene-linear out.
// The positive is pivoted at 18% gray so the reference gray maps to itself
// (each channel divides out its own gray-reference density) — this is what
// makes a gray-balanced profile neutral-preserving by construction.
static inline float toneChannel(float lin, int ch, const SpeakProfile& p)
{
    const float stops = std::log2((lin < kLinTiny ? kLinTiny : lin) / k18Gray);
    const float Dprn  = chainDensity(stops, ch, p);
    const float Dref  = chainDensity(0.0f, ch, p);   // print density at 18% gray
    return k18Gray * pow10f(-(Dprn - Dref));
}

// ---------------------------------------------------------------------------
// Live H&D curve scope (deterministic — a pure function of the params, so it is
// parity-trivial). The curves are drawn by evaluating the SAME production
// hdCurve()/chainDensity() the pixels use, so the plot can never disagree with
// the filter (the "measured-transfer renderer" discipline). Panel anchors in
// DISPLAY space (yd = H-1-y) exactly like Hush's scopes.
//
// Slice 1 draws the applied per-channel system curve (input stops -> output
// stops, i.e. the whole negative->printer->print tone scale) on a stops grid,
// with an 18% gray crosshair and R/G/B legend swatches. Text labels and the
// live exposure histogram land in the next increment.
// ---------------------------------------------------------------------------
// The APPLIED transform for one input exposure, in stops. It mirrors the pixel
// path EXACTLY — including the Strength mix and the enable toggle — so the plot
// can never disagree with the pixels (at strength 0 it collapses to the y=x
// diagonal, matching the identity pass-through). Output encode is a display
// transform applied equally to curve and diagonal, so it cancels in stops.
static inline float scopeYStops(float inStops, int ch, const SpeakParams& pr)
{
    const float lin = k18Gray * std::exp2(inStops);
    float outLin = lin;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        const float s = clampf(pr.strength, 0.0f, 1.0f);
        outLin = lerpf(lin, toneChannel(lin, ch, pr.profile), s);
    }
    return std::log2((outLin < kLinTiny ? kLinTiny : outLin) / k18Gray);
}

// Returns true and writes an (r,g,b) display-space color if (x,y) is a scope
// pixel. `out*` are only touched when it returns true.
static inline bool hdScopePixel(int x, int y, int W, int H, const SpeakParams& pr,
                                float& outR, float& outG, float& outB)
{
    if (pr.scopeHD == 0) return false;

    const int sc = (H / 540) > 1 ? (H / 540) : 1;      // panel scale (as Hush)
    const int panelW = 220 * sc, panelH = 150 * sc;
    const int margin = 12 * sc;
    const int px0 = margin, py0 = margin;              // top-left in DISPLAY space
    const int yd = H - 1 - y;                          // display row
    const int lx = x - px0, ly = yd - py0;             // panel-local coords
    if (lx < 0 || ly < 0 || lx >= panelW || ly >= panelH) return false;

    // Plot area inside a small inset.
    const int pad = 6 * sc;
    const int plotW = panelW - 2 * pad, plotH = panelH - 2 * pad;
    const int gx = lx - pad, gy = ly - pad;            // plot-local (gy down)

    // Opaque dark panel with a 1px border.
    outR = outG = outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        outR = outG = outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    // gy runs DOWN the plot; convert to an output-stops value at this row.
    const float rowStops = 6.0f - 12.0f * (static_cast<float>(gy) / (plotH - 1));

    // Grid every 2 stops + the 18% gray crosshair (input & output = 0 stops).
    const int gcol0 = static_cast<int>((0.0f + 6.0f) / 12.0f * (plotW - 1) + 0.5f);
    const int grow0 = static_cast<int>((6.0f - 0.0f) / 12.0f * (plotH - 1) + 0.5f);
    if (gx == gcol0 || gy == grow0) { outR = outG = outB = 0.24f; return true; }
    if ((gx % (plotW / 6)) == 0 || (gy % (plotH / 6)) == 0) { outR = outG = outB = 0.13f; }

    // The three applied per-channel curves. A column is "on" a curve when the
    // row's output-stops straddles the curve value between this and the next
    // column (so the trace stays connected on steep segments).
    const int chR[3] = { 1, 0, 0 }, chG[3] = { 0, 1, 0 }, chB[3] = { 0, 0, 1 };
    for (int ch = 0; ch < 3; ++ch) {
        const float inS  = -6.0f + 12.0f * (static_cast<float>(gx)     / (plotW - 1));
        const float inS2 = -6.0f + 12.0f * (static_cast<float>(gx + 1) / (plotW - 1));
        float y0 = scopeYStops(inS,  ch, pr);
        float y1 = scopeYStops(inS2, ch, pr);
        if (y0 > y1) { const float t = y0; y0 = y1; y1 = t; }
        const float lo = y1 < y0 ? y1 : y0, hi = y1 > y0 ? y1 : y0;
        if (rowStops <= hi + 0.09f && rowStops >= lo - 0.09f) {
            outR = 0.10f + 0.85f * chR[ch];
            outG = 0.10f + 0.85f * chG[ch];
            outB = 0.10f + 0.85f * chB[ch];
            return true;
        }
    }

    // Legend swatches, bottom-left of the plot.
    if (gy >= plotH - 5 * sc && gy < plotH - 1 * sc) {
        const int sw = gx / (6 * sc);
        if (gx % (6 * sc) < 4 * sc) {
            if (sw == 0) { outR = 0.95f; outG = 0.10f; outB = 0.10f; return true; }
            if (sw == 1) { outR = 0.10f; outG = 0.95f; outB = 0.10f; return true; }
            if (sw == 2) { outR = 0.10f; outG = 0.10f; outB = 0.95f; return true; }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// The whole per-pixel operation. `x,y` are buffer coords (y up, OFX-native);
// the scope converts to display space internally. RGBA in -> RGBA out (alpha
// passes through). This is the ONE function the four backends must agree on.
// ---------------------------------------------------------------------------
static inline void processPixel(float r, float g, float b,
                                int x, int y, int W, int H,
                                const SpeakParams& pr,
                                float& outR, float& outG, float& outB)
{
    const SpeakProfile& p = pr.profile;
    const int cs = pr.inputColorSpace;
    const bool toneOn = (pr.enableTone != 0) && (pr.strength > 0.0f);
    const bool bake   = (pr.outputMode == SPEAK_OUT_BAKE_REC709);

    if (!toneOn && !bake) {
        // Working space + no look: bit-exact pass-through (identity). Scopes
        // may still overwrite below.
        outR = r; outG = g; outB = b;
    } else {
        // Working-space linear result (look applied if any).
        const float lr = decodeToLinear(cs, r);
        const float lg = decodeToLinear(cs, g);
        const float lb = decodeToLinear(cs, b);
        float mr = lr, mg = lg, mb = lb;
        if (toneOn) {
            const float s = clampf(pr.strength, 0.0f, 1.0f);
            mr = lerpf(lr, toneChannel(lr, 0, p), s);
            mg = lerpf(lg, toneChannel(lg, 1, p), s);
            mb = lerpf(lb, toneChannel(lb, 2, p), s);
        }
        if (bake) {
            // Output CST: gamut-convert to Rec.709 and encode gamma 2.4. Applies
            // regardless of the look (it is delivery, not a look) — a hard gamut
            // clip for now (the constant-hue soft guardrail is a later module).
            float rr, rg, rb;
            gamutToRec709Lin(cs, mr, mg, mb, rr, rg, rb);
            rr = rr < 0.0f ? 0.0f : rr;
            rg = rg < 0.0f ? 0.0f : rg;
            rb = rb < 0.0f ? 0.0f : rb;
            outR = encodeFromLinear(SPEAK_CS_REC709_G24, rr);
            outG = encodeFromLinear(SPEAK_CS_REC709_G24, rg);
            outB = encodeFromLinear(SPEAK_CS_REC709_G24, rb);
        } else {
            // Working space: re-encode to the input space and let RCM deliver.
            outR = encodeFromLinear(cs, mr);
            outG = encodeFromLinear(cs, mg);
            outB = encodeFromLinear(cs, mb);
        }
    }

    // View modes (Split / Input) operate on the buffer coordinate.
    if (pr.viewMode == SPEAK_VIEW_INPUT) { outR = r; outG = g; outB = b; }
    else if (pr.viewMode == SPEAK_VIEW_SPLIT && x < W / 2) { outR = r; outG = g; outB = b; }

    // Scopes render last, over any view.
    float sr, sg, sb;
    if (hdScopePixel(x, y, W, H, pr, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
}

// ---------------------------------------------------------------------------
// Whole-frame CPU entry point (the reference the GPU kernels are ported from).
// Interleaved RGBA float, row-major, y up (OFX-native buffer order).
// ---------------------------------------------------------------------------
inline void speakFrame(const float* src, int W, int H, const SpeakParams& pr, float* dst)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            float oR, oG, oB;
            processPixel(src[i + 0], src[i + 1], src[i + 2], x, y, W, H, pr, oR, oG, oB);
            dst[i + 0] = oR;
            dst[i + 1] = oG;
            dst[i + 2] = oB;
            dst[i + 3] = src[i + 3];   // alpha passes through
        }
    }
}

// A canonical "Neutral" gray-balanced profile: three identical channels, a mild
// negative + a stronger print, zero printer lights. Neutral-preserving by
// construction (identical channels => equal outputs for equal inputs). Built-in
// stock families and Shoot-a-Chart calibration emit the SAME struct.
static inline SpeakProfile neutralProfile()
{
    // The negative and print stages are registered so that 18% gray (stops=0)
    // lands in the STRAIGHT region of BOTH curves: the negative maps gray to
    // D_neg ~= 1.05, and the print's speed point is set below that print
    // exposure so gray sits mid-straight on the print (full print gamma). The
    // product neg_gamma * print_gamma gives the ~1.5 system gamma of a film
    // print. Printer lights slide this operating point at use time.
    SpeakProfile p;
    for (int c = 0; c < 3; ++c) {
        p.negDmin[c] = 0.15f; p.negDmax[c] = 2.90f; p.negGamma[c] = 0.62f;
        p.negToe[c]  = 2.5f;  p.negShoulder[c] = 2.5f; p.negSpeed[c] = -1.5f;
        p.prnDmin[c] = 0.10f; p.prnDmax[c] = 3.30f; p.prnGamma[c] = 2.60f;
        p.prnToe[c]  = 3.5f;  p.prnShoulder[c] = 2.2f; p.prnSpeed[c] = -1.75f;
        p.printerLights[c] = 0.0f;
        p.subSat[c] = 0.0f; p.subSatKnee[c] = 1.0f;
        p.splitShadow[c] = 0.0f; p.splitHigh[c] = 0.0f;
    }
    for (int k = 0; k < 9; ++k) p.dyeCouple[k] = 0.0f;
    p.printerMaster = 0.0f;
    p.splitPivot = 0.0f; p.splitBalance = 0.5f;
    p.systemGamma = 1.6f; p.residualLUT = 0; p.profileVersion = 1; p._pad0 = 0;
    return p;
}

} // namespace speakcore

#endif // OPENNR_SPEAK_CORE_H
