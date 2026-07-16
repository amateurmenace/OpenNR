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
#include <vector>

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

// CIE L*a*b* (D65) — the perceptual metric the Phase-2 control arm scores in.
// Used by the Macbeth gate and (later) the subtractive-sat vector scope.
static inline float labF(float t)
{
    const float d = 6.0f / 29.0f;
    return (t > d * d * d) ? std::cbrt(t) : (t / (3.0f * d * d) + 4.0f / 29.0f);
}
static inline void xyzToLab(float X, float Y, float Z, float& L, float& a, float& b)
{
    const float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;
    const float fx = labF(X / Xn), fy = labF(Y / Yn), fz = labF(Z / Zn);
    L = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b = 200.0f * (fy - fz);
}
static inline void dwgLinToLab(float r, float g, float b, float& L, float& aa, float& bb)
{
    float X, Y, Z;
    mul3(kDWG_to_XYZ, r, g, b, X, Y, Z);
    xyzToLab(X, Y, Z, L, aa, bb);
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
// Density-Space Subtractive Saturation + inter-image coupler (Phase 2). This
// is *why film looks like film*: it works in log-density (where dye density
// adds and transmittance multiplies), not in linear or HSL. Converting to
// density, amplifying each dye's deviation from the neutral (gray) density, and
// viewing back through 10^-D produces two structural signatures a linear 3x3
// saturation cannot fake — highlight chroma self-compresses toward base white,
// and hues skew toward the dye axes (from the asymmetric coupler). Neutral is
// invariant by construction: the transform acts on deviations from D-bar, which
// are zero on the gray axis. Standalone (usable on any grade).
// ---------------------------------------------------------------------------
static inline float density10(float lin)   // linear -> optical density (log10)
{
    return -std::log2(lin < 1e-6f ? 1e-6f : lin) * kLog10_2;  // log10 via log2 (parity-safe)
}
static inline float softCapKnee(float d, float cap)  // soft cap density at `cap`
{
    if (cap <= 0.0f) return d;                          // disabled
    return cap - (1.0f / 8.0f) * softplusf(8.0f * (cap - d));
}
static inline void subtractiveColor(float r, float g, float b, const SpeakProfile& p,
                                    float& oR, float& oG, float& oB)
{
    const float DR = density10(r), DG = density10(g), DB = density10(b);
    const float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    const float devR = DR - Dbar, devG = DG - Dbar, devB = DB - Dbar;
    // diagonal = per-dye subtractive saturation; off-diagonals = inter-image
    // coupler (unwanted-absorption cross terms). dyeCouple diagonal is unused.
    const float cR = (1.0f + p.subSat[0]) * devR - (p.dyeCouple[1] * devG + p.dyeCouple[2] * devB);
    const float cG = (1.0f + p.subSat[1]) * devG - (p.dyeCouple[3] * devR + p.dyeCouple[5] * devB);
    const float cB = (1.0f + p.subSat[2]) * devB - (p.dyeCouple[6] * devR + p.dyeCouple[7] * devG);
    const float DpR = softCapKnee(Dbar + cR, p.subSatKnee[0]);
    const float DpG = softCapKnee(Dbar + cG, p.subSatKnee[1]);
    const float DpB = softCapKnee(Dbar + cB, p.subSatKnee[2]);
    oR = pow10f(-DpR); oG = pow10f(-DpG); oB = pow10f(-DpB);
}
static inline bool dyeActive(const SpeakProfile& p)
{
    return p.subSat[0] != 0.0f || p.subSat[1] != 0.0f || p.subSat[2] != 0.0f ||
           p.dyeCouple[1] != 0.0f || p.dyeCouple[2] != 0.0f || p.dyeCouple[3] != 0.0f ||
           p.dyeCouple[5] != 0.0f || p.dyeCouple[6] != 0.0f || p.dyeCouple[7] != 0.0f;
}

// ---------------------------------------------------------------------------
// Split toning / film-referred tonal-zone balance (Phase 3) — the lift-gamma-
// gain replacement, done in the density domain.
//
// Per-channel density offsets weighted by a 3-zone partition of unity (toe /
// mid / shoulder) anchored to the WORKING H&D CURVE rather than fixed luma
// cuts: the tone position is the pixel's own neutral density, and the spine
// pivots 18% gray to D = -log10(0.18) = 0.745, so that IS the mid anchor.
//
// Two structural properties fall out, both of which LGG cannot offer:
//  - MIDS STAY NEUTRAL BY CONSTRUCTION: at the pivot both zone weights are
//    exactly 0, so the mid tone receives no offset at all — you cannot tint
//    shadows and highlights and accidentally drag mid-gray.
//  - HUE-STABLE: additive in density == MULTIPLICATIVE in linear, so a tint
//    scales a channel rather than offsetting it the way LGG's lift does (which
//    is what desaturates and swings hue in the shadows).
// Chromogenic crossover (cool shadows / warm highlights) is simply opposite
// shadow and highlight offsets; cross-process is a more extreme pair.
// ---------------------------------------------------------------------------
static inline float smooth01(float t)
{
    t = clampf(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
static inline void splitWeights(float Dbar, const SpeakProfile& p, float& wShadow, float& wHigh)
{
    const float grayD  = 0.744727f;                          // D of 18% gray (the spine's pivot)
    const float pivotD = grayD - p.splitPivot * kLog10_2;    // pivot: stops -> density
    const float halfW  = 0.25f + 1.5f * clampf(p.splitBalance, 0.0f, 1.0f);
    const float x = (Dbar - pivotD) / halfW;                 // signed: + = darker, - = brighter
    wShadow = smooth01(x);        // -> 1 into the toe (shadows)
    wHigh   = smooth01(-x);       // -> 1 into the shoulder (highlights)
}                                 // at the pivot both are 0 => the mid zone is untouched
static inline void splitTone(float r, float g, float b, const SpeakProfile& p,
                             float& oR, float& oG, float& oB)
{
    const float DR = density10(r), DG = density10(g), DB = density10(b);
    const float Dbar = (DR + DG + DB) * (1.0f / 3.0f);
    float wS, wH;
    splitWeights(Dbar, p, wS, wH);
    oR = pow10f(-(DR + wS * p.splitShadow[0] + wH * p.splitHigh[0]));
    oG = pow10f(-(DG + wS * p.splitShadow[1] + wH * p.splitHigh[1]));
    oB = pow10f(-(DB + wS * p.splitShadow[2] + wH * p.splitHigh[2]));
}
static inline bool splitActive(const SpeakProfile& p)
{
    return p.splitShadow[0] != 0.0f || p.splitShadow[1] != 0.0f || p.splitShadow[2] != 0.0f ||
           p.splitHigh[0] != 0.0f || p.splitHigh[1] != 0.0f || p.splitHigh[2] != 0.0f;
}

// The generic dye cross-absorption pattern. The hue skew lives ENTIRELY in the
// within-row asymmetry (the density deviations sum to zero, so equal cross terms
// in a row would collapse to a plain saturation boost). Follows the classic
// published unwanted absorptions — cyan absorbs mostly green, magenta mostly
// blue, yellow mostly green — behavior-named, cloning no stock. Verified by the
// Macbeth CIELAB control arm (test/test_speak_macbeth.cpp).
static const float kCouplerRG = 0.28f, kCouplerRB = 0.06f;
static const float kCouplerGR = 0.08f, kCouplerGB = 0.30f;
static const float kCouplerBR = 0.04f, kCouplerBG = 0.22f;
static inline void setDyeCoupler(SpeakProfile& p, float amount)
{
    p.dyeCouple[1] = kCouplerRG * amount; p.dyeCouple[2] = kCouplerRB * amount;
    p.dyeCouple[3] = kCouplerGR * amount; p.dyeCouple[5] = kCouplerGB * amount;
    p.dyeCouple[6] = kCouplerBR * amount; p.dyeCouple[7] = kCouplerBG * amount;
}

// ---------------------------------------------------------------------------
// HALATION (Phase 4) — Speak's first spatial module.
//
// THE PHYSICS, AND WHY THE INJECTION POINT IS WHERE IT IS.
// Light focused by the lens passes THROUGH the emulsion, reflects off the
// base/air interface, and re-enters the emulsion displaced sideways. Therefore:
//   * the SOURCE of the scattering light is the SCENE image on the emulsion;
//   * what the scattered light ADDS TO is the negative's EXPOSURE H;
//   * exposure is additive in linear light (photons add).
// So H_c'(x) = H_c(x) + amount * w_c * (PSF * excess(H_c))(x), and the WHOLE
// negative -> printer -> print cascade then sees H'. The print's shoulder is
// what compresses the halo, which is why the halo self-limits and goes
// white-hot in its core instead of pumping unbounded red (test/proto_halation).
//
// There is NO halation injection between the negative and the print. The print's
// exposure is 10^-Dneg (chainDensity, above): it is BRIGHT where the SCENE was
// DARK, so a highlight-driven injection there DARKENS highlight edges instead of
// haloing them — measured, a bright disc inverts 8.056 -> 0.007. (Scatter in the
// optical printer, and the print stock's own halation, ARE gap-sited; they are
// real but different phenomena with the opposite polarity, and are not this
// module.) toneChannel/chainDensity/hdCurve are therefore untouched by halation.
//
// WHY IT IS RED, AND WHAT THAT CLAIM IS WORTH. The light that reaches the base
// has already been depleted of blue and green — by the yellow filter layer and
// the upper emulsion layers — and the red-sensitive layer sits closest to the
// base. That MECHANISM is published sensitometry. The specific {1, 0.30, 0.10}
// ratio is NOT: it is our MODELLED DEFAULT for it, gated by H3/H4 in
// test/proto_halation.cpp rather than cited to anyone. It is not a user control,
// and no shipping hint attributes it to a stock or to a measurement.
// (An earlier revision called these "Beer-Lambert-shaped, published AH
// behaviour" with no citation — that fitted a made-up ratio and then cited
// Beer-Lambert for it, and is withdrawn. Nor is this the eye's glare-spread
// function: Vos & van den Berg / CIE 135 model corneal and lens scatter in
// degrees of visual angle for a human observer. Citing them for base reflection
// in an emulsion would be laundering a different physical system's authority.)
//
// The weight is applied PER CHANNEL to that channel's OWN scattered light.
// Collapsing RGB to a luminous mean and tinting it red — what the prototype
// originally did, and what the arm could not see — lets BLUE photons manufacture
// a RED halo (measured: a (0,0,2.0) source got a red scatter 10x its blue).
// Gated by H3/H4 in test/proto_halation.cpp.
// ---------------------------------------------------------------------------
static const float kHalWeight[3] = { 1.0f, 0.30f, 0.10f };

// The scatter source in scene-linear light, one channel. Pure, so the pyramid
// builder and the tests call the exact same function.
static inline float halExcess(float lin, float thresh)
{
    const float l = lin < 0.0f ? 0.0f : lin;
    return l > thresh ? (l - thresh) : 0.0f;
}

// Effective halation amount: 0 unless the optics stage is on. Callers ALSO gate
// on the tone spine — halation re-exposes the NEGATIVE, so with no spine there
// is no negative, and injecting scatter into linear with no curve downstream is
// EXACTLY the end-chain overlay the control arm rejected. The UI disables the
// controls when Film Tone is off and says why.
static inline float halAmountOf(const SpeakParams& pr)
{
    return (pr.enableOptics != 0) ? pr.profile.halAmount : 0.0f;
}

// Halation spreads a fixed distance in mm on the film, so its radius is
// FORMAT-relative: sigma scales with frame height, never with pixel count. This
// is what makes the look survive a proxy/full-res switch (measured: the halo
// profile matches to ~1% across a 4x resolution change, G16).
//
// KNOWN AND MEASURED LIMIT, stated rather than papered over: the halo's actual
// half-width tracks this sigma closely once sigma is well resolved (HWHM/sigma
// = 0.93..1.00 above ~20 px) but UNDERSIZES below that, down to ~0.6x at a
// sigma of ~1 px — a pyramid cannot represent a halo a few pixels wide. There is
// also a ~8% ripple as the target level slides between octaves. Neither is
// visible in use, and the Radius hint promises a percentage of frame height, not
// an exact half-width. Pinned by G13.
static inline float halSigmaPx(int H, const SpeakParams& pr)
{
    const float s = pr.profile.halRadius * 0.01f * static_cast<float>(H);
    return s < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : s;
}
static inline bool halActive(const SpeakParams& pr)
{
    return (pr.enableTone != 0) && (pr.strength > 0.0f) && (halAmountOf(pr) > 0.0f);
}

// ---- pyramid geometry (identical arithmetic in all four backends) ----
static inline int halLevelCount(int W, int H)
{
    int n = 1, w = W, h = H;
    while (n < SPEAK_HAL_MAXLEV && w > SPEAK_HAL_MINDIM && h > SPEAK_HAL_MINDIM) {
        w = (w + 1) / 2; h = (h + 1) / 2; n++;
    }
    return n;
}
// Dimensions and the packed arena offset (in PIXELS, x3 for RGB) of level L.
static inline void halLevelInfo(int W, int H, int L, int& lw, int& lh, int& off)
{
    int w = W, h = H, o = 0;
    for (int i = 0; i < L; ++i) { o += w * h; w = (w + 1) / 2; h = (h + 1) / 2; }
    lw = w; lh = h; off = o;
}
static inline int halArenaPixels(int W, int H)
{
    int lw, lh, off;
    const int n = halLevelCount(W, H);
    halLevelInfo(W, H, n, lw, lh, off);   // offset just past the last level
    return off;
}

// Level L's effective full-res sigma. Two terms, and BOTH are load-bearing:
//   1. DECIMATION. The [1,3,3,1]/8 kernel has variance 0.75 in its own level's
//      pixel units, so going from level j to j+1 adds 0.75*(2^j)^2 full-res
//      px^2:   sum_{j<L} 0.75*4^j = 0.25*(4^L - 1).
//   2. UPSAMPLE. Reading level L back at full res bilinearly interpolates
//      samples 2^L px apart, and a tent filter of spacing h has variance h^2/6,
//      adding 4^L/6. (Level 0 needs no interpolation, so it takes no such term.)
//   var_L = 0.25*(4^L - 1) + [L>0] * 4^L/6      =>   sigma_L -> 0.6455 * 2^L
//
// Term 2 was missing from the first draft and the ladder came out 1.29x low at
// EVERY level — a constant scale error, so the octave spacing still looked
// perfect and only an absolute measurement could catch it. G13 caught it:
// predicted 15.99 vs 20.65 measured at level 5. It is PINNED THERE against the
// real impulse response, because a wrong ladder silently resizes the shipped
// halo and parity would never see it.
static const float kHalSigmaC = 0.645497f;   // sqrt(0.25 + 1/6): sigma_L / 2^L
static inline float halLevelSigma(int L)
{
    const float q = std::exp2(2.0f * static_cast<float>(L));   // 4^L
    const float up = (L > 0) ? (q * (1.0f / 6.0f)) : 0.0f;
    return std::sqrt(0.25f * (q - 1.0f) + up);
}

// The mixture weights. Level L contributes a Gaussian of sigma_L ~ 2^(L-1); a
// weighted sum over octaves is a MULTI-SCALE scatter — a bright core plus a wide
// faint skirt — which a single Gaussian cannot produce at any sigma. Levels
// tighter than the target fall off fast (kHalCoreFall); broader levels decay
// slowly (kHalSkirtFall = 1 => weight halves per octave), which is the skirt.
//
// The tail this produces is a power law, by construction: at radius r the
// dominant term is the level with sigma_L ~ r, contributing w_L/sigma_L^2 (a 2D
// Gaussian's peak amplitude goes as 1/sigma^2). With w_L ~ 2^-d and sigma_L ~
// 2^d that is 2^-d/4^d = 8^-d, and r ~ 2^d, so the contribution falls as r^-3 —
// versus a Gaussian's exp(-r^2/2sigma^2). Measured against a matched-core
// Gaussian in G14; kHalSkirtFall is a MODELLED DEFAULT, not a cited constant.
static const float kHalCoreFall  = 3.0f;   // octaves tighter than target: fast cut
static const float kHalSkirtFall = 1.0f;   // octaves broader: halve per octave
static inline float halLevelWeight(int L, float sigmaTarget)
{
    // Target level from sigma_L -> kHalSigmaC * 2^L  =>  L_t = log2(sigma/C).
    // Continuous in L_t, so the radius slider is smooth: it does NOT snap to
    // octaves as the level bracket moves (gated by G16 across a 4x resolution
    // change, which is the same machinery).
    const float s = sigmaTarget < SPEAK_HAL_SIGMA_MIN ? SPEAK_HAL_SIGMA_MIN : sigmaTarget;
    const float Lt = std::log2(s / kHalSigmaC);
    const float d = static_cast<float>(L) - Lt;
    return (d <= 0.0f) ? std::exp2(kHalCoreFall * d) : std::exp2(-kHalSkirtFall * d);
}

// ---- pyramid taps (per-pixel, so the GPU kernels are textual ports) ----
// Clamp-to-edge fetch. NOTE the consequence, which is stated in the UI rather
// than hidden: a highlight AT the frame edge scatters against a clamped border,
// so its halo carries slightly less energy than one in frame centre. There is
// no closed form to compensate; the energy gate (G15) therefore measures the
// INTERIOR, at least 2 sigma from every border, where the identity is exact.
static inline float halFetch(const float* arena, int off, int lw, int lh, int x, int y, int c)
{
    const int xx = x < 0 ? 0 : (x >= lw ? lw - 1 : x);
    const int yy = y < 0 ? 0 : (y >= lh ? lh - 1 : y);
    return arena[(static_cast<size_t>(off) + static_cast<size_t>(yy) * lw + xx) * 3 + c];
}

// The [1,3,3,1]/8 separable binomial decimation. Its taps sum to 1 on each axis,
// so it is MEAN-PRESERVING: every level carries the same total energy as level 0
// once upsampled, which is what makes the mixture energy-normalized by Sum(w)=1
// alone. Deterministic and atomics-free (lesson: Apple's OpenCL miscompiles
// global int32 atomics, so no spatial pass may depend on them).
static const float kHalDec[4] = { 0.125f, 0.375f, 0.375f, 0.125f };
static inline void halDecimatePixel(const float* arena, int sOff, int sW, int sH,
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
static inline void halBSpline(float t, float* w)
{
    const float t2 = t * t, t3 = t2 * t;
    w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) * (1.0f / 6.0f);
    w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) * (1.0f / 6.0f);
    w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) * (1.0f / 6.0f);
    w[3] = t3 * (1.0f / 6.0f);
}
static inline float halSampleLevel(const float* arena, int off, int lw, int lh,
                                   int W, int H, int x, int y, int c)
{
    const float fx = (static_cast<float>(x) + 0.5f) * static_cast<float>(lw) / static_cast<float>(W) - 0.5f;
    const float fy = (static_cast<float>(y) + 0.5f) * static_cast<float>(lh) / static_cast<float>(H) - 0.5f;
    const int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
    float wx[4], wy[4];
    halBSpline(fx - static_cast<float>(x0), wx);
    halBSpline(fy - static_cast<float>(y0), wy);
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
static inline float halWeightSum(int nLev, float sigmaTarget)
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
// resolution) because none of them measured isotropy, and it was plainly visible
// the moment the scatter field was rendered: the PSF had ~9% angular variation
// at every radius and 372% at r=128 (G18 now measures this; it read 3.72 before
// this change and ~0.02 after).
//
// Going one octave at a time fixes it because each step interpolates only
// between ADJACENT samples and is then re-filtered by every step below it. It is
// also strictly CHEAPER: the total work is sum_L (level L's pixels) = 4/3 N,
// against nLev*N for the full-res reads.
//
// In-place is safe: a thread reads its OWN (x,y) at level L and a neighbourhood
// at level L+1 (a disjoint region), and writes only its own (x,y) at level L.
static inline void halAccumPixel(float* arena, int W, int H, int L, int nLev,
                                 float sigmaTarget, int x, int y, float* out)
{
    int lw, lh, off;
    halLevelInfo(W, H, L, lw, lh, off);
    const float wl = halLevelWeight(L, sigmaTarget);
    if (L >= nLev - 1) {                       // the coarsest level: nothing above it
        for (int c = 0; c < 3; ++c) out[c] = wl * halFetch(arena, off, lw, lh, x, y, c);
        return;
    }
    int cw, ch, coff;
    halLevelInfo(W, H, L + 1, cw, ch, coff);
    for (int c = 0; c < 3; ++c)
        out[c] = wl * halFetch(arena, off, lw, lh, x, y, c)
               + halSampleLevel(arena, coff, cw, ch, lw, lh, x, y, c);
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
// The LOOK in working-space linear: decode, halation re-exposure, tone spine,
// subtractive color. Shared by the pixel path and the density scope's
// measurement pass, so the scope measures exactly what the pixels became — it
// cannot drift from them.
//
// `scat*` are the energy-normalized, blurred, PER-CHANNEL scene-linear highlight
// excesses at this pixel (0 when halation is off). They are deliberately NOT
// defaulted: the density scope calls this function, and a scope that silently
// passed 0 would measure an image the pixels never became — an L3-class bug that
// parity CANNOT catch, because all four backends would agree on the same wrong
// parade. Forcing every call site to pass scatter is the gate. The failure would
// even be directional: halation raises E, so toneChannel(E) rises and density
// falls, and a scatter-blind parade would draw the halo DARKER than it is —
// exactly where the user opened the scope to look.
static inline void lookLinear(float r, float g, float b,
                              float scatR, float scatG, float scatB,
                              const SpeakParams& pr,
                              float& oR, float& oG, float& oB)
{
    const SpeakProfile& p = pr.profile;
    const int cs = pr.inputColorSpace;
    const float lr = decodeToLinear(cs, r);
    const float lg = decodeToLinear(cs, g);
    const float lb = decodeToLinear(cs, b);
    float mr = lr, mg = lg, mb = lb;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        const float s = clampf(pr.strength, 0.0f, 1.0f);
        const float a = halAmountOf(pr);
        // RE-EXPOSURE: the scattered light adds to the scene light ENTERING the
        // negative, so it goes in the CURVE'S ARGUMENT. The dry side of the mix
        // stays lr, not the re-exposed value — otherwise strength 0 would leave
        // the raw scatter added to linear with no curve downstream, which is
        // precisely the end-chain overlay the arm rejected (and would break the
        // identity gate G3). Branch rather than multiply by zero so amount 0 is
        // provably bit-exact with the pre-halation build; `a` is frame-uniform,
        // so the branch costs no GPU divergence.
        float er = lr, eg = lg, eb = lb;
        if (a > 0.0f) {
            er = lr + a * kHalWeight[0] * scatR;
            eg = lg + a * kHalWeight[1] * scatG;
            eb = lb + a * kHalWeight[2] * scatB;
        }
        mr = lerpf(lr, toneChannel(er, 0, p), s);
        mg = lerpf(lg, toneChannel(eg, 1, p), s);
        mb = lerpf(lb, toneChannel(eb, 2, p), s);
    }
    if ((pr.enableDye != 0) && dyeActive(p)) subtractiveColor(mr, mg, mb, p, mr, mg, mb);
    if ((pr.enableSplit != 0) && splitActive(p)) splitTone(mr, mg, mb, p, mr, mg, mb);
    oR = mr; oG = mg; oB = mb;
}

// ---------------------------------------------------------------------------
// Scope statistics — the frame's own exposure distribution, and the Status-M
// density waveform of the RESULT, measured on a stride-2 grid and binned with
// integer counts (order-independent, so all four backends land on identical
// bins).
// ---------------------------------------------------------------------------
static inline int wfColOf(int x, int W)
{
    const int c = x * SPEAK_WF_COLS / (W > 0 ? W : 1);
    return c < 0 ? 0 : (c >= SPEAK_WF_COLS ? SPEAK_WF_COLS - 1 : c);
}
static inline int wfRowOf(float D)
{
    const int r = static_cast<int>(D / SPEAK_WF_DMAX * SPEAK_WF_ROWS);
    return r < 0 ? 0 : (r >= SPEAK_WF_ROWS ? SPEAK_WF_ROWS - 1 : r);
}
static inline int wfIdx(int ch, int col, int row)
{
    return SPEAK_STATS_WF + ch * (SPEAK_WF_COLS * SPEAK_WF_ROWS) + col * SPEAK_WF_ROWS + row;
}
static inline int expBinOf(float stops)
{
    const int b = static_cast<int>((stops + 6.0f) / 12.0f * SPEAK_EXP_BINS);
    return b < 0 ? 0 : (b >= SPEAK_EXP_BINS ? SPEAK_EXP_BINS - 1 : b);
}
// A pixel's scene exposure in stops (mean of the linear channels — color-space
// agnostic, so the histogram means the same thing in any declared input space).
static inline float pixelStops(int cs, float r, float g, float b)
{
    const float m = (decodeToLinear(cs, r) + decodeToLinear(cs, g) + decodeToLinear(cs, b)) * (1.0f / 3.0f);
    return std::log2((m < kLinTiny ? kLinTiny : m) / k18Gray);
}
// `scat` is the W*H*3 interleaved scatter field, or null when halation is off.
// The density parade MUST see it — see the lookLinear header for why a
// scatter-blind scope is an L3 bug parity cannot catch.
inline void computeStats(const float* src, const float* scat, int W, int H,
                         const SpeakParams& pr, uint32_t* stats)
{
    for (int i = 0; i < SPEAK_STATS_UINTS; ++i) stats[i] = 0u;
    if (pr.scopeHD == 0 && pr.scopeDensity == 0) return;   // only measured when shown
    const int cs = pr.inputColorSpace;
    for (int y = 0; y < H; y += 2)
        for (int x = 0; x < W; x += 2) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const size_t j = (static_cast<size_t>(y) * W + x) * 3;
            if (pr.scopeHD != 0)
                stats[SPEAK_STATS_HIST_EXP + expBinOf(pixelStops(cs, src[i], src[i + 1], src[i + 2]))]++;
            if (pr.scopeDensity != 0) {
                float mr, mg, mb;
                lookLinear(src[i], src[i + 1], src[i + 2],
                           scat ? scat[j + 0] : 0.0f, scat ? scat[j + 1] : 0.0f,
                           scat ? scat[j + 2] : 0.0f, pr, mr, mg, mb);
                const int col = wfColOf(x, W);
                stats[wfIdx(0, col, wfRowOf(density10(mr)))]++;
                stats[wfIdx(1, col, wfRowOf(density10(mg)))]++;
                stats[wfIdx(2, col, wfRowOf(density10(mb)))]++;
            }
        }
    uint32_t mx = 0u;
    for (int b = 0; b < SPEAK_EXP_BINS; ++b)
        if (stats[SPEAK_STATS_HIST_EXP + b] > mx) mx = stats[SPEAK_STATS_HIST_EXP + b];
    stats[SPEAK_STATS_HIST_MAX] = mx;
    uint32_t wmx = 0u;
    for (int k = 0; k < SPEAK_WF_COLS * SPEAK_WF_ROWS * 3; ++k)
        if (stats[SPEAK_STATS_WF + k] > wmx) wmx = stats[SPEAK_STATS_WF + k];
    stats[SPEAK_STATS_WF_MAX] = wmx;
}

// The APPLIED TONE SCALE for one input exposure, in stops — the Strength mix and
// the enable toggle included, so the plot cannot drift from the curve the pixels
// use (at strength 0 it collapses to the y=x diagonal, matching the identity
// pass-through). Output encode is a display transform applied equally to curve
// and diagonal, so it cancels in stops.
//
// WHAT THIS PLOT DOES NOT SHOW, deliberately: halation. This is a function of
// EXPOSURE ALONE, and halation is SPATIAL — a pixel's re-exposure depends on its
// neighbours, so there is no single scatter value for an exposure and no honest
// way to draw it as one curve (threading a scatter value in here would turn one
// curve into a per-pixel family, and the per-channel AH weight into three). The
// curve remains exactly true of the tone scale; it is not the whole transform
// once Light is on. The two scopes that DO see halation are the Status-M density
// parade (which measures the halated result — G17) and the isolated-scatter
// view. Do not restore an "it can never disagree with the pixels" claim here.
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
                                const uint32_t* stats,
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

    // The frame's own exposure histogram, projected onto the logE axis under
    // the curves — so you can see where THIS shot's tones sit on the curve.
    const uint32_t hmax = stats[SPEAK_STATS_HIST_MAX];
    if (hmax > 0u) {
        const int hb = expBinOf(-6.0f + 12.0f * (static_cast<float>(gx) / (plotW - 1)));
        const float f = static_cast<float>(stats[SPEAK_STATS_HIST_EXP + hb]) / static_cast<float>(hmax);
        const int barH = static_cast<int>(std::sqrt(f) * (plotH * 0.45f) + 0.5f);
        if (gy >= plotH - barH) { outR = 0.16f; outG = 0.19f; outB = 0.24f; }
    }

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
// Status-M density scope: an RGB parade of the RESULT's film density, measured
// from the frame by the same look the pixels get. Vertical axis is optical
// density (0 = paper white at the top, SPEAK_WF_DMAX at the bottom) with
// markers at paper white and 18% gray (D = -log10(0.18) = 0.745). Anchored
// top-RIGHT in display space, so it sits beside the H&D scope.
// ---------------------------------------------------------------------------
static inline bool densityScopePixel(int x, int y, int W, int H, const SpeakParams& pr,
                                     const uint32_t* stats,
                                     float& outR, float& outG, float& outB)
{
    if (pr.scopeDensity == 0) return false;

    const int sc = (H / 540) > 1 ? (H / 540) : 1;
    const int panelW = 220 * sc, panelH = 150 * sc;
    const int margin = 12 * sc;
    const int px0 = W - margin - panelW, py0 = margin;   // top-right, display space
    const int yd = H - 1 - y;
    const int lx = x - px0, ly = yd - py0;
    if (lx < 0 || ly < 0 || lx >= panelW || ly >= panelH) return false;

    const int pad = 6 * sc;
    const int plotW = panelW - 2 * pad, plotH = panelH - 2 * pad;
    const int gx = lx - pad, gy = ly - pad;

    outR = outG = outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        outR = outG = outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    // Density markers: paper white (D=0, the top) and 18% gray.
    const int rowGray = static_cast<int>(0.744727f / SPEAK_WF_DMAX * (plotH - 1) + 0.5f);
    if (gy == 0) { outR = 0.35f; outG = 0.35f; outB = 0.35f; return true; }
    if (gy == rowGray) { outR = 0.30f; outG = 0.26f; outB = 0.12f; return true; }

    // The parade: three channel panes side by side.
    const int chW = plotW / 3;
    int ch = gx / (chW > 0 ? chW : 1);
    if (ch > 2) ch = 2;
    if (gx - ch * chW == 0 && ch > 0) { outR = outG = outB = 0.16f; return true; }  // pane divider

    const uint32_t wmax = stats[SPEAK_STATS_WF_MAX];
    if (wmax > 0u) {
        const int within = gx - ch * chW;
        const int wcol = within * SPEAK_WF_COLS / (chW > 0 ? chW : 1);
        const int wrow = gy * SPEAK_WF_ROWS / plotH;
        const uint32_t c = stats[wfIdx(ch, wcol < SPEAK_WF_COLS ? wcol : SPEAK_WF_COLS - 1,
                                       wrow < SPEAK_WF_ROWS ? wrow : SPEAK_WF_ROWS - 1)];
        if (c > 0u) {
            const float inten = std::sqrt(static_cast<float>(c) / static_cast<float>(wmax));
            const float v = 0.12f + 0.88f * (inten > 1.0f ? 1.0f : inten);
            outR = (ch == 0) ? v : 0.05f;
            outG = (ch == 1) ? v : 0.05f;
            outB = (ch == 2) ? v : 0.05f;
        }
    }
    return true;
}

// The INPUT pixel delivered through the same output CST as the result (no look
// applied). Split/Input views use this so both halves of a comparison are in
// the SAME space: in Bake mode both are Rec.709 (a valid look A/B), in Working
// mode both are the untouched working space (bit-identical to the raw input).
static inline void deliverInput(const SpeakParams& pr, float r, float g, float b,
                                float& oR, float& oG, float& oB)
{
    if (pr.outputMode == SPEAK_OUT_BAKE_REC709) {
        const int cs = pr.inputColorSpace;
        const float lr = decodeToLinear(cs, r);
        const float lg = decodeToLinear(cs, g);
        const float lb = decodeToLinear(cs, b);
        float rr, rg, rb;
        gamutToRec709Lin(cs, lr, lg, lb, rr, rg, rb);
        rr = rr < 0.0f ? 0.0f : rr;
        rg = rg < 0.0f ? 0.0f : rg;
        rb = rb < 0.0f ? 0.0f : rb;
        oR = encodeFromLinear(SPEAK_CS_REC709_G24, rr);
        oG = encodeFromLinear(SPEAK_CS_REC709_G24, rg);
        oB = encodeFromLinear(SPEAK_CS_REC709_G24, rb);
    } else {
        oR = r; oG = g; oB = b;
    }
}

// ---------------------------------------------------------------------------
// The whole per-pixel operation. `x,y` are buffer coords (y up, OFX-native);
// the scope converts to display space internally. RGBA in -> RGBA out (alpha
// passes through). This is the ONE function the four backends must agree on.
// ---------------------------------------------------------------------------
static inline void processPixel(float r, float g, float b,
                                float scatR, float scatG, float scatB,
                                int x, int y, int W, int H,
                                const SpeakParams& pr, const uint32_t* stats,
                                float& outR, float& outG, float& outB)
{
    const SpeakProfile& p = pr.profile;
    const int cs = pr.inputColorSpace;
    const bool toneOn  = (pr.enableTone != 0) && (pr.strength > 0.0f);
    const bool dyeOn   = (pr.enableDye != 0) && dyeActive(p);
    const bool splitOn = (pr.enableSplit != 0) && splitActive(p);
    const bool bake    = (pr.outputMode == SPEAK_OUT_BAKE_REC709);

    if (!toneOn && !dyeOn && !splitOn && !bake) {
        // Working space + no look: bit-exact pass-through (identity). Scopes
        // may still overwrite below. Halation cannot reach here: halActive()
        // requires toneOn, so this branch already excludes it.
        outR = r; outG = g; outB = b;
    } else {
        // The look in working-space linear (halation + tone spine + subtractive
        // color) — the SAME function the density scope measures.
        float mr, mg, mb;
        lookLinear(r, g, b, scatR, scatG, scatB, pr, mr, mg, mb);
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

    // View modes (Split / Input): show the input delivered through the same
    // output CST, so a Split isolates the LOOK, not the color space.
    if (pr.viewMode == SPEAK_VIEW_INPUT ||
        (pr.viewMode == SPEAK_VIEW_SPLIT && x < W / 2))
        deliverInput(pr, r, g, b, outR, outG, outB);

    // Isolated-scatter view (spec 1B.5: "an isolated-scatter scope"; 1B "show
    // halation-only"). It shows the ACTUAL injected re-exposure, a * w_c * S_c,
    // delivered through the SAME output transform as the picture — so it is in
    // the same units as the image and cannot overstate itself. Deliberately NOT
    // auto-normalized or gained up: a per-frame normalisation would make a
    // negligible halo look enormous, which is the exact dishonesty this view
    // exists to prevent. If it reads dark, that IS the measurement — halation is
    // a small addition everywhere except beside a real highlight.
    if (pr.viewMode == SPEAK_VIEW_SCATTER) {
        const float a = halAmountOf(pr);
        const float on = ((pr.enableTone != 0) && (pr.strength > 0.0f)) ? 1.0f : 0.0f;
        float sr = on * a * kHalWeight[0] * scatR;
        float sg = on * a * kHalWeight[1] * scatG;
        float sb = on * a * kHalWeight[2] * scatB;
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, sr, sg, sb, rr, rg, rb);
            sr = rr < 0.0f ? 0.0f : rr;
            sg = rg < 0.0f ? 0.0f : rg;
            sb = rb < 0.0f ? 0.0f : rb;
            outR = encodeFromLinear(SPEAK_CS_REC709_G24, sr);
            outG = encodeFromLinear(SPEAK_CS_REC709_G24, sg);
            outB = encodeFromLinear(SPEAK_CS_REC709_G24, sb);
        } else {
            outR = encodeFromLinear(cs, sr);
            outG = encodeFromLinear(cs, sg);
            outB = encodeFromLinear(cs, sb);
        }
    }

    // Scopes render last, over any view (each owns its own corner).
    float sr, sg, sb;
    if (hdScopePixel(x, y, W, H, pr, stats, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
    if (densityScopePixel(x, y, W, H, pr, stats, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
}

// ---------------------------------------------------------------------------
// Whole-frame CPU entry point (the reference the GPU kernels are ported from).
// Interleaved RGBA float, row-major, y up (OFX-native buffer order).
// ---------------------------------------------------------------------------
// Builds the scatter pyramid for a frame. `arena` must hold
// halArenaPixels(W,H)*3 floats and `scat` W*H*3. Only called when halation is
// live — see speakFrame.
inline void buildHalScatter(const float* src, int W, int H, const SpeakParams& pr,
                            float* arena, float* scat)
{
    const int cs = pr.inputColorSpace;
    const float th = pr.profile.halThresh;
    // Level 0: the per-channel scene-linear highlight excess.
    // THRESHOLD BEFORE DECIMATION — mean(max(0, l-t)) != max(0, mean(l)-t), so
    // thresholding a downsampled level would scatter a different field than the
    // one the physics names (and would leak sub-threshold light into the halo).
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const size_t o = (static_cast<size_t>(y) * W + x) * 3;
            arena[o + 0] = halExcess(decodeToLinear(cs, src[i + 0]), th);
            arena[o + 1] = halExcess(decodeToLinear(cs, src[i + 1]), th);
            arena[o + 2] = halExcess(decodeToLinear(cs, src[i + 2]), th);
        }
    const int nLev = halLevelCount(W, H);
    for (int L = 1; L < nLev; ++L) {
        int sw, sh, so, dw, dh, doff;
        halLevelInfo(W, H, L - 1, sw, sh, so);
        halLevelInfo(W, H, L,     dw, dh, doff);
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x) {
                float v[3];
                halDecimatePixel(arena, so, sw, sh, x, y, v);
                const size_t o = (static_cast<size_t>(doff) + static_cast<size_t>(y) * dw + x) * 3;
                arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
            }
    }
    // Coarse-to-fine accumulate, in place, one octave at a time.
    const float sig = halSigmaPx(H, pr);
    for (int L = nLev - 1; L >= 0; --L) {
        int lw, lh, off;
        halLevelInfo(W, H, L, lw, lh, off);
        for (int y = 0; y < lh; ++y)
            for (int x = 0; x < lw; ++x) {
                float v[3];
                halAccumPixel(arena, W, H, L, nLev, sig, x, y, v);
                const size_t o = (static_cast<size_t>(off) + static_cast<size_t>(y) * lw + x) * 3;
                arena[o + 0] = v[0]; arena[o + 1] = v[1]; arena[o + 2] = v[2];
            }
    }
    // Normalize level 0 into the scatter plane: sum(w) = 1 => energy preserved.
    const float inv = 1.0f / halWeightSum(nLev, sig);
    for (size_t k = 0; k < static_cast<size_t>(W) * H * 3; ++k) scat[k] = arena[k] * inv;
}

inline void speakFrame(const float* src, int W, int H, const SpeakParams& pr, float* dst)
{
    // Heap, not stack: SPEAK_STATS_UINTS is 36994 uints = ~148 KB, which is at
    // or past the default stack limit on some hosts' worker threads.
    std::vector<uint32_t> stats(SPEAK_STATS_UINTS, 0u);
    std::vector<float> arena, scat;
    const bool hal = halActive(pr) || (pr.viewMode == SPEAK_VIEW_SCATTER);
    if (hal) {
        arena.assign(static_cast<size_t>(halArenaPixels(W, H)) * 3, 0.0f);
        scat.assign(static_cast<size_t>(W) * H * 3, 0.0f);
        buildHalScatter(src, W, H, pr, arena.data(), scat.data());
    }
    const float* sc = hal ? scat.data() : nullptr;
    computeStats(src, sc, W, H, pr, stats.data());   // measure the frame, then render
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const size_t j = (static_cast<size_t>(y) * W + x) * 3;
            float oR, oG, oB;
            processPixel(src[i + 0], src[i + 1], src[i + 2],
                         sc ? sc[j + 0] : 0.0f, sc ? sc[j + 1] : 0.0f, sc ? sc[j + 2] : 0.0f,
                         x, y, W, H, pr, stats.data(), oR, oG, oB);
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
        p.subSat[c] = 0.0f; p.subSatKnee[c] = 0.0f;   // 0 = subtractive color off / knee disabled
        p.splitShadow[c] = 0.0f; p.splitHigh[c] = 0.0f;
    }
    for (int k = 0; k < 9; ++k) p.dyeCouple[k] = 0.0f;
    p.printerMaster = 0.0f;
    p.splitPivot = 0.0f; p.splitBalance = 0.5f;
    // Halation: OFF by default (amount 0 => bit-exact identity, and the whole
    // multi-pass scatter chain is skipped). The radius default is a modelled
    // characteristic scale, not a measured stock: base-reflection geometry puts
    // the TIR onset around 2*t*tan(asin(1/n)) ~ 0.23 mm for a 0.13 mm base,
    // ~1% of a Super-35 frame. That sets the ORDER of the default; it is not a
    // precision claim, and the hint does not make one.
    p.halAmount = 0.0f; p.halRadius = 1.0f; p.halThresh = 0.6f;
    p.systemGamma = 1.6f; p.residualLUT = 0; p.profileVersion = 1; p._pad0 = 0;
    return p;
}

} // namespace speakcore

#endif // OPENNR_SPEAK_CORE_H
