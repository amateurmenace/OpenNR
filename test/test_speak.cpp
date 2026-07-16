// Speak — CPU gate suite. Validates the density-spine math against the Phase-1
// control arms BEFORE the GPU ports:
//   G1 struct layout parity (the cardinal-rule layout check)
//   G2 color-management round-trip is lossless (the CST scaffold gate)
//   G3 identity at default (strength 0 => bit-exact pass-through)
//   G4 neutral-in => neutral-out is exact for a gray-balanced profile
//   G5 H&D curve + tone scale are monotone
//   G6 gray pivots to gray (18% in => 18% out) by construction
//   G7 the on-screen scope curve is sampled from the production kernel
//      (plot == pixels)
//
// Build:
//   c++ -O2 -std=c++14 -I../plugin test_speak.cpp -o test_speak && ./test_speak

#include <cstdio>
#include <cstddef>
#include <cmath>
#include <vector>
#include <cstdint>

#include "speak_core.h"

using namespace speakcore;

static int g_fail = 0;
static void check(bool ok, const char* name, const char* detail = "")
{
    printf("  [%s] %s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok) g_fail++;
}

// ----------------------------------------------------------------- G1 layout
static void gateLayout()
{
    printf("G1 struct layout parity\n");
    // All-4-byte-field invariant: sizeof must equal the field count * 4.
    const size_t profFields = 67;   // see SpeakParams.h; keep in sync
    const size_t parFields  = 13 + profFields;
    check(sizeof(float) == 4 && sizeof(int) == 4, "float/int are 4 bytes");
    check(sizeof(SpeakProfile) == profFields * 4, "sizeof(SpeakProfile)==268",
          (std::to_string(sizeof(SpeakProfile))).c_str());
    check(sizeof(SpeakParams) == parFields * 4, "sizeof(SpeakParams)==320",
          (std::to_string(sizeof(SpeakParams))).c_str());
    check(offsetof(SpeakParams, profile) == 13 * 4, "profile offset==52",
          (std::to_string(offsetof(SpeakParams, profile))).c_str());
    // A few anchor offsets the GPU struct declarations must match.
    check(offsetof(SpeakProfile, printerLights) == 18 * 4, "printerLights offset==72");
    check(offsetof(SpeakProfile, prnDmin) == 22 * 4, "prnDmin offset==88");
    check(offsetof(SpeakProfile, dyeCouple) == 40 * 4, "dyeCouple offset==160");
}

// ------------------------------------------------------------ G2 round-trip
static void gateRoundTrip()
{
    printf("G2 color-management round-trip is lossless\n");
    const int spaces[] = { SPEAK_CS_DWG_INTERMEDIATE, SPEAK_CS_REC709_G24,
                           SPEAK_CS_ACESCCT, SPEAK_CS_LINEAR };
    for (int si = 0; si < 4; ++si) {
        const int cs = spaces[si];
        float maxErr = 0.0f;
        for (int i = 0; i <= 4000; ++i) {
            const float L = std::pow(10.0f, -4.0f + 8.0f * (i / 4000.0f)); // 1e-4..1e4
            const float v = encodeFromLinear(cs, L);
            const float L2 = decodeToLinear(cs, v);
            const float rel = std::fabs(L2 - L) / (std::fabs(L) + 1e-6f);
            if (rel > maxErr) maxErr = rel;
        }
        char buf[64]; snprintf(buf, sizeof(buf), "cs=%d maxRelErr=%.2e", cs, maxErr);
        check(maxErr < 1e-4f, "encode->decode round-trips", buf);
    }
    // Verify DI continuity at the segment cut (encode branches must meet).
    const float aLin = kDI_LIN_CUT * kDI_M;
    const float aLog = (std::log2(kDI_LIN_CUT + kDI_A) + kDI_B) * kDI_C;
    check(std::fabs(aLin - aLog) < 2e-4f, "DI encode is continuous at the cut");
}

// ----------------------------------------------------------- G3 identity
static void gateIdentity()
{
    printf("G3 identity at default (strength 0)\n");
    SpeakParams pr = {};
    pr.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
    pr.enableTone = 1;
    pr.strength = 0.0f;               // default: no look
    pr.profile = neutralProfile();
    // A deterministic pseudo-random-ish tile of values.
    const int W = 17, H = 11;
    std::vector<float> src(W * H * 4), dst(W * H * 4);
    for (int i = 0; i < W * H * 4; ++i)
        src[i] = std::fmod(std::sin(i * 12.9898f) * 43758.5453f, 1.0f) * 0.5f + 0.5f;
    speakFrame(src.data(), W, H, pr, dst.data());
    float maxAbs = 0.0f;
    for (int i = 0; i < W * H * 4; ++i)
        maxAbs = std::fmax(maxAbs, std::fabs(dst[i] - src[i]));
    check(maxAbs == 0.0f, "strength 0 => bit-exact pass-through",
          (std::string("maxAbs=") + std::to_string(maxAbs)).c_str());

    // enableTone 0 with strength 1 is also identity.
    pr.strength = 1.0f; pr.enableTone = 0;
    speakFrame(src.data(), W, H, pr, dst.data());
    maxAbs = 0.0f;
    for (int i = 0; i < W * H * 4; ++i) maxAbs = std::fmax(maxAbs, std::fabs(dst[i] - src[i]));
    check(maxAbs == 0.0f, "enableTone 0 => bit-exact pass-through");
}

// ----------------------------------------------------------- G4 neutral
static void gateNeutral()
{
    printf("G4 neutral-in => neutral-out is exact (gray-balanced profile)\n");
    SpeakProfile p = neutralProfile();
    float maxChroma = 0.0f;
    for (int i = 0; i <= 500; ++i) {
        const float lin = std::pow(10.0f, -3.0f + 5.0f * (i / 500.0f)); // 1e-3..1e2
        const float oR = toneChannel(lin, 0, p);
        const float oG = toneChannel(lin, 1, p);
        const float oB = toneChannel(lin, 2, p);
        maxChroma = std::fmax(maxChroma, std::fmax(std::fabs(oR - oG), std::fabs(oG - oB)));
    }
    check(maxChroma < 1e-6f, "R==G==B out for R==G==B in",
          (std::string("maxChroma=") + std::to_string(maxChroma)).c_str());
}

// ----------------------------------------------------------- G5 monotone
static void gateMonotone()
{
    printf("G5 H&D curve + tone scale are monotone\n");
    SpeakProfile p = neutralProfile();
    // H&D curve monotone in logH.
    float prev = -1e30f; bool mono = true;
    for (int i = 0; i <= 6000; ++i) {
        const float logH = -6.0f + 12.0f * (i / 6000.0f);
        const float D = hdCurve(logH, p.negDmin[0], p.negDmax[0], p.negGamma[0],
                                p.negToe[0], p.negShoulder[0], p.negSpeed[0]);
        if (D < prev - 1e-6f) { mono = false; break; }
        prev = D;
    }
    check(mono, "hdCurve is non-decreasing in logH");

    // Full tone scale monotone in scene-linear.
    prev = -1e30f; mono = true;
    for (int i = 0; i <= 6000; ++i) {
        const float lin = std::pow(10.0f, -4.0f + 8.0f * (i / 6000.0f));
        const float o = toneChannel(lin, 0, p);
        if (o < prev - 1e-7f) { mono = false; break; }
        prev = o;
    }
    check(mono, "toneChannel is non-decreasing in scene-linear");

    // The curve is a real S (has contrast), not a straight identity. Measure
    // the mid-gray slope over a tight +/-0.25 stop window (the design contrast;
    // a wide window dips into toe/shoulder and reads lower, as real film does).
    const float dS = 0.25f;
    const float sysGamma = (std::log2(toneChannel(k18Gray * std::exp2(dS), 0, p) / k18Gray) -
                            std::log2(toneChannel(k18Gray * std::exp2(-dS), 0, p) / k18Gray)) / (2.0f * dS);
    char buf[48]; snprintf(buf, sizeof(buf), "systemGamma~%.2f", sysGamma);
    check(sysGamma > 1.15f && sysGamma < 2.4f, "system gamma is filmic (~1.6)", buf);
}

// ----------------------------------------------------------- G6 gray pivot
static void gateGrayPivot()
{
    printf("G6 gray pivots to gray (18%% in => 18%% out)\n");
    SpeakProfile p = neutralProfile();
    // Also test an intentionally un-balanced (per-channel) profile: the pivot
    // is per-channel, so gray still maps to 0.18 on every channel.
    p.negGamma[0] = 0.70f; p.prnGamma[2] = 2.9f; p.negSpeed[1] = -2.2f;
    float maxErr = 0.0f;
    for (int ch = 0; ch < 3; ++ch)
        maxErr = std::fmax(maxErr, std::fabs(toneChannel(k18Gray, ch, p) - k18Gray));
    check(maxErr < 1e-5f, "each channel maps 0.18 -> 0.18",
          (std::string("maxErr=") + std::to_string(maxErr)).c_str());
}

// ----------------------------------------------------------- G7 scope==kernel
static void gateScopeMatchesKernel()
{
    printf("G7 scope curve tracks the pixels at every strength (plot == pixels)\n");
    SpeakProfile prof = neutralProfile();
    prof.negGamma[0] = 0.66f; prof.prnGamma[1] = 2.7f;  // non-trivial, per-channel
    const float strengths[] = { 0.0f, 0.5f, 1.0f };     // incl. identity (s=0)
    float maxErr = 0.0f;
    for (int si = 0; si < 3; ++si) {
        SpeakParams pr = {};
        pr.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
        pr.enableTone = 1; pr.strength = strengths[si]; pr.viewMode = SPEAK_VIEW_RESULT;
        pr.scopeHD = 0;                                  // measure the transform, not the overlay
        pr.profile = prof;
        for (int i = 0; i <= 200; ++i) {
            const float inStops = -6.0f + 12.0f * (i / 200.0f);
            const float lin = k18Gray * std::exp2(inStops);
            const float enc = diEncode(lin);
            float oR, oG, oB;
            processPixel(enc, enc, enc, 4, 4, 100, 100, pr, oR, oG, oB); // the REAL pixel path
            for (int ch = 0; ch < 3; ++ch) {
                const float scopeOut = scopeYStops(inStops, ch, pr);
                const float outCh = (ch == 0) ? oR : (ch == 1) ? oG : oB;
                const float pxLin = decodeToLinear(pr.inputColorSpace, outCh);
                const float pxStops = std::log2((pxLin < kLinTiny ? kLinTiny : pxLin) / k18Gray);
                maxErr = std::fmax(maxErr, std::fabs(scopeOut - pxStops));
            }
        }
    }
    // Bounded by the CST encode/decode round-trip (~1e-6), not a scope discrepancy.
    check(maxErr < 1e-4f, "scope value == pixel value at every strength",
          (std::string("maxErr=") + std::to_string(maxErr)).c_str());
}

// -------------------------------------------------------------- G8 bake CST
static void gateBakeCST()
{
    printf("G8 Bake-to-Rec.709 CST scaffold (neutral-identity + round-trip)\n");
    SpeakParams pr = {};
    pr.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
    pr.outputMode = SPEAK_OUT_BAKE_REC709;
    pr.enableTone = 0; pr.strength = 0.0f;      // pure CST, no look
    pr.profile = neutralProfile();

    // Neutral in -> neutral out: a DWG gray ramp bakes to Rec.709 with equal
    // channels (bounded by the published matrix's own rounding).
    float maxChroma = 0.0f;
    for (int i = 0; i <= 400; ++i) {
        const float lin = std::pow(10.0f, -3.0f + 5.0f * (i / 400.0f));
        const float enc = diEncode(lin);
        float oR, oG, oB;
        processPixel(enc, enc, enc, 4, 4, 100, 100, pr, oR, oG, oB);
        maxChroma = std::fmax(maxChroma, std::fmax(std::fabs(oR - oG), std::fabs(oG - oB)));
    }
    check(maxChroma < 2e-3f, "DWG neutral bakes to Rec.709 neutral",
          (std::string("maxChroma=") + std::to_string(maxChroma)).c_str());

    // 18% gray bakes to the correct Rec.709 code value (pow(0.18, 1/2.4)).
    {
        const float enc = diEncode(k18Gray);
        float oR, oG, oB;
        processPixel(enc, enc, enc, 4, 4, 100, 100, pr, oR, oG, oB);
        const float expect = std::pow(k18Gray, 1.0f / 2.4f);
        check(std::fabs(oR - expect) < 3e-3f, "18% gray -> correct Rec.709 code",
              (std::string("got=") + std::to_string(oR) + " want=" + std::to_string(expect)).c_str());
    }

    // Round-trip DWG-linear -> Rec.709-linear -> DWG-linear ~ identity (proves
    // the forward matrices are internally consistent).
    const float XYZ_to_DWG[9] = {
        1.51667205f,-0.28147806f,-0.14696364f,
       -0.46491710f, 1.25142377f, 0.17488461f,
        0.06484904f, 0.10913935f, 0.76141462f };
    const float Rec709_to_XYZ[9] = {
        0.41245643f, 0.35757608f, 0.18043748f,
        0.21267285f, 0.71515217f, 0.07217500f,
        0.01933390f, 0.11919203f, 0.95030407f };
    const float cols[4][3] = { {0.2f,0.2f,0.2f}, {0.4f,0.1f,0.05f}, {0.05f,0.3f,0.2f}, {0.6f,0.5f,0.1f} };
    float maxErr = 0.0f;
    for (int t = 0; t < 4; ++t) {
        float rr, rg, rb;
        gamutToRec709Lin(SPEAK_CS_DWG_LINEAR, cols[t][0], cols[t][1], cols[t][2], rr, rg, rb);
        float X, Y, Z, br, bg, bb;
        mul3(Rec709_to_XYZ, rr, rg, rb, X, Y, Z);
        mul3(XYZ_to_DWG, X, Y, Z, br, bg, bb);
        maxErr = std::fmax(maxErr, std::fmax(std::fabs(br - cols[t][0]),
                           std::fmax(std::fabs(bg - cols[t][1]), std::fabs(bb - cols[t][2]))));
    }
    check(maxErr < 1e-4f, "DWG->Rec.709->DWG round-trips",
          (std::string("maxErr=") + std::to_string(maxErr)).c_str());
}

int main()
{
    printf("=== Speak CPU gate suite ===\n");
    gateLayout();
    gateRoundTrip();
    gateIdentity();
    gateNeutral();
    gateMonotone();
    gateGrayPivot();
    gateScopeMatchesKernel();
    gateBakeCST();
    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL GATES GREEN", g_fail);
    return g_fail ? 1 : 0;
}
