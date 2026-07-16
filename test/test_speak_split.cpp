// Speak — Phase 3 control arm: split toning vs a lift/gamma/gain baseline.
//
// The house rule: declare the cheap baseline and the measurement BEFORE the
// module ships. Baseline = classic LGG (lift = additive in linear, gain =
// multiplicative), matched to the split's tint strength on a shadow patch and a
// highlight patch. The density-domain split must then show two structural
// properties LGG cannot offer:
//   N1 MID NEUTRALITY  — tint shadows + highlights hard, and 18% gray comes out
//                        EXACTLY neutral and EXACTLY unchanged (the mid zone's
//                        weight is 0 at the pivot by construction). LGG's lift
//                        provably drags mid-gray off neutral.
//   N2 HUE STABILITY   — over the 24 Macbeth patches, the split rotates hue in
//                        CIELAB LESS than the matched LGG baseline (density
//                        offsets are multiplicative in linear; LGG's lift is
//                        additive, which swings hue in the shadows).
//
// Build: c++ -O2 -std=c++14 -I../plugin test_speak_split.cpp -o test_speak_split

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "speak_core.h"
using namespace speakcore;

static int g_fail = 0;
static void check(bool ok, const char* name, const std::string& detail = "")
{
    printf("  [%s] %s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) g_fail++;
}

static const int kMacbeth[24][3] = {
    {115,82,68}, {194,150,130}, {98,122,157}, {87,108,67}, {133,128,177}, {103,189,170},
    {214,126,44}, {80,91,166}, {193,90,99}, {94,60,108}, {157,188,64}, {224,163,46},
    {56,61,150}, {70,148,73}, {175,54,60}, {231,199,31}, {187,86,149}, {8,133,161},
    {243,243,242}, {200,200,200}, {160,160,160}, {122,122,121}, {85,85,85}, {52,52,52} };
static const float kRec709_to_XYZ[9] = {
    0.41245643f, 0.35757608f, 0.18043748f,
    0.21267285f, 0.71515217f, 0.07217500f,
    0.01933390f, 0.11919203f, 0.95030407f };
static const float kXYZ_to_DWG[9] = {
    1.51667205f,-0.28147806f,-0.14696364f,
   -0.46491710f, 1.25142377f, 0.17488461f,
    0.06484904f, 0.10913935f, 0.76141462f };

static float srgbToLin(int v8)
{
    const float v = v8 / 255.0f;
    return (v <= 0.04045f) ? (v / 12.92f) : std::pow((v + 0.055f) / 1.055f, 2.4f);
}
static void patchToDWG(int i, float& r, float& g, float& b)
{
    const float lr = srgbToLin(kMacbeth[i][0]), lg = srgbToLin(kMacbeth[i][1]), lb = srgbToLin(kMacbeth[i][2]);
    float X, Y, Z; mul3(kRec709_to_XYZ, lr, lg, lb, X, Y, Z);
    mul3(kXYZ_to_DWG, X, Y, Z, r, g, b);
}

// The CHEAP BASELINE: classic lift/gamma/gain colour balance.
// lift  = additive in linear (the shadow tint), gain = multiplicative (highlight tint).
struct LGG { float lift[3]; float gain[3]; };
static void applyLGG(float r, float g, float b, const LGG& l, float& oR, float& oG, float& oB)
{
    oR = (r + l.lift[0]) * l.gain[0];
    oG = (g + l.lift[1]) * l.gain[1];
    oB = (b + l.lift[2]) * l.gain[2];
}

struct Lab { float L, a, b; };
static Lab labOf(float r, float g, float b) { Lab o; dwgLinToLab(r, g, b, o.L, o.a, o.b); return o; }
static float chroma(const Lab& c) { return std::sqrt(c.a * c.a + c.b * c.b); }
static float hueDeg(const Lab& c) { return std::atan2(c.b, c.a) * 57.29577951f; }
static float hueDiff(float h1, float h0)
{
    float d = h1 - h0;
    while (d > 180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

int main()
{
    printf("=== Speak Phase 3 control arm: split toning vs lift/gamma/gain ===\n");

    // Subject: a chromogenic crossover — cool shadows, warm highlights, as
    // per-channel DENSITY offsets (more density = less of that channel).
    SpeakProfile p = neutralProfile();
    p.splitShadow[0] =  0.10f; p.splitShadow[1] = 0.0f; p.splitShadow[2] = -0.10f; // shadows -> cool
    p.splitHigh[0]   = -0.08f; p.splitHigh[1]   = 0.0f; p.splitHigh[2]   =  0.08f; // highlights -> warm
    p.splitPivot = 0.0f; p.splitBalance = 0.5f;

    // ---- N1: mid neutrality, and the LGG baseline's failure at the same job --
    {
        const float gray = k18Gray;
        float sR, sG, sB;
        splitTone(gray, gray, gray, p, sR, sG, sB);
        const float chromaOut = std::fmax(std::fabs(sR - sG), std::fabs(sG - sB));
        check(chromaOut < 1e-6f, "N1a 18% gray stays EXACTLY neutral through the split",
              "max channel spread " + std::to_string(chromaOut));
        check(std::fabs(sR - gray) < 1e-6f, "N1b 18% gray is EXACTLY unchanged (mid weight is 0)",
              "got " + std::to_string(sR) + " want " + std::to_string(gray));

        // Match an LGG baseline to the split's tint on a shadow and a highlight
        // gray, then look at what it did to mid-gray.
        const float shadowLin = k18Gray * std::exp2(-3.0f), highLin = k18Gray * std::exp2(2.5f);
        float ssR, ssG, ssB, hhR, hhG, hhB;
        splitTone(shadowLin, shadowLin, shadowLin, p, ssR, ssG, ssB);
        splitTone(highLin, highLin, highLin, p, hhR, hhG, hhB);
        LGG l;
        for (int c = 0; c < 3; ++c) {
            const float sTgt = (c == 0) ? ssR : (c == 1) ? ssG : ssB;
            const float hTgt = (c == 0) ? hhR : (c == 1) ? hhG : hhB;
            // solve (x + lift)*gain = target at the two anchors
            l.gain[c] = (hTgt - sTgt) / (highLin - shadowLin);
            l.lift[c] = hTgt / l.gain[c] - highLin;
        }
        float lR, lG, lB;
        applyLGG(gray, gray, gray, l, lR, lG, lB);
        const float lggChroma = std::fmax(std::fabs(lR - lG), std::fabs(lG - lB));
        printf("  matched LGG: lift %.4f/%.4f/%.4f  gain %.4f/%.4f/%.4f\n",
               l.lift[0], l.lift[1], l.lift[2], l.gain[0], l.gain[1], l.gain[2]);
        printf("  mid-gray chroma  — split %.2e  |  LGG %.2e\n", chromaOut, lggChroma);
        check(lggChroma > 20.0f * (chromaOut + 1e-9f) && lggChroma > 1e-4f,
              "N1c the matched LGG baseline DOES drag mid-gray off neutral",
              "LGG spread " + std::to_string(lggChroma));

        // ---- N2: hue stability over the Macbeth patches ----
        float splitHue = 0.0f, lggHue = 0.0f; int n = 0;
        for (int i = 0; i < 24; ++i) {
            float r, g, b; patchToDWG(i, r, g, b);
            const Lab in = labOf(r, g, b);
            if (chroma(in) < 5.0f) continue;          // neutrals: hue undefined
            float a1, a2, a3; splitTone(r, g, b, p, a1, a2, a3);
            float b1, b2, b3; applyLGG(r, g, b, l, b1, b2, b3);
            splitHue += std::fabs(hueDiff(hueDeg(labOf(a1, a2, a3)), hueDeg(in)));
            lggHue   += std::fabs(hueDiff(hueDeg(labOf(b1, b2, b3)), hueDeg(in)));
            n++;
        }
        splitHue /= n; lggHue /= n;
        printf("  mean |hue rotation| — split %.2f deg  |  matched LGG %.2f deg\n", splitHue, lggHue);
        check(splitHue < lggHue, "N2 split toning is more hue-stable than matched LGG",
              std::to_string(splitHue) + " deg < " + std::to_string(lggHue) + " deg");
    }

    printf("\n%s (%d failures)\n", g_fail ? "CONTROL ARM FAILED" : "CONTROL ARM PASSED", g_fail);
    return g_fail ? 1 : 0;
}
