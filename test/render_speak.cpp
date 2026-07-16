// Speak — dev render tool (never ships). Renders a DWG/DI test frame (a
// scene-linear gray ramp + primary/secondary color bars, DI-encoded) through
// the CPU reference with a behavioral stock look and the H&D scope on, and
// writes a display-order PPM (top row first) so it looks like the viewer.

#include <cstdio>
#include <vector>
#include <cmath>
#include "speak_core.h"
using namespace speakcore;

int main(int argc, char** argv)
{
    const int W = 900, H = 520;
    std::vector<float> src(W * H * 4, 0.0f);

    // Build a DWG/DI frame: top 70% is a horizontal scene-linear ramp
    // (-6..+6 stops around gray); bottom 30% is 8 color bars.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t(y) * W + x) * 4;
            float r, g, b;
            if (y < (H * 7) / 10) {
                const float stops = -6.0f + 12.0f * (float(x) / (W - 1));
                const float lin = 0.18f * std::exp2(stops);
                r = g = b = lin;
            } else {
                const int bar = (x * 8) / W;
                const float lo = 0.18f * std::exp2(-1.0f), hi = 0.18f * std::exp2(2.0f);
                const float cols[8][3] = {
                    {hi,hi,hi},{hi,hi,lo},{lo,hi,hi},{lo,hi,lo},
                    {hi,lo,hi},{hi,lo,lo},{lo,lo,hi},{lo,lo,lo}};
                r = cols[bar][0]; g = cols[bar][1]; b = cols[bar][2];
            }
            src[i + 0] = diEncode(r); src[i + 1] = diEncode(g); src[i + 2] = diEncode(b); src[i + 3] = 1.0f;
        }

    SpeakParams p = {};
    p.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
    p.outputMode = (argc > 3 && atoi(argv[3]) == 1) ? SPEAK_OUT_BAKE_REC709 : SPEAK_OUT_WORKING;
    p.enableTone = 1;
    p.strength = (argc > 1) ? float(atof(argv[1])) : 0.85f;
    p.scopeHD = 1;
    p.profile = neutralProfile();
    // A warm behavioral stock: lower blue speed + printer trims (all
    // neutral-preserving on the gray axis).
    p.profile.negGamma[0] = 0.66f; p.profile.negGamma[2] = 0.58f;
    p.profile.prnGamma[1] = 2.8f;  p.profile.printerLights[0] = 2.0f;
    p.profile.printerLights[2] = -1.5f; p.profile.printerMaster = 0.5f;

    std::vector<float> dst(W * H * 4);
    speakFrame(src.data(), W, H, p, dst.data());

    // Write display-order PPM (row H-1 first) so the scope reads top-left.
    FILE* f = fopen(argc > 2 ? argv[2] : "speak_render.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int y = H - 1; y >= 0; --y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t(y) * W + x) * 4;
            for (int c = 0; c < 3; ++c) {
                float v = dst[i + c]; v = v < 0 ? 0 : (v > 1 ? 1 : v);
                unsigned char b8 = (unsigned char)(v * 255.0f + 0.5f);
                fwrite(&b8, 1, 1, f);
            }
        }
    fclose(f);
    printf("wrote render (%dx%d, strength %.2f)\n", W, H, p.strength);
    return 0;
}
