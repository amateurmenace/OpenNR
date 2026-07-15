// Renders the 1280x720 website shots of each scope/view individually:
// scope_measure / scope_eq / scope_motion (with camera drift so the map has
// red) / scope_snr. PPMs come out in buffer order; flip vertically and
// convert for the site (see CLAUDE.md "site build"):
//   c++ -O2 -std=c++14 -I../plugin site_shots.cpp -o site_shots && ./site_shots
//   sips -s format jpeg -s formatOptions 88 --flip vertical scope_measure.ppm \
//        --out ../site/assets/scope-measure.jpg   (repeat per shot)
#include "nr_core.h"
#include <cstdio>
#include <random>
#include <vector>

static const int W = 1280, H = 720;

static void scene(std::vector<float>& img, float ox, float oy)
{
    img.assign(static_cast<size_t>(W) * H * 4, 1.0f);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            const float xs = x - ox, ys = y - oy;
            const float gx = static_cast<float>(x) / W, gy = static_cast<float>(y) / H;
            p[0] = 0.42f - 0.18f * gy + 0.06f * gx;
            p[1] = 0.55f - 0.20f * gy;
            p[2] = 0.47f - 0.16f * gy + 0.03f * (1.0f - gx);
            if (xs > 90 && xs < 330 && ys > 320 && ys < 560) { p[0] = 0.90f; p[1] = 0.88f; p[2] = 0.82f; }
            if (xs > 90 && xs < 330 && ys > 90 && ys < 260) { p[0] = 0.72f; p[1] = 0.42f; p[2] = 0.27f; }
            const float dx1 = xs - 640.0f, dy1 = ys - 420.0f;
            if (dx1 * dx1 + dy1 * dy1 < 120.0f * 120.0f) { p[0] = 0.13f; p[1] = 0.19f; p[2] = 0.12f; }
            const float dx2 = xs - 640.0f, dy2 = ys - 150.0f;
            if (dx2 * dx2 + dy2 * dy2 < 80.0f * 80.0f) { p[0] = 0.93f; p[1] = 0.90f; p[2] = 0.81f; }
            if (xs > 880 && xs < 1200 && ys > 400 && ys < 620) {
                const float t = 0.07f * std::sin(xs * 0.9f) * std::sin(ys * 0.8f);
                p[0] = 0.50f + t; p[1] = 0.54f + t; p[2] = 0.50f + t;
            }
            if (xs > 880 && xs < 1200 && ys > 90 && ys < 260)
                if (((static_cast<int>(xs + 4096) / 14) & 1) == 0) { p[0] = 0.14f; p[1] = 0.19f; p[2] = 0.12f; }
            p[3] = 1.0f;
        }
}

static void addNoise(std::vector<float>& img, float sigma, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, sigma);
    for (size_t i = 0; i < img.size(); i += 4) {
        img[i] += N(rng); img[i + 1] += N(rng); img[i + 2] += N(rng);
    }
}

static void writePPM(const char* path, const std::vector<float>& img)
{
    FILE* f = fopen(path, "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (size_t i = 0; i < img.size(); i += 4)
        for (int c = 0; c < 3; ++c) {
            const float v = img[i + c];
            fputc(static_cast<int>(std::min(1.0f, std::max(0.0f, v)) * 255.0f + 0.5f), f);
        }
    fclose(f);
}

static void run(const char* path, float driftX, float driftY,
                int sm, int se, int smo, int view)
{
    std::vector<float> frames[7];
    for (int k = 0; k < 7; ++k) {
        scene(frames[k], (k - 3) * driftX, (k - 3) * driftY);
        addNoise(frames[k], 0.035f, 100 + (k - 3) + 2);   // offset-keyed seeds
    }
    const float* fptr[7] = { frames[0].data(), frames[1].data(), frames[2].data(),
                             frames[3].data(), frames[4].data(), frames[5].data(),
                             frames[6].data() };
    nrcore::Params p;
    p.temporalFrames = 5;
    p.temporalLuma = 0.85f; p.temporalChroma = 1.05f;
    p.spatialRadius = 5; p.spatialLuma = 0.80f; p.spatialChroma = 1.25f;
    p.preserveDetail = 0.30f; p.detailRescue = 0.40f;
    p.chromaBlotch = 0.55f; p.eqFine = 1.05f; p.eqMedium = 0.30f; p.eqCoarse = 0.10f;
    p.scopeMeasure = sm; p.scopeEq = se; p.scopeMotion = smo;
    p.viewMode = view;
    std::vector<float> out(frames[0].size()), scratch;
    nrcore::denoiseFrame(fptr, W, H, p, out.data(), scratch);
    writePPM(path, out);
    printf("wrote %s\n", path);
}

int main()
{
    run("scope_measure.ppm", 0, 0,   1, 0, 0, 0);
    run("scope_eq.ppm",      0, 0,   0, 1, 0, 0);
    run("scope_motion.ppm",  6, 2.5f, 0, 0, 1, 0);   // drifting camera: red + green
    run("scope_snr.ppm",     0, 0,   0, 0, 0, 7);
    return 0;
}
