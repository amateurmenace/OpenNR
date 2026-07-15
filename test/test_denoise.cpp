// OpenNR test harness — validates nr_core.h without DaVinci Resolve.
//
// Covers:
//   1. estimator accuracy on iid Gaussian noise (all levels)
//   2. estimator accuracy on spatially CORRELATED noise (the real-world case
//      that broke v1.0's Laplacian-only estimator)
//   3. temporal-difference estimator sanity under global motion (clamped)
//   4. PSNR gains, static and moving
//   5. identity: master = 0, and both stages disabled
//   6. manual profile passthrough, region profiling
//   7. renders of every view mode incl. the Noise Analysis HUD (PPM out)
//
// Build: c++ -O2 -std=c++14 -I../plugin test_denoise.cpp -o test_denoise

#include "nr_core.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static const int W = 640;
static const int H = 360;

// ---------------------------------------------------------------------------
static void scenePixel(float fx, float fy, float& r, float& g, float& b)
{
    r = 0.18f + 0.35f * (fx / W);
    g = 0.20f + 0.30f * (fx / W) + 0.05f * (fy / H);
    b = 0.24f + 0.28f * (fx / W);

    if (fx > 60 && fx < 180 && fy > 50 && fy < 150) { r = g = b = 0.45f; }
    if (fx > 60 && fx < 180 && fy > 200 && fy < 300) { r = 0.55f; g = 0.30f; b = 0.25f; }
    {
        const float dx = fx - 320, dy = fy - 120;
        if (dx * dx + dy * dy < 60 * 60) { r = 0.85f; g = 0.80f; b = 0.70f; }
    }
    {
        const float dx = fx - 320, dy = fy - 260;
        if (dx * dx + dy * dy < 45 * 45) { r = 0.08f; g = 0.09f; b = 0.12f; }
    }
    if (fx > 440 && fx < 620 && fy > 40 && fy < 160) {
        const float t = 0.10f * std::sin(fx * 0.9f) * std::sin(fy * 0.8f);
        r += t; g += t; b += t;
    }
    if (fx > 440 && fx < 620 && fy > 200 && fy < 320) {
        const int bar = static_cast<int>(fx) % 12;
        if (bar < 2) { r *= 0.35f; g *= 0.35f; b *= 0.35f; }
    }
}

static void renderScene(std::vector<float>& img, float shiftX, float shiftY)
{
    img.resize(static_cast<size_t>(W) * H * 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float r, g, b;
            scenePixel(x + shiftX, y + shiftY, r, g, b);
            float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            p[0] = r; p[1] = g; p[2] = b; p[3] = 1.0f;
        }
}

static void addNoise(std::vector<float>& img, float sigma, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, sigma);
    for (size_t i = 0; i < img.size(); i += 4) {
        img[i + 0] += N(rng);
        img[i + 1] += N(rng);
        img[i + 2] += N(rng);
    }
}

// Spatially correlated noise: iid noise generated at half resolution, then
// nearest-upsampled 2x — every 2x2 block shares one noise value per channel.
// Mimics debayer/chroma-subsampling/compression correlation.
static void addCorrelatedNoise(std::vector<float>& img, float sigma, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, sigma);
    const int hw = (W + 1) / 2, hh = (H + 1) / 2;
    std::vector<float> half(static_cast<size_t>(hw) * hh * 3);
    for (auto& v : half) v = N(rng);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t s = (static_cast<size_t>(y / 2) * hw + (x / 2)) * 3;
            float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            p[0] += half[s + 0];
            p[1] += half[s + 1];
            p[2] += half[s + 2];
        }
}

// ---------------------------------------------------------------------------
static double psnr(const std::vector<float>& a, const std::vector<float>& b, int border = 8)
{
    double mse = 0.0;
    size_t n = 0;
    for (int y = border; y < H - border; ++y)
        for (int x = border; x < W - border; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            for (int c = 0; c < 3; ++c) {
                const double d = a[i + c] - b[i + c];
                mse += d * d;
                ++n;
            }
        }
    mse /= static_cast<double>(n);
    return 10.0 * std::log10(1.0 / std::max(mse, 1e-12));
}

static float trueSigmaY(const std::vector<float>& noisy, const std::vector<float>& clean)
{
    double acc = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < noisy.size(); i += 4) {
        float yn, cb, cr, yc;
        nrcore::rgb2ycc(noisy[i], noisy[i + 1], noisy[i + 2], yn, cb, cr);
        nrcore::rgb2ycc(clean[i], clean[i + 1], clean[i + 2], yc, cb, cr);
        const double d = yn - yc;
        acc += d * d;
        ++n;
    }
    return static_cast<float>(std::sqrt(acc / n));
}

static void writePPM(const std::string& path, const std::vector<float>& img)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::vector<unsigned char> row(static_cast<size_t>(W) * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            for (int c = 0; c < 3; ++c)
                row[x * 3 + c] = static_cast<unsigned char>(nrcore::clampf(p[c], 0.0f, 1.0f) * 255.0f + 0.5f);
        }
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
}

// ---------------------------------------------------------------------------
static int g_failures = 0;
static void check(bool ok, const char* what)
{
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

struct CaseResult { double before, after; nrcore::Stats stats; float trueY; };

static CaseResult runCase(float sigma, float stepX, float stepY, bool correlated, const nrcore::Params& p,
                          std::vector<float>* outDenoised = nullptr,
                          std::vector<float>* outNoisy = nullptr)
{
    std::vector<float> clean;
    renderScene(clean, 0, 0);

    std::vector<std::vector<float>> frames(5);
    for (int k = 0; k < 5; ++k) {
        const float sx = (k - 2) * stepX;
        const float sy = (k - 2) * stepY;
        renderScene(frames[k], sx, sy);
        if (correlated)
            addCorrelatedNoise(frames[k], sigma, 1000 + k);
        else
            addNoise(frames[k], sigma, 1000 + k);
    }

    const float* fptr[5] = { frames[0].data(), frames[1].data(), frames[2].data(),
                             frames[3].data(), frames[4].data() };

    std::vector<float> out(static_cast<size_t>(W) * H * 4);
    std::vector<float> scratch;
    const nrcore::Stats s = nrcore::denoiseFrame(fptr, W, H, p, out.data(), scratch);

    CaseResult r;
    r.before = psnr(frames[2], clean);
    r.after  = psnr(out, clean);
    r.stats  = s;
    r.trueY  = trueSigmaY(frames[2], clean);
    if (outDenoised) *outDenoised = out;
    if (outNoisy)    *outNoisy = frames[2];
    return r;
}

int main()
{
    nrcore::Params p; // defaults

    printf("OpenNR algorithm validation v1.1 (%dx%d synthetic scene)\n", W, H);
    printf("=========================================================\n");

    // --- iid noise, static, 5 frames
    p.temporalFrames = 5;
    for (float sigma : { 0.02f, 0.05f, 0.10f }) {
        std::vector<float> denoised, noisy;
        const bool keep = (sigma == 0.05f);
        CaseResult r = runCase(sigma, 0.0f, 0.0f, false, p, keep ? &denoised : nullptr, keep ? &noisy : nullptr);
        printf("static iid s=%.2f: PSNR %5.2f -> %5.2f dB   sigS %.4f sigT %.4f / true %.4f\n",
               sigma, r.before, r.after, r.stats.sy, r.stats.ty, r.trueY);
        char buf[160];
        snprintf(buf, sizeof(buf), "spatial sigma within 30%% of truth (iid s=%.2f)", sigma);
        check(std::fabs(r.stats.sy - r.trueY) / r.trueY < 0.30f, buf);
        snprintf(buf, sizeof(buf), "temporal sigma within 25%% of truth (iid s=%.2f)", sigma);
        check(std::fabs(r.stats.ty - r.trueY) / r.trueY < 0.25f, buf);
        snprintf(buf, sizeof(buf), "PSNR gain >= 6 dB (static iid s=%.2f)", sigma);
        check(r.after >= r.before + 6.0, buf);
        if (keep) {
            std::vector<float> clean;
            renderScene(clean, 0, 0);
            writePPM("out_clean.ppm", clean);
            writePPM("out_noisy.ppm", noisy);
            writePPM("out_denoised.ppm", denoised);
        }
    }

    // --- CORRELATED noise (the v1.0 failure case): estimator must still see it
    {
        CaseResult r = runCase(0.05f, 0.0f, 0.0f, true, p);
        printf("static CORR s=0.05: PSNR %5.2f -> %5.2f dB   sigS %.4f sigT %.4f / true %.4f\n",
               r.before, r.after, r.stats.sy, r.stats.ty, r.trueY);
        check(r.stats.ty > 0.60f * r.trueY, "temporal sigma catches correlated noise (>60% of truth)");
        check(r.stats.sy > 0.55f * r.trueY, "spatial sigma catches correlated noise (>55% of truth)");
        check(r.after >= r.before + 5.0, "PSNR gain >= 5 dB on correlated noise");
    }

    // --- motion: no ghosting collapse, temporal sigma stays clamped
    {
        CaseResult r = runCase(0.05f, 3.0f, 1.5f, false, p);
        printf("motion iid s=0.05: PSNR %5.2f -> %5.2f dB   sigS %.4f sigT %.4f / true %.4f\n",
               r.before, r.after, r.stats.sy, r.stats.ty, r.trueY);
        check(r.after >= r.before + 4.0, "PSNR gain >= 4 dB under motion");
        check(r.stats.ty <= 1.55f * r.stats.sy, "motion cannot inflate temporal sigma past clamp");
    }

    // --- SLOW motion (the ghosting regime): sub-pixel drift like real
    //     handheld/panning footage. v1.1's soft Gaussian gate blended
    //     mismatched pixels here ("ghost soup"). Requirement: temporal NR
    //     must never do worse than spatial-only.
    {
        nrcore::Params ps = p;
        ps.enableTemporal = 0;
        CaseResult spatialOnly = runCase(0.05f, 0.8f, 0.5f, false, ps);
        CaseResult full        = runCase(0.05f, 0.8f, 0.5f, false, p);
        printf("slow drift s=0.05: spatial-only %5.2f dB, temporal+spatial %5.2f dB (sigT %.4f)\n",
               spatialOnly.after, full.after, full.stats.ty);
        check(full.after >= spatialOnly.after - 0.3, "slow motion: temporal never hurts (no ghosting)");
        check(full.after >= full.before + 4.0, "slow motion: still >= 4 dB gain");
    }

    // --- identity: master = 0
    {
        nrcore::Params pid = p;
        pid.master = 0.0f;
        std::vector<float> clean, noisy, out(static_cast<size_t>(W) * H * 4), scratch;
        renderScene(clean, 0, 0);
        noisy = clean;
        addNoise(noisy, 0.05f, 7);
        const float* fptr[5] = { noisy.data(), noisy.data(), noisy.data(), noisy.data(), noisy.data() };
        nrcore::denoiseFrame(fptr, W, H, pid, out.data(), scratch);
        float maxd = 0.0f;
        for (size_t i = 0; i < out.size(); ++i)
            maxd = std::max(maxd, std::fabs(out[i] - noisy[i]));
        check(maxd < 2e-5f, "master=0 is identity");
    }

    // --- identity: both stages disabled
    {
        nrcore::Params pid = p;
        pid.enableTemporal = 0;
        pid.enableSpatial = 0;
        std::vector<float> clean, noisy, out(static_cast<size_t>(W) * H * 4), scratch;
        renderScene(clean, 0, 0);
        noisy = clean;
        addNoise(noisy, 0.05f, 8);
        const float* fptr[5] = { noisy.data(), noisy.data(), noisy.data(), noisy.data(), noisy.data() };
        nrcore::denoiseFrame(fptr, W, H, pid, out.data(), scratch);
        float maxd = 0.0f;
        for (size_t i = 0; i < out.size(); ++i)
            maxd = std::max(maxd, std::fabs(out[i] - noisy[i]));
        check(maxd < 2e-5f, "both stages disabled is identity");
    }

    // --- stage isolation: spatial-only must still denoise
    {
        nrcore::Params ps = p;
        ps.enableTemporal = 0;
        CaseResult r = runCase(0.05f, 0.0f, 0.0f, false, ps);
        printf("spatial only s=0.05: PSNR %5.2f -> %5.2f dB\n", r.before, r.after);
        check(r.after >= r.before + 4.0, "spatial-only gain >= 4 dB");
    }

    // --- manual profile passthrough
    {
        nrcore::Params pm = p;
        pm.profileSource = 2;
        pm.sigmaY = 0.037f;
        pm.sigmaC = 0.021f;
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<float> noisy = clean;
        addNoise(noisy, 0.05f, 11);
        nrcore::Stats s = nrcore::estimateSigma(noisy.data(), nullptr, W, H, pm);
        check(s.sy == 0.037f && s.tc == 0.021f, "manual sigma values pass through to both stages");
    }

    // --- profile adjust scales the auto estimate
    {
        nrcore::Params pa = p;
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<float> noisy = clean;
        addNoise(noisy, 0.05f, 12);
        nrcore::Stats s1 = nrcore::estimateSigma(noisy.data(), nullptr, W, H, pa);
        pa.profileAdjust = 2.0f;
        nrcore::Stats s2 = nrcore::estimateSigma(noisy.data(), nullptr, W, H, pa);
        check(std::fabs(s2.sy - 2.0f * s1.sy) < 1e-5f, "profile adjust x2 doubles the estimate");
    }

    // --- duplicate partner frame (clip boundary): zero temporal diff must not
    //     collapse the temporal sigma; it falls back to the spatial estimate
    {
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<float> noisy = clean;
        addNoise(noisy, 0.05f, 21);
        std::vector<float> duplicate = noisy; // distinct buffer, identical content
        nrcore::Stats s = nrcore::estimateSigma(noisy.data(), duplicate.data(), W, H, p);
        printf("duplicate partner: sigT %.4f (spatial fallback %.4f)\n", s.ty, s.sy);
        check(s.ty > 0.01f, "duplicate partner does not collapse temporal sigma");
    }

    // --- region profiling on the flat gray card
    {
        nrcore::Params pr = p;
        pr.profileSource = 1;
        pr.regionCX = 120.0f / W;
        pr.regionCY = 100.0f / H;
        pr.regionSize = 0.20f;
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<float> noisy = clean;
        addNoise(noisy, 0.05f, 13);
        nrcore::Stats s = nrcore::estimateSigma(noisy.data(), nullptr, W, H, pr);
        const float t = trueSigmaY(noisy, clean);
        printf("region profile: est %.4f / true %.4f\n", s.sy, t);
        check(std::fabs(s.sy - t) / t < 0.30f, "region-based sigma within 30% of truth");
    }

    // --- render every view mode (HUD and activity map get a visual check)
    {
        for (int vm = 1; vm <= 4; ++vm) {
            nrcore::Params pv = p;
            pv.viewMode = vm;
            if (vm == 3) pv.profileSource = 1; // show the region rect in analysis
            std::vector<float> denoised;
            runCase(0.05f, (vm == 4) ? 3.0f : 0.0f, (vm == 4) ? 1.5f : 0.0f, false, pv, &denoised, nullptr);
            char name[64];
            snprintf(name, sizeof(name), "out_view%d.ppm", vm);
            writePPM(name, denoised);
        }
        printf("view renders written: out_view1..4.ppm\n");
        check(true, "view modes rendered (visual check)");
    }

    printf("=========================================================\n");
    if (g_failures == 0)
        printf("ALL CHECKS PASSED\n");
    else
        printf("%d CHECK(S) FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
