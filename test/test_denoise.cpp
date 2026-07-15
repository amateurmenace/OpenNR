// OpenNR test harness — validates nr_core.h (v2) without DaVinci Resolve.
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

// 2x2-correlated RGB noise (v1.0 failure case)
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

// large soft chroma blotches (8x8 correlated, chroma only) — what 4:2:0 +
// camera NR leaves behind; the NLM window cannot span these.
static void addChromaBlotchNoise(std::vector<float>& img, float sigma, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, sigma);
    const int bw = (W + 7) / 8, bh = (H + 7) / 8;
    std::vector<float> blk(static_cast<size_t>(bw) * bh * 2);
    for (auto& v : blk) v = N(rng);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t s = (static_cast<size_t>(y / 8) * bw + (x / 8)) * 2;
            const float ncb = blk[s + 0], ncr = blk[s + 1];
            const float dR = 1.5748f * ncr;
            const float dB = 1.8556f * ncb;
            const float dG = -(0.2126f * dR + 0.0722f * dB) / 0.7152f;
            float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            p[0] += dR; p[1] += dG; p[2] += dB;
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

enum NoiseKind { NOISE_IID, NOISE_CORR, NOISE_BLOTCH };

static CaseResult runCase(float sigma, float stepX, float stepY, NoiseKind kind, const nrcore::Params& p,
                          std::vector<float>* outDenoised = nullptr,
                          std::vector<float>* outNoisy = nullptr)
{
    std::vector<float> clean;
    renderScene(clean, 0, 0);

    std::vector<std::vector<float>> frames(5);
    for (int k = 0; k < 5; ++k) {
        renderScene(frames[k], (k - 2) * stepX, (k - 2) * stepY);
        if (kind == NOISE_CORR) addCorrelatedNoise(frames[k], sigma, 1000 + k);
        else if (kind == NOISE_BLOTCH) { addNoise(frames[k], sigma * 0.5f, 1000 + k); addChromaBlotchNoise(frames[k], sigma, 2000 + k); }
        else addNoise(frames[k], sigma, 1000 + k);
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
    nrcore::Params p; // v2 defaults

    printf("OpenNR algorithm validation v2 (%dx%d synthetic scene)\n", W, H);
    printf("=========================================================\n");

    // --- iid noise, static, 5 frames
    p.temporalFrames = 5;
    for (float sigma : { 0.02f, 0.05f, 0.10f }) {
        std::vector<float> denoised, noisy;
        const bool keep = (sigma == 0.05f);
        CaseResult r = runCase(sigma, 0.0f, 0.0f, NOISE_IID, p, keep ? &denoised : nullptr, keep ? &noisy : nullptr);
        printf("static iid s=%.2f: PSNR %5.2f -> %5.2f dB   sigS %.4f sigT %.4f sigR %.4f effN %.1f / true %.4f\n",
               sigma, r.before, r.after, r.stats.sy, r.stats.ty, r.stats.ry, r.stats.effNMed, r.trueY);
        char buf[160];
        snprintf(buf, sizeof(buf), "spatial sigma within 30%% of truth (iid s=%.2f)", sigma);
        check(std::fabs(r.stats.sy - r.trueY) / r.trueY < 0.30f, buf);
        snprintf(buf, sizeof(buf), "residual < 75%% of input sigma (temporal worked, s=%.2f)", sigma);
        check(r.stats.ry < 0.75f * r.stats.sy, buf);
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

    // --- correlated noise (v1.0 failure case)
    {
        CaseResult r = runCase(0.05f, 0.0f, 0.0f, NOISE_CORR, p);
        printf("static CORR s=0.05: PSNR %5.2f -> %5.2f dB   sigS %.4f sigT %.4f sigR %.4f / true %.4f\n",
               r.before, r.after, r.stats.sy, r.stats.ty, r.stats.ry, r.trueY);
        check(r.stats.ty > 0.60f * r.trueY, "temporal sigma catches correlated noise");
        check(r.after >= r.before + 5.0, "PSNR gain >= 5 dB on correlated noise");
    }

    // --- chroma blotches: the large-radius chroma pass must matter
    {
        nrcore::Params pb0 = p; pb0.chromaBlotch = 0.0f;
        nrcore::Params pb1 = p; pb1.chromaBlotch = 1.0f;
        CaseResult r0 = runCase(0.04f, 0.0f, 0.0f, NOISE_BLOTCH, pb0);
        CaseResult r1 = runCase(0.04f, 0.0f, 0.0f, NOISE_BLOTCH, pb1);
        printf("chroma blotches: PSNR %5.2f -> without pass %5.2f dB, with pass %5.2f dB\n",
               r0.before, r0.after, r1.after);
        check(r1.after >= r0.after + 1.0, "blotch pass gains >= 1 dB on chroma stains");
    }

    // --- fast motion
    {
        CaseResult r = runCase(0.05f, 3.0f, 1.5f, NOISE_IID, p);
        printf("motion iid s=0.05: PSNR %5.2f -> %5.2f dB   sigT %.4f\n", r.before, r.after, r.stats.ty);
        check(r.after >= r.before + 4.0, "PSNR gain >= 4 dB under motion");
        check(r.stats.ty <= 1.55f * r.stats.sy, "motion cannot inflate temporal sigma past clamp");
    }

    // --- slow drift (the ghosting regime)
    {
        nrcore::Params ps = p;
        ps.enableTemporal = 0;
        CaseResult spatialOnly = runCase(0.05f, 0.8f, 0.5f, NOISE_IID, ps);
        CaseResult full        = runCase(0.05f, 0.8f, 0.5f, NOISE_IID, p);
        printf("slow drift s=0.05: spatial-only %5.2f dB, temporal+spatial %5.2f dB\n",
               spatialOnly.after, full.after);
        check(full.after >= spatialOnly.after - 0.3, "slow motion: temporal never hurts (no ghosting)");
        check(full.after >= full.before + 4.0, "slow motion: still >= 4 dB gain");
    }

    // --- master > 1 must actually do something now
    {
        nrcore::Params m1 = p;
        nrcore::Params m2 = p; m2.master = 2.0f;
        std::vector<float> o1, o2;
        runCase(0.10f, 0.0f, 0.0f, NOISE_IID, m1, &o1);
        runCase(0.10f, 0.0f, 0.0f, NOISE_IID, m2, &o2);
        double diff = 0.0;
        for (size_t i = 0; i < o1.size(); ++i) diff += std::fabs(o1[i] - o2[i]);
        diff /= o1.size();
        printf("master 1 vs 2: mean abs difference %.5f\n", diff);
        check(diff > 5e-4, "master > 1 changes the result (stronger filtering)");
    }

    // --- identity: master = 0; both stages + refine off
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
    {
        nrcore::Params pid = p;
        pid.enableTemporal = 0; pid.enableSpatial = 0; pid.enableRefine = 0;
        std::vector<float> clean, noisy, out(static_cast<size_t>(W) * H * 4), scratch;
        renderScene(clean, 0, 0);
        noisy = clean;
        addNoise(noisy, 0.05f, 8);
        const float* fptr[5] = { noisy.data(), noisy.data(), noisy.data(), noisy.data(), noisy.data() };
        nrcore::denoiseFrame(fptr, W, H, pid, out.data(), scratch);
        float maxd = 0.0f;
        for (size_t i = 0; i < out.size(); ++i)
            maxd = std::max(maxd, std::fabs(out[i] - noisy[i]));
        check(maxd < 2e-5f, "all stages disabled is identity");
    }

    // --- refine: grain determinism, grain identity at 0, shadow desat effect
    {
        nrcore::Params pg = p;
        pg.grainAmount = 0.5f; pg.frameIndex = 42;
        std::vector<float> a, b;
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pg, &a);
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pg, &b);
        bool same = true;
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) { same = false; break; }
        check(same, "grain is deterministic for a fixed frame");

        nrcore::Params pg2 = pg; pg2.frameIndex = 43;
        std::vector<float> c2;
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pg2, &c2);
        double d = 0.0;
        for (size_t i = 0; i < a.size(); ++i) d += std::fabs(a[i] - c2[i]);
        check(d / a.size() > 1e-4, "grain animates between frames");

        nrcore::Params p0 = p; p0.enableRefine = 1;   // defaults: desat 0, tex 0, grain 0
        nrcore::Params pOff = p; pOff.enableRefine = 0;
        std::vector<float> r1, r0;
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, p0, &r1);
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pOff, &r0);
        float maxd = 0.0f;
        for (size_t i = 0; i < r1.size(); ++i) maxd = std::max(maxd, std::fabs(r1[i] - r0[i]));
        check(maxd < 2e-5f, "refine at neutral settings is identity");
    }
    {
        nrcore::Params pd2 = p; pd2.shadowDesat = 1.0f; pd2.desatRange = 0.3f;
        std::vector<float> withD, withoutD;
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pd2, &withD);
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, p, &withoutD);
        // chroma energy in the dark circle region (around 320,260) must drop
        double cw = 0.0, cwo = 0.0;
        for (int y = 240; y < 280; ++y)
            for (int x = 300; x < 340; ++x) {
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                float yy, cb, cr;
                nrcore::rgb2ycc(withD[i], withD[i+1], withD[i+2], yy, cb, cr);
                cw += std::fabs(cb) + std::fabs(cr);
                nrcore::rgb2ycc(withoutD[i], withoutD[i+1], withoutD[i+2], yy, cb, cr);
                cwo += std::fabs(cb) + std::fabs(cr);
            }
        printf("shadow desat: dark-region chroma %.4f -> %.4f\n", cwo, cw);
        check(cw < cwo * 0.7, "shadow desaturation reduces dark-region chroma");
    }

    // --- manual profile passthrough (applies to all three sigma pairs)
    {
        nrcore::Params pm = p;
        pm.profileSource = 2;
        pm.sigmaY = 0.037f;
        pm.sigmaC = 0.021f;
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<float> noisy = clean;
        addNoise(noisy, 0.05f, 11);
        nrcore::Stats s;
        nrcore::estimateInput(noisy.data(), nullptr, W, H, pm, s);
        check(s.sy == 0.037f && s.tc == 0.021f && s.ry == 0.037f, "manual sigma values pass through");
    }

    // --- duplicate partner guard
    {
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<float> noisy = clean;
        addNoise(noisy, 0.05f, 21);
        std::vector<float> duplicate = noisy;
        nrcore::Stats s;
        nrcore::estimateInput(noisy.data(), duplicate.data(), W, H, p, s);
        check(s.ty > 0.01f, "duplicate partner does not collapse temporal sigma");
    }

    // --- region profiling
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
        nrcore::Stats s;
        nrcore::estimateInput(noisy.data(), nullptr, W, H, pr, s);
        const float t = trueSigmaY(noisy, clean);
        check(std::fabs(s.sy - t) / t < 0.30f, "region-based sigma within 30% of truth");
    }

    // --- brightness profile sanity: gains near 1 for flat-spectrum noise
    {
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<float> noisy = clean;
        addNoise(noisy, 0.05f, 15);
        nrcore::Stats s;
        nrcore::estimateInput(noisy.data(), nullptr, W, H, p, s);
        bool sane = true;
        for (int b2 = 4; b2 < 12; ++b2)   // mid bins have plenty of samples
            if (s.gainY[b2] < 0.6f || s.gainY[b2] > 1.6f) sane = false;
        check(sane, "brightness-dependent gains near 1 on uniform noise");
    }

    // --- render every view mode
    {
        for (int vm = 1; vm <= 7; ++vm) {
            nrcore::Params pv = p;
            pv.viewMode = vm;
            if (vm == 5) pv.profileSource = 1;
            const bool motion = (vm == 6);
            std::vector<float> denoised;
            runCase(0.05f, motion ? 3.0f : 0.0f, motion ? 1.5f : 0.0f, NOISE_IID, pv, &denoised);
            char name[64];
            snprintf(name, sizeof(name), "out_view%d.ppm", vm);
            writePPM(name, denoised);
        }
        printf("view renders written: out_view1..7.ppm\n");
        check(true, "view modes rendered (visual check)");
    }

    printf("=========================================================\n");
    if (g_failures == 0)
        printf("ALL CHECKS PASSED\n");
    else
        printf("%d CHECK(S) FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
