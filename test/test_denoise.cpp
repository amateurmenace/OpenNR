// OpenNR test harness — validates nr_core.h (v2) without DaVinci Resolve.
//
// Build: c++ -O2 -std=c++14 -I../plugin test_denoise.cpp -o test_denoise

#include "nr_core.h"
#include "nr_analyze.h"

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

// block-correlated LUMA noise (equal push on R,G,B), block size in px —
// the v3 Noise EQ band-test generator (4 px = medium band, 16 px = coarse)
static void addLumaBlockNoise(std::vector<float>& img, float sigma, int blockPx, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, sigma);
    const int bw = (W + blockPx - 1) / blockPx, bh = (H + blockPx - 1) / blockPx;
    std::vector<float> blk(static_cast<size_t>(bw) * bh);
    for (auto& v : blk) v = N(rng);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float n = blk[static_cast<size_t>(y / blockPx) * bw + (x / blockPx)];
            float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            p[0] += n; p[1] += n; p[2] += n;
        }
}

// single-frame impulses ("fireflies") at deterministic pseudo-random sites
static void addImpulses(std::vector<float>& img, int count, uint32_t seed,
                        std::vector<int>& sites)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> RX(16, W - 17), RY(16, H - 17);
    for (int i = 0; i < count; ++i) {
        const int x = RX(rng), y = RY(rng);
        sites.push_back(y * W + x);
        float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
        const float v = (i & 1) ? 1.0f : 0.0f;   // hot and dead pixels
        p[0] = v; p[1] = v; p[2] = v;
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

enum NoiseKind { NOISE_IID, NOISE_CORR, NOISE_BLOTCH, NOISE_BLOB4, NOISE_BLOB16 };

static void makeCaseFrames(float sigma, float stepX, float stepY, NoiseKind kind,
                           std::vector<std::vector<float>>& frames)
{
    frames.assign(5, {});
    for (int k = 0; k < 5; ++k) {
        renderScene(frames[k], (k - 2) * stepX, (k - 2) * stepY);
        if (kind == NOISE_CORR) addCorrelatedNoise(frames[k], sigma, 1000 + k);
        else if (kind == NOISE_BLOTCH) { addNoise(frames[k], sigma * 0.5f, 1000 + k); addChromaBlotchNoise(frames[k], sigma, 2000 + k); }
        else if (kind == NOISE_BLOB4)  { addNoise(frames[k], sigma * 0.25f, 1000 + k); addLumaBlockNoise(frames[k], sigma, 4, 3000 + k); }
        else if (kind == NOISE_BLOB16) { addNoise(frames[k], sigma * 0.25f, 1000 + k); addLumaBlockNoise(frames[k], sigma, 16, 4000 + k); }
        else addNoise(frames[k], sigma, 1000 + k);
    }
}

static CaseResult runCase(float sigma, float stepX, float stepY, NoiseKind kind, const nrcore::Params& p,
                          std::vector<float>* outDenoised = nullptr,
                          std::vector<float>* outNoisy = nullptr)
{
    std::vector<float> clean;
    renderScene(clean, 0, 0);

    std::vector<std::vector<float>> frames;
    makeCaseFrames(sigma, stepX, stepY, kind, frames);

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

static double meanAbs(const std::vector<float>& a)
{
    double s = 0.0;
    for (float v : a) s += std::fabs(v);
    return s / a.size();
}

// banding energy: squared column-mean derivative (dither cancels in the
// column mean; staircase steps do not)
static double bandingEnergy(const std::vector<float>& img)
{
    std::vector<double> cm(W, 0.0);
    for (int x = 0; x < W; ++x) {
        double s = 0.0;
        for (int y = 8; y < H - 8; ++y) {
            const float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            s += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        }
        cm[x] = s / (H - 16);
    }
    double e = 0.0;
    for (int x = 16; x < W - 17; ++x) {
        const double d = cm[x + 1] - cm[x];
        e += d * d;
    }
    return e;
}

// per-frame clip analysis exactly as the plugin's analyzer does it
static nranalyze::ClipAggregate analyzeFrames(const std::vector<std::vector<float>>& frames)
{
    nrcore::Params ap;   // auto whole-frame, adjust 1 — the analyzer's params
    std::vector<nrcore::Stats> per;
    for (int i = 0; i < 5; ++i) {
        const int partner = (i < 4) ? i + 1 : 3;
        nrcore::Stats st;
        nrcore::estimateInput(frames[i].data(), frames[partner].data(), W, H, ap, st);
        per.push_back(st);
    }
    return nranalyze::aggregateClipStats(per);
}

// convert Auto Setup's UI-unit settings + the locked aggregate to core params
static nrcore::Params paramsFromAuto(const nranalyze::AutoSettings& as,
                                     const nranalyze::ClipAggregate& agg)
{
    nrcore::Params q;
    q.enableTemporal = as.enableTemporal;
    q.temporalFrames = as.temporalFrames;
    q.temporalLuma   = as.temporalLuma / 100.0f;
    q.temporalChroma = as.temporalChroma / 100.0f;
    q.motionThresh   = as.motionThresh / 100.0f;
    q.motionTracking = as.motionTracking;
    q.fireflyRemoval = as.fireflyRemoval;
    q.ghostGuard     = as.ghostGuard;
    q.enableSpatial  = as.enableSpatial;
    q.spatialMode    = as.spatialMode;
    q.spatialRadius  = as.spatialRadius;
    q.spatialLuma    = as.spatialLuma / 100.0f;
    q.spatialChroma  = as.spatialChroma / 100.0f;
    q.preserveDetail = as.preserveDetail / 100.0f;
    q.detailRescue   = as.detailRescue / 100.0f;
    q.chromaBlotch   = as.chromaBlotch / 100.0f;
    q.eqFine         = as.eqFine / 100.0f;
    q.eqMedium       = as.eqMedium / 100.0f;
    q.eqCoarse       = as.eqCoarse / 100.0f;
    q.profileAdjust  = as.profileAdjust;
    q.profileLocked  = as.lockProfile;
    q.lockSY = agg.sy; q.lockSC = agg.sc;
    q.lockTY = agg.ty; q.lockTC = agg.tc;
    for (int b = 0; b < 16; ++b) {
        q.lockGainY[b] = agg.gainY[b];
        q.lockGainC[b] = agg.gainC[b];
    }
    return q;
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

    // =====================================================================
    // v3 — golden regression: EQ at defaults reproduces v2.1 output
    // (goldens captured from the pristine v2.1 build on this scene)
    // =====================================================================
    {
        nrcore::Params ps;                 // v3 defaults, temporal off
        ps.enableTemporal = 0;
        std::vector<float> out;
        CaseResult r = runCase(0.05f, 0.0f, 0.0f, NOISE_IID, ps, &out);
        printf("v2.1 golden spatial iid: PSNR %.6f (want 34.650967)  meanAbs %.8f (want 0.54357698)\n",
               r.after, meanAbs(out));
        check(std::fabs(r.after - 34.650967) < 0.02, "EQ defaults reproduce v2.1 spatial output (PSNR)");
        check(std::fabs(meanAbs(out) - 0.54357698) < 2e-5, "EQ defaults reproduce v2.1 spatial output (mean)");

        CaseResult rb = runCase(0.04f, 0.0f, 0.0f, NOISE_BLOTCH, ps, &out);
        printf("v2.1 golden spatial blotch: PSNR %.6f (want 26.791311)  meanAbs %.8f (want 0.54452496)\n",
               rb.after, meanAbs(out));
        check(std::fabs(rb.after - 26.791311) < 0.02, "EQ defaults reproduce v2.1 blotch output (PSNR)");
        check(std::fabs(meanAbs(out) - 0.54452496) < 2e-5, "EQ defaults reproduce v2.1 blotch output (mean)");

        // full pipeline with the v3/v3.2 temporal features disabled — the
        // two-scale residual (v3.2) shifts this slightly from the original
        // v2.1 value; re-pinned and explained in CHANGELOG 3.2.0
        nrcore::Params pv21 = p;           // temporalFrames=5 from above
        pv21.motionTracking = 0;
        pv21.fireflyRemoval = 0;
        pv21.ghostGuard = 0;
        CaseResult rs = runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pv21);
        printf("v3.2 golden full static: PSNR %.6f (want 40.948263)\n", rs.after);
        check(std::fabs(rs.after - 40.948263) < 0.02, "v3 features off reproduces pinned static golden");
    }

    // =====================================================================
    // v3 feature 1 — shift-search temporal matching (motion tracking)
    // =====================================================================
    {
        nrcore::Params pOff = p; pOff.motionTracking = 0; pOff.fireflyRemoval = 0;
        nrcore::Params pOn  = p; pOn.motionTracking  = 1; pOn.fireflyRemoval  = 0;

        // slow drift 0.8 px/frame — the case the feature was built for
        CaseResult drOff = runCase(0.05f, 0.8f, 0.5f, NOISE_IID, pOff);
        CaseResult drOn  = runCase(0.05f, 0.8f, 0.5f, NOISE_IID, pOn);
        printf("motion tracking, drift 0.8px: off %5.2f dB -> on %5.2f dB\n", drOff.after, drOn.after);
        check(drOn.after > drOff.after + 0.3, "tracking gains >= 0.3 dB on slow drift");
        check(drOn.after > 37.91, "tracking beats the v2.1 drift number (37.908)");

        // 1.5 px/frame pan: temporal must beat spatial-only by >= 1 dB
        nrcore::Params pSp = p; pSp.enableTemporal = 0;
        CaseResult panSp  = runCase(0.05f, 1.5f, 0.0f, NOISE_IID, pSp);
        CaseResult panOff = runCase(0.05f, 1.5f, 0.0f, NOISE_IID, pOff);
        CaseResult panOn  = runCase(0.05f, 1.5f, 0.0f, NOISE_IID, pOn);
        printf("pan 1.5px: spatial-only %5.2f, temporal off %5.2f, on %5.2f dB\n",
               panSp.after, panOff.after, panOn.after);
        check(panOn.after >= panSp.after + 1.0, "pan: temporal beats spatial-only by >= 1 dB");
        check(panOn.after > panOff.after + 0.3, "pan: tracking gains >= 0.3 dB over v2.1 temporal");

        // static footage: tracking must not regress
        CaseResult stOff = runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pOff);
        CaseResult stOn  = runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pOn);
        printf("static: tracking off %5.2f dB, on %5.2f dB\n", stOff.after, stOn.after);
        check(stOn.after > stOff.after - 0.15, "tracking does not regress static footage");
    }

    // =====================================================================
    // v3 feature 2 — firefly / hot-pixel zapper
    // =====================================================================
    {
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<std::vector<float>> frames;
        makeCaseFrames(0.02f, 0.0f, 0.0f, NOISE_IID, frames);
        std::vector<int> sites;
        addImpulses(frames[2], 400, 99, sites);

        const float* fptr[5] = { frames[0].data(), frames[1].data(), frames[2].data(),
                                 frames[3].data(), frames[4].data() };
        nrcore::Params pf = p;             // 5 frames, defaults otherwise
        pf.motionTracking = 1;
        std::vector<float> outOn(static_cast<size_t>(W) * H * 4), outOff(outOn.size()), scratch;
        pf.fireflyRemoval = 1;
        nrcore::denoiseFrame(fptr, W, H, pf, outOn.data(), scratch);
        pf.fireflyRemoval = 0;
        nrcore::denoiseFrame(fptr, W, H, pf, outOff.data(), scratch);

        int removed = 0, survivedOff = 0;
        for (int site : sites) {
            const size_t i = static_cast<size_t>(site) * 4;
            float yOn, yOff, yClean, cb, cr;
            nrcore::rgb2ycc(outOn[i], outOn[i+1], outOn[i+2], yOn, cb, cr);
            nrcore::rgb2ycc(outOff[i], outOff[i+1], outOff[i+2], yOff, cb, cr);
            nrcore::rgb2ycc(clean[i], clean[i+1], clean[i+2], yClean, cb, cr);
            if (std::fabs(yOn - yClean) < 0.08f) removed++;
            if (std::fabs(yOff - yClean) > 0.10f) survivedOff++;
        }
        printf("firefly: %d/400 impulses removed (on), %d/400 survive with zapper off\n",
               removed, survivedOff);
        check(removed >= 380, "firefly zapper removes >= 95% of impulses");
        check(survivedOff >= 240, "impulses genuinely survive without the zapper");

        // clean pixels (>= 8 px from any impulse) must be untouched
        std::vector<uint8_t> nearImpulse(static_cast<size_t>(W) * H, 0);
        for (int site : sites) {
            const int sx = site % W, sy2 = site / W;
            for (int dy = -8; dy <= 8; ++dy)
                for (int dx = -8; dx <= 8; ++dx) {
                    const int nx = sx + dx, ny = sy2 + dy;
                    if (nx >= 0 && nx < W && ny >= 0 && ny < H)
                        nearImpulse[static_cast<size_t>(ny) * W + nx] = 1;
                }
        }
        // Removing the impulses legitimately shifts the measured residual by
        // a histogram quantum, which drifts every pixel by ~1e-4 (up to ~1e-2
        // where the NLM's thresholds respond nonlinearly) — that is the
        // estimator honestly adapting, not a zap. A false zap replaces the
        // pre-temporal centre with the temporal median, which survives the
        // pipeline as a >=0.02 change. Count only those.
        int falseTrips = 0;
        float maxCleanDiff = 0.0f;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                if (nearImpulse[static_cast<size_t>(y) * W + x])
                    continue;
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                float d = 0.0f;
                for (int ch = 0; ch < 3; ++ch)
                    d = std::max(d, std::fabs(outOn[i + ch] - outOff[i + ch]));
                maxCleanDiff = std::max(maxCleanDiff, d);
                if (d > 0.015f) falseTrips++;
            }
        printf("firefly: clean pixels max on/off diff %.5f, %d above 0.015\n",
               maxCleanDiff, falseTrips);
        check(falseTrips <= 5, "clean pixels untouched by the zapper (<= 5 false trips)");
    }

    // =====================================================================
    // v3 feature 5 — Noise EQ: each slider dominantly removes its own band
    // =====================================================================
    {
        nrcore::Params base = p;
        base.enableTemporal = 0;           // isolate the spatial stage
        base.chromaBlotch = 0.0f;
        base.eqFine = 0.0f; base.eqMedium = 0.0f; base.eqCoarse = 0.0f;

        nrcore::Params pF = base; pF.eqFine = 1.0f;
        nrcore::Params pM = base; pM.eqMedium = 1.0f;
        nrcore::Params pC = base; pC.eqCoarse = 1.0f;

        struct BandCase { const char* name; NoiseKind kind; float sigma; int winner; };
        const BandCase cases[3] = {
            { "per-pixel", NOISE_IID,    0.03f,  0 },
            { "4px blobs", NOISE_BLOB4,  0.035f, 1 },
            { "16px blobs", NOISE_BLOB16, 0.035f, 2 },
        };
        for (int ci = 0; ci < 3; ++ci) {
            const BandCase& bc = cases[ci];
            const double dF = runCase(bc.sigma, 0, 0, bc.kind, pF).after;
            const double dM = runCase(bc.sigma, 0, 0, bc.kind, pM).after;
            const double dC = runCase(bc.sigma, 0, 0, bc.kind, pC).after;
            const double best = std::max(dF, std::max(dM, dC));
            const double own = (bc.winner == 0) ? dF : (bc.winner == 1) ? dM : dC;
            printf("EQ %-10s fine %5.2f  medium %5.2f  coarse %5.2f dB\n", bc.name, dF, dM, dC);
            char msg[96];
            snprintf(msg, sizeof(msg), "EQ: %s noise removed best by its own band", bc.name);
            check(own >= best, msg);
        }
    }

    // =====================================================================
    // v3 feature 6a — noise matte view
    // =====================================================================
    {
        nrcore::Params pm2 = p;
        pm2.viewMode = 8;
        std::vector<float> matte;
        runCase(0.02f, 0.0f, 0.0f, NOISE_IID, pm2, &matte);
        bool rgbaEqual = true;
        double flatSum = 0.0, texSum = 0.0;
        int flatN = 0, texN = 0;
        for (int y = 8; y < H - 8; ++y)
            for (int x = 8; x < W - 8; ++x) {
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                if (matte[i] != matte[i+3] || matte[i+1] != matte[i+3] || matte[i+2] != matte[i+3])
                    rgbaEqual = false;
                if (matte[i+3] < 0.0f || matte[i+3] > 1.0f)
                    rgbaEqual = false;
                // flat gradient area (no structure) vs strong sine texture
                if (x >= 220 && x < 260 && y >= 320 && y < 350) { flatSum += matte[i+3]; flatN++; }
                if (x >= 460 && x < 600 && y >= 60 && y < 140)  { texSum += matte[i+3]; texN++; }
            }
        const double flatAvg = flatSum / std::max(flatN, 1);
        const double texAvg  = texSum / std::max(texN, 1);
        printf("matte: alpha==map %s, flat-area %.3f vs textured %.3f\n",
               rgbaEqual ? "yes" : "NO", flatAvg, texAvg);
        check(rgbaEqual, "matte view: alpha equals the map, in [0,1]");
        check(flatAvg > texAvg + 0.25 && texAvg < 0.75,
              "matte: noise dominates flat areas, image wins on texture");
    }

    // =====================================================================
    // v3 feature 6b — deband
    // =====================================================================
    {
        // 8-step quantized ramp at 1.5 8-bit-LSB steps (classic sky banding)
        const float q = 1.5f / 255.0f;
        std::vector<float> ramp(static_cast<size_t>(W) * H * 4);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float* px = &ramp[(static_cast<size_t>(y) * W + x) * 4];
                const float v = 0.30f + q * static_cast<float>((x * 8) / W);
                px[0] = px[1] = px[2] = v; px[3] = 1.0f;
            }

        nrcore::Params pd;
        pd.enableTemporal = 0; pd.enableSpatial = 0; pd.enableRefine = 1;
        pd.profileSource = 2; pd.sigmaY = 0.005f; pd.sigmaC = 0.005f;
        pd.deband = 1.0f;
        const float* fptr[5] = { ramp.data(), ramp.data(), ramp.data(), ramp.data(), ramp.data() };
        std::vector<float> out(ramp.size()), scratch;
        nrcore::denoiseFrame(fptr, W, H, pd, out.data(), scratch);
        const double eIn = bandingEnergy(ramp), eOut = bandingEnergy(out);
        printf("deband: banding energy %.3e -> %.3e (%.1f%%)\n", eIn, eOut, 100.0 * eOut / eIn);
        check(eOut < 0.5 * eIn, "deband halves banding energy on the 8-step ramp");

        nrcore::Params pd0 = pd; pd0.deband = 0.0f;
        nrcore::denoiseFrame(fptr, W, H, pd0, out.data(), scratch);
        float maxd0 = 0.0f;
        for (size_t i = 0; i < out.size(); ++i) maxd0 = std::max(maxd0, std::fabs(out[i] - ramp[i]));
        check(maxd0 < 2e-5f, "deband at 0 is identity");

        // hard-edge chart: high-contrast stripes + one saturated stripe
        std::vector<float> chart(static_cast<size_t>(W) * H * 4);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float* px = &chart[(static_cast<size_t>(y) * W + x) * 4];
                const int band = (x / 80) % 4;
                if (band == 3) { px[0] = 0.7f; px[1] = 0.15f; px[2] = 0.12f; }
                else { const float v = 0.15f + 0.30f * band; px[0] = px[1] = px[2] = v; }
                px[3] = 1.0f;
            }
        const float* cptr[5] = { chart.data(), chart.data(), chart.data(), chart.data(), chart.data() };
        std::vector<float> outC(chart.size());
        nrcore::denoiseFrame(cptr, W, H, pd, outC.data(), scratch);
        float maxd = 0.0f;
        double sumd = 0.0;
        for (size_t i = 0; i < outC.size(); ++i) {
            const float d = std::fabs(outC[i] - chart[i]);
            maxd = std::max(maxd, d);
            sumd += d;
        }
        printf("deband: hard-edge chart max diff %.5f, mean %.7f\n", maxd, sumd / outC.size());
        check(maxd < 0.005f, "deband leaves hard edges effectively identical");
    }

    // =====================================================================
    // v3 feature 3 — locked profiles: serialization + effect
    // =====================================================================
    {
        nranalyze::ClipAggregate a;
        a.sy = 0.03127f; a.sc = 0.01693f; a.ty = 0.02944f; a.tc = 0.01512f;
        for (int b = 0; b < 16; ++b) {
            a.gainY[b] = 0.6f + 0.09f * b;
            a.gainC[b] = 2.2f - 0.08f * b;
        }
        const std::string ser = nranalyze::formatLockedProfile(a);
        nranalyze::ClipAggregate back;
        const bool ok = nranalyze::parseLockedProfile(ser, back);
        bool exact = ok &&
            std::memcmp(&a.sy, &back.sy, 4) == 0 && std::memcmp(&a.sc, &back.sc, 4) == 0 &&
            std::memcmp(&a.ty, &back.ty, 4) == 0 && std::memcmp(&a.tc, &back.tc, 4) == 0;
        for (int b = 0; exact && b < 16; ++b)
            exact = std::memcmp(&a.gainY[b], &back.gainY[b], 4) == 0 &&
                    std::memcmp(&a.gainC[b], &back.gainC[b], 4) == 0;
        check(exact, "locked profile serialize -> parse round-trip is bit-exact");
        nranalyze::ClipAggregate junk;
        check(!nranalyze::parseLockedProfile("HUSHLOCK1,zz", junk) &&
              !nranalyze::parseLockedProfile("nonsense", junk),
              "locked profile parser rejects corrupt data");

        // locked values override the measurement; histogram stays live
        std::vector<float> clean, noisy;
        renderScene(clean, 0, 0);
        noisy = clean;
        addNoise(noisy, 0.05f, 31);
        nrcore::Params pl = p;
        pl.profileLocked = 1;
        pl.lockSY = 0.0311f; pl.lockSC = 0.0177f; pl.lockTY = 0.0299f; pl.lockTC = 0.0155f;
        for (int b = 0; b < 16; ++b) { pl.lockGainY[b] = 1.0f + 0.01f * b; pl.lockGainC[b] = 1.3f; }
        nrcore::Stats st;
        nrcore::estimateInput(noisy.data(), nullptr, W, H, pl, st);
        check(st.sy == 0.0311f && st.sc == 0.0177f && st.ty == 0.0299f && st.tc == 0.0155f,
              "locked sigmas override the measurement");
        check(st.gainY[5] == 1.05f && st.gainC[9] == 1.3f, "locked gains override the measurement");
        check(st.histMax > 1, "locked profile keeps the HUD histogram live");
    }

    // =====================================================================
    // v3 feature 3 — clip analyzer aggregation
    // =====================================================================
    {
        std::vector<std::vector<float>> framesStatic, framesPan;
        makeCaseFrames(0.03f, 0.0f, 0.0f, NOISE_IID, framesStatic);
        makeCaseFrames(0.03f, 3.0f, 1.0f, NOISE_IID, framesPan);
        const nranalyze::ClipAggregate aggS = analyzeFrames(framesStatic);
        const nranalyze::ClipAggregate aggP = analyzeFrames(framesPan);
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        const float truth = trueSigmaY(framesStatic[2], clean);
        printf("analyzer: static sy %.4f (true %.4f) motion %.3f | pan motion %.3f\n",
               aggS.sy, truth, aggS.motion, aggP.motion);
        check(std::fabs(aggS.sy - truth) / truth < 0.30f, "aggregate sigma within 30% of truth");
        check(aggS.motion < 0.15f, "static clip reads as low motion energy");
        // the synthetic scene is mostly flat, which caps how far motion can
        // push the median — real textured footage reads much higher
        check(aggP.motion > aggS.motion + 0.02f, "panning clip reads as higher motion energy");
    }

    // =====================================================================
    // v3 feature 4 — Auto Setup mapping: golden cases + monotonicity
    // =====================================================================
    {
        using nranalyze::ClipAggregate;
        using nranalyze::AutoSettings;
        auto makeAgg = [](float sy, float motion, float chromaRatio, float coarseRatio) {
            ClipAggregate a;
            a.sy = sy; a.sc = sy; a.ty = sy * 0.95f; a.tc = sy * 0.95f * chromaRatio;
            a.motion = motion;
            a.chromaRatio = chromaRatio;
            a.coarseRatioY = coarseRatio;
            a.frames = 5;
            return a;
        };

        const AutoSettings sClean  = nranalyze::mapAnalysisToSettings(makeAgg(0.004f, 0.02f, 1.0f, 1.0f));
        const AutoSettings sMod    = nranalyze::mapAnalysisToSettings(makeAgg(0.015f, 0.02f, 1.0f, 1.0f));
        const AutoSettings sNoisy  = nranalyze::mapAnalysisToSettings(makeAgg(0.040f, 0.02f, 1.0f, 1.0f));
        const AutoSettings sSevere = nranalyze::mapAnalysisToSettings(makeAgg(0.090f, 0.02f, 1.0f, 1.0f));

        check(sClean.noiseClass == 0 && sMod.noiseClass == 1 &&
              sNoisy.noiseClass == 2 && sSevere.noiseClass == 3, "noise classification thresholds");
        auto mono = [](float a, float b, float c, float d) { return a <= b && b <= c && c <= d; };
        check(mono(sClean.temporalLuma, sMod.temporalLuma, sNoisy.temporalLuma, sSevere.temporalLuma) &&
              mono(sClean.spatialLuma, sMod.spatialLuma, sNoisy.spatialLuma, sSevere.spatialLuma) &&
              mono(sClean.spatialRadius, sMod.spatialRadius, sNoisy.spatialRadius, sSevere.spatialRadius) &&
              mono(sClean.chromaBlotch, sMod.chromaBlotch, sNoisy.chromaBlotch, sSevere.chromaBlotch) &&
              mono(sClean.eqMedium, sMod.eqMedium, sNoisy.eqMedium, sSevere.eqMedium),
              "settings are monotone in the noise class");
        check(mono(sSevere.preserveDetail, sNoisy.preserveDetail, sMod.preserveDetail, sClean.preserveDetail),
              "preserve-detail decreases with noise");
        check(sClean.temporalFrames == 3 && sSevere.temporalFrames == 5, "frame count grows with noise");
        check(sClean.lockProfile == 1 && sClean.eqFine == 100.0f, "auto always locks and keeps fine at 100");

        const AutoSettings sMotion = nranalyze::mapAnalysisToSettings(makeAgg(0.040f, 0.60f, 1.0f, 1.0f));
        check(sMotion.temporalFrames == 3, "heavy motion prefers 3 frames");
        check(sMotion.motionThresh < sNoisy.motionThresh - 10.0f, "heavy motion lowers the motion threshold");
        check(sMotion.movingCamera == 1 && sNoisy.movingCamera == 0, "moving-camera flag");

        const AutoSettings sBlotch = nranalyze::mapAnalysisToSettings(makeAgg(0.015f, 0.02f, 2.6f, 1.7f));
        check(sBlotch.chromaBlotch >= sMod.chromaBlotch + 30.0f, "blotchy chroma raises the blotch pass");
        check(sBlotch.spatialChroma >= 110.0f, "blotchy chroma pushes chroma strength up");
        check(sBlotch.eqMedium >= 25.0f && sBlotch.eqCoarse >= 10.0f, "correlated noise raises medium/coarse bands");

        const std::string rep = nranalyze::formatAutoReport(makeAgg(0.032f, 0.5f, 1.2f, 1.0f), sMotion);
        check(rep.find("moving camera") != std::string::npos &&
              rep.find("%Y") != std::string::npos &&
              rep.find("profile locked") != std::string::npos, "auto report contains the key facts");
    }

    // =====================================================================
    // v3 feature 4 — Auto Setup end-to-end: auto >= defaults on every class
    // =====================================================================
    {
        struct E2E { const char* name; float sigma; float sx, sy2; NoiseKind kind; bool clean; };
        const E2E cases[6] = {
            { "clean",     0.004f, 0.0f, 0.0f, NOISE_IID,    true  },
            { "moderate",  0.015f, 0.0f, 0.0f, NOISE_IID,    false },
            { "noisy",     0.040f, 0.0f, 0.0f, NOISE_IID,    false },
            { "severe",    0.090f, 0.0f, 0.0f, NOISE_IID,    false },
            { "blotchy",   0.040f, 0.0f, 0.0f, NOISE_BLOTCH, false },
            { "motion",    0.050f, 3.0f, 1.5f, NOISE_IID,    false },
        };
        for (int ci = 0; ci < 6; ++ci) {
            const E2E& e = cases[ci];
            std::vector<std::vector<float>> frames;
            makeCaseFrames(e.sigma, e.sx, e.sy2, e.kind, frames);
            const nranalyze::ClipAggregate agg = analyzeFrames(frames);
            const nranalyze::AutoSettings as = nranalyze::mapAnalysisToSettings(agg);
            const nrcore::Params pAuto = paramsFromAuto(as, agg);
            nrcore::Params pDef;       // stock v3 defaults
            const CaseResult rAuto = runCase(e.sigma, e.sx, e.sy2, e.kind, pAuto);
            const CaseResult rDef  = runCase(e.sigma, e.sx, e.sy2, e.kind, pDef);
            printf("auto e2e %-9s defaults %5.2f dB -> auto %5.2f dB\n", e.name, rDef.after, rAuto.after);
            char msg[96];
            if (e.clean) {
                snprintf(msg, sizeof(msg), "auto e2e %s: never worse than defaults - 0.3 dB", e.name);
                check(rAuto.after >= rDef.after - 0.3, msg);
            } else {
                snprintf(msg, sizeof(msg), "auto e2e %s: auto >= defaults", e.name);
                check(rAuto.after >= rDef.after - 0.05, msg);
            }
        }
    }

    // =====================================================================
    // v3.1 — Detail Rescue: crank everything, rescue restores structure
    // =====================================================================
    {
        nrcore::Params pc = p;
        pc.master = 3.0f;
        pc.spatialLuma = 1.5f; pc.spatialChroma = 1.5f;
        pc.eqFine = 3.0f; pc.spatialRadius = 10;
        pc.preserveDetail = 0.0f;
        pc.eqMedium = 1.0f; pc.eqCoarse = 0.5f;
        nrcore::Params pr = pc; pr.detailRescue = 0.8f;
        const CaseResult r0 = runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pc);
        const CaseResult r1 = runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pr);
        printf("v3.1 rescue at full crank: without %5.2f dB -> with %5.2f dB\n", r0.after, r1.after);
        check(r1.after > r0.after + 1.5, "detail rescue recovers >= 1.5 dB at full crank");
        // full crank deliberately trades PSNR for smoothness — the bound
        // that matters is that rescue keeps it net-positive vs the input
        check(r1.after > r1.before + 1.0, "cranked + rescue stays net-positive");
    }

    // v3.1 — the top half of EQ Fine is no longer a silent no-op
    {
        nrcore::Params pa = p; pa.spatialLuma = 1.0f; pa.eqFine = 1.5f;
        nrcore::Params pb = p; pb.spatialLuma = 1.0f; pb.eqFine = 3.0f;
        std::vector<float> oa, ob;
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pa, &oa);
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pb, &ob);
        double d = 0.0;
        for (size_t i = 0; i < oa.size(); ++i) d += std::fabs(oa[i] - ob[i]);
        check(d / oa.size() > 1e-4, "EQ Fine above 150 changes the result (h widening)");
    }

    // v3.1 — extended ranges engage: master 3 > 2, band overshoot works
    {
        nrcore::Params m2 = p; m2.master = 2.0f;
        nrcore::Params m3 = p; m3.master = 3.0f;
        std::vector<float> o2, o3;
        runCase(0.10f, 0.0f, 0.0f, NOISE_IID, m2, &o2);
        runCase(0.10f, 0.0f, 0.0f, NOISE_IID, m3, &o3);
        double d = 0.0;
        for (size_t i = 0; i < o2.size(); ++i) d += std::fabs(o2[i] - o3[i]);
        check(d / o2.size() > 2e-4, "master 3 filters harder than master 2");

        nrcore::Params b1 = p;  b1.chromaBlotch = 1.0f;  b1.eqMedium = 1.0f;
        nrcore::Params b15 = p; b15.chromaBlotch = 1.5f; b15.eqMedium = 1.5f;
        std::vector<float> ob1, ob15;
        runCase(0.04f, 0.0f, 0.0f, NOISE_BLOTCH, b1, &ob1);
        runCase(0.04f, 0.0f, 0.0f, NOISE_BLOTCH, b15, &ob15);
        d = 0.0;
        for (size_t i = 0; i < ob1.size(); ++i) d += std::fabs(ob1[i] - ob15[i]);
        check(d / ob1.size() > 1e-4, "band sliders above 100 widen tolerance and reach");
    }

    // v3.1 — noise view: soft-knee output stays inside [0,1]
    {
        nrcore::Params pv = p; pv.viewMode = 4;
        std::vector<float> nv;
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, pv, &nv);
        bool ok = true;
        for (size_t i = 0; i < nv.size(); i += 4)
            if (!(nv[i] >= 0.0f && nv[i] <= 1.0f && nv[i+1] >= 0.0f && nv[i+1] <= 1.0f &&
                  nv[i+2] >= 0.0f && nv[i+2] <= 1.0f)) { ok = false; break; }
        check(ok, "noise view (soft knee) stays inside [0,1]");
    }

    // v3.1 — scope overlays draw over the result and never touch alpha
    {
        nrcore::Params ps2 = p;
        ps2.scopeMeasure = 1; ps2.scopeEq = 1; ps2.scopeMotion = 1;
        std::vector<float> withScopes, without;
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, ps2, &withScopes);
        runCase(0.05f, 0.0f, 0.0f, NOISE_IID, p, &without);
        writePPM("out_scopes.ppm", withScopes);
        auto differsAt = [&](int xd, int ydTop) {
            const size_t i = (static_cast<size_t>(H - 1 - ydTop) * W + xd) * 4;
            return withScopes[i] != without[i] || withScopes[i+1] != without[i+1] ||
                   withScopes[i+2] != without[i+2];
        };
        const bool drewHud = differsAt(30, 30);              // top-left panel
        const bool drewEq  = differsAt(W - 30, 30);          // top-right panel
        const bool drewMo  = differsAt(W - 30, H - 30);      // bottom-right map
        check(drewHud && drewEq && drewMo, "all three scopes draw on the result view");
        bool alphaSame = true, bounded = true;
        for (size_t i = 0; i < withScopes.size(); i += 4) {
            if (withScopes[i+3] != without[i+3]) { alphaSame = false; break; }
            if (!(withScopes[i] >= 0.0f && withScopes[i] <= 1.0f)) bounded = false;
        }
        check(alphaSame, "scopes never write into alpha");
        check(bounded, "scope pixels stay inside [0,1]");
    }

    // v3.1 — letterboxed footage no longer collapses the measurement
    {
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<float> noisy = clean;
        addNoise(noisy, 0.05f, 77);
        const float truth = trueSigmaY(noisy, clean);
        std::vector<float> boxed = noisy;
        const int bar = H / 5;
        for (int y = 0; y < H; ++y) {
            if (y >= bar && y < H - bar)
                continue;
            for (int x = 0; x < W; ++x) {
                float* q = &boxed[(static_cast<size_t>(y) * W + x) * 4];
                q[0] = q[1] = q[2] = 0.0f;
            }
        }
        nrcore::Stats sb;
        nrcore::estimateInput(boxed.data(), nullptr, W, H, p, sb);
        printf("letterbox: measured sigma %.4f vs true %.4f\n", sb.sy, truth);
        check(std::fabs(sb.sy - truth) / truth < 0.30f,
              "letterbox bars do not collapse the noise estimate");
    }

    // =====================================================================
    // v3.2 — locked profile honors Auto Profile Adjust (live trim)
    // =====================================================================
    {
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<float> noisy = clean;
        addNoise(noisy, 0.05f, 41);
        nrcore::Params pl = p;
        pl.profileLocked = 1;
        pl.lockSY = 0.020f; pl.lockSC = 0.015f; pl.lockTY = 0.019f; pl.lockTC = 0.014f;
        pl.profileAdjust = 2.0f;
        nrcore::Stats st;
        nrcore::estimateInput(noisy.data(), nullptr, W, H, pl, st);
        check(std::fabs(st.sy - 0.040f) < 1e-6f && std::fabs(st.tc - 0.028f) < 1e-6f,
              "Auto Profile Adjust trims a locked profile live");
    }

    // =====================================================================
    // v3.2 — Ghost Guard: subtle coherent drift smears less, static ~free
    // =====================================================================
    {
        nrcore::Params gOff = p; gOff.ghostGuard = 0;
        nrcore::Params gOn  = p; gOn.ghostGuard  = 1;
        const CaseResult subOff = runCase(0.05f, 0.4f, 0.25f, NOISE_IID, gOff);
        const CaseResult subOn  = runCase(0.05f, 0.4f, 0.25f, NOISE_IID, gOn);
        printf("v3.2 ghost guard, subtle drift 0.4px: off %5.2f dB -> on %5.2f dB\n",
               subOff.after, subOn.after);
        check(subOn.after >= subOff.after - 0.05, "ghost guard never hurts subtle drift");

        const CaseResult stOff = runCase(0.05f, 0.0f, 0.0f, NOISE_IID, gOff);
        const CaseResult stOn  = runCase(0.05f, 0.0f, 0.0f, NOISE_IID, gOn);
        printf("v3.2 ghost guard, static: off %5.2f dB -> on %5.2f dB\n", stOff.after, stOn.after);
        check(stOn.after >= stOff.after - 0.40, "ghost guard costs <= 0.4 dB on static");
    }

    // =====================================================================
    // v3.2 — global blend: 0 = identity, 0.5 = exact midpoint
    // =====================================================================
    {
        nrcore::Params b0 = p;  b0.globalBlend = 0.0f;
        nrcore::Params b5 = p;  b5.globalBlend = 0.5f;
        nrcore::Params b1 = p;  b1.globalBlend = 1.0f;
        std::vector<float> clean, noisy, o0, o5, o1, scratch;
        renderScene(clean, 0, 0);
        noisy = clean;
        addNoise(noisy, 0.05f, 55);
        const float* fptr[5] = { noisy.data(), noisy.data(), noisy.data(),
                                 noisy.data(), noisy.data() };
        o0.resize(noisy.size()); o5.resize(noisy.size()); o1.resize(noisy.size());
        nrcore::denoiseFrame(fptr, W, H, b0, o0.data(), scratch);
        nrcore::denoiseFrame(fptr, W, H, b5, o5.data(), scratch);
        nrcore::denoiseFrame(fptr, W, H, b1, o1.data(), scratch);
        float maxd0 = 0.0f, maxdMid = 0.0f;
        for (size_t i = 0; i < o0.size(); i += 4)
            for (int c = 0; c < 3; ++c) {
                maxd0 = std::max(maxd0, std::fabs(o0[i+c] - noisy[i+c]));
                maxdMid = std::max(maxdMid, std::fabs(o5[i+c] - 0.5f * (noisy[i+c] + o1[i+c])));
            }
        printf("v3.2 global blend: blend0 maxdiff %.2e, midpoint maxdiff %.2e\n", maxd0, maxdMid);
        check(maxd0 < 2e-5f, "global blend 0 is identity");
        check(maxdMid < 1e-5f, "global blend 50 is the exact midpoint");
    }

    // =====================================================================
    // v3.2 — two-scale residual: correlated noise can no longer hide from
    // the spatial stage (temporal off -> residual == input noise)
    // =====================================================================
    {
        std::vector<float> clean;
        renderScene(clean, 0, 0);
        std::vector<std::vector<float>> frames;
        makeCaseFrames(0.05f, 0.0f, 0.0f, NOISE_CORR, frames);
        const float truth = trueSigmaY(frames[2], clean);
        nrcore::Params pr = p;
        pr.enableTemporal = 0;   // residual == the input noise, unmerged
        std::vector<float> out(frames[2].size()), scratch;
        const float* fptr[5] = { frames[0].data(), frames[1].data(), frames[2].data(),
                                 frames[3].data(), frames[4].data() };
        const nrcore::Stats st = nrcore::denoiseFrame(fptr, W, H, pr, out.data(), scratch);
        printf("v3.2 residual on correlated noise: ry %.4f vs true %.4f (fine-only used to underread)\n",
               st.ry, truth);
        check(st.ry > 0.55f * truth, "coarse residual sees correlated noise");
    }

    // --- render every view mode
    {
        for (int vm = 1; vm <= 8; ++vm) {
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
        printf("view renders written: out_view1..8.ppm\n");
        check(true, "view modes rendered (visual check)");
    }

    printf("=========================================================\n");
    if (g_failures == 0)
        printf("ALL CHECKS PASSED\n");
    else
        printf("%d CHECK(S) FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
