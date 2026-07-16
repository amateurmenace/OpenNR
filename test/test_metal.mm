// OpenNR GPU parity test (v2) — runs the real Metal pipeline (RunMetalNR, the
// same entry point DaVinci Resolve calls) against the CPU reference and
// verifies the outputs match within fast-math tolerance.
//
// Build:
//   c++ -O2 -std=c++14 -I../plugin test_metal.mm ../plugin/MetalKernel.mm \
//       -framework Metal -framework Foundation -o test_metal

#import <Metal/Metal.h>

#include <cstdio>
#include <random>
#include <vector>

#include "nr_core.h"
#include "NRParams.h"

extern void RunMetalNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                       const float* const p_Srcs[7], float* p_Dst);

static const int W = 512;
static const int H = 288;

static void makeFrame(std::vector<float>& img, int k, uint32_t seed, float panPx = 2.0f)
{
    img.resize(static_cast<size_t>(W) * H * 4);
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, 0.05f);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            const float fx = x + k * panPx;
            float r = 0.2f + 0.4f * (fx / W);
            float g = 0.3f + 0.2f * (static_cast<float>(y) / H);
            float b = 0.25f + 0.3f * (fx / W);
            const float dx = fx - 250, dy = y - 140;
            if (dx * dx + dy * dy < 70 * 70) { r = 0.8f; g = 0.75f; b = 0.6f; }
            p[0] = r + N(rng);
            p[1] = g + N(rng);
            p[2] = b + N(rng);
            p[3] = 1.0f;
        }
    }
}

static void toCpuParams(const NRParams& gp, nrcore::Params& cp)
{
    cp.profileSource  = gp.profileSource;
    cp.sigmaY         = gp.sigmaY;
    cp.sigmaC         = gp.sigmaC;
    cp.profileAdjust  = gp.profileAdjust;
    cp.regionCX       = gp.regionCX;
    cp.regionCY       = gp.regionCY;
    cp.regionSize     = gp.regionSize;
    cp.enableTemporal = gp.enableTemporal;
    cp.temporalFrames = gp.temporalFrames;
    cp.temporalLuma   = gp.temporalLuma;
    cp.temporalChroma = gp.temporalChroma;
    cp.motionThresh   = gp.motionThresh;
    cp.enableSpatial  = gp.enableSpatial;
    cp.spatialMode    = gp.spatialMode;
    cp.spatialRadius  = gp.spatialRadius;
    cp.spatialLuma    = gp.spatialLuma;
    cp.spatialChroma  = gp.spatialChroma;
    cp.preserveDetail = gp.preserveDetail;
    cp.chromaBlotch   = gp.chromaBlotch;
    cp.enableRefine   = gp.enableRefine;
    cp.shadowDesat    = gp.shadowDesat;
    cp.desatRange     = gp.desatRange;
    cp.lumaTexture    = gp.lumaTexture;
    cp.grainAmount    = gp.grainAmount;
    cp.grainSize      = gp.grainSize;
    cp.grainChroma    = gp.grainChroma;
    cp.grainBlue      = gp.grainBlue;
    cp.acutance       = gp.acutance;
    cp.chromaSpeckle  = gp.chromaSpeckle;
    cp.frameIndex     = gp.frameIndex;
    cp.master         = gp.master;
    cp.viewMode       = gp.viewMode;
    cp.motionTracking = gp.motionTracking;
    cp.fireflyRemoval = gp.fireflyRemoval;
    cp.eqFine         = gp.eqFine;
    cp.eqMedium       = gp.eqMedium;
    cp.eqCoarse       = gp.eqCoarse;
    cp.deband         = gp.deband;
    cp.profileLocked  = gp.profileLocked;
    cp.lockSY         = gp.lockSY;
    cp.lockSC         = gp.lockSC;
    cp.lockTY         = gp.lockTY;
    cp.lockTC         = gp.lockTC;
    for (int b = 0; b < 16; ++b) {
        cp.lockGainY[b] = gp.lockGainY[b];
        cp.lockGainC[b] = gp.lockGainC[b];
    }
    cp.detailRescue   = gp.detailRescue;
    cp.scopeMeasure   = gp.scopeMeasure;
    cp.scopeMotion    = gp.scopeMotion;
    cp.scopeEq        = gp.scopeEq;
    cp.ghostGuard     = gp.ghostGuard;
    cp.globalBlend    = gp.globalBlend;
    cp.deepClean      = gp.deepClean;
    cp.lockSCr        = gp.lockSCr;
    cp.lockTCr        = gp.lockTCr;
    cp.renderBoost    = gp.renderBoost;
    cp.histValid      = gp.histValid;
}

// sparseOK: cases that exercise the v3 shift-search selection. Picking the
// minimum of nine near-equal patch scores has measure-zero decision
// boundaries, so a 1-ulp fast-math difference can legitimately choose a
// different (equally good) candidate at a handful of pixels. Those pixels
// blend a different noise sample — individually visible to the diff, but
// bounded in count. Assert tight mean agreement plus a bounded outlier
// fraction instead of a strict max.
// hudOK: cases that render scope panels. Bar heights and value digits are
// integer-quantized display math (int(sqrt(...)+0.5) etc); a 1-ulp fast-math
// difference legitimately flips a bar-edge row or a rounded digit — huge
// per-pixel diff, but bounded to a handful of panel pixels. Assert a tiny
// mean and a tiny absolute count instead of a strict max.
static int compareRun(id<MTLCommandQueue> queue, const NRParams& gp, const char* label,
                      bool sparseOK = false, bool injectImpulses = false, bool hudOK = false,
                      float panPx = 2.0f, float expStep = 0.0f)
{
    // 7 frames, centre 3; seeds keyed to the offset so the middle five are
    // bit-identical to the 5-frame era's
    std::vector<std::vector<float>> frames(7);
    for (int k = 0; k < 7; ++k)
        makeFrame(frames[k], k - 3, 100 + (k - 3) + 2, panPx);
    if (expStep != 0.0f)   // v3.5 P1: abrupt exposure step after the centre
        for (int k = 4; k < 7; ++k)
            for (size_t i = 0; i < frames[k].size(); i += 4) {
                frames[k][i] += expStep;
                frames[k][i + 1] += expStep;
                frames[k][i + 2] += expStep;
            }
    if (injectImpulses) {
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> RX(8, W - 9), RY(8, H - 9);
        for (int i = 0; i < 200; ++i) {
            float* p = &frames[3][(static_cast<size_t>(RY(rng)) * W + RX(rng)) * 4];
            const float v = (i & 1) ? 1.0f : 0.0f;
            p[0] = v; p[1] = v; p[2] = v;
        }
    }

    nrcore::Params cp;
    toCpuParams(gp, cp);

    const float* fptr[7] = { frames[0].data(), frames[1].data(), frames[2].data(),
                             frames[3].data(), frames[4].data(), frames[5].data(),
                             frames[6].data() };
    std::vector<float> cpuOut(static_cast<size_t>(W) * H * 4), scratch;
    nrcore::denoiseFrame(fptr, W, H, cp, cpuOut.data(), scratch);

    id<MTLDevice> device = queue.device;
    const size_t bytes = static_cast<size_t>(W) * H * 4 * sizeof(float);

    id<MTLBuffer> srcBuf[7];
    for (int i = 0; i < 7; ++i)
        srcBuf[i] = [device newBufferWithBytes:frames[i].data() length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    const float* srcs[7];
    for (int i = 0; i < 7; ++i)
        srcs[i] = (const float*)srcBuf[i];

    RunMetalNR((void*)queue, W, H, gp, srcs, (float*)dstBuf);

    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];

    const float* gpuOut = static_cast<const float*>(dstBuf.contents);

    double maxd = 0.0, sumd = 0.0;
    size_t nOver = 0;
    size_t n = static_cast<size_t>(W) * H * 4;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(gpuOut[i] - cpuOut[i]);
        maxd = std::max(maxd, d);
        sumd += d;
        if (d > 5e-3) ++nOver;
    }
    const double meand = sumd / n;
    const double overFrac = static_cast<double>(nOver) / n;
    bool pass = (maxd < 5e-3) && (meand < 1e-4);
    if (!pass && sparseOK)
        pass = (meand < 1.5e-4) && (overFrac < 5e-3) && (maxd < 0.25);
    if (!pass && hudOK)
        pass = (meand < 5e-5) && (nOver <= 400);
    printf("  [%s] %-36s max %.2e  mean %.2e  over %zu\n",
           pass ? "PASS" : "FAIL", label, maxd, meand, nOver);
    return pass ? 0 : 1;
}

// v3.5 R1 Render Boost parity: two sequential calls through the REAL entry
// point — the first primes the host history cache (frame 100), the second
// consumes it (frame 101; its CPU mirror gets the CPU tmp plane from call 1
// with histValid=1). A third call at a NON-consecutive frame (200) must fall
// back to boost-off exactly — that one verifies the host validity logic
// rather than the kernel math. Static scene (panPx 0) so the boost gates
// actually open; stacks A and B use different seeds.
static int compareBoost(id<MTLCommandQueue> queue, NRParams gp)
{
    std::vector<std::vector<float>> fA(7), fB(7);
    for (int k = 0; k < 7; ++k) {
        makeFrame(fA[k], k - 3, 300 + (k - 3) + 2, 0.0f);
        makeFrame(fB[k], k - 3, 400 + (k - 3) + 2, 0.0f);
    }
    const float* fpA[7] = { fA[0].data(), fA[1].data(), fA[2].data(), fA[3].data(),
                            fA[4].data(), fA[5].data(), fA[6].data() };
    const float* fpB[7] = { fB[0].data(), fB[1].data(), fB[2].data(), fB[3].data(),
                            fB[4].data(), fB[5].data(), fB[6].data() };

    gp.renderBoost = 1;
    gp.histValid = 0;   // the host decides this; the incoming value is ignored

    nrcore::Params cp;
    toCpuParams(gp, cp);

    const size_t n = static_cast<size_t>(W) * H * 4;
    std::vector<float> cpuOut1(n), cpuOut2(n), cpuOut3(n);
    std::vector<float> scratchA, scratchB, scratchC;
    cp.frameIndex = 100;                       // cold cache: boost-off math
    nrcore::denoiseFrame(fpA, W, H, cp, cpuOut1.data(), scratchA);
    cp.frameIndex = 101; cp.histValid = 1;     // scratchA's first plane IS
    nrcore::denoiseFrame(fpB, W, H, cp, cpuOut2.data(), scratchB,
                         scratchA.data());     // call 1's temporal result
    cp.frameIndex = 200; cp.histValid = 0;     // validity miss: boost-off
    nrcore::denoiseFrame(fpB, W, H, cp, cpuOut3.data(), scratchC);

    id<MTLDevice> device = queue.device;
    const size_t bytes = n * sizeof(float);
    id<MTLBuffer> srcA[7], srcB[7];
    const float* srcsA[7];
    const float* srcsB[7];
    for (int i = 0; i < 7; ++i) {
        srcA[i] = [device newBufferWithBytes:fA[i].data() length:bytes options:MTLResourceStorageModeShared];
        srcB[i] = [device newBufferWithBytes:fB[i].data() length:bytes options:MTLResourceStorageModeShared];
        srcsA[i] = (const float*)srcA[i];
        srcsB[i] = (const float*)srcB[i];
    }
    id<MTLBuffer> dstA = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstB = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstC = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    gp.frameIndex = 100;
    RunMetalNR((void*)queue, W, H, gp, srcsA, (float*)dstA);
    gp.frameIndex = 101;
    RunMetalNR((void*)queue, W, H, gp, srcsB, (float*)dstB);
    gp.frameIndex = 200;
    RunMetalNR((void*)queue, W, H, gp, srcsB, (float*)dstC);

    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];

    int failures = 0;
    const struct { id<MTLBuffer> gpu; const float* cpu; const char* label; } cases[3] = {
        { dstA, cpuOut1.data(), "v3.5 boost call 1 (cold = boost-off)" },
        { dstB, cpuOut2.data(), "v3.5 boost call 2 (history engaged)" },
        { dstC, cpuOut3.data(), "v3.5 boost call 3 (miss = boost-off)" },
    };
    for (int c = 0; c < 3; ++c) {
        const float* g = static_cast<const float*>(cases[c].gpu.contents);
        double maxd = 0.0, sumd = 0.0;
        size_t nOver = 0;
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(g[i] - cases[c].cpu[i]);
            maxd = std::max(maxd, d);
            sumd += d;
            if (d > 5e-3) ++nOver;
        }
        const double meand = sumd / n;
        const bool pass = (maxd < 5e-3) && (meand < 1e-4);
        printf("  [%s] %-36s max %.2e  mean %.2e  over %zu\n",
               pass ? "PASS" : "FAIL", cases[c].label, maxd, meand, nOver);
        if (!pass) ++failures;
    }
    return failures;
}

int main()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("no Metal device — skipping GPU parity test\n");
        return 0;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    printf("GPU parity test v2 on %s\n", device.name.UTF8String);

    int failures = 0;

    NRParams p = {};
    p.profileSource = 0;
    p.sigmaY = 0.02f; p.sigmaC = 0.02f;
    p.profileAdjust = 1.0f;
    p.regionCX = 0.5f; p.regionCY = 0.5f; p.regionSize = 0.25f;
    p.hasTemporalDiff = 0;
    p.enableTemporal = 1; p.temporalFrames = 5;
    p.temporalLuma = 0.6f; p.temporalChroma = 0.8f; p.motionThresh = 0.4f;
    p.enableSpatial = 1; p.spatialMode = 1; p.spatialRadius = 3;
    p.spatialLuma = 0.6f; p.spatialChroma = 1.0f;
    p.preserveDetail = 0.35f; p.chromaBlotch = 0.25f;
    p.enableRefine = 1; p.shadowDesat = 0.0f; p.desatRange = 0.15f;
    p.lumaTexture = 0.0f; p.grainAmount = 0.0f; p.grainSize = 1.6f; p.grainChroma = 0.25f;
    p.frameIndex = 0;
    p.master = 1.0f; p.viewMode = 0;
    // v3 fields: base cases run tracking OFF (deterministic selection-free
    // math, strict tolerance); dedicated cases below cover tracking.
    p.motionTracking = 0; p.fireflyRemoval = 1;
    p.eqFine = 1.0f; p.eqMedium = 0.0f; p.eqCoarse = 0.0f; p.deband = 0.0f;
    p.profileLocked = 0;
    p.ghostGuard = 1; p.globalBlend = 1.0f;   // v3.2 defaults
    p.lockSY = 0.02f; p.lockSC = 0.02f; p.lockTY = 0.02f; p.lockTC = 0.02f;
    p.lockSCr = 0.02f; p.lockTCr = 0.02f;     // v3.3 B5
    for (int b = 0; b < 16; ++b) { p.lockGainY[b] = 1.0f; p.lockGainC[b] = 1.0f; }

    failures += compareRun(queue, p, "defaults (auto, 5f, NLM, blotch)");

    NRParams p2 = p; p2.enableTemporal = 0;
    failures += compareRun(queue, p2, "temporal disabled");

    NRParams p3 = p; p3.profileSource = 2; p3.sigmaY = 0.035f; p3.sigmaC = 0.028f;
    failures += compareRun(queue, p3, "manual profile");

    NRParams p4 = p; p4.spatialMode = 0; p4.spatialRadius = 2;
    failures += compareRun(queue, p4, "bilateral mode");

    NRParams p5 = p; p5.enableSpatial = 0;
    failures += compareRun(queue, p5, "spatial disabled");

    NRParams p6 = p; p6.profileSource = 1; p6.regionCX = 0.3f; p6.regionCY = 0.4f; p6.regionSize = 0.3f;
    failures += compareRun(queue, p6, "region profiling");

    NRParams p7 = p; p7.master = 0.0f;
    failures += compareRun(queue, p7, "master 0 (identity)");

    NRParams p8 = p; p8.master = 2.0f;
    failures += compareRun(queue, p8, "master 2 (boosted)");

    NRParams p9 = p; p9.chromaBlotch = 1.0f;
    failures += compareRun(queue, p9, "blotch pass full");

    NRParams p10 = p; p10.shadowDesat = 0.6f; p10.lumaTexture = 0.3f;
    p10.grainAmount = 0.5f; p10.grainChroma = 0.5f; p10.frameIndex = 42;
    failures += compareRun(queue, p10, "refine: desat+texture+grain");

    // v3.6 reconstructed grain: shadow-loud amplitude (gainYv), contrast-mask
    // (redg 3x3), blue-noise high-pass (4-neighbour valueNoise) — all new
    // per-pixel refine math that must match CPU across the backend
    NRParams p10b = p; p10b.grainAmount = 0.8f; p10b.grainChroma = 0.4f;
    p10b.grainBlue = 0.7f; p10b.frameIndex = 42;
    failures += compareRun(queue, p10b, "v3.6 grain: shadow-loud + blue-noise");

    // v3.6 optical acutance: the 3x3 min/max clamp + edginess-gated high-pass
    // (a gain>1 stage) must match CPU
    NRParams p10c = p; p10c.acutance = 1.2f;
    failures += compareRun(queue, p10c, "v3.6 acutance: overshoot-bounded high-pass");
    NRParams p10d = p; p10d.acutance = 0.8f; p10d.grainAmount = 0.6f; p10d.grainBlue = 0.5f;
    p10d.lumaTexture = 0.3f; p10d.frameIndex = 7;
    failures += compareRun(queue, p10d, "v3.6 acutance + grain + texture together");

    // v3.6 chroma-speckle (WEAK-1): wide luma-guided chroma ring (block means,
    // exp luma weight) must match CPU
    NRParams p10e = p; p10e.chromaSpeckle = 1.0f;
    failures += compareRun(queue, p10e, "v3.6 chroma-speckle: luma-guided wide chroma");

    NRParams p11 = p; p11.spatialRadius = 8;
    failures += compareRun(queue, p11, "radius 8");

    NRParams p12 = p; p12.viewMode = 3;
    failures += compareRun(queue, p12, "after-temporal view");

    NRParams p13 = p; p13.viewMode = 5; p13.profileSource = 1;
    failures += compareRun(queue, p13, "analysis HUD view", false, false, true);

    NRParams p14 = p; p14.viewMode = 7;
    failures += compareRun(queue, p14, "SNR map view");

    NRParams p15 = p; p15.viewMode = 6;
    failures += compareRun(queue, p15, "temporal activity view");

    NRParams p16 = p; p16.viewMode = 4;
    failures += compareRun(queue, p16, "noise removed view");

    // ---- v3 cases ----
    NRParams v1 = p; v1.motionTracking = 1;
    failures += compareRun(queue, v1, "v3 motion tracking (2px pan)", true);

    NRParams v2 = p; v2.motionTracking = 1; v2.temporalFrames = 3;
    failures += compareRun(queue, v2, "v3 tracking, 3 frames", true);

    NRParams v3 = p;
    failures += compareRun(queue, v3, "v3 firefly + injected impulses", false, true);

    NRParams v4 = p; v4.eqMedium = 0.8f; v4.eqCoarse = 0.6f; v4.chromaBlotch = 0.5f;
    failures += compareRun(queue, v4, "v3 noise EQ (medium+coarse)");

    NRParams v5 = p; v5.eqFine = 1.6f; v5.deband = 1.0f;
    failures += compareRun(queue, v5, "v3 fine boost + deband");

    NRParams v6 = p; v6.profileLocked = 1;
    v6.lockSY = 0.031f; v6.lockSC = 0.017f; v6.lockTY = 0.029f; v6.lockTC = 0.015f;
    v6.lockSCr = 0.009f; v6.lockTCr = 0.008f;   // v3.3 B5: distinct Cr pair
    for (int b = 0; b < 16; ++b) { v6.lockGainY[b] = 0.8f + 0.05f * b; v6.lockGainC[b] = 1.4f; }
    failures += compareRun(queue, v6, "v3 locked profile");

    NRParams v7 = v6; v7.profileSource = 2; v7.sigmaY = 0.04f; v7.sigmaC = 0.04f;
    failures += compareRun(queue, v7, "v3 locked overrides manual");

    NRParams v8 = v6; v8.viewMode = 5;
    failures += compareRun(queue, v8, "v3 HUD with LOCKED tag", false, false, true);

    NRParams v9 = p; v9.viewMode = 8;
    failures += compareRun(queue, v9, "v3 noise matte view");

    NRParams v9b = p; v9b.viewMode = 9;   // v3.6 clean-confidence matte (effN)
    failures += compareRun(queue, v9b, "v3.6 clean-confidence matte view");

    // ---- v3.1 cases ----
    NRParams w1 = p; w1.detailRescue = 0.8f; w1.spatialLuma = 1.5f;
    w1.spatialChroma = 1.5f; w1.eqFine = 3.0f; w1.spatialRadius = 10; w1.master = 3.0f;
    failures += compareRun(queue, w1, "v3.1 full crank + detail rescue");

    NRParams w2 = p; w2.detailRescue = 0.4f;
    failures += compareRun(queue, w2, "v3.1 detail rescue at defaults");

    NRParams w3 = p; w3.chromaBlotch = 1.5f; w3.eqMedium = 1.5f; w3.eqCoarse = 1.2f;
    failures += compareRun(queue, w3, "v3.1 band overshoot (150s)");

    NRParams w4 = p; w4.scopeMeasure = 1; w4.scopeMotion = 1; w4.scopeEq = 1;
    failures += compareRun(queue, w4, "v3.1 all scopes on result view", false, false, true);

    NRParams w5 = p; w5.scopeEq = 1; w5.profileSource = 2;
    w5.sigmaY = 0.03f; w5.sigmaC = 0.02f;
    failures += compareRun(queue, w5, "v3.1 EQ scope, manual profile", false, false, true);

    NRParams w6 = p; w6.viewMode = 4; w6.temporalLuma = 1.25f; w6.motionThresh = 1.5f;
    failures += compareRun(queue, w6, "v3.1 noise view + extended temporal");

    // ---- v3.2 cases ----
    NRParams x1 = p; x1.ghostGuard = 0;
    failures += compareRun(queue, x1, "v3.2 ghost guard off");

    NRParams x2 = p; x2.globalBlend = 0.5f;
    failures += compareRun(queue, x2, "v3.2 global blend 50");

    NRParams x3 = p; x3.globalBlend = 0.0f;
    failures += compareRun(queue, x3, "v3.2 global blend 0 (identity mix)");

    NRParams x4 = p; x4.profileLocked = 1; x4.profileAdjust = 2.5f;
    x4.lockSY = 0.02f; x4.lockSC = 0.015f; x4.lockTY = 0.019f; x4.lockTC = 0.014f;
    failures += compareRun(queue, x4, "v3.2 locked profile x adjust");

    NRParams x5 = p; x5.motionTracking = 1; x5.ghostGuard = 1;
    failures += compareRun(queue, x5, "v3.2 guard + tracking (2px pan)", true);

    // ---- v3.3 cases ----
    // lock fast path: the "v3 locked profile" case above (locked, scopes
    // off) now skips the NoiseEst/FinalizeStats dispatches on the GPU and
    // the measurement pass on the CPU — these cover the boundaries.
    NRParams y1 = v6; y1.scopeEq = 1;
    failures += compareRun(queue, y1, "v3.3 locked + EQ scope (live stats)", false, false, true);

    NRParams y2 = v6; y2.scopeMotion = 1;
    failures += compareRun(queue, y2, "v3.3 locked + motion scope (fast path)", false, false, true);

    NRParams y3 = v6; y3.viewMode = 4;
    failures += compareRun(queue, y3, "v3.3 locked fast path, noise view");

    NRParams y4 = v6; y4.profileSource = 1; y4.regionCX = 0.3f; y4.regionCY = 0.4f;
    failures += compareRun(queue, y4, "v3.3 locked fast path, region source");

    // B1 hierarchical search: a 5 px/frame pan exercises the coarse step-4
    // level + refine walk (the 2 px cases above only reach the walk)
    NRParams y5 = p; y5.motionTracking = 1;
    failures += compareRun(queue, y5, "v3.3 tracking, 5px pan (coarse)", true, false, false, 5.0f);

    NRParams y6 = y5; y6.temporalFrames = 3; y6.ghostGuard = 0;
    failures += compareRun(queue, y6, "v3.3 tracking 5px, 3f, no guard", true, false, false, 5.0f);

    // B3 Deep Clean: pre-pass + residual re-measure + dual-buffer views
    NRParams z1 = p; z1.deepClean = 1;
    failures += compareRun(queue, z1, "v3.3 deep clean, defaults");

    NRParams z2 = z1; z2.viewMode = 3;
    failures += compareRun(queue, z2, "v3.3 deep clean, after-temporal view");

    NRParams z3 = z1; z3.profileSource = 2; z3.sigmaY = 0.04f; z3.sigmaC = 0.03f;
    failures += compareRun(queue, z3, "v3.3 deep clean, manual profile");

    NRParams z4 = v6; z4.deepClean = 1;
    failures += compareRun(queue, z4, "v3.3 deep clean + locked fast path");

    NRParams z5 = z1; z5.spatialLuma = 1.5f; z5.eqFine = 3.0f; z5.detailRescue = 0.8f;
    z5.spatialRadius = 8; z5.scopeMotion = 1;
    failures += compareRun(queue, z5, "v3.3 deep clean, crank + motion scope", false, false, true);

    // v3.5 P1: exposure step exercises the offset estimation + compensation
    NRParams e1 = p; e1.temporalFrames = 7;
    failures += compareRun(queue, e1, "v3.5 exposure step +3%", false, false, false, 2.0f, 0.03f);

    NRParams e2 = p; e2.motionTracking = 1;
    failures += compareRun(queue, e2, "v3.5 exposure step + tracking", true, false, false, 2.0f, 0.03f);

    // B2: 7-frame stack
    NRParams s7 = p; s7.temporalFrames = 7;
    failures += compareRun(queue, s7, "v3.3 7-frame stack");

    NRParams s7t = s7; s7t.motionTracking = 1;
    failures += compareRun(queue, s7t, "v3.3 7 frames + tracking (2px pan)", true);

    NRParams s7d = s7; s7d.deepClean = 1; s7d.viewMode = 6;
    failures += compareRun(queue, s7d, "v3.3 7f + deep clean, activity view");

    // v3.5 R1 Render Boost: sequential-call history cache + kernel block
    failures += compareBoost(queue, p);

    printf(failures == 0 ? "ALL GPU PARITY CHECKS PASSED\n" : "%d GPU PARITY CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
