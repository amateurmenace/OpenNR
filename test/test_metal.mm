// OpenNR GPU parity test — runs the real Metal pipeline (RunMetalNR, the same
// entry point DaVinci Resolve calls) against the CPU reference implementation
// and verifies the outputs match within fast-math tolerance.
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
                       const float* const p_Srcs[5], float* p_Dst);

static const int W = 512;
static const int H = 288;

static void makeFrame(std::vector<float>& img, int k, uint32_t seed)
{
    img.resize(static_cast<size_t>(W) * H * 4);
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, 0.05f);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            const float fx = x + k * 2.0f;
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

static int compareRun(id<MTLCommandQueue> queue, const NRParams& gp, const char* label)
{
    std::vector<std::vector<float>> frames(5);
    for (int k = 0; k < 5; ++k)
        makeFrame(frames[k], k - 2, 100 + k);

    nrcore::Params cp;
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
    cp.master         = gp.master;
    cp.viewMode       = gp.viewMode;

    const float* fptr[5] = { frames[0].data(), frames[1].data(), frames[2].data(),
                             frames[3].data(), frames[4].data() };
    std::vector<float> cpuOut(static_cast<size_t>(W) * H * 4), scratch;
    nrcore::denoiseFrame(fptr, W, H, cp, cpuOut.data(), scratch);

    id<MTLDevice> device = queue.device;
    const size_t bytes = static_cast<size_t>(W) * H * 4 * sizeof(float);

    id<MTLBuffer> srcBuf[5];
    for (int i = 0; i < 5; ++i)
        srcBuf[i] = [device newBufferWithBytes:frames[i].data() length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    const float* srcs[5];
    for (int i = 0; i < 5; ++i)
        srcs[i] = (const float*)srcBuf[i];

    RunMetalNR((void*)queue, W, H, gp, srcs, (float*)dstBuf);

    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];

    const float* gpuOut = static_cast<const float*>(dstBuf.contents);

    double maxd = 0.0, sumd = 0.0;
    size_t n = static_cast<size_t>(W) * H * 4;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(gpuOut[i] - cpuOut[i]);
        maxd = std::max(maxd, d);
        sumd += d;
    }
    const double meand = sumd / n;
    const bool pass = (maxd < 5e-3) && (meand < 1e-4);
    printf("  [%s] %-34s max %.2e  mean %.2e\n", pass ? "PASS" : "FAIL", label, maxd, meand);
    return pass ? 0 : 1;
}

int main()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("no Metal device — skipping GPU parity test\n");
        return 0;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    printf("GPU parity test on %s\n", device.name.UTF8String);

    int failures = 0;

    NRParams p = {};
    p.profileSource = 0;
    p.sigmaY = 0.02f; p.sigmaC = 0.02f;
    p.profileAdjust = 1.0f;
    p.regionCX = 0.5f; p.regionCY = 0.5f; p.regionSize = 0.25f;
    p.hasTemporalDiff = 0; // host recomputes
    p.enableTemporal = 1; p.temporalFrames = 5;
    p.temporalLuma = 0.6f; p.temporalChroma = 0.8f; p.motionThresh = 0.4f;
    p.enableSpatial = 1; p.spatialMode = 1; p.spatialRadius = 3;
    p.spatialLuma = 0.45f; p.spatialChroma = 0.75f;
    p.preserveDetail = 0.35f; p.master = 1.0f; p.viewMode = 0;

    failures += compareRun(queue, p, "defaults (auto, 5f, NLM)");

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

    NRParams p8 = p; p8.viewMode = 2;
    failures += compareRun(queue, p8, "noise-only view");

    NRParams p9 = p; p9.viewMode = 3; p9.profileSource = 1;
    failures += compareRun(queue, p9, "analysis HUD view");

    NRParams p10 = p; p10.viewMode = 4;
    failures += compareRun(queue, p10, "temporal activity view");

    NRParams p11 = p; p11.viewMode = 1;
    failures += compareRun(queue, p11, "split view");

    NRParams p12 = p; p12.profileAdjust = 2.0f;
    failures += compareRun(queue, p12, "profile adjust x2");

    printf(failures == 0 ? "ALL GPU PARITY CHECKS PASSED\n" : "%d GPU PARITY CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
