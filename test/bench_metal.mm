// OpenNR — GPU performance benchmark (Metal). Reports ms/frame for common
// resolutions and quality settings.
//
// Build:
//   c++ -O2 -std=c++14 -I../plugin bench_metal.mm ../plugin/MetalKernel.mm \
//       -framework Metal -framework Foundation -o bench_metal

#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "NRParams.h"

extern void RunMetalNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                       const float* const p_Srcs[5], float* p_Dst);

static double benchOne(id<MTLCommandQueue> queue, int W, int H, const NRParams& p, int iters)
{
    id<MTLDevice> device = queue.device;
    const size_t bytes = static_cast<size_t>(W) * H * 4 * sizeof(float);

    std::vector<float> frame(static_cast<size_t>(W) * H * 4);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    for (auto& v : frame) v = U(rng);

    id<MTLBuffer> srcBuf[5];
    for (int i = 0; i < 5; ++i)
        srcBuf[i] = [device newBufferWithBytes:frame.data() length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    const float* srcs[5];
    for (int i = 0; i < 5; ++i)
        srcs[i] = (const float*)srcBuf[i];

    auto sync = [&] {
        id<MTLCommandBuffer> fence = [queue commandBuffer];
        [fence commit];
        [fence waitUntilCompleted];
    };

    // warmup (includes kernel compile on first call)
    for (int i = 0; i < 3; ++i)
        RunMetalNR((void*)queue, W, H, p, srcs, (float*)dstBuf);
    sync();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        RunMetalNR((void*)queue, W, H, p, srcs, (float*)dstBuf);
    sync();
    const auto t1 = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("no Metal device\n"); return 0; }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    printf("OpenNR benchmark on %s\n", device.name.UTF8String);

    NRParams best = {};
    best.profileSource = 0; best.sigmaY = 0.02f; best.sigmaC = 0.02f; best.profileAdjust = 1.0f;
    best.regionCX = 0.5f; best.regionCY = 0.5f; best.regionSize = 0.25f; best.hasTemporalDiff = 0;
    best.enableTemporal = 1; best.temporalFrames = 5; best.temporalLuma = 0.6f; best.temporalChroma = 0.8f;
    best.motionThresh = 0.3f; best.enableSpatial = 1; best.spatialMode = 1; best.spatialRadius = 3;
    best.spatialLuma = 0.6f; best.spatialChroma = 1.0f; best.preserveDetail = 0.35f;
    best.chromaBlotch = 0.25f; best.enableRefine = 1; best.shadowDesat = 0.0f; best.desatRange = 0.15f;
    best.lumaTexture = 0.0f; best.grainAmount = 0.0f; best.grainSize = 1.6f; best.grainChroma = 0.25f;
    best.frameIndex = 0; best.master = 1.0f; best.viewMode = 0;

    NRParams fast = best;
    fast.temporalFrames = 3;
    fast.spatialMode = 0;   // bilateral
    fast.spatialRadius = 2;

    struct { const char* name; int w, h; } sizes[] = {
        { "HD  1920x1080", 1920, 1080 },
        { "UHD 3840x2160", 3840, 2160 },
    };

    for (auto& s : sizes) {
        const double msBest = benchOne(queue, s.w, s.h, best, 20);
        const double msFast = benchOne(queue, s.w, s.h, fast, 20);
        printf("%s   Better(NLM,R3,5f): %6.2f ms/frame (%5.1f fps)   Faster(Bilat,R2,3f): %6.2f ms/frame (%5.1f fps)\n",
               s.name, msBest, 1000.0 / msBest, msFast, 1000.0 / msFast);
    }
    return 0;
}
