// OpenNR — GPU performance benchmark (Metal). Reports ms/frame for common
// resolutions and quality settings.
//
// v3: frames are a synthetic scene with per-frame noise (and optionally a
// pan), so the motion-tracking shift search pays its real cost — the v2.1
// bench used five identical buffers, which a content-adaptive temporal stage
// can short-circuit entirely.
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

static void makeFrame(std::vector<float>& img, int W, int H, float shift, uint32_t seed)
{
    img.resize(static_cast<size_t>(W) * H * 4);
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, 0.03f);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float fx = x + shift;
            float r = 0.2f + 0.4f * (fx / W);
            float g = 0.3f + 0.2f * (static_cast<float>(y) / H);
            float b = 0.25f + 0.3f * (fx / W);
            const float dx = fx - W * 0.4f, dy = y - H * 0.45f;
            if (dx * dx + dy * dy < (H * 0.2f) * (H * 0.2f)) { r = 0.8f; g = 0.75f; b = 0.6f; }
            if (fx > W * 0.7f && static_cast<int>(fx) % 12 < 2) { r *= 0.4f; g *= 0.4f; b *= 0.4f; }
            float* p = &img[(static_cast<size_t>(y) * W + x) * 4];
            p[0] = r + N(rng); p[1] = g + N(rng); p[2] = b + N(rng); p[3] = 1.0f;
        }
    }
}

static double benchOne(id<MTLCommandQueue> queue, int W, int H, const NRParams& p,
                       float panPerFrame, int iters)
{
    // min of 3 interleaved passes, each in its own autorelease pool, so
    // thermal drift and buffer lifetime can't skew one configuration
    double best = 1e30;
    for (int pass = 0; pass < 3; ++pass) {
        @autoreleasepool {
            id<MTLDevice> device = queue.device;
            const size_t bytes = static_cast<size_t>(W) * H * 4 * sizeof(float);

            id<MTLBuffer> srcBuf[5];
            for (int i = 0; i < 5; ++i) {
                std::vector<float> frame;
                makeFrame(frame, W, H, (i - 2) * panPerFrame, 100 + i);
                srcBuf[i] = [device newBufferWithBytes:frame.data() length:bytes options:MTLResourceStorageModeShared];
            }
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

            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
            if (ms < best) best = ms;
            for (int i = 0; i < 5; ++i) srcBuf[i] = nil;
            dstBuf = nil;
        }
    }
    return best;
}

int main()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("no Metal device\n"); return 0; }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    printf("OpenNR v3 benchmark on %s\n", device.name.UTF8String);

    NRParams best = {};
    best.profileSource = 0; best.sigmaY = 0.02f; best.sigmaC = 0.02f; best.profileAdjust = 1.0f;
    best.regionCX = 0.5f; best.regionCY = 0.5f; best.regionSize = 0.25f; best.hasTemporalDiff = 0;
    best.enableTemporal = 1; best.temporalFrames = 5; best.temporalLuma = 0.6f; best.temporalChroma = 0.8f;
    best.motionThresh = 0.3f; best.enableSpatial = 1; best.spatialMode = 1; best.spatialRadius = 3;
    best.spatialLuma = 0.6f; best.spatialChroma = 1.0f; best.preserveDetail = 0.35f;
    best.chromaBlotch = 0.25f; best.enableRefine = 1; best.shadowDesat = 0.0f; best.desatRange = 0.15f;
    best.lumaTexture = 0.0f; best.grainAmount = 0.0f; best.grainSize = 1.6f; best.grainChroma = 0.25f;
    best.frameIndex = 0; best.master = 1.0f; best.viewMode = 0;
    best.motionTracking = 1; best.fireflyRemoval = 1; best.eqFine = 1.0f;
    best.lockSY = 0.02f; best.lockSC = 0.02f; best.lockTY = 0.02f; best.lockTC = 0.02f;
    for (int b = 0; b < 16; ++b) { best.lockGainY[b] = 1.0f; best.lockGainC[b] = 1.0f; }

    NRParams fast = best;
    fast.temporalFrames = 3;
    fast.spatialMode = 0;   // bilateral
    fast.spatialRadius = 2;

    NRParams noTrack = best;
    noTrack.motionTracking = 0;   // the v2.1-equivalent temporal path

    // v3.3 lock fast path: locked profile + no scopes skips the input
    // estimation dispatches entirely
    NRParams locked = best;
    locked.profileLocked = 1;
    locked.lockSY = 0.03f; locked.lockSC = 0.02f; locked.lockTY = 0.028f; locked.lockTC = 0.018f;

    struct { const char* name; int w, h; int iters; } sizes[] = {
        { "HD  1920x1080", 1920, 1080, 40 },
        { "UHD 3840x2160", 3840, 2160, 15 },
    };

    for (auto& s : sizes) {
        const double msStatic = benchOne(queue, s.w, s.h, best, 0.0f, s.iters);
        const double msPan    = benchOne(queue, s.w, s.h, best, 1.5f, s.iters);
        const double msNoTrk  = benchOne(queue, s.w, s.h, noTrack, 0.0f, s.iters);
        const double msLock   = benchOne(queue, s.w, s.h, locked, 0.0f, s.iters);
        const double msFast   = benchOne(queue, s.w, s.h, fast, 0.0f, s.iters);
        printf("%s   Better(NLM,R3,5f): static %6.2f ms (%5.1f fps)  panning %6.2f ms (%5.1f fps)  tracking-off %6.2f ms  locked %6.2f ms (%5.1f fps)\n",
               s.name, msStatic, 1000.0 / msStatic, msPan, 1000.0 / msPan, msNoTrk,
               msLock, 1000.0 / msLock);
        printf("%s   Faster(Bilat,R2,3f): %6.2f ms/frame (%5.1f fps)\n",
               s.name, msFast, 1000.0 / msFast);
    }
    return 0;
}
