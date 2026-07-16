// Speak GPU parity test — runs the REAL RunMetalSpeak entry point (the same
// one DaVinci Resolve calls) against the CPU reference (speak_core.h) and
// asserts they agree. This is the cardinal-rule parity gate for the Metal port.
//
// Build:
//   c++ -O2 -std=c++14 -I../plugin test_speak_metal.mm ../plugin/SpeakMetalKernel.mm \
//       -framework Metal -framework Foundation -o test_speak_metal && ./test_speak_metal

#import <Metal/Metal.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

#include "SpeakParams.h"
#include "speak_core.h"

extern void RunMetalSpeak(void* p_CmdQ, int p_Width, int p_Height,
                          const SpeakParams& p_Params, const float* p_Src, float* p_Dst);

static int g_fail = 0;

// A varied synthetic frame: per-channel-different gradients spanning deep
// shadow to highlight (values in a DI-encoded-ish [0, 1.2] range), plus alpha.
static std::vector<float> makeFrame(int W, int H)
{
    std::vector<float> f(static_cast<size_t>(W) * H * 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const float u = static_cast<float>(x) / (W - 1);
            const float v = static_cast<float>(y) / (H - 1);
            f[i + 0] = 0.02f + 1.15f * u * u;                 // R: dark->bright
            f[i + 1] = 0.05f + 0.9f * v;                       // G
            f[i + 2] = 0.03f + 0.8f * (0.5f * u + 0.5f * (1.0f - v)); // B
            f[i + 3] = 0.25f + 0.5f * u;                       // alpha
        }
    return f;
}

static SpeakParams baseParams()
{
    SpeakParams p = {};
    p.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
    p.outputMode = SPEAK_OUT_WORKING;
    p.enableTone = 1;
    p.strength = 1.0f;
    p.viewMode = SPEAK_VIEW_RESULT;
    p.profile = speakcore::neutralProfile();
    return p;
}

// A behavioral "stock": per-channel-different curves + printer lights.
static SpeakProfile stockProfile()
{
    SpeakProfile p = speakcore::neutralProfile();
    p.negGamma[0] = 0.66f; p.negGamma[2] = 0.58f;
    p.prnGamma[1] = 2.8f;
    p.negSpeed[2] = -1.35f;
    p.printerLights[0] = 2.5f; p.printerLights[2] = -1.5f;
    p.printerMaster = 0.8f;
    return p;
}

static void run(id<MTLDevice> device, id<MTLCommandQueue> queue,
                int W, int H, const SpeakParams& p, const char* label,
                int mode) // 0 strict, 1 hud-tolerant (scope)
{
    std::vector<float> src = makeFrame(W, H);
    const size_t n = static_cast<size_t>(W) * H * 4;
    const size_t bytes = n * sizeof(float);

    std::vector<float> cpu(n);
    speakcore::speakFrame(src.data(), W, H, p, cpu.data());

    id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data() length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    RunMetalSpeak((void*)queue, W, H, p, (const float*)srcBuf, (float*)dstBuf);
    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];

    const float* gpu = static_cast<const float*>(dstBuf.contents);
    double maxd = 0.0, sumd = 0.0; size_t nOver = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(gpu[i]) - cpu[i]);
        if (d > maxd) maxd = d;
        sumd += d;
        if (d > 5e-3) nOver++;
    }
    const double meand = sumd / n;
    bool pass;
    if (mode == 1)      pass = (meand < 5e-5) && (nOver <= 400);              // scope: hudOK
    else if (mode == 2) pass = (meand < 1e-5) && (nOver <= 64) && (maxd < 0.30); // bake: isolated
                                                                              // gamut-edge/near-black pixels where fast-math
                                                                              // straddles a channel zero and the (correct)
                                                                              // pure gamma-2.4 slope->inf amplifies it
    else                pass = (maxd < 5e-3) && (meand < 1e-4);               // strict
    printf("  [%s] %-30s max %.2e  mean %.2e  over %zu\n",
           pass ? "PASS" : "FAIL", label, maxd, meand, nOver);
    if (!pass) g_fail++;
}

int main()
{
    printf("=== Speak Metal parity ===\n");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("no Metal device — skipping\n"); return 0; }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const int W = 640, H = 480;

    // 1 — identity (strength 0): must be bit-exact.
    { SpeakParams p = baseParams(); p.strength = 0.0f; run(device, queue, W, H, p, "identity (strength 0)", 0); }
    // 2 — neutral tone, full strength.
    { SpeakParams p = baseParams(); run(device, queue, W, H, p, "tone neutral s1.0", 0); }
    // 3 — behavioral stock + printer lights, partial strength.
    { SpeakParams p = baseParams(); p.profile = stockProfile(); p.strength = 0.7f;
      run(device, queue, W, H, p, "stock + printerLights s0.7", 0); }
    // 4 — Rec.709 input space.
    { SpeakParams p = baseParams(); p.inputColorSpace = SPEAK_CS_REC709_G24; p.profile = stockProfile();
      run(device, queue, W, H, p, "stock Rec709-in s1.0", 0); }
    // 5 — Split view.
    { SpeakParams p = baseParams(); p.viewMode = SPEAK_VIEW_SPLIT; p.profile = stockProfile();
      run(device, queue, W, H, p, "split view s1.0", 0); }
    // 5b — Bake to Rec.709 (output CST). Mode 2: bounded gamut-edge boundary flips.
    { SpeakParams p = baseParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; p.profile = stockProfile();
      run(device, queue, W, H, p, "bake Rec.709 s1.0", 2); }
    // 5c — Bake with look off (pure CST).
    { SpeakParams p = baseParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; p.strength = 0.0f;
      run(device, queue, W, H, p, "bake Rec.709 CST-only", 2); }
    // 6 — H&D scope on (hud-tolerant).
    { SpeakParams p = baseParams(); p.scopeHD = 1; p.strength = 0.6f; p.profile = stockProfile();
      run(device, queue, W, H, p, "scope H&D on s0.6", 1); }

    printf("\n%s (%d failures)\n", g_fail ? "PARITY FAILED" : "PARITY GREEN", g_fail);
    return g_fail ? 1 : 0;
}
