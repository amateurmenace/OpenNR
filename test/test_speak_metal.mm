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

// A frame with a REAL HIGHLIGHT: a hot, slightly-warm disc on a dim blue-ish
// field, plus a saturated blue lamp. A smooth gradient (makeFrame) barely
// exercises a scatter pyramid — nothing has a sharp enough highlight edge to
// reveal a wrong octave, and no source is saturated enough to reveal a
// per-channel scatter error. This is the frame the halation cases use.
static std::vector<float> makeHotFrame(int W, int H)
{
    std::vector<float> f(static_cast<size_t>(W) * H * 4);
    const float cx = W * 0.42f, cy = H * 0.45f, rad = W * 0.09f;
    const float bx = W * 0.75f, by = H * 0.70f, brad = W * 0.05f;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            const float db = std::sqrt((x - bx) * (x - bx) + (y - by) * (y - by));
            if (d <= rad)        { f[i + 0] = 9.0f;  f[i + 1] = 8.2f;  f[i + 2] = 7.0f; }
            else if (db <= brad) { f[i + 0] = 0.02f; f[i + 1] = 0.05f; f[i + 2] = 6.0f; } // blue lamp
            else                 { f[i + 0] = 0.05f + 0.20f * (static_cast<float>(x) / (W - 1));
                                   f[i + 1] = 0.07f;
                                   f[i + 2] = 0.12f + 0.10f * (static_cast<float>(y) / (H - 1)); }
            f[i + 3] = 1.0f;
        }
    return f;
}

static void runSrc(id<MTLDevice> device, id<MTLCommandQueue> queue,
                   int W, int H, const SpeakParams& p, const char* label,
                   int mode, const std::vector<float>& src)
{
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

static void run(id<MTLDevice> device, id<MTLCommandQueue> queue,
                int W, int H, const SpeakParams& p, const char* label,
                int mode) // 0 strict, 1 hud-tolerant (scope)
{
    runSrc(device, queue, W, H, p, label, mode, makeFrame(W, H));
}
static void runHot(id<MTLDevice> device, id<MTLCommandQueue> queue,
                   int W, int H, const SpeakParams& p, const char* label, int mode)
{
    runSrc(device, queue, W, H, p, label, mode, makeHotFrame(W, H));
}
static SpeakParams halParams(float amount, float radius)
{
    SpeakParams p = baseParams();
    p.inputColorSpace = SPEAK_CS_LINEAR;   // scene-linear in, so the disc is a real highlight
    p.enableOptics = 1;
    p.profile.halAmount = amount;
    p.profile.halRadius = radius;
    p.profile.halThresh = 0.6f;
    return p;
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
    // 5d — Bake + Split view (both halves must land in Rec.709).
    { SpeakParams p = baseParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; p.viewMode = SPEAK_VIEW_SPLIT; p.profile = stockProfile();
      run(device, queue, W, H, p, "bake + split view", 2); }
    // 5e — Subtractive color (Phase 2): dye standalone, and dye + tone.
    { SpeakParams p = baseParams(); p.strength = 0.0f; p.enableDye = 1;
      p.profile.subSat[0] = p.profile.subSat[1] = p.profile.subSat[2] = 0.55f;
      speakcore::setDyeCoupler(p.profile, 1.0f);
      p.profile.subSatKnee[0] = p.profile.subSatKnee[1] = p.profile.subSatKnee[2] = 2.2f;
      run(device, queue, W, H, p, "subtractive color standalone", 0); }
    { SpeakParams p = baseParams(); p.profile = stockProfile(); p.enableDye = 1;
      p.profile.subSat[0] = p.profile.subSat[1] = p.profile.subSat[2] = 0.8f;
      speakcore::setDyeCoupler(p.profile, 0.7f);
      p.profile.subSatKnee[0] = p.profile.subSatKnee[1] = p.profile.subSatKnee[2] = 2.2f;
      run(device, queue, W, H, p, "tone + subtractive color", 0); }
    // 5f — Status-M density scope (hud-tolerant: bar/parade display math).
    { SpeakParams p = baseParams(); p.scopeDensity = 1; p.strength = 0.7f; p.profile = stockProfile();
      run(device, queue, W, H, p, "density scope on", 1); }
    { SpeakParams p = baseParams(); p.scopeHD = 1; p.scopeDensity = 1; p.strength = 0.7f; p.profile = stockProfile();
      run(device, queue, W, H, p, "both scopes on", 1); }
    // 5g — Split toning (Phase 3): crossover standalone, and the full stack.
    { SpeakParams p = baseParams(); p.strength = 0.0f; p.enableSplit = 1;
      p.profile.splitShadow[0] = 0.10f; p.profile.splitShadow[2] = -0.10f;
      p.profile.splitHigh[0] = -0.08f;  p.profile.splitHigh[2] = 0.08f;
      p.profile.splitBalance = 0.5f;
      run(device, queue, W, H, p, "split toning standalone", 0); }
    { SpeakParams p = baseParams(); p.profile = stockProfile(); p.enableDye = 1; p.enableSplit = 1;
      p.profile.subSat[0] = p.profile.subSat[1] = p.profile.subSat[2] = 0.6f;
      speakcore::setDyeCoupler(p.profile, 0.8f);
      p.profile.subSatKnee[0] = p.profile.subSatKnee[1] = p.profile.subSatKnee[2] = 2.2f;
      p.profile.splitShadow[2] = -0.09f; p.profile.splitHigh[0] = -0.07f;
      p.profile.splitBalance = 0.4f;
      run(device, queue, W, H, p, "tone + dye + split (full stack)", 0); }
    // 6 — H&D scope on (hud-tolerant).
    { SpeakParams p = baseParams(); p.scopeHD = 1; p.strength = 0.6f; p.profile = stockProfile();
      run(device, queue, W, H, p, "scope H&D on s0.6", 1); }

    // ------------------------------------------------------------ 7 halation
    // The scatter pyramid is multi-pass and size-dependent, so it needs its own
    // coverage: BEFORE these cases existed the suite reported PARITY GREEN on a
    // completely broken pyramid, because no case ever set halAmount.
    printf("  -- halation (the scatter pyramid) --\n");
    // 7a — halation off but optics on: must still be bit-exact (skip path).
    { SpeakParams p = halParams(0.0f, 1.0f);
      runHot(device, queue, W, H, p, "halation amount 0 (skip path)", 0); }
    // 7b — halation standalone on a real highlight.
    { SpeakParams p = halParams(1.0f, 1.0f);
      runHot(device, queue, W, H, p, "halation s1.0 r1.0%", 0); }
    // 7c — a WIDE radius: drives the mixture up into the small top levels, whose
    // dimensions fall below one threadgroup. That is exactly where a wrong grid
    // or a wrong level ladder hides.
    { SpeakParams p = halParams(1.2f, 6.0f);
      runHot(device, queue, W, H, p, "halation wide r6.0%", 0); }
    // 7d — a TIGHT radius (the mixture sits on the bottom levels).
    { SpeakParams p = halParams(1.0f, 0.1f);
      runHot(device, queue, W, H, p, "halation tight r0.1%", 0); }
    // 7e — the isolated-scatter view.
    { SpeakParams p = halParams(1.0f, 2.0f); p.viewMode = SPEAK_VIEW_SCATTER;
      runHot(device, queue, W, H, p, "halation scatter view", 0); }
    // 7f — halation through the whole look (dye + split + stock) and baked.
    { SpeakParams p = halParams(0.9f, 1.5f); p.profile = stockProfile();
      p.profile.halAmount = 0.9f; p.profile.halRadius = 1.5f; p.profile.halThresh = 0.6f;
      p.enableDye = 1; p.enableSplit = 1;
      p.profile.subSat[0] = p.profile.subSat[1] = p.profile.subSat[2] = 0.6f;
      speakcore::setDyeCoupler(p.profile, 0.8f);
      p.profile.subSatKnee[0] = p.profile.subSatKnee[1] = p.profile.subSatKnee[2] = 2.2f;
      p.profile.splitShadow[2] = -0.09f; p.profile.splitHigh[0] = -0.07f;
      runHot(device, queue, W, H, p, "halation + full stack", 0); }
    // 7g — halation + the density parade: the parade must measure the HALATED
    // result, so this case is what proves the scatter reaches the stats pass.
    { SpeakParams p = halParams(1.0f, 2.0f); p.scopeDensity = 1;
      runHot(device, queue, W, H, p, "halation + density scope", 1); }
    // 7h — ODD dimensions: every level's dims go odd, exercising the (w+1)/2
    // ladder and the clamped decimation taps.
    { SpeakParams p = halParams(1.0f, 2.0f);
      runHot(device, queue, 333, 197, p, "halation odd dims 333x197", 0); }
    // 7h' — W and H both == 1 (mod 32). The stats dispatch indexes x = gid*2, so
    // it needs ceil(W/2) sample threads; dispatching floor(W/2) falls exactly one
    // sample column short at these sizes and nowhere else. Metal always
    // over-covered, but OpenCL and CUDA did NOT (1921 and 3841 are both 1 mod 32),
    // so pin the size here to keep every backend honest.
    { SpeakParams p = halParams(1.0f, 2.0f); p.scopeDensity = 1; p.scopeHD = 1;
      runHot(device, queue, 353, 225, p, "scopes at 353x225 (W,H = 1 mod 32)", 1); }
    // 7i — SIZE CHANGES on ONE queue: proxy -> full -> proxy -> bigger. The
    // scatter buffers are size-dependent (the stats buffer is not), so this is
    // the realloc path — and a stale buffer here is a silent overrun.
    { SpeakParams p = halParams(1.0f, 2.0f);
      runHot(device, queue, 320, 240,  p, "halation size 320x240", 0);
      runHot(device, queue, 1280, 720, p, "halation size 1280x720", 0);
      runHot(device, queue, 320, 240,  p, "halation size back to 320x240", 0);
      runHot(device, queue, 960, 540,  p, "halation size 960x540", 0); }

    printf("\n%s (%d failures)\n", g_fail ? "PARITY FAILED" : "PARITY GREEN", g_fail);
    return g_fail ? 1 : 0;
}
