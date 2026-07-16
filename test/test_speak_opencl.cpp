// Speak GPU parity test (OpenCL) — runs the REAL RunOpenCLSpeak entry point
// against the CPU reference (speak_core.h). OpenCL is Resolve's render path on
// AMD/Intel/NVIDIA-without-CUDA and the primary Windows/Linux path, so this
// gives a THIRD verified backend (CPU + Metal + OpenCL); CUDA stays a faithful
// hardware-unverified port like Hush's.
//
// Build (macOS): c++ -O2 -std=c++14 -I../plugin test_speak_opencl.cpp \
//     ../plugin/SpeakOpenCLKernel.cpp -framework OpenCL -o test_speak_opencl

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdio>
#include <cmath>
#include <vector>

#include "SpeakParams.h"
#include "speak_core.h"

extern void RunOpenCLSpeak(void* p_CmdQ, int p_Width, int p_Height,
                           const SpeakParams& p_Params, const float* p_Src, float* p_Dst);

static int g_fail = 0;

static std::vector<float> makeFrame(int W, int H)
{
    std::vector<float> f(static_cast<size_t>(W) * H * 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const float u = static_cast<float>(x) / (W - 1);
            const float v = static_cast<float>(y) / (H - 1);
            f[i + 0] = 0.02f + 1.15f * u * u;
            f[i + 1] = 0.05f + 0.9f * v;
            f[i + 2] = 0.03f + 0.8f * (0.5f * u + 0.5f * (1.0f - v));
            f[i + 3] = 0.25f + 0.5f * u;
        }
    return f;
}

static SpeakParams baseParams()
{
    SpeakParams p = {};
    p.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
    p.outputMode = SPEAK_OUT_WORKING;
    p.enableTone = 1; p.strength = 1.0f; p.viewMode = SPEAK_VIEW_RESULT;
    p.profile = speakcore::neutralProfile();
    return p;
}
static SpeakProfile stockProfile()
{
    SpeakProfile p = speakcore::neutralProfile();
    p.negGamma[0] = 0.66f; p.negGamma[2] = 0.58f; p.prnGamma[1] = 2.8f;
    p.negSpeed[2] = -1.35f; p.printerLights[0] = 2.5f; p.printerLights[2] = -1.5f;
    p.printerMaster = 0.8f;
    return p;
}
// Halation on: the scatter pyramid is live, so the three extra passes and the
// size-dependent arena/scat buffers all get exercised. enableOptics gates the
// amount (halAmountOf) and the tone spine must be on for any of it to reach the
// pixels — halation re-exposes the NEGATIVE.
static SpeakParams halParams()
{
    SpeakParams p = baseParams();
    p.inputColorSpace = SPEAK_CS_LINEAR;   // scene-linear in: the disc is a real highlight
    p.profile = stockProfile();
    p.enableOptics = 1;
    p.profile.halAmount = 0.5f;
    p.profile.halRadius = 1.0f;
    p.profile.halThresh = 0.6f;
    return p;
}

static cl_context g_ctx; static cl_command_queue g_q; static cl_device_id g_dev;

// Apple's DEPRECATED OpenCL runtime miscompiles global int32 atomics: a minimal
// 1000-work-item atomic_inc over 4 bins returns ~1.7e9 in every bin instead of
// 250, even though the device advertises cl_khr_global_int32_base_atomics (plain
// __global writes on the same buffer are fine). The scope's measurement pass
// bins the frame with atomics, so on such a device its result is meaningless.
//
// We probe for it and SKIP the stats-dependent cases loudly rather than silently
// passing them or reporting a false parity failure — the kernel source is
// correct and IS verified here on CPU + Metal, and OpenCL's real targets
// (NVIDIA/AMD/Intel on Windows/Linux) have working atomics. macOS production
// renders via Metal, so no shipping path depends on this.
// (This very likely also explains Hush's long-standing ~2e-3 Apple-OpenCL
// divergence — its noise estimator accumulates histograms with atomics too.)
static bool atomicsWork()
{
    static const char* K =
        "__kernel void probe(volatile __global uint* s){ atomic_inc(&s[get_global_id(0) % 4]); }\n";
    cl_int e;
    cl_program pr = clCreateProgramWithSource(g_ctx, 1, &K, NULL, &e);
    if (clBuildProgram(pr, 1, &g_dev, NULL, NULL, NULL) != CL_SUCCESS) return false;
    cl_kernel k = clCreateKernel(pr, "probe", &e);
    cl_mem b = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 16, NULL, &e);
    cl_uint z = 0, out[4] = { 0, 0, 0, 0 };
    clEnqueueFillBuffer(g_q, b, &z, sizeof(cl_uint), 0, 16, 0, NULL, NULL);
    clSetKernelArg(k, 0, sizeof(cl_mem), &b);
    size_t g = 1000;
    clEnqueueNDRangeKernel(g_q, k, 1, NULL, &g, NULL, 0, NULL, NULL);
    clFinish(g_q);
    clEnqueueReadBuffer(g_q, b, CL_TRUE, 0, 16, out, 0, NULL, NULL);
    clReleaseMemObject(b); clReleaseKernel(k); clReleaseProgram(pr);
    return (out[0] + out[1] + out[2] + out[3]) == 1000u;
}

// A frame with a REAL HIGHLIGHT (mirrors test_speak_metal.mm): a hot disc plus a
// saturated blue lamp on a dim field. The smooth gradient in makeFrame barely
// exercises a scatter pyramid — no sharp highlight edge to reveal a wrong
// octave, no saturated source to reveal a per-channel scatter error.
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
            else if (db <= brad) { f[i + 0] = 0.02f; f[i + 1] = 0.05f; f[i + 2] = 6.0f; }
            else                 { f[i + 0] = 0.05f + 0.20f * (static_cast<float>(x) / (W - 1));
                                   f[i + 1] = 0.07f;
                                   f[i + 2] = 0.12f + 0.10f * (static_cast<float>(y) / (H - 1)); }
            f[i + 3] = 1.0f;
        }
    return f;
}

static void runSrc(int W, int H, const SpeakParams& p, const char* label, int mode,
                   const std::vector<float>& src)
{
    const size_t n = static_cast<size_t>(W) * H * 4;
    const size_t bytes = n * sizeof(float);

    std::vector<float> cpu(n);
    speakcore::speakFrame(src.data(), W, H, p, cpu.data());

    cl_int err;
    cl_mem srcBuf = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                                   const_cast<float*>(src.data()), &err);
    cl_mem dstBuf = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

    RunOpenCLSpeak((void*)g_q, W, H, p, (const float*)srcBuf, (float*)dstBuf);
    clFinish(g_q);

    std::vector<float> gpu(n);
    clEnqueueReadBuffer(g_q, dstBuf, CL_TRUE, 0, bytes, gpu.data(), 0, NULL, NULL);

    double maxd = 0.0, sumd = 0.0; size_t nOver = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(gpu[i]) - cpu[i]);
        if (d > maxd) maxd = d; sumd += d; if (d > 5e-3) nOver++;
    }
    const double meand = sumd / n;
    bool pass;
    if (mode == 1)      pass = (meand < 5e-5) && (nOver <= 400);                 // scope: hudOK
    else if (mode == 2) pass = (meand < 1e-5) && (nOver <= 64) && (maxd < 0.30); // bake gamut-edge boundary flips
    else                pass = (maxd < 5e-3) && (meand < 1e-4);                  // strict
    printf("  [%s] %-30s max %.2e  mean %.2e  over %zu\n", pass ? "PASS" : "FAIL", label, maxd, meand, nOver);
    if (!pass) g_fail++;
    clReleaseMemObject(srcBuf); clReleaseMemObject(dstBuf);
}

static void run(int W, int H, const SpeakParams& p, const char* label, int mode)
{
    runSrc(W, H, p, label, mode, makeFrame(W, H));
}
static void runHot(int W, int H, const SpeakParams& p, const char* label, int mode)
{
    runSrc(W, H, p, label, mode, makeHotFrame(W, H));
}

int main()
{
    printf("=== Speak OpenCL parity ===\n");
    cl_platform_id plat; if (clGetPlatformIDs(1, &plat, NULL) != CL_SUCCESS) { printf("no platform\n"); return 0; }
    cl_device_id dev;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) != CL_SUCCESS &&
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL) != CL_SUCCESS) { printf("no device\n"); return 0; }
    cl_int err;
    g_dev = dev;
    g_ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    g_q = clCreateCommandQueue(g_ctx, dev, 0, &err);
    const bool atomicsOK = atomicsWork();
    if (!atomicsOK) {
        char nm[128] = { 0 };
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(nm) - 1, nm, NULL);
        printf("  NOTE: this device's OpenCL global int32 atomics are BROKEN (%s).\n"
               "        The scope's measurement pass depends on them, so stats-dependent\n"
               "        cases are SKIPPED here. The kernel is verified on CPU + Metal;\n"
               "        OpenCL's real targets (Win/Linux NVIDIA/AMD/Intel) are unaffected.\n", nm);
    }
    const int W = 640, H = 480;

    { SpeakParams p = baseParams(); p.strength = 0.0f; run(W, H, p, "identity (strength 0)", 0); }
    { SpeakParams p = baseParams(); run(W, H, p, "tone neutral s1.0", 0); }
    { SpeakParams p = baseParams(); p.profile = stockProfile(); p.strength = 0.7f; run(W, H, p, "stock + printerLights s0.7", 0); }
    { SpeakParams p = baseParams(); p.inputColorSpace = SPEAK_CS_REC709_G24; p.profile = stockProfile(); run(W, H, p, "stock Rec709-in s1.0", 0); }
    { SpeakParams p = baseParams(); p.viewMode = SPEAK_VIEW_SPLIT; p.profile = stockProfile(); run(W, H, p, "split view s1.0", 0); }
    { SpeakParams p = baseParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; p.profile = stockProfile(); run(W, H, p, "bake Rec.709 s1.0", 2); }
    { SpeakParams p = baseParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; p.strength = 0.0f; run(W, H, p, "bake Rec.709 CST-only", 2); }
    { SpeakParams p = baseParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; p.viewMode = SPEAK_VIEW_SPLIT; p.profile = stockProfile(); run(W, H, p, "bake + split view", 2); }
    { SpeakParams p = baseParams(); p.strength = 0.0f; p.enableDye = 1;
      p.profile.subSat[0] = p.profile.subSat[1] = p.profile.subSat[2] = 0.55f;
      speakcore::setDyeCoupler(p.profile, 1.0f);
      p.profile.subSatKnee[0] = p.profile.subSatKnee[1] = p.profile.subSatKnee[2] = 2.2f;
      run(W, H, p, "subtractive color standalone", 0); }
    { SpeakParams p = baseParams(); p.profile = stockProfile(); p.enableDye = 1;
      p.profile.subSat[0] = p.profile.subSat[1] = p.profile.subSat[2] = 0.8f;
      speakcore::setDyeCoupler(p.profile, 0.7f);
      p.profile.subSatKnee[0] = p.profile.subSatKnee[1] = p.profile.subSatKnee[2] = 2.2f;
      run(W, H, p, "tone + subtractive color", 0); }
    // ---- halation: the scatter pyramid (atomics-free, so it runs everywhere) ----
    { SpeakParams p = halParams(); runHot(W, H, p, "halation s1.0 r1.0%", 0); }
    { SpeakParams p = halParams(); p.profile.halRadius = 4.0f; p.profile.halAmount = 0.9f;
      runHot(W, H, p, "halation wide r4.0% a0.9", 0); }
    { SpeakParams p = halParams(); p.profile.halRadius = 0.15f; runHot(W, H, p, "halation tight r0.15%", 0); }
    { SpeakParams p = halParams(); p.strength = 0.4f; runHot(W, H, p, "halation s0.4 (dry side = lr)", 0); }
    { SpeakParams p = halParams(); p.viewMode = SPEAK_VIEW_SCATTER; runHot(W, H, p, "scatter view", 0); }
    { SpeakParams p = halParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; runHot(W, H, p, "halation + bake Rec.709", 2); }
    { SpeakParams p = halParams(); p.enableDye = 1;
      p.profile.subSat[0] = p.profile.subSat[1] = p.profile.subSat[2] = 0.8f;
      speakcore::setDyeCoupler(p.profile, 0.7f);
      runHot(W, H, p, "halation + subtractive color", 0); }
    // enableOptics off => halAmountOf 0 => the whole chain is skipped; must be
    // bit-identical to the no-halation build (the identity gate on the GPU side).
    { SpeakParams p = halParams(); p.enableOptics = 0; runHot(W, H, p, "halation optics-off (skipped)", 0); }
    { SpeakParams p = halParams(); p.profile.halAmount = 0.0f; runHot(W, H, p, "halation amount 0 (skipped)", 0); }
    // Odd, non-power-of-two sizes: exercises the (w+1)/2 level ladder, the
    // clamped edge fetch and the per-level dispatch rounding.
    { SpeakParams p = halParams(); runHot(173, 97, p, "halation odd size 173x97", 0); }
    { SpeakParams p = halParams(); runHot(31, 19, p, "halation tiny 31x19", 0); }
    // Same params at two sizes: the size-dependent buffers must grow, not overrun
    // (the proxy -> full-res switch, on ONE cached queue).
    { SpeakParams p = halParams(); runHot(320, 240, p, "halation proxy 320x240", 0); }
    { SpeakParams p = halParams(); runHot(1024, 576, p, "halation full-res 1024x576", 0); }

    if (atomicsOK) {
        { SpeakParams p = baseParams(); p.scopeHD = 1; p.strength = 0.6f; p.profile = stockProfile(); runHot(W, H, p, "scope H&D on s0.6", 1); }
        // The density parade must measure the HALATED result — a scatter-blind
        // scope is a bug parity cannot catch, so this case only means anything
        // because the CPU reference it is compared against reads the plane too.
        { SpeakParams p = halParams(); p.scopeDensity = 1; runHot(W, H, p, "density scope + halation", 1); }
        { SpeakParams p = baseParams(); p.scopeDensity = 1; p.strength = 0.7f; p.profile = stockProfile(); runHot(W, H, p, "density scope on", 1); }
        { SpeakParams p = baseParams(); p.scopeHD = 1; p.scopeDensity = 1; p.strength = 0.7f; p.profile = stockProfile(); runHot(W, H, p, "both scopes on", 1); }
        // W and H both == 1 (mod 32). The stats kernel indexes x = gid*2, so the
        // dispatch needs ceil(W/2) sample threads; floor(W/2) falls exactly one
        // sample column short at these sizes and nowhere else (1921 and 3841 are
        // both 1 mod 32). Every OTHER scope case here runs at 640x480 — 0 mod 32
        // — and the one odd-sized case (173x97, whose 97 IS 1 mod 32) has no
        // scope on, so the stats pass never fires there. This suite was blind to
        // it. NOTE it still skips on Apple's OpenCL (broken atomics); it earns
        // its place on OpenCL's real targets (Win/Linux NVIDIA/AMD/Intel).
        { SpeakParams p = halParams(); p.scopeDensity = 1; p.scopeHD = 1;
          runHot(353, 225, p, "scopes at 353x225 (W,H = 1 mod 32)", 1); }
    } else {
        printf("  [SKIP] scope H&D / density / both    (device's OpenCL atomics are broken)\n");
    }

    printf("\n%s (%d failures)\n", g_fail ? "PARITY FAILED" : "PARITY GREEN", g_fail);
    return g_fail ? 1 : 0;
}
