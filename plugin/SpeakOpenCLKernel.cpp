// Speak — OpenCL implementation of the film-reconstruction pipeline.
// Line-by-line port of plugin/speak_core.h; keep the two in sync. This is the
// render path Resolve uses on NVIDIA/AMD/Intel when CUDA isn't advertised, and
// the primary path on Windows/Linux.

#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include <cstdio>
#include <map>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "SpeakParams.h"

static const char* kSpeakKernelSource = R"CLC(

typedef struct SpeakProfile
{
    float negDmin[3];   float negDmax[3];   float negGamma[3];
    float negToe[3];    float negShoulder[3]; float negSpeed[3];
    float printerLights[3]; float printerMaster;
    float prnDmin[3];   float prnDmax[3];   float prnGamma[3];
    float prnToe[3];    float prnShoulder[3]; float prnSpeed[3];
    float dyeCouple[9]; float subSat[3];    float subSatKnee[3];
    float splitShadow[3]; float splitHigh[3]; float splitPivot; float splitBalance;
    float systemGamma;  int residualLUT;    int profileVersion; int _pad0;
} SpeakProfile;

typedef struct SpeakParams
{
    int inputColorSpace; int outputMode; int grainRef; float strength;
    int frameIndex; int viewMode;
    int enableTone; int enableDye; int enableSplit; int enableOptics;
    int scopeHD; int scopeDensity; int scopeVector;
    SpeakProfile profile;
} SpeakParams;

#define kLog10_2 0.301029996f
#define k18Gray  0.18f
#define kPrinterPt 0.025f
#define kLinTiny 1e-8f
#define kKneeMin 0.05f
#define kDI_A 0.0075f
#define kDI_B 7.0f
#define kDI_C 0.07329248f
#define kDI_M 10.44426855f
#define kDI_LIN_CUT 0.00262409f
#define kDI_LOG_CUT 0.02740668f

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float pow10f(float x) { return exp2(x * 3.32192809f); }

inline float softplusf(float z)
{
    float az = z < 0.0f ? -z : z;
    return (z > 0.0f ? z : 0.0f) + log(1.0f + exp(-az));
}

inline float diDecode(float v)
{
    return (v <= kDI_LOG_CUT) ? (v / kDI_M) : (exp2(v / kDI_C - kDI_B) - kDI_A);
}
inline float diEncode(float L)
{
    return (L <= kDI_LIN_CUT) ? (L * kDI_M) : ((log2(L + kDI_A) + kDI_B) * kDI_C);
}
inline float acesCctDecode(float v)
{
    if (v <= 0.155251141552511f) return (v - 0.0729055341958355f) / 10.5402377416545f;
    return exp2(v * 17.52f - 9.72f);
}
inline float acesCctEncode(float L)
{
    if (L <= 0.0078125f) return 10.5402377416545f * L + 0.0729055341958355f;
    return (log2(L) + 9.72f) / 17.52f;
}
inline float decodeToLinear(int cs, float v)
{
    if (cs == 0) return diDecode(v);
    if (cs == 1) return v <= 0.0f ? 0.0f : pow(v, 2.4f);
    if (cs == 3) return acesCctDecode(v);
    return v;
}
inline float encodeFromLinear(int cs, float L)
{
    if (cs == 0) return diEncode(L);
    if (cs == 1) return L <= 0.0f ? 0.0f : pow(L, 1.0f / 2.4f);
    if (cs == 3) return acesCctEncode(L);
    return L;
}

__constant float kDWG_to_XYZ[9] = {
    0.70062239f, 0.14877482f, 0.10105872f,
    0.27411851f, 0.87363190f,-0.14775041f,
   -0.09896291f,-0.13789533f, 1.32591599f };
__constant float kXYZ_to_Rec709[9] = {
    3.24045420f,-1.53713850f,-0.49853140f,
   -0.96926600f, 1.87601080f, 0.04155600f,
    0.05564340f,-0.20402590f, 1.05722520f };
inline void mul3(__constant float* m, float r, float g, float b,
                 float* oR, float* oG, float* oB)
{
    *oR = m[0] * r + m[1] * g + m[2] * b;
    *oG = m[3] * r + m[4] * g + m[5] * b;
    *oB = m[6] * r + m[7] * g + m[8] * b;
}
inline void gamutToRec709Lin(int cs, float r, float g, float b,
                             float* oR, float* oG, float* oB)
{
    if (cs == 0 || cs == 2) {
        float X, Y, Z;
        mul3(kDWG_to_XYZ, r, g, b, &X, &Y, &Z);
        mul3(kXYZ_to_Rec709, X, Y, Z, oR, oG, oB);
    } else {
        *oR = r; *oG = g; *oB = b;
    }
}

inline float hdCurve(float logH, float Dmin, float Dmax, float gamma,
                     float toe, float shoulder, float speed)
{
    float t = toe < kKneeMin ? kKneeMin : toe;
    float s = shoulder < kKneeMin ? kKneeMin : shoulder;
    float d1 = Dmin + (gamma / t) * softplusf(t * (logH - speed));
    return Dmax - (1.0f / s) * softplusf(s * (Dmax - d1));
}
inline float chainDensity(float stops, int ch, const SpeakProfile* p)
{
    float logH = stops * kLog10_2;
    float Dneg = hdCurve(logH, p->negDmin[ch], p->negDmax[ch], p->negGamma[ch],
                         p->negToe[ch], p->negShoulder[ch], p->negSpeed[ch]);
    float printerOff = (p->printerMaster + p->printerLights[ch]) * kPrinterPt;
    float logHprn = -Dneg + printerOff;
    return hdCurve(logHprn, p->prnDmin[ch], p->prnDmax[ch], p->prnGamma[ch],
                   p->prnToe[ch], p->prnShoulder[ch], p->prnSpeed[ch]);
}
inline float toneChannel(float lin, int ch, const SpeakProfile* p)
{
    float stops = log2((lin < kLinTiny ? kLinTiny : lin) / k18Gray);
    float Dprn = chainDensity(stops, ch, p);
    float Dref = chainDensity(0.0f, ch, p);
    return k18Gray * pow10f(-(Dprn - Dref));
}
inline float scopeYStops(float inStops, int ch, const SpeakParams* pr)
{
    float lin = k18Gray * exp2(inStops);
    float outLin = lin;
    if ((pr->enableTone != 0) && (pr->strength > 0.0f)) {
        float s = clampf(pr->strength, 0.0f, 1.0f);
        outLin = lerpf(lin, toneChannel(lin, ch, &pr->profile), s);
    }
    return log2((outLin < kLinTiny ? kLinTiny : outLin) / k18Gray);
}

inline bool hdScopePixel(int x, int y, int W, int H, const SpeakParams* pr,
                         float* outR, float* outG, float* outB)
{
    if (pr->scopeHD == 0) return false;
    int sc = (H / 540) > 1 ? (H / 540) : 1;
    int panelW = 220 * sc, panelH = 150 * sc;
    int margin = 12 * sc;
    int px0 = margin, py0 = margin;
    int yd = H - 1 - y;
    int lx = x - px0, ly = yd - py0;
    if (lx < 0 || ly < 0 || lx >= panelW || ly >= panelH) return false;

    int pad = 6 * sc;
    int plotW = panelW - 2 * pad, plotH = panelH - 2 * pad;
    int gx = lx - pad, gy = ly - pad;

    *outR = 0.06f; *outG = 0.06f; *outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        *outR = 0.30f; *outG = 0.30f; *outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    float rowStops = 6.0f - 12.0f * ((float)gy / (plotH - 1));
    int gcol0 = (int)((0.0f + 6.0f) / 12.0f * (plotW - 1) + 0.5f);
    int grow0 = (int)((6.0f - 0.0f) / 12.0f * (plotH - 1) + 0.5f);
    if (gx == gcol0 || gy == grow0) { *outR = 0.24f; *outG = 0.24f; *outB = 0.24f; return true; }
    if ((gx % (plotW / 6)) == 0 || (gy % (plotH / 6)) == 0) { *outR = 0.13f; *outG = 0.13f; *outB = 0.13f; }

    int chR[3]; chR[0]=1; chR[1]=0; chR[2]=0;
    int chG[3]; chG[0]=0; chG[1]=1; chG[2]=0;
    int chB[3]; chB[0]=0; chB[1]=0; chB[2]=1;
    for (int ch = 0; ch < 3; ++ch) {
        float inS  = -6.0f + 12.0f * ((float)gx       / (plotW - 1));
        float inS2 = -6.0f + 12.0f * ((float)(gx + 1) / (plotW - 1));
        float y0 = scopeYStops(inS,  ch, pr);
        float y1 = scopeYStops(inS2, ch, pr);
        if (y0 > y1) { float tt = y0; y0 = y1; y1 = tt; }
        float lo = y1 < y0 ? y1 : y0, hi = y1 > y0 ? y1 : y0;
        if (rowStops <= hi + 0.09f && rowStops >= lo - 0.09f) {
            *outR = 0.10f + 0.85f * chR[ch];
            *outG = 0.10f + 0.85f * chG[ch];
            *outB = 0.10f + 0.85f * chB[ch];
            return true;
        }
    }
    if (gy >= plotH - 5 * sc && gy < plotH - 1 * sc) {
        int sw = gx / (6 * sc);
        if (gx % (6 * sc) < 4 * sc) {
            if (sw == 0) { *outR = 0.95f; *outG = 0.10f; *outB = 0.10f; return true; }
            if (sw == 1) { *outR = 0.10f; *outG = 0.95f; *outB = 0.10f; return true; }
            if (sw == 2) { *outR = 0.10f; *outG = 0.10f; *outB = 0.95f; return true; }
        }
    }
    return true;
}

inline void processPixel(float r, float g, float b, int x, int y, int W, int H,
                         const SpeakParams* pr, float* outR, float* outG, float* outB)
{
    int cs = pr->inputColorSpace;
    bool toneOn = (pr->enableTone != 0) && (pr->strength > 0.0f);
    bool bake = (pr->outputMode == 1);
    if (!toneOn && !bake) {
        *outR = r; *outG = g; *outB = b;
    } else {
        float lr = decodeToLinear(cs, r);
        float lg = decodeToLinear(cs, g);
        float lb = decodeToLinear(cs, b);
        float mr = lr, mg = lg, mb = lb;
        if (toneOn) {
            float s = clampf(pr->strength, 0.0f, 1.0f);
            mr = lerpf(lr, toneChannel(lr, 0, &pr->profile), s);
            mg = lerpf(lg, toneChannel(lg, 1, &pr->profile), s);
            mb = lerpf(lb, toneChannel(lb, 2, &pr->profile), s);
        }
        if (bake) {
            float rr, rg, rb;
            gamutToRec709Lin(cs, mr, mg, mb, &rr, &rg, &rb);
            rr = rr < 0.0f ? 0.0f : rr;
            rg = rg < 0.0f ? 0.0f : rg;
            rb = rb < 0.0f ? 0.0f : rb;
            *outR = encodeFromLinear(1, rr);
            *outG = encodeFromLinear(1, rg);
            *outB = encodeFromLinear(1, rb);
        } else {
            *outR = encodeFromLinear(cs, mr);
            *outG = encodeFromLinear(cs, mg);
            *outB = encodeFromLinear(cs, mb);
        }
    }
    if (pr->viewMode == 2) { *outR = r; *outG = g; *outB = b; }
    else if (pr->viewMode == 1 && x < W / 2) { *outR = r; *outG = g; *outB = b; }

    float sr, sg, sb;
    if (hdScopePixel(x, y, W, H, pr, &sr, &sg, &sb)) { *outR = sr; *outG = sg; *outB = sb; }
}

__kernel void SpeakKernel(SpeakParams p, int W, int H,
                          __global const float* src, __global float* dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int i = (y * W + x) * 4;
    float oR, oG, oB;
    processPixel(src[i + 0], src[i + 1], src[i + 2], x, y, W, H, &p, &oR, &oG, &oB);
    dst[i + 0] = oR; dst[i + 1] = oG; dst[i + 2] = oB; dst[i + 3] = src[i + 3];
}
)CLC";

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
namespace {

void SpeakCheck(cl_int e, const char* m) { if (e != CL_SUCCESS) fprintf(stderr, "Speak: %s [%d]\n", m, e); }

class SpeakLocker
{
public:
#ifdef _WIN64
    SpeakLocker() { InitializeCriticalSection(&m); }
    ~SpeakLocker() { DeleteCriticalSection(&m); }
    void Lock() { EnterCriticalSection(&m); }
    void Unlock() { LeaveCriticalSection(&m); }
    CRITICAL_SECTION m;
#else
    SpeakLocker() { pthread_mutex_init(&m, NULL); }
    ~SpeakLocker() { pthread_mutex_destroy(&m); }
    void Lock() { pthread_mutex_lock(&m); }
    void Unlock() { pthread_mutex_unlock(&m); }
    pthread_mutex_t m;
#endif
};

struct SpeakRes { cl_kernel k = NULL; };

} // namespace

void RunOpenCLSpeak(void* p_CmdQ, int p_Width, int p_Height,
                    const SpeakParams& p_Params, const float* p_Src, float* p_Dst)
{
    cl_int error;
    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    static std::map<void*, SpeakRes> s_res;
    static SpeakLocker s_lock;

    cl_context clContext = NULL;
    error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
    SpeakCheck(error, "get context");

    SpeakParams params = p_Params;

    cl_kernel kernel = NULL;
    s_lock.Lock();
    {
        SpeakRes& r = s_res[p_CmdQ];
        if (!r.k) {
            cl_program program = clCreateProgramWithSource(clContext, 1, &kSpeakKernelSource, NULL, &error);
            SpeakCheck(error, "create program");
            error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (error != CL_SUCCESS) {
                cl_device_id dev = NULL;
                clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &dev, NULL);
                char log[16384] = { 0 };
                clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
                fprintf(stderr, "Speak OpenCL build log:\n%s\n", log);
                s_lock.Unlock();
                return;
            }
            r.k = clCreateKernel(program, "SpeakKernel", &error);
            SpeakCheck(error, "create kernel");
        }
        kernel = r.k;
    }
    s_lock.Unlock();

    cl_mem src = reinterpret_cast<cl_mem>(const_cast<float*>(p_Src));
    cl_mem dst = reinterpret_cast<cl_mem>(p_Dst);
    int W = p_Width, H = p_Height;
    int c = 0;
    error  = clSetKernelArg(kernel, c++, sizeof(SpeakParams), &params);
    error |= clSetKernelArg(kernel, c++, sizeof(int), &W);
    error |= clSetKernelArg(kernel, c++, sizeof(int), &H);
    error |= clSetKernelArg(kernel, c++, sizeof(cl_mem), &src);
    error |= clSetKernelArg(kernel, c++, sizeof(cl_mem), &dst);
    SpeakCheck(error, "set args");

    const size_t local[2]  = { 16, 16 };
    const size_t global[2] = { static_cast<size_t>((p_Width + 15) / 16) * 16,
                               static_cast<size_t>((p_Height + 15) / 16) * 16 };
    error = clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, global, local, 0, NULL, NULL);
    SpeakCheck(error, "enqueue");
}
