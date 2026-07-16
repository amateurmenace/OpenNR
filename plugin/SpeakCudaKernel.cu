// Speak — CUDA implementation of the film-reconstruction pipeline (NVIDIA).
// Line-by-line port of plugin/speak_core.h; keep the two in sync. Like Hush's
// CudaKernel.cu this cannot be RUN on the dev Mac — it is a faithful textual
// port, syntax/type-checked via test/check_cuda_syntax.sh and treated as
// unverified until it has run on real NVIDIA hardware (CUDA stays OFF in the
// Windows build; Resolve renders Speak via OpenCL there).

#include <cuda_runtime.h>

#include <cstdio>
#include <map>

#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include "SpeakParams.h"

namespace {

#define kLog10_2   0.301029996f
#define k18Gray    0.18f
#define kPrinterPt 0.025f
#define kLinTiny   1e-8f
#define kKneeMin   0.05f
#define kDI_A      0.0075f
#define kDI_B      7.0f
#define kDI_C      0.07329248f
#define kDI_M      10.44426855f
#define kDI_LIN_CUT 0.00262409f
#define kDI_LOG_CUT 0.02740668f

__device__ inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline float pow10f(float x) { return exp2f(x * 3.32192809f); }

__device__ inline float softplusf(float z)
{
    float az = z < 0.0f ? -z : z;
    return (z > 0.0f ? z : 0.0f) + logf(1.0f + expf(-az));
}

__device__ inline float diDecode(float v)
{
    return (v <= kDI_LOG_CUT) ? (v / kDI_M) : (exp2f(v / kDI_C - kDI_B) - kDI_A);
}
__device__ inline float diEncode(float L)
{
    return (L <= kDI_LIN_CUT) ? (L * kDI_M) : ((log2f(L + kDI_A) + kDI_B) * kDI_C);
}
__device__ inline float acesCctDecode(float v)
{
    if (v <= 0.155251141552511f) return (v - 0.0729055341958355f) / 10.5402377416545f;
    return exp2f(v * 17.52f - 9.72f);
}
__device__ inline float acesCctEncode(float L)
{
    if (L <= 0.0078125f) return 10.5402377416545f * L + 0.0729055341958355f;
    return (log2f(L) + 9.72f) / 17.52f;
}
__device__ inline float decodeToLinear(int cs, float v)
{
    if (cs == 0) return diDecode(v);
    if (cs == 1) return v <= 0.0f ? 0.0f : powf(v, 2.4f);
    if (cs == 3) return acesCctDecode(v);
    return v;
}
__device__ inline float encodeFromLinear(int cs, float L)
{
    if (cs == 0) return diEncode(L);
    if (cs == 1) return L <= 0.0f ? 0.0f : powf(L, 1.0f / 2.4f);
    if (cs == 3) return acesCctEncode(L);
    return L;
}

__device__ inline float hdCurve(float logH, float Dmin, float Dmax, float gamma,
                                float toe, float shoulder, float speed)
{
    float t = toe < kKneeMin ? kKneeMin : toe;
    float s = shoulder < kKneeMin ? kKneeMin : shoulder;
    float d1 = Dmin + (gamma / t) * softplusf(t * (logH - speed));
    return Dmax - (1.0f / s) * softplusf(s * (Dmax - d1));
}
__device__ inline float chainDensity(float stops, int ch, const SpeakProfile& p)
{
    float logH = stops * kLog10_2;
    float Dneg = hdCurve(logH, p.negDmin[ch], p.negDmax[ch], p.negGamma[ch],
                         p.negToe[ch], p.negShoulder[ch], p.negSpeed[ch]);
    float printerOff = (p.printerMaster + p.printerLights[ch]) * kPrinterPt;
    float logHprn = -Dneg + printerOff;
    return hdCurve(logHprn, p.prnDmin[ch], p.prnDmax[ch], p.prnGamma[ch],
                   p.prnToe[ch], p.prnShoulder[ch], p.prnSpeed[ch]);
}
__device__ inline float toneChannel(float lin, int ch, const SpeakProfile& p)
{
    float stops = log2f((lin < kLinTiny ? kLinTiny : lin) / k18Gray);
    float Dprn = chainDensity(stops, ch, p);
    float Dref = chainDensity(0.0f, ch, p);
    return k18Gray * pow10f(-(Dprn - Dref));
}
__device__ inline float scopeYStops(float inStops, int ch, const SpeakParams& pr)
{
    float lin = k18Gray * exp2f(inStops);
    float outLin = lin;
    if ((pr.enableTone != 0) && (pr.strength > 0.0f)) {
        float s = clampf(pr.strength, 0.0f, 1.0f);
        outLin = lerpf(lin, toneChannel(lin, ch, pr.profile), s);
    }
    return log2f((outLin < kLinTiny ? kLinTiny : outLin) / k18Gray);
}

__device__ inline bool hdScopePixel(int x, int y, int W, int H, const SpeakParams& pr,
                                    float& outR, float& outG, float& outB)
{
    if (pr.scopeHD == 0) return false;
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

    outR = 0.06f; outG = 0.06f; outB = 0.06f;
    if (lx < sc || ly < sc || lx >= panelW - sc || ly >= panelH - sc) {
        outR = 0.30f; outG = 0.30f; outB = 0.30f; return true;
    }
    if (gx < 0 || gy < 0 || gx >= plotW || gy >= plotH) return true;

    float rowStops = 6.0f - 12.0f * ((float)gy / (plotH - 1));
    int gcol0 = (int)((0.0f + 6.0f) / 12.0f * (plotW - 1) + 0.5f);
    int grow0 = (int)((6.0f - 0.0f) / 12.0f * (plotH - 1) + 0.5f);
    if (gx == gcol0 || gy == grow0) { outR = 0.24f; outG = 0.24f; outB = 0.24f; return true; }
    if ((gx % (plotW / 6)) == 0 || (gy % (plotH / 6)) == 0) { outR = 0.13f; outG = 0.13f; outB = 0.13f; }

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
            outR = 0.10f + 0.85f * chR[ch];
            outG = 0.10f + 0.85f * chG[ch];
            outB = 0.10f + 0.85f * chB[ch];
            return true;
        }
    }
    if (gy >= plotH - 5 * sc && gy < plotH - 1 * sc) {
        int sw = gx / (6 * sc);
        if (gx % (6 * sc) < 4 * sc) {
            if (sw == 0) { outR = 0.95f; outG = 0.10f; outB = 0.10f; return true; }
            if (sw == 1) { outR = 0.10f; outG = 0.95f; outB = 0.10f; return true; }
            if (sw == 2) { outR = 0.10f; outG = 0.10f; outB = 0.95f; return true; }
        }
    }
    return true;
}

__device__ inline void processPixel(float r, float g, float b, int x, int y, int W, int H,
                                    const SpeakParams& pr, float& outR, float& outG, float& outB)
{
    bool toneOn = (pr.enableTone != 0) && (pr.strength > 0.0f);
    if (!toneOn) {
        outR = r; outG = g; outB = b;
    } else {
        int cs = pr.inputColorSpace;
        float lr = decodeToLinear(cs, r);
        float lg = decodeToLinear(cs, g);
        float lb = decodeToLinear(cs, b);
        float tr = toneChannel(lr, 0, pr.profile);
        float tg = toneChannel(lg, 1, pr.profile);
        float tb = toneChannel(lb, 2, pr.profile);
        float s = clampf(pr.strength, 0.0f, 1.0f);
        float mr = lerpf(lr, tr, s);
        float mg = lerpf(lg, tg, s);
        float mb = lerpf(lb, tb, s);
        outR = encodeFromLinear(cs, mr);
        outG = encodeFromLinear(cs, mg);
        outB = encodeFromLinear(cs, mb);
    }
    if (pr.viewMode == 2) { outR = r; outG = g; outB = b; }
    else if (pr.viewMode == 1 && x < W / 2) { outR = r; outG = g; outB = b; }

    float sr, sg, sb;
    if (hdScopePixel(x, y, W, H, pr, sr, sg, sb)) { outR = sr; outG = sg; outB = sb; }
}

__global__ void SpeakKernel(SpeakParams p, int W, int H,
                            const float4* src, float4* dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
    float4 s = src[i];
    float oR, oG, oB;
    processPixel(s.x, s.y, s.z, x, y, W, H, p, oR, oG, oB);
    float4 o; o.x = oR; o.y = oG; o.z = oB; o.w = s.w;
    dst[i] = o;
}

} // namespace

void RunCudaSpeak(void* p_Stream, int p_Width, int p_Height,
                  const SpeakParams& p_Params, const float* p_Src, float* p_Dst)
{
    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);
    dim3 block(16, 16, 1);
    dim3 grid((p_Width + block.x - 1) / block.x, (p_Height + block.y - 1) / block.y, 1);
    SpeakKernel<<<grid, block, 0, stream>>>(p_Params, p_Width, p_Height,
                                            reinterpret_cast<const float4*>(p_Src),
                                            reinterpret_cast<float4*>(p_Dst));
}
