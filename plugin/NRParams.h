// OpenNR — shared parameter block passed from the OFX plugin to GPU kernels.
// All members are 4 bytes so the struct layout is identical in C++, Metal,
// CUDA and OpenCL (no padding on any of them). A field added here must also
// be added to the struct declarations inside MetalKernel.mm and
// OpenCLKernel.cpp kernel sources (CUDA includes this header directly).

#ifndef OPENNR_NRPARAMS_H
#define OPENNR_NRPARAMS_H

typedef struct NRParams
{
    int   profileSource;   // 0 auto whole frame, 1 auto from region, 2 manual
    float sigmaY;
    float sigmaC;
    float profileAdjust;
    float regionCX;
    float regionCY;
    float regionSize;
    int   hasTemporalDiff;

    int   enableTemporal;
    int   temporalFrames;  // 3 or 5
    float temporalLuma;
    float temporalChroma;
    float motionThresh;

    int   enableSpatial;
    int   spatialMode;     // 0 faster (bilateral), 1 better (NLM)
    int   spatialRadius;   // 1..8
    float spatialLuma;
    float spatialChroma;
    float preserveDetail;
    float chromaBlotch;    // 0..1

    int   enableRefine;
    float shadowDesat;     // 0..1
    float desatRange;      // luma pivot
    float lumaTexture;     // 0..1
    float grainAmount;     // 0..1
    float grainSize;       // px
    float grainChroma;     // 0..1
    int   frameIndex;

    float master;          // 0..2
    int   viewMode;        // 0 result 1 split 2 input 3 after-temporal
                           // 4 noise-removed 5 analysis 6 activity 7 snr
                           // 8 noise matte (map in RGB + alpha)

    // ---- v3 ----
    int   motionTracking;  // temporal shift-search matching (0/1)
    int   fireflyRemoval;  // 3-frame temporal median impulse clip (0/1)
    float eqFine;          // Noise EQ: fine band gain, 0..2 (1 = v2.1)
    float eqMedium;        // Noise EQ: medium band amount, 0..1
    float eqCoarse;        // Noise EQ: coarse band LUMA amount, 0..1
    float deband;          // gradient-aware debanding, 0..1
    int   profileLocked;   // 1 = use lock* values instead of measuring
    float lockSY;          // locked input profile (spatial pair)
    float lockSC;
    float lockTY;          // locked temporal pair
    float lockTC;
    float lockGainY[16];   // locked brightness-dependent gains
    float lockGainC[16];
} NRParams;

// stats buffer layout (uint32 slots)
#define NR_STATS_HIST_YF    0        // input fine luma |laplacian| (256)
#define NR_STATS_HIST_CF    256
#define NR_STATS_HIST_Y2    512      // coarse (256)
#define NR_STATS_HIST_C2    768
#define NR_STATS_HIST_YT    1024     // |temporal diff| (256)
#define NR_STATS_HIST_CT    1280
#define NR_STATS_LUMA_Y     1536     // 16 luma bins x 64 sub-bins of |lapY|
#define NR_STATS_LUMA_C     2560     // same for chroma
#define NR_STATS_HIST_YR    3584     // residual (on tmp) luma (256)
#define NR_STATS_HIST_CR    3840
#define NR_STATS_HIST_EFFN  4096     // effN histogram (32)
#define NR_STATS_SIGMA_SY   4128     // float bits from here on
#define NR_STATS_SIGMA_SC   4129
#define NR_STATS_SIGMA_TY   4130
#define NR_STATS_SIGMA_TC   4131
#define NR_STATS_SIGMA_RY   4132
#define NR_STATS_SIGMA_RC   4133
#define NR_STATS_MEDBIN_Y   4134
#define NR_STATS_HISTMAX_Y  4135
#define NR_STATS_GAINY      4136     // 16 float-bit gains
#define NR_STATS_GAINC      4152     // 16 float-bit gains
#define NR_STATS_EFFN_MED   4168
#define NR_STATS_UINTS      4176

#endif // OPENNR_NRPARAMS_H
