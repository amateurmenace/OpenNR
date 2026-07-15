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
    int   temporalFrames;  // 3, 5 or 7 (v3.3)
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
    float eqFine;          // Noise EQ: fine band gain, 0..3 (1 = v2.1)
    float eqMedium;        // Noise EQ: medium band amount, 0..1.5
    float eqCoarse;        // Noise EQ: coarse band LUMA amount, 0..1.5
    float deband;          // gradient-aware debanding, 0..1
    int   profileLocked;   // 1 = use lock* values instead of measuring
    float lockSY;          // locked input profile (spatial pair)
    float lockSC;
    float lockTY;          // locked temporal pair
    float lockTC;
    float lockGainY[16];   // locked brightness-dependent gains
    float lockGainC[16];

    // ---- v3.1 ----
    float detailRescue;    // 0..1: restore fine-band changes beyond noise size
    int   scopeMeasure;    // overlay: Noise Analysis panel (0/1)
    int   scopeMotion;     // overlay: temporal-activity mini map (0/1)
    int   scopeEq;         // overlay: Noise EQ panel (0/1)

    // ---- v3.2 ----
    int   ghostGuard;      // temporal signed-mean coherence gate (0/1)
    float globalBlend;     // 0..1 final crossfade original -> result

    // ---- v3.3 ----
    int   deepClean;       // fine-NLM pre-pass at 0.6h before the main
                           // spatial stage (0/1)
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
#define NR_STATS_HIST_EFFN  4096     // effN histogram (64; v3.3 — was 32,
                                     // which saturated at effN 4.875 and the
                                     // 7-frame stack reaches ~6.7; everything
                                     // below shifted +32)
#define NR_STATS_HIST_YR2   4208     // v3.2: coarse residual (2x2 blocks, 256)
#define NR_STATS_HIST_CR2   4464
#define NR_STATS_SIGMA_SY   4160     // float bits from here on
#define NR_STATS_SIGMA_SC   4161
#define NR_STATS_SIGMA_TY   4162
#define NR_STATS_SIGMA_TC   4163
#define NR_STATS_SIGMA_RY   4164
#define NR_STATS_SIGMA_RC   4165
#define NR_STATS_MEDBIN_Y   4166
#define NR_STATS_HISTMAX_Y  4167
#define NR_STATS_GAINY      4168     // 16 float-bit gains
#define NR_STATS_GAINC      4184     // 16 float-bit gains
#define NR_STATS_EFFN_MED   4200
#define NR_STATS_FINE_Y     4201     // v3.1: per-band estimates for the EQ
#define NR_STATS_FINE_C     4202     //       scope (float bits)
#define NR_STATS_COARSE_Y   4203
#define NR_STATS_UINTS      4720     // v3.3: effN histogram grew 32 -> 64

#endif // OPENNR_NRPARAMS_H
