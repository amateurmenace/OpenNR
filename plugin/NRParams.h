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
    float lockSCr;         // locked Cr sigmas (B5 per-channel chroma:
    float lockTCr;         // lockSC/lockTC hold the Cb pair; old projects
                           // load with Cr = Cb)
} NRParams;

// stats buffer layout (uint32 slots). v3.3 B5 re-layout: every chroma
// histogram splits into a Cb/Cr pair (blue-channel night noise reads very
// differently on the two axes), and ALL residual histograms are now
// contiguous — [HIST_YR, SIGMA_SY) is one zeroable range for the Deep Clean
// re-measure. Bump carefully: this table is re-declared inside the Metal and
// OpenCL kernel sources (CUDA includes this header).
#define NR_STATS_HIST_YF    0        // input fine luma |laplacian| (256)
#define NR_STATS_HIST_CFB   256      // input fine |lap Cb| (256)
#define NR_STATS_HIST_CFR   512      // input fine |lap Cr| (256)
#define NR_STATS_HIST_Y2    768      // coarse luma (256)
#define NR_STATS_HIST_C2B   1024     // coarse Cb / Cr
#define NR_STATS_HIST_C2R   1280
#define NR_STATS_HIST_YT    1536     // |temporal diff| luma (256)
#define NR_STATS_HIST_CTB   1792     // |temporal diff| Cb / Cr
#define NR_STATS_HIST_CTR   2048
#define NR_STATS_LUMA_Y     2304     // 16 luma bins x 64 sub-bins of |lapY|
#define NR_STATS_LUMA_C     3328     // same for chroma (both channels feed it)
#define NR_STATS_HIST_YR    4352     // residual (on tmp) luma (256)
#define NR_STATS_HIST_CRB   4608     // residual Cb / Cr
#define NR_STATS_HIST_CRR   4864
#define NR_STATS_HIST_EFFN  5120     // effN histogram (64)
#define NR_STATS_HIST_YR2   5184     // v3.2: coarse residual (2x2 blocks, 256)
#define NR_STATS_HIST_CR2B  5440     // coarse residual Cb / Cr
#define NR_STATS_HIST_CR2R  5696
#define NR_STATS_SIGMA_SY   5952     // float bits from here on
#define NR_STATS_SIGMA_SCB  5953
#define NR_STATS_SIGMA_SCR  5954
#define NR_STATS_SIGMA_TY   5955
#define NR_STATS_SIGMA_TCB  5956
#define NR_STATS_SIGMA_TCR  5957
#define NR_STATS_SIGMA_RY   5958
#define NR_STATS_SIGMA_RCB  5959
#define NR_STATS_SIGMA_RCR  5960
#define NR_STATS_MEDBIN_Y   5961
#define NR_STATS_HISTMAX_Y  5962
#define NR_STATS_GAINY      5963     // 16 float-bit gains
#define NR_STATS_GAINC      5979     // 16 float-bit gains
#define NR_STATS_EFFN_MED   5995
#define NR_STATS_FINE_Y     5996     // v3.1: per-band estimates for the EQ
#define NR_STATS_FINE_C     5997     //       scope (float bits; Cb/Cr mean)
#define NR_STATS_COARSE_Y   5998
#define NR_STATS_UINTS      5999

#endif // OPENNR_NRPARAMS_H
