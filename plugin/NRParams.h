// OpenNR — shared parameter block passed from the OFX plugin to GPU kernels.
// All members are 4 bytes so the struct layout is identical in C++, Metal,
// CUDA and OpenCL (no padding on any of them).

#ifndef OPENNR_NRPARAMS_H
#define OPENNR_NRPARAMS_H

typedef struct NRParams
{
    int   profileSource;   // 0 auto whole frame, 1 auto from region, 2 manual
    float sigmaY;          // manual luma sigma (signal units 0..1)
    float sigmaC;          // manual chroma sigma
    float profileAdjust;   // 0.25..4 multiplier on the auto estimate
    float regionCX;        // region center x, normalized
    float regionCY;        // region center y, normalized
    float regionSize;      // region edge / min(W,H)
    int   hasTemporalDiff; // 1 if a distinct neighbor frame is bound for the estimator

    int   enableTemporal;
    int   temporalFrames;  // 3 or 5
    float temporalLuma;    // 0..1
    float temporalChroma;  // 0..1
    float motionThresh;    // 0..1

    int   enableSpatial;
    int   spatialMode;     // 0 = faster (bilateral), 1 = better (NLM)
    int   spatialRadius;   // 1..5
    float spatialLuma;     // 0..1
    float spatialChroma;   // 0..1
    float preserveDetail;  // 0..1

    float master;          // 0..2
    int   viewMode;        // 0 result, 1 split, 2 noise, 3 analysis, 4 temporal map
} NRParams;

// stats buffer layout (uint32 slots)
#define NR_STATS_HIST_YF   0        // fine luma |laplacian| histogram
#define NR_STATS_HIST_CF   256      // fine chroma
#define NR_STATS_HIST_Y2   512      // coarse luma
#define NR_STATS_HIST_C2   768      // coarse chroma
#define NR_STATS_HIST_YT   1024     // luma |temporal diff|
#define NR_STATS_HIST_CT   1280     // chroma |temporal diff|
#define NR_STATS_SIGMA_SY  1536     // float bits, written by finalize kernel
#define NR_STATS_SIGMA_SC  1537
#define NR_STATS_SIGMA_TY  1538
#define NR_STATS_SIGMA_TC  1539
#define NR_STATS_MEDBIN_Y  1540     // median bin of fine luma hist (HUD marker)
#define NR_STATS_HISTMAX_Y 1541     // max bin count of fine luma hist (HUD scale)
#define NR_STATS_UINTS     1544

#endif // OPENNR_NRPARAMS_H
