// OpenNR — clip analysis, profile locking and Auto Setup mapping (v3).
//
// CPU-side only (host plugin + tests); nothing here has a GPU mirror. The
// per-frame measurement is nrcore::estimateInput — this header aggregates
// those measurements robustly across frames, serializes locked profiles
// bit-exactly, and maps measurements to slider settings.
//
// mapAnalysisToSettings is a PURE function of the aggregate, unit-tested with
// golden cases in test/test_denoise.cpp; the plugin only shuttles its output
// into the visible params. Settings come out in UI units (percent sliders,
// frame counts) — divide by 100 where nrcore::Params wants 0..1.
//
// MIT License — see LICENSE.

#ifndef OPENNR_NR_ANALYZE_H
#define OPENNR_NR_ANALYZE_H

#include "nr_core.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace nranalyze {

// ---------------------------------------------------------------------------
// Aggregate of per-frame measurements
// ---------------------------------------------------------------------------
struct ClipAggregate {
    float sy = 0.02f, sc = 0.02f;      // median input sigmas (spatial family)
    float ty = 0.02f, tc = 0.02f;      // median temporal-gating sigmas
    float gainY[16];                   // averaged brightness gains
    float gainC[16];
    float motion = 0.0f;               // mean motion energy, >= 0 (0 = static)
    float coarseRatioY = 1.0f;         // median coarse/fine luma sigma ratio
    float chromaRatio = 1.0f;          // tc/ty (chroma-to-luma noise)
    int   frames = 0;                  // frames measured
    ClipAggregate()
    {
        for (int i = 0; i < 16; ++i) { gainY[i] = 1.0f; gainC[i] = 1.0f; }
    }
};

inline float medianOf(std::vector<float> v)
{
    if (v.empty())
        return 0.0f;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

inline ClipAggregate aggregateClipStats(const std::vector<nrcore::Stats>& all)
{
    ClipAggregate a;
    a.frames = static_cast<int>(all.size());
    if (all.empty())
        return a;

    std::vector<float> sy, sc, ty, tc, cr;
    float motionSum = 0.0f;
    int motionN = 0;
    for (size_t i = 0; i < all.size(); ++i) {
        const nrcore::Stats& s = all[i];
        sy.push_back(s.sy);
        sc.push_back(s.sc);
        ty.push_back(s.ty);
        tc.push_back(s.tc);
        cr.push_back(s.coarseY / std::max(s.fineY, 1e-6f));
        if (s.hadTemporal) {
            motionSum += std::max(0.0f, s.motionRatio - 1.0f);
            motionN++;
        }
        for (int b = 0; b < 16; ++b) {
            a.gainY[b] += s.gainY[b];
            a.gainC[b] += s.gainC[b];
        }
    }
    // gains started at 1.0 in the constructor; fold that into the average
    for (int b = 0; b < 16; ++b) {
        a.gainY[b] = (a.gainY[b] - 1.0f) / static_cast<float>(all.size());
        a.gainC[b] = (a.gainC[b] - 1.0f) / static_cast<float>(all.size());
    }
    a.sy = medianOf(sy);
    a.sc = medianOf(sc);
    a.ty = medianOf(ty);
    a.tc = medianOf(tc);
    a.coarseRatioY = medianOf(cr);
    a.motion = (motionN > 0) ? motionSum / static_cast<float>(motionN) : 0.0f;
    a.chromaRatio = a.tc / std::max(a.ty, 1e-6f);
    return a;
}

// ---------------------------------------------------------------------------
// Locked-profile serialization (bit-exact: floats stored as hex bit patterns)
// ---------------------------------------------------------------------------
inline std::string formatLockedProfile(const ClipAggregate& a)
{
    std::string s = "HUSHLOCK1";
    char buf[16];
    const float vals[4] = { a.sy, a.sc, a.ty, a.tc };
    for (int i = 0; i < 4; ++i) {
        uint32_t u;
        std::memcpy(&u, &vals[i], 4);
        snprintf(buf, sizeof(buf), ",%08x", u);
        s += buf;
    }
    for (int b = 0; b < 32; ++b) {
        const float g = (b < 16) ? a.gainY[b] : a.gainC[b - 16];
        uint32_t u;
        std::memcpy(&u, &g, 4);
        snprintf(buf, sizeof(buf), ",%08x", u);
        s += buf;
    }
    return s;
}

inline bool parseLockedProfile(const std::string& s, ClipAggregate& out)
{
    if (s.compare(0, 9, "HUSHLOCK1") != 0)
        return false;
    float vals[36];
    size_t pos = 9;
    for (int i = 0; i < 36; ++i) {
        if (pos >= s.size() || s[pos] != ',')
            return false;
        char* end = nullptr;
        const unsigned long u = strtoul(s.c_str() + pos + 1, &end, 16);
        if (!end || end == s.c_str() + pos + 1)
            return false;
        const uint32_t u32 = static_cast<uint32_t>(u);
        std::memcpy(&vals[i], &u32, 4);
        pos = static_cast<size_t>(end - s.c_str());
    }
    out.sy = vals[0];
    out.sc = vals[1];
    out.ty = vals[2];
    out.tc = vals[3];
    for (int b = 0; b < 16; ++b) {
        out.gainY[b] = vals[4 + b];
        out.gainC[b] = vals[20 + b];
    }
    return true;
}

// ---------------------------------------------------------------------------
// Auto Setup mapping policy — pure function, golden-tested
// ---------------------------------------------------------------------------
// Noise classes by luma sigma (fraction of full range):
//   0 clean (<0.8%), 1 moderate (0.8-2.5%), 2 noisy (2.5-6%), 3 severe (>6%)
inline float clampfLocal(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

struct AutoSettings {
    int   enableTemporal  = 1;
    int   temporalFrames  = 3;      // 3, 5 or 7 (v3.3)
    float temporalLuma    = 60.0f;  // UI percent
    float temporalChroma  = 80.0f;
    float motionThresh    = 30.0f;
    int   motionTracking  = 1;
    int   fireflyRemoval  = 1;
    int   ghostGuard      = 1;      // v3.2
    int   enableSpatial   = 1;
    int   spatialMode     = 1;      // NLM
    int   spatialRadius   = 3;
    float spatialLuma     = 60.0f;
    float spatialChroma   = 100.0f;
    float preserveDetail  = 35.0f;
    float detailRescue    = 0.0f;   // v3.1
    int   deepClean       = 0;      // v3.3: fine pre-pass, severe class only
    float chromaBlotch    = 25.0f;
    float eqFine          = 100.0f;
    float eqMedium        = 0.0f;
    float eqCoarse        = 0.0f;
    float profileAdjust   = 1.0f;   // raw multiplier, not percent
    int   lockProfile     = 1;
    int   noiseClass      = 1;      // for the report
    int   movingCamera    = 0;
};

inline AutoSettings mapAnalysisToSettings(const ClipAggregate& a)
{
    AutoSettings s;
    const float syPct = a.sy * 100.0f;

    s.noiseClass = (syPct < 0.8f) ? 0 : (syPct < 2.5f) ? 1 : (syPct < 6.0f) ? 2 : 3;
    s.movingCamera = (a.motion > 0.15f) ? 1 : 0;

    // Base table per class. Values are monotone in the class so "noisier
    // footage never gets gentler settings" holds by construction. Clean
    // keeps the plugin defaults; from moderate up, v3.1 deliberately uses
    // the extended ranges — field feedback was that Auto had to be beaten
    // to reach Resolve's built-in NR. Detail Rescue rises with class so the
    // extra aggression flattens noise without blurring structure.
    // Tuned empirically against the synthetic-scene PSNR suite: the noise
    // energy lives in the temporal stack, the bands and the reach — pushing
    // the fine-band blend past ~85 buys smoothness that PSNR (and faces)
    // pay for, so the fine band stays moderate and Detail Rescue climbs
    // instead. The e2e tests gate every class against stock defaults.
    static const float kTL[4]  = { 60, 70, 85, 100 };   // temporal luma
    static const float kTC2[4] = { 80, 90, 105, 125 };  // temporal chroma
    static const float kSL[4]  = { 60, 70, 80, 85 };    // spatial luma
    static const float kSC2[4] = { 100, 110, 125, 150 };// spatial chroma
    static const int   kR[4]   = { 3, 4, 5, 7 };        // search radius
    static const int   kF[4]   = { 3, 5, 5, 5 };        // frames
    static const float kPD[4]  = { 45, 35, 30, 15 };    // preserve detail
    static const float kRS[4]  = { 0, 15, 40, 50 };     // detail rescue
    static const float kCB[4]  = { 25, 35, 55, 80 };    // chroma blotch
    static const float kEF[4]  = { 100, 100, 105, 110 };// fine band drive
    static const float kEM[4]  = { 0, 10, 30, 50 };     // medium band
    static const float kEC[4]  = { 0, 0, 10, 25 };      // coarse luma band

    const int c = s.noiseClass;
    s.temporalLuma   = kTL[c];
    s.temporalChroma = kTC2[c];
    s.spatialLuma    = kSL[c];
    s.spatialChroma  = kSC2[c];
    s.spatialRadius  = kR[c];
    s.temporalFrames = kF[c];
    s.preserveDetail = kPD[c];
    s.detailRescue   = kRS[c];
    s.chromaBlotch   = kCB[c];
    s.eqFine         = kEF[c];
    s.eqMedium       = kEM[c];
    s.eqCoarse       = kEC[c];

    // Motion energy: heavy motion prefers 3 frames and a lower (more
    // cautious) motion threshold. The base stays cautious even for noisy
    // classes — sweeps show a wide gate buys almost nothing on static
    // footage but visibly costs quality the moment anything moves.
    // v3.3: the Deep Clean pre-pass earns its render cost only when there
    // is serious noise to decorrelate
    if (s.noiseClass >= 3)
        s.deepClean = 1;

    const float mNorm = std::min(a.motion / 0.4f, 1.0f);
    s.motionThresh = clampfLocal(26.0f + 4.0f * c - 18.0f * mNorm, 12.0f, 40.0f);
    if (a.motion > 0.15f)
        s.temporalFrames = 3;
    // v3.3 B2: a 7-frame stack pays off only where frames genuinely repeat —
    // noisy/severe clips from a steady camera
    else if (s.noiseClass >= 2)
        s.temporalFrames = 7;

    // Chroma-heavy noise: push chroma strength and the blotch pass. The
    // blotch radius scales with the slider, and the rings must clear the
    // stain to see clean context, so heavy chroma noise wants it well up.
    if (a.chromaRatio > 1.6f) {
        s.spatialChroma = std::max(s.spatialChroma, 125.0f);
        s.chromaBlotch += 20.0f;
    }
    if (a.chromaRatio > 2.5f)
        s.chromaBlotch += 20.0f;

    // Spatially-correlated noise (coarse estimator reads above fine): the
    // fine band can't see it — raise the medium band and the blotch reach,
    // and for strong correlation bring in the coarse luma band.
    if (a.coarseRatioY > 1.25f) {
        s.eqMedium += 15.0f;
        s.chromaBlotch += 10.0f;
    }
    if (a.coarseRatioY > 1.6f) {
        s.eqMedium += 10.0f;
        s.eqCoarse += 10.0f;
    }

    s.chromaBlotch = clampfLocal(s.chromaBlotch, 0.0f, 150.0f);
    s.eqMedium     = clampfLocal(s.eqMedium, 0.0f, 150.0f);
    s.eqCoarse     = clampfLocal(s.eqCoarse, 0.0f, 150.0f);
    return s;
}

// ---------------------------------------------------------------------------
// Human-readable report for the Auto Setup status line
// ---------------------------------------------------------------------------
inline std::string formatAutoReport(const ClipAggregate& a, const AutoSettings& s,
                                    int fromRegion = 0)
{
    static const char* kClassName[4] = { "clean", "moderate noise", "noisy", "severe noise" };
    char buf[256];
    snprintf(buf, sizeof(buf),
             "Analyzed %d frame%s%s \xc2\xb7 noise %.1f%%Y / %.1f%%C (%s) \xc2\xb7 %s \xc2\xb7 profile locked",
             a.frames, a.frames == 1 ? "" : "s",
             fromRegion ? " (from region)" : "",
             a.sy * 100.0f, a.sc * 100.0f,
             kClassName[s.noiseClass],
             s.movingCamera ? "moving camera" : "steady camera");
    return std::string(buf);
}

} // namespace nranalyze

#endif // OPENNR_NR_ANALYZE_H
