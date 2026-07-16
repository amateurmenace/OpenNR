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
    // v3.3 B5: per-channel chroma pairs (sc/tc above stay the pair means)
    float scb = 0.02f, scr = 0.02f;
    float tcb = 0.02f, tcr = 0.02f;
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
    std::vector<float> scb, scr, tcb, tcr;
    float motionSum = 0.0f;
    int motionN = 0;
    for (size_t i = 0; i < all.size(); ++i) {
        const nrcore::Stats& s = all[i];
        sy.push_back(s.sy);
        sc.push_back(s.sc);
        ty.push_back(s.ty);
        tc.push_back(s.tc);
        scb.push_back(s.scb);
        scr.push_back(s.scr);
        tcb.push_back(s.tcb);
        tcr.push_back(s.tcr);
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
    a.scb = medianOf(scb);
    a.scr = medianOf(scr);
    a.tcb = medianOf(tcb);
    a.tcr = medianOf(tcr);
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
    // v3.3 B5: HUSHLOCK2 carries the split chroma pairs; parseLockedProfile
    // still reads HUSHLOCK1 (Cr loads as Cb) so old projects keep working
    std::string s = "HUSHLOCK2";
    char buf[16];
    const float vals[6] = { a.sy, a.scb, a.scr, a.ty, a.tcb, a.tcr };
    for (int i = 0; i < 6; ++i) {
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
    const bool v2 = (s.compare(0, 9, "HUSHLOCK2") == 0);
    if (!v2 && s.compare(0, 9, "HUSHLOCK1") != 0)
        return false;
    const int nSig = v2 ? 6 : 4;
    float vals[38];
    size_t pos = 9;
    for (int i = 0; i < nSig + 32; ++i) {
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
    if (v2) {
        out.sy = vals[0];
        out.scb = vals[1];
        out.scr = vals[2];
        out.ty = vals[3];
        out.tcb = vals[4];
        out.tcr = vals[5];
    } else {
        // v1: combined chroma — both channels load the combined value
        out.sy = vals[0];
        out.scb = out.scr = vals[1];
        out.ty = vals[2];
        out.tcb = out.tcr = vals[3];
    }
    out.sc = 0.5f * (out.scb + out.scr);
    out.tc = 0.5f * (out.tcb + out.tcr);
    for (int b = 0; b < 16; ++b) {
        out.gainY[b] = vals[nSig + b];
        out.gainC[b] = vals[nSig + 16 + b];
    }
    return true;
}

// ---------------------------------------------------------------------------
// v3.3 B4 — flattest-patch scan: where a sampling region SHOULD go
// ---------------------------------------------------------------------------
// The From Region profile is only as good as the rectangle the user drew —
// on a flat area it is exact, on texture it overreads. The scan walks a
// grid of candidate patches and scores each by structure content: mean 3x3
// local luma variance minus the expected noise variance at that brightness
// (so dark noisy flats are not penalized for their noise). Auto Setup only
// REPORTS the lowest-scoring centre in the Analysis line (it never moves the
// user's rectangle); the v3.4 Auto Region button applies it deliberately —
// pressing that button IS the consent to move the box.
// v3.4: the patch size follows the caller's region size (the box lands where
// a box THAT size is flattest) and the grid tightened 7 -> 9.
struct FlatPatch {
    float cx = 0.5f, cy = 0.5f;   // normalized centre of the flattest patch
    int   valid = 0;
};

inline FlatPatch findFlattestPatch(const float* rgba, int W, int H,
                                   const nrcore::Stats& st,
                                   float regionSize = 0.25f)
{
    FlatPatch fp;
    if (W < 64 || H < 64)
        return fp;
    // patch half-size mirrors estimateInput's region half (0.5 * size * minDim),
    // capped so a large region still scans meaningfully inside the frame
    const float rs = std::min(0.5f, std::max(0.05f, regionSize));
    const int half = std::max(8, static_cast<int>(0.5f * rs * static_cast<float>(std::min(W, H))));
    const int grid = 9;
    float bestScore = 0.0f;
    for (int gy = 0; gy < grid; ++gy) {
        for (int gx = 0; gx < grid; ++gx) {
            const float fx = 0.15f + 0.70f * static_cast<float>(gx) / (grid - 1);
            const float fy = 0.15f + 0.70f * static_cast<float>(gy) / (grid - 1);
            const int cx = static_cast<int>(fx * W);
            const int cy = static_cast<int>(fy * H);
            const int x0 = nrcore::clampi(cx - half, 1, W - 2);
            const int x1 = nrcore::clampi(cx + half, 1, W - 2);
            const int y0 = nrcore::clampi(cy - half, 1, H - 2);
            const int y1 = nrcore::clampi(cy + half, 1, H - 2);
            double score = 0.0;
            int n = 0;
            for (int y = y0; y < y1; y += 4) {
                for (int x = x0; x < x1; x += 4) {
                    float mean = 0.0f, m2 = 0.0f;
                    float Yc = 0.0f, cb, cr;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            float Y;
                            nrcore::sampleYCC(rgba, W, H, x + dx, y + dy, Y, cb, cr);
                            if (dx == 0 && dy == 0) Yc = Y;
                            mean += Y; m2 += Y * Y;
                        }
                    mean *= (1.0f / 9.0f);
                    const float var = std::max(0.0f, m2 * (1.0f / 9.0f) - mean * mean);
                    const int lb = nrcore::clampi(static_cast<int>(Yc * nrcore::kLumaBins),
                                                  0, nrcore::kLumaBins - 1);
                    const float sig = st.sy * st.gainY[lb];
                    score += static_cast<double>(var) - static_cast<double>(sig) * sig;
                    ++n;
                }
            }
            if (n < 16)
                continue;
            score /= n;
            if (!fp.valid || score < bestScore) {
                bestScore = static_cast<float>(score);
                fp.cx = fx;
                fp.cy = fy;
                fp.valid = 1;
            }
        }
    }
    return fp;
}

// ---------------------------------------------------------------------------
// v3.4 — lock sweep: freeze what the user SEES, hardened by tracking
// ---------------------------------------------------------------------------
// Until v3.3 the lock measured 5 frames spread across the WHOLE clip and
// took the median, while the live (unlocked) render measures the playhead
// frame — so on anything but a locked-off shot the two disagreed: the
// screen-fixed region covers different content at those distant times, the
// median collapsed, and locking visibly brought the noise back. (The second
// half of the "lock brings the noise back" bug — v3.2 fixed the
// region-ignored half.) v3.4 locks the PLAYHEAD measurement, hardened by a
// short sweep of the frames around it: the patch content is tracked
// frame-to-frame (SAD over sampled luma with a zero-motion prior, so
// featureless patches stay put instead of random-walking, drift-capped),
// measured at its tracked position, and a sweep frame only votes when its
// measurement AGREES with the playhead's (sigma ratio + patch brightness) —
// a car crossing the patch, an occlusion or a lost track simply drops out.
// Worst case every helper is rejected and the lock is exactly the playhead
// measurement: the user always gets what they were looking at.

struct RegionTrack {
    int dx = 0, dy = 0;    // cumulative pixel offset of the tracked patch
    int ok = 1;            // 0 = track lost (drift cap exceeded)
};

// Mean sampled luma of the (offset) region — the cheap surface fingerprint
// the acceptance test uses to notice "different content under the box".
// Pass sizeN >= 2 for a whole-frame fingerprint.
inline float regionMeanLuma(const float* rgba, int W, int H,
                            float cxN, float cyN, float sizeN, int dx, int dy)
{
    const int half = std::max(4, static_cast<int>(0.5f * sizeN * static_cast<float>(std::min(W, H))));
    const int cx = static_cast<int>(cxN * W) + dx;
    const int cy = static_cast<int>(cyN * H) + dy;
    const int step = std::max(1, half / 12);
    float sum = 0.0f;
    int n = 0;
    for (int y = cy - half; y <= cy + half; y += step)
        for (int x = cx - half; x <= cx + half; x += step) {
            float Y, cb, cr;
            nrcore::sampleYCC(rgba, W, H, x, y, Y, cb, cr);
            sum += Y;
            ++n;
        }
    return (n > 0) ? sum / static_cast<float>(n) : 0.0f;
}

// One tracking step: where did the patch under (cxN,cyN)+prevT move between
// frame `prev` and frame `cur`? Integer SAD search over sampled luma within
// +/-searchR of the previous offset, with a mild zero-motion prior — on a
// featureless patch every candidate scores alike and the prior keeps the box
// still. ok drops to 0 once the cumulative drift leaves the patch's own
// neighbourhood (the measurement there would no longer be "the user's spot").
inline RegionTrack trackRegionStep(const float* prev, const float* cur, int W, int H,
                                   float cxN, float cyN, float sizeN,
                                   const RegionTrack& prevT,
                                   int searchR = 6, int maxDriftPx = 0)
{
    RegionTrack t = prevT;
    if (!prevT.ok)
        return t;
    const int half = std::max(4, static_cast<int>(0.5f * sizeN * static_cast<float>(std::min(W, H))));
    if (maxDriftPx <= 0)
        maxDriftPx = (3 * half) / 4;
    const int cx = static_cast<int>(cxN * W) + prevT.dx;
    const int cy = static_cast<int>(cyN * H) + prevT.dy;
    const int step = std::max(1, half / 12);

    // reference grid from the previous frame, sampled once
    std::vector<float> ref;
    std::vector<int> gx, gy;
    for (int y = cy - half; y <= cy + half; y += step)
        for (int x = cx - half; x <= cx + half; x += step) {
            float Y, cb, cr;
            nrcore::sampleYCC(prev, W, H, x, y, Y, cb, cr);
            ref.push_back(Y);
            gx.push_back(x);
            gy.push_back(y);
        }
    if (ref.size() < 16) {
        t.ok = 0;
        return t;
    }

    float bestScore = 0.0f;
    int bestOx = 0, bestOy = 0;
    bool first = true;
    for (int oy = -searchR; oy <= searchR; ++oy) {
        for (int ox = -searchR; ox <= searchR; ++ox) {
            float sad = 0.0f;
            for (size_t i = 0; i < ref.size(); ++i) {
                float Y, cb, cr;
                nrcore::sampleYCC(cur, W, H, gx[i] + ox, gy[i] + oy, Y, cb, cr);
                sad += std::fabs(Y - ref[i]);
            }
            const float score = sad * (1.0f + 0.02f * static_cast<float>(std::abs(ox) + std::abs(oy)));
            if (first || score < bestScore) {
                bestScore = score;
                bestOx = ox;
                bestOy = oy;
                first = false;
            }
        }
    }
    t.dx = prevT.dx + bestOx;
    t.dy = prevT.dy + bestOy;
    t.ok = (std::abs(t.dx) <= maxDriftPx && std::abs(t.dy) <= maxDriftPx) ? 1 : 0;
    return t;
}

// Sweep acceptance: does a tracked measurement describe the SAME noise the
// user locked? Sigmas within [1/1.6, 1.6]x of the playhead's (the temporal
// pair only when both frames had a diff partner) and patch brightness within
// 0.13 — beyond either, the content under the box has changed and the frame
// must not vote.
inline bool sweepMeasurementMatches(const nrcore::Stats& anchor, float anchorLuma,
                                    const nrcore::Stats& cand, float candLuma)
{
    const float kLo = 1.0f / 1.6f, kHi = 1.6f;
    const float rS = cand.sy / std::max(anchor.sy, 1e-6f);
    if (rS < kLo || rS > kHi)
        return false;
    if (anchor.hadTemporal && cand.hadTemporal) {
        const float rT = cand.ty / std::max(anchor.ty, 1e-6f);
        if (rT < kLo || rT > kHi)
            return false;
    }
    return std::fabs(candLuma - anchorLuma) <= 0.13f;
}

// One measured frame of the sweep.
struct SweepSample {
    nrcore::Stats st;
    float luma = 0.0f;     // region fingerprint at the tracked position
    int dx = 0, dy = 0;    // tracked offset the measurement was taken at
};

struct LockSweep {
    ClipAggregate agg;     // what to lock
    int measured = 0;      // frames measured (anchor included)
    int accepted = 0;      // frames that agreed with the playhead
    int tracked = 0;       // 1 = region mode with at least one sweep frame
    int driftPx = 0;       // largest accepted |offset| component
};

// Measure one frame at a tracked offset. ap carries profileSource/region and
// MUST keep profileAdjust at 1 — locks store the RAW measurement, the trim
// is applied at use time (v3.2 rule).
inline void measureAtOffset(const float* frame, const float* partner, int W, int H,
                            const nrcore::Params& ap, int dx, int dy, SweepSample& out)
{
    nrcore::Params fp = ap;
    if (ap.profileSource == 1) {
        const float nx = ap.regionCX + static_cast<float>(dx) / static_cast<float>(W);
        const float ny = ap.regionCY + static_cast<float>(dy) / static_cast<float>(H);
        fp.regionCX = std::min(1.0f, std::max(0.0f, nx));
        fp.regionCY = std::min(1.0f, std::max(0.0f, ny));
    }
    nrcore::estimateInput(frame, partner, W, H, fp, out.st);
    out.luma = (ap.profileSource == 1)
             ? regionMeanLuma(frame, W, H, ap.regionCX, ap.regionCY, ap.regionSize, dx, dy)
             : regionMeanLuma(frame, W, H, 0.5f, 0.5f, 2.0f, 0, 0);
    out.dx = dx;
    out.dy = dy;
}

// The playhead measurement always votes; helpers vote only when they agree.
inline LockSweep composeLockAggregate(const SweepSample& anchor,
                                      const std::vector<SweepSample>& others,
                                      int tracked)
{
    LockSweep out;
    out.tracked = tracked;
    out.measured = 1 + static_cast<int>(others.size());
    std::vector<nrcore::Stats> keep;
    keep.push_back(anchor.st);
    out.accepted = 1;
    for (size_t i = 0; i < others.size(); ++i) {
        const SweepSample& s = others[i];
        if (!sweepMeasurementMatches(anchor.st, anchor.luma, s.st, s.luma))
            continue;
        keep.push_back(s.st);
        ++out.accepted;
        const int d = std::max(std::abs(s.dx), std::abs(s.dy));
        if (d > out.driftPx)
            out.driftPx = d;
    }
    out.agg = aggregateClipStats(keep);
    return out;
}

// Reference orchestration over frames already in memory (tests use this; the
// plugin runs a streaming port of the same loop — 2 frames resident instead
// of 9 — feeding the same helpers, so the logic cannot diverge).
inline LockSweep lockSweepAnalyze(const std::vector<const float*>& frames, int W, int H,
                                  int anchorIdx, const nrcore::Params& ap)
{
    LockSweep bad;
    const int n = static_cast<int>(frames.size());
    if (n <= 0 || anchorIdx < 0 || anchorIdx >= n)
        return bad;
    const bool region = (ap.profileSource == 1);

    SweepSample anchor;
    const float* aPartner = (anchorIdx + 1 < n) ? frames[anchorIdx + 1]
                          : (anchorIdx > 0 ? frames[anchorIdx - 1] : nullptr);
    measureAtOffset(frames[anchorIdx], aPartner, W, H, ap, 0, 0, anchor);

    std::vector<SweepSample> others;
    RegionTrack tr;
    for (int i = anchorIdx + 1; i < n; ++i) {           // forward
        if (region) {
            tr = trackRegionStep(frames[i - 1], frames[i], W, H,
                                 ap.regionCX, ap.regionCY, ap.regionSize, tr);
            if (!tr.ok)
                break;
        }
        SweepSample s;
        measureAtOffset(frames[i], frames[i - 1], W, H, ap, tr.dx, tr.dy, s);
        others.push_back(s);
    }
    tr = RegionTrack();
    for (int i = anchorIdx - 1; i >= 0; --i) {          // backward
        if (region) {
            tr = trackRegionStep(frames[i + 1], frames[i], W, H,
                                 ap.regionCX, ap.regionCY, ap.regionSize, tr);
            if (!tr.ok)
                break;
        }
        SweepSample s;
        measureAtOffset(frames[i], frames[i + 1], W, H, ap, tr.dx, tr.dy, s);
        others.push_back(s);
    }
    return composeLockAggregate(anchor, others, (region && !others.empty()) ? 1 : 0);
}

// Status line for a Lock Profile click.
inline std::string formatLockReport(const LockSweep& sw, int fromRegion)
{
    char buf[224];
    if (sw.measured <= 0)
        return std::string("Lock failed \xe2\x80\x94 could not read the clip.");
    if (fromRegion && sw.tracked)
        snprintf(buf, sizeof(buf),
                 "Profile locked \xc2\xb7 region at the playhead, %d of %d nearby frames agreed "
                 "(patch tracked, drift %d px) \xc2\xb7 noise %.1f%%Y / %.1f%%C",
                 sw.accepted, sw.measured, sw.driftPx,
                 sw.agg.sy * 100.0f, sw.agg.sc * 100.0f);
    else
        snprintf(buf, sizeof(buf),
                 "Profile locked \xc2\xb7 %s at the playhead, %d of %d nearby frames agreed "
                 "\xc2\xb7 noise %.1f%%Y / %.1f%%C",
                 fromRegion ? "region" : "whole frame",
                 sw.accepted, sw.measured,
                 sw.agg.sy * 100.0f, sw.agg.sc * 100.0f);
    return std::string(buf);
}

// Status line for the v3.4 Auto Region button.
inline std::string formatRegionReport(const FlatPatch& fp, const nrcore::Stats& st)
{
    char buf[224];
    snprintf(buf, sizeof(buf),
             "Region placed on the flattest patch (%d%%, %d%%) \xc2\xb7 noise there %.1f%%Y / %.1f%%C "
             "\xc2\xb7 it measures live \xe2\x80\x94 Lock Profile to freeze it, or run Auto Setup",
             static_cast<int>(fp.cx * 100.0f + 0.5f),
             static_cast<int>(fp.cy * 100.0f + 0.5f),
             st.sy * 100.0f, st.sc * 100.0f);
    return std::string(buf);
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
    float effnSteer       = 0.0f;   // v3.6 S1 (UI percent); the class table
                                    // leaves it off — the SURE descent turns
                                    // it on only when measurement says so
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
                                    int fromRegion = 0,
                                    const FlatPatch& flat = FlatPatch())
{
    static const char* kClassName[4] = { "clean", "moderate noise", "noisy", "severe noise" };
    char buf[352];
    snprintf(buf, sizeof(buf),
             "Analyzed %d frame%s%s \xc2\xb7 noise %.1f%%Y / %.1f%%C (%s) \xc2\xb7 %s \xc2\xb7 profile locked",
             a.frames, a.frames == 1 ? "" : "s",
             fromRegion ? " (from region)" : "",
             a.sy * 100.0f, a.sc * 100.0f,
             kClassName[s.noiseClass],
             s.movingCamera ? "moving camera" : "steady camera");
    std::string out(buf);
    // v3.3 B4: report-only — the user's rectangle is never moved
    if (flat.valid) {
        snprintf(buf, sizeof(buf),
                 " \xc2\xb7 flattest patch at %d%%, %d%% \xe2\x80\x94 %s",
                 static_cast<int>(flat.cx * 100.0f + 0.5f),
                 static_cast<int>(flat.cy * 100.0f + 0.5f),
                 fromRegion ? "compare with your region"
                            : "a good spot for a sampling region");
        out += buf;
    }
    return out;
}

// ---------------------------------------------------------------------------
// v3.5 X1 — Monte-Carlo SURE self-tuning for Auto Setup.
//
// Stein's unbiased risk estimate scores a denoiser's true MSE against the
// UNKNOWN clean image using only the noisy input and the denoiser itself:
//
//   SURE = ||f(y) - y||^2 / N  -  sigma^2  +  (2 sigma^2 / N) * div f(y)
//
// with the divergence estimated Monte-Carlo (Ramani et al.):
//
//   div f(y) ~= sum_i b_i * (f(y + eps*b)_i - f(y)_i) / eps
//
// b = +/-1 Rademacher from nrcore::hashNoise at a FIXED seed, so the whole
// tuner is deterministic across runs and machines. Luma only: y is the
// centre frame's luma, the perturbation shifts R,G,B together (luma moves,
// chroma untouched), and f() is the FULL pipeline at the candidate settings
// (neighbour frames ride along as side information — their noise is
// independent of the centre's, which is all Stein's identity needs).
//
// sureTuneGrid sweeps a 3x3 grid of (temporalLuma, spatialLuma) around the
// base settings (+/-25%, the slider range's natural neighbourhood) and
// returns the SURE argmin. 18 denoiseFrame runs on the caller's crop —
// size the crop, not the grid.
// ---------------------------------------------------------------------------
// Build the CPU Params that Auto Setup's sliders would produce — the tuner
// must score EXACTLY the pipeline the user gets after Apply (UI percents
// become fractions; everything not in AutoSettings keeps plugin defaults).
inline nrcore::Params paramsFromAutoSettings(const AutoSettings& s)
{
    nrcore::Params p;
    p.enableTemporal = s.enableTemporal;
    p.temporalFrames = s.temporalFrames;
    p.temporalLuma   = s.temporalLuma / 100.0f;
    p.temporalChroma = s.temporalChroma / 100.0f;
    p.motionThresh   = s.motionThresh / 100.0f;
    p.motionTracking = s.motionTracking;
    p.fireflyRemoval = s.fireflyRemoval;
    p.ghostGuard     = s.ghostGuard;
    p.enableSpatial  = s.enableSpatial;
    p.spatialMode    = s.spatialMode;
    p.spatialRadius  = s.spatialRadius;
    p.spatialLuma    = s.spatialLuma / 100.0f;
    p.spatialChroma  = s.spatialChroma / 100.0f;
    p.preserveDetail = s.preserveDetail / 100.0f;
    p.detailRescue   = s.detailRescue / 100.0f;
    p.deepClean      = s.deepClean;
    p.chromaBlotch   = s.chromaBlotch / 100.0f;
    p.eqFine         = s.eqFine / 100.0f;
    p.eqMedium       = s.eqMedium / 100.0f;
    p.eqCoarse       = s.eqCoarse / 100.0f;
    p.effnSteer      = s.effnSteer / 100.0f;
    p.profileAdjust  = s.profileAdjust;
    return p;
}

struct SureTune {
    int    ran = 0;              // 1 when the grid ran (fetch etc. succeeded)
    int    ti = 1, si = 1;       // winning grid indices (1,1 = table values)
    float  temporalLuma = 0.0f;  // winning values, PARAM fractions (UI/100)
    float  spatialLuma  = 0.0f;
    float  gridT[3] = {0, 0, 0}; // the swept fractions (tests re-run these)
    float  gridS[3] = {0, 0, 0};
    double sure[3][3];           // the surface (tests assert determinism)
    float  sigma = 0.0f, eps = 0.0f;
};

inline float sureLuma(const float* px)
{
    float y, cb, cr;
    nrcore::rgb2ycc(px[0], px[1], px[2], y, cb, cr);
    return y;
}

// One luma SURE evaluation: two full-pipeline runs (clean + perturbed
// stack). Verbatim extraction of the v3.5 X1 grid loop body — the tests
// pin that surface bit-for-bit, so the arithmetic order must not change.
inline double sureEvalLuma(const float* const frames[7], const float* const pf[7],
                           const std::vector<signed char>& rad,
                           int W, int H, const nrcore::Params& p,
                           float sigma, float eps,
                           std::vector<float>& outA, std::vector<float>& outB,
                           std::vector<float>& scratch)
{
    const size_t nl = static_cast<size_t>(W) * H;
    nrcore::denoiseFrame(frames, W, H, p, outA.data(), scratch);
    nrcore::denoiseFrame(pf, W, H, p, outB.data(), scratch);
    double rss = 0.0, div = 0.0;
    for (size_t i = 0; i < nl; ++i) {
        const float ly  = sureLuma(frames[3] + i * 4);
        const float fy  = sureLuma(outA.data() + i * 4);
        const float fyp = sureLuma(outB.data() + i * 4);
        const double d = static_cast<double>(fy) - ly;
        rss += d * d;
        div += static_cast<double>(rad[i]) * (static_cast<double>(fyp) - fy);
    }
    div /= eps;
    return rss / static_cast<double>(nl)
         - static_cast<double>(sigma) * sigma
         + (2.0 * sigma * sigma / static_cast<double>(nl)) * div;
}

inline SureTune sureTuneGrid(const float* const frames[7], int W, int H,
                             const nrcore::Params& base)
{
    SureTune r;
    if (W < 32 || H < 32)
        return r;
    const size_t n  = static_cast<size_t>(W) * H * 4;
    const size_t nl = static_cast<size_t>(W) * H;

    // sigma once, from the input (settings don't change the input measure);
    // the temporal-pair estimate is immune to spatial correlation and is
    // what "the noise" means for SURE's iid model — fall back to spatial
    // when the stack has no distinct neighbour
    const float* partner = 0;
    if (frames[2] != frames[3])      partner = frames[2];
    else if (frames[4] != frames[3]) partner = frames[4];
    nrcore::Params p0 = base;
    nrcore::Stats s0;
    nrcore::estimateInput(frames[3], partner, W, H, p0, s0);
    const float sigma = s0.hadTemporal ? s0.ty : s0.sy;
    // eps = sigma/20: small enough that f stays in its linear regime,
    // large enough that float cancellation in f(y+eps*b)-f(y) stays well
    // above the mantissa floor (validated on the synthetic suite)
    const float eps = std::max(sigma * 0.05f, 1e-4f);
    r.sigma = sigma;
    r.eps = eps;

    // fixed-seed Rademacher field + the perturbed centre frame. Neighbour
    // slots that ALIAS the centre (clip edges) must alias the perturbed
    // centre too, or the pointer-equality partner logic inside denoiseFrame
    // flips between the two runs and the divergence measures the estimator
    // flapping instead of the filter.
    std::vector<float> pert(n);
    std::vector<signed char> rad(nl);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = static_cast<size_t>(y) * W + x;
            const signed char b =
                (nrcore::hashNoise(static_cast<uint32_t>(x),
                                   static_cast<uint32_t>(y), 977u, 7u) >= 0.0f) ? 1 : -1;
            rad[i] = b;
            const float d = eps * static_cast<float>(b);
            pert[i * 4 + 0] = frames[3][i * 4 + 0] + d;
            pert[i * 4 + 1] = frames[3][i * 4 + 1] + d;
            pert[i * 4 + 2] = frames[3][i * 4 + 2] + d;
            pert[i * 4 + 3] = frames[3][i * 4 + 3];
        }
    const float* pf[7];
    for (int k = 0; k < 7; ++k)
        pf[k] = (frames[k] == frames[3]) ? pert.data() : frames[k];

    static const float kStep[3] = { 0.75f, 1.0f, 1.25f };
    for (int i = 0; i < 3; ++i) {
        r.gridT[i] = clampfLocal(base.temporalLuma * kStep[i], 0.0f, 1.25f);
        r.gridS[i] = clampfLocal(base.spatialLuma * kStep[i], 0.0f, 1.25f);
    }

    std::vector<float> outA(n), outB(n), scratch;
    double bestSure = 0.0;
    for (int ti = 0; ti < 3; ++ti)
        for (int si = 0; si < 3; ++si) {
            nrcore::Params p = base;
            p.temporalLuma = r.gridT[ti];
            p.spatialLuma  = r.gridS[si];
            // v3.6 #19: the evaluation lives in sureEvalLuma (verbatim
            // extraction — the tests pin this surface bit-for-bit)
            const double sure = sureEvalLuma(frames, pf, rad, W, H, p,
                                             sigma, eps, outA, outB, scratch);
            r.sure[ti][si] = sure;
            if ((ti == 0 && si == 0) || sure < bestSure) {
                bestSure = sure;
                r.ti = ti;
                r.si = si;
            }
        }

    r.temporalLuma = r.gridT[r.ti];
    r.spatialLuma  = r.gridS[r.si];
    r.ran = 1;
    return r;
}

// ---------------------------------------------------------------------------
// v3.6 #19 — SURE everything: coordinate descent + chroma SURE.
//
// X1 proved the machinery (full-pipeline MC-SURE argmin matched true-MSE
// argmin exactly, bit-deterministic); this widens it two ways:
//
//   1. LUMA coordinate descent over the axes the 3x3 grid never touches —
//      motionThresh, preserveDetail, detailRescue, eqMedium and the v3.6
//      Adaptive Strength (effnSteer). One pass, one axis at a time,
//      +/- one step around the incumbent, adopt only on improvement,
//      ride a winning direction at most one extra step. The incumbent's
//      score is reused across axes, so each axis costs 2-3 evaluations
//      (4-6 crop denoises).
//
//   2. A CHROMA-SURE pass over (temporalChroma, spatialChroma,
//      chromaBlotch) — the same estimator on Cb/Cr against their OWN
//      measured sigmas (tcb/tcr), so the chroma sliders come from
//      measurement instead of the class table. The perturbation moves
//      each chroma axis in RGB along the exact direction that leaves the
//      other two YCC axes fixed (dY == 0 by construction), with
//      INDEPENDENT fixed-seed Rademacher fields per channel — Stein's
//      identity then splits cleanly per channel (cross-divergences vanish
//      in expectation).
//
// Runs on a centre SUB-crop (<=384x216) of the tuner crop: the descent
// costs ~2x the grid's denoise count, the smaller crop pays that bill.
// Deterministic end to end (fixed seeds, fixed order). Any failure path
// returns ran=0 and Auto Setup keeps the grid/table values.
// ---------------------------------------------------------------------------
// One chroma SURE evaluation: SURE_cb + SURE_cr, each channel against its
// own sigma and its own Rademacher field / epsilon.
inline double sureEvalChroma(const float* const frames[7], const float* const pfc[7],
                             const std::vector<signed char>& radCb,
                             const std::vector<signed char>& radCr,
                             int W, int H, const nrcore::Params& p,
                             float sigmaCb, float sigmaCr,
                             float epsCb, float epsCr,
                             std::vector<float>& outA, std::vector<float>& outB,
                             std::vector<float>& scratch)
{
    const size_t nl = static_cast<size_t>(W) * H;
    nrcore::denoiseFrame(frames, W, H, p, outA.data(), scratch);
    nrcore::denoiseFrame(pfc, W, H, p, outB.data(), scratch);
    double rssCb = 0.0, rssCr = 0.0, divCb = 0.0, divCr = 0.0;
    for (size_t i = 0; i < nl; ++i) {
        float y0, cb0, cr0, ya, cba, cra, yb, cbb, crb;
        const float* src = frames[3] + i * 4;
        nrcore::rgb2ycc(src[0], src[1], src[2], y0, cb0, cr0);
        const float* pa = outA.data() + i * 4;
        nrcore::rgb2ycc(pa[0], pa[1], pa[2], ya, cba, cra);
        const float* pb = outB.data() + i * 4;
        nrcore::rgb2ycc(pb[0], pb[1], pb[2], yb, cbb, crb);
        const double dcb = static_cast<double>(cba) - cb0;
        const double dcr = static_cast<double>(cra) - cr0;
        rssCb += dcb * dcb;
        rssCr += dcr * dcr;
        divCb += static_cast<double>(radCb[i]) * (static_cast<double>(cbb) - cba);
        divCr += static_cast<double>(radCr[i]) * (static_cast<double>(crb) - cra);
    }
    divCb /= epsCb;
    divCr /= epsCr;
    const double sCb = rssCb / static_cast<double>(nl)
                     - static_cast<double>(sigmaCb) * sigmaCb
                     + (2.0 * sigmaCb * sigmaCb / static_cast<double>(nl)) * divCb;
    const double sCr = rssCr / static_cast<double>(nl)
                     - static_cast<double>(sigmaCr) * sigmaCr
                     + (2.0 * sigmaCr * sigmaCr / static_cast<double>(nl)) * divCr;
    return sCb + sCr;
}

// Chroma noise spatial-correlation length, for the descent's chroma-SURE
// probe. White MC-SURE (a per-pixel Rademacher probe) estimates tr(A) and is
// an unbiased MSE estimator ONLY for white noise: on the spatially-correlated
// shadow chroma speckle (the WEAK-1 case — "sits above the chroma bands'
// reach") the divergence term must be tr(A*Sigma), not sigma^2*tr(A), so a
// white probe is blind to the win and the descent walks the chroma sliders
// the WRONG way (measured: it dropped true PSNR 0.9 dB by de-tuning a good
// seed). Feeding the probe the noise's own correlation makes cov(b)=Sigma/
// sigma^2, and the SAME estimator formula becomes generalized SURE
// (tr(A*Sigma) exactly) — so it tracks true MSE across white AND correlated
// chroma without any change to sureEvalChroma.
//
// L is read from the temporal chroma difference (signal cancels on the static
// shadows this targets): block-mean chroma std at box sizes 1,2,4,8. White
// noise falls as 1/k; a field correlated over L px stays flat out to k~L. L is
// the largest size still above 0.6x the per-pixel level. Capped at 8 (the
// conservative direction — a block-8 probe already tracks larger-scale speckle,
// and capping bounds the over-credit if frame-to-frame MOTION, not noise,
// inflates the coarse levels). Returns 1 (the exact white probe, bit-for-bit)
// with no partner, negligible chroma noise, or an immediate fall-off.
inline int chromaCorrLen(const float* cur, const float* partner, int W, int H)
{
    if (!partner || W < 32 || H < 32) return 1;
    auto blockSig = [&](int k) -> double {
        const int bw = W / k, bh = H / k;
        if (bw < 2 || bh < 2) return 0.0;
        double s2 = 0.0; size_t c = 0;
        for (int by = 0; by < bh; ++by)
            for (int bx = 0; bx < bw; ++bx) {
                double acb = 0.0, acr = 0.0;
                for (int j = 0; j < k; ++j)
                    for (int i = 0; i < k; ++i) {
                        const size_t p = ((static_cast<size_t>(by) * k + j)
                                          * W + (static_cast<size_t>(bx) * k + i)) * 4;
                        float y1, cb1, cr1, y2, cb2, cr2;
                        nrcore::rgb2ycc(cur[p],     cur[p + 1],     cur[p + 2],     y1, cb1, cr1);
                        nrcore::rgb2ycc(partner[p], partner[p + 1], partner[p + 2], y2, cb2, cr2);
                        acb += (cb1 - cb2); acr += (cr1 - cr2);
                    }
                const double inv = 1.0 / (static_cast<double>(k) * static_cast<double>(k));
                acb *= inv; acr *= inv;
                s2 += acb * acb + acr * acr; c += 2;
            }
        // /sqrt(2): the difference of two independent noise fields has 2x
        // the per-field variance
        return (c > 0) ? std::sqrt(s2 / static_cast<double>(c)) * 0.70710678f : 0.0;
    };
    const double s1 = blockSig(1);
    if (s1 < 1e-4) return 1;   // negligible chroma noise: the white probe
    int L = 1;
    for (int k = 2; k <= 8; k *= 2) {
        if (blockSig(k) >= 0.6 * s1) L = k; else break;   // stop at fall-off
    }
    return L;
}

struct SureDescent {
    int    ran = 0;
    int    evals = 0;            // SURE evaluations spent (2 denoises each)
    // tuned values, PARAM fractions (UI/100) — valid when ran
    float  motionThresh = 0.0f, preserveDetail = 0.0f, detailRescue = 0.0f;
    float  eqMedium = 0.0f, effnSteer = 0.0f;
    float  temporalChroma = 0.0f, spatialChroma = 0.0f, chromaBlotch = 0.0f;
    double lumaBase = 0.0, lumaTuned = 0.0;      // SURE, for the report
    double chromaBase = 0.0, chromaTuned = 0.0;
    int    chromaCorr = 1;                        // probe correlation length
};

// Axis ids for the descent get/set switches (explicit — no member pointers,
// the GPU-struct 4-byte discipline has made us allergic to layout games).
enum SureAxisId {
    kAxMotionThresh, kAxPreserveDetail, kAxDetailRescue,
    kAxEqMedium, kAxEffnSteer,
    kAxTemporalChroma, kAxSpatialChroma, kAxChromaBlotch
};

inline float sureAxisGet(const nrcore::Params& p, int id)
{
    switch (id) {
        case kAxMotionThresh:   return p.motionThresh;
        case kAxPreserveDetail: return p.preserveDetail;
        case kAxDetailRescue:   return p.detailRescue;
        case kAxEqMedium:       return p.eqMedium;
        case kAxEffnSteer:      return p.effnSteer;
        case kAxTemporalChroma: return p.temporalChroma;
        case kAxSpatialChroma:  return p.spatialChroma;
        default:                return p.chromaBlotch;
    }
}

inline void sureAxisSet(nrcore::Params& p, int id, float v)
{
    switch (id) {
        case kAxMotionThresh:   p.motionThresh = v;   break;
        case kAxPreserveDetail: p.preserveDetail = v; break;
        case kAxDetailRescue:   p.detailRescue = v;   break;
        case kAxEqMedium:       p.eqMedium = v;       break;
        case kAxEffnSteer:      p.effnSteer = v;      break;
        case kAxTemporalChroma: p.temporalChroma = v; break;
        case kAxSpatialChroma:  p.spatialChroma = v;  break;
        default:                p.chromaBlotch = v;   break;
    }
}

inline SureDescent sureTuneDescent(const float* const frames[7], int W, int H,
                                   const nrcore::Params& base)
{
    SureDescent r;
    if (W < 32 || H < 32)
        return r;

    // ---- centre sub-crop (<=384x216, even-aligned) with the aliasing
    // structure preserved: slots that shared a pointer share a crop ----
    const int cw = (W > 384 ? 384 : W) & ~1;
    const int ch = (H > 216 ? 216 : H) & ~1;
    const int x0 = ((W - cw) / 2) & ~1;
    const int y0 = ((H - ch) / 2) & ~1;
    std::vector<float> crops[7];
    const float* cf[7] = { 0, 0, 0, 0, 0, 0, 0 };
    for (int k = 0; k < 7; ++k) {
        int prior = -1;
        for (int j = 0; j < k; ++j)
            if (frames[j] == frames[k]) { prior = j; break; }
        if (prior >= 0) { cf[k] = cf[prior]; continue; }
        crops[k].resize(static_cast<size_t>(cw) * ch * 4);
        for (int y = 0; y < ch; ++y)
            std::memcpy(&crops[k][static_cast<size_t>(y) * cw * 4],
                        frames[k] + ((static_cast<size_t>(y0 + y) * W + x0) * 4),
                        static_cast<size_t>(cw) * 4 * sizeof(float));
        cf[k] = crops[k].data();
    }

    const size_t n  = static_cast<size_t>(cw) * ch * 4;
    const size_t nl = static_cast<size_t>(cw) * ch;

    // ---- sigmas once, from the sub-crop (X1 pattern) ----
    const float* partner = 0;
    if (cf[2] != cf[3])      partner = cf[2];
    else if (cf[4] != cf[3]) partner = cf[4];
    nrcore::Params p0 = base;
    nrcore::Stats s0;
    nrcore::estimateInput(cf[3], partner, cw, ch, p0, s0);
    const float sigma   = s0.hadTemporal ? s0.ty  : s0.sy;
    const float sigmaCb = s0.hadTemporal ? s0.tcb : s0.scb;
    const float sigmaCr = s0.hadTemporal ? s0.tcr : s0.scr;
    const float eps   = std::max(sigma * 0.05f, 1e-4f);
    const float epsCb = std::max(sigmaCb * 0.05f, 1e-4f);
    const float epsCr = std::max(sigmaCr * 0.05f, 1e-4f);

    // ---- fixed-seed Rademacher fields + the two perturbed stacks ----
    // Luma: R,G,B together (channel 7 — the X1 seed). Chroma: independent
    // fields per channel (8, 9); the RGB delta moves exactly one YCC axis:
    //   pure Cb: (0, -0.0722*1.8556/0.7152, 1.8556) * dcb
    //   pure Cr: (1.5748, -0.2126*1.5748/0.7152, 0) * dcr
    // so dY == 0 identically and Cb/Cr move by exactly dcb/dcr (rgb2ycc's
    // BT.709 constants; verified against nr_core.h:224).
    //
    // The chroma field is block-constant at the measured correlation length
    // cB (see chromaCorrLen): cov(b)=Sigma/sigma^2, so chroma SURE becomes
    // generalized SURE and tracks true MSE on correlated speckle. cB==1 (the
    // white-noise / no-partner case) is x/1 == x — bit-for-bit the old probe,
    // so the luma path and every white-noise result stay unchanged.
    const int cB = chromaCorrLen(cf[3], partner, cw, ch);
    // Two independent probe families (seeds 977 and 1013): the first drives
    // the descent, the second holds it out for cross-validation (see below).
    std::vector<float> pertL(n), pertC(n), pertL2(n), pertC2(n);
    std::vector<signed char> rad(nl), radCb(nl), radCr(nl),
                             rad2(nl), radCb2(nl), radCr2(nl);
    for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x) {
            const size_t i = static_cast<size_t>(y) * cw + x;
            for (int pass = 0; pass < 2; ++pass) {
                const uint32_t sd = (pass == 0) ? 977u : 1013u;
                const signed char b =
                    (nrcore::hashNoise(static_cast<uint32_t>(x),
                                       static_cast<uint32_t>(y), sd, 7u) >= 0.0f) ? 1 : -1;
                const signed char bcb =
                    (nrcore::hashNoise(static_cast<uint32_t>(x / cB),
                                       static_cast<uint32_t>(y / cB), sd, 8u) >= 0.0f) ? 1 : -1;
                const signed char bcr =
                    (nrcore::hashNoise(static_cast<uint32_t>(x / cB),
                                       static_cast<uint32_t>(y / cB), sd, 9u) >= 0.0f) ? 1 : -1;
                float* pL = (pass == 0) ? pertL.data() : pertL2.data();
                float* pC = (pass == 0) ? pertC.data() : pertC2.data();
                (pass == 0 ? rad   : rad2)[i]   = b;
                (pass == 0 ? radCb : radCb2)[i] = bcb;
                (pass == 0 ? radCr : radCr2)[i] = bcr;
                const float d = eps * static_cast<float>(b);
                pL[i * 4 + 0] = cf[3][i * 4 + 0] + d;
                pL[i * 4 + 1] = cf[3][i * 4 + 1] + d;
                pL[i * 4 + 2] = cf[3][i * 4 + 2] + d;
                pL[i * 4 + 3] = cf[3][i * 4 + 3];
                const float dcb = epsCb * static_cast<float>(bcb);
                const float dcr = epsCr * static_cast<float>(bcr);
                pC[i * 4 + 0] = cf[3][i * 4 + 0] + 1.5748f * dcr;
                pC[i * 4 + 1] = cf[3][i * 4 + 1]
                              - (0.0722f * 1.8556f / 0.7152f) * dcb
                              - (0.2126f * 1.5748f / 0.7152f) * dcr;
                pC[i * 4 + 2] = cf[3][i * 4 + 2] + 1.8556f * dcb;
                pC[i * 4 + 3] = cf[3][i * 4 + 3];
            }
        }
    const float* pfL[7];  const float* pfC[7];
    const float* pfL2[7]; const float* pfC2[7];
    for (int k = 0; k < 7; ++k) {
        pfL[k]  = (cf[k] == cf[3]) ? pertL.data()  : cf[k];
        pfC[k]  = (cf[k] == cf[3]) ? pertC.data()  : cf[k];
        pfL2[k] = (cf[k] == cf[3]) ? pertL2.data() : cf[k];
        pfC2[k] = (cf[k] == cf[3]) ? pertC2.data() : cf[k];
    }

    std::vector<float> outA(n), outB(n), scratch;
    nrcore::Params cur = base;

    // ---- luma coordinate descent ----
    // {axis, step, lo, hi} in Params fractions. Steps sized to the sliders'
    // useful granularity; one pass, adopt only on improvement, ride a
    // winning direction at most one extra step.
    struct Axis { int id; float step, lo, hi; };
    static const Axis kLumaAxes[] = {
        { kAxMotionThresh,   0.15f, 0.0f, 1.5f  },
        { kAxPreserveDetail, 0.15f, 0.0f, 1.0f  },
        { kAxDetailRescue,   0.25f, 0.0f, 1.0f  },
        { kAxEqMedium,       0.35f, 0.0f, 1.5f  },
        { kAxEffnSteer,      0.50f, 0.0f, 1.0f  },
    };
    static const Axis kChromaAxes[] = {
        { kAxTemporalChroma, 0.25f, 0.0f, 1.25f },
        { kAxSpatialChroma,  0.25f, 0.0f, 1.5f  },
        { kAxChromaBlotch,   0.30f, 0.0f, 1.5f  },
    };

    double curScore = sureEvalLuma(cf, pfL, rad, cw, ch, cur, sigma, eps,
                                   outA, outB, scratch);
    ++r.evals;
    r.lumaBase = curScore;
    for (size_t a = 0; a < sizeof(kLumaAxes) / sizeof(kLumaAxes[0]); ++a) {
        const Axis& ax = kLumaAxes[a];
        for (int dir = -1; dir <= 1; dir += 2) {
            int rides = 0;
            for (;;) {
                const float v0 = sureAxisGet(cur, ax.id);
                const float v1 = clampfLocal(v0 + ax.step * static_cast<float>(dir),
                                             ax.lo, ax.hi);
                if (v1 == v0)
                    break;
                nrcore::Params cand = cur;
                sureAxisSet(cand, ax.id, v1);
                const double sc = sureEvalLuma(cf, pfL, rad, cw, ch, cand, sigma, eps,
                                               outA, outB, scratch);
                ++r.evals;
                if (sc >= curScore - 1e-12)
                    break;               // no improvement — stay put
                cur = cand;
                curScore = sc;
                if (++rides > 1)
                    break;               // ride a winner one extra step, max
            }
        }
    }
    r.lumaTuned = curScore;

    // ---- hold-out cross-validation (luma) ----
    // Greedy adoption can chase the driving probe's own MC variance on a
    // near-clean crop (measured: a 46 dB stack over-cleaned by ~1.4 dB of
    // true PSNR). Re-score the seed vs the luma winner on an INDEPENDENT
    // probe (seed 1013) and keep the winner only if the improvement survives
    // noise it never saw. A genuine denoising win (the whole point on real
    // footage) generalizes; an overfit to one probe does not.
    {
        const double base2 = sureEvalLuma(cf, pfL2, rad2, cw, ch, base, sigma, eps,
                                          outA, outB, scratch);
        const double cur2  = sureEvalLuma(cf, pfL2, rad2, cw, ch, cur,  sigma, eps,
                                          outA, outB, scratch);
        r.evals += 2;
        if (cur2 >= base2) {              // did not generalize — revert luma axes
            for (size_t a = 0; a < sizeof(kLumaAxes) / sizeof(kLumaAxes[0]); ++a)
                sureAxisSet(cur, kLumaAxes[a].id, sureAxisGet(base, kLumaAxes[a].id));
            r.lumaTuned = curScore = r.lumaBase;
        }
    }

    // ---- chroma descent (scored by chroma SURE, from the luma winner) ----
    double curC = sureEvalChroma(cf, pfC, radCb, radCr, cw, ch, cur,
                                 sigmaCb, sigmaCr, epsCb, epsCr,
                                 outA, outB, scratch);
    ++r.evals;
    r.chromaBase = curC;
    const nrcore::Params preChroma = cur;   // hold-out anchor for the chroma pass
    for (size_t a = 0; a < sizeof(kChromaAxes) / sizeof(kChromaAxes[0]); ++a) {
        const Axis& ax = kChromaAxes[a];
        for (int dir = -1; dir <= 1; dir += 2) {
            int rides = 0;
            for (;;) {
                const float v0 = sureAxisGet(cur, ax.id);
                const float v1 = clampfLocal(v0 + ax.step * static_cast<float>(dir),
                                             ax.lo, ax.hi);
                if (v1 == v0)
                    break;
                nrcore::Params cand = cur;
                sureAxisSet(cand, ax.id, v1);
                const double sc = sureEvalChroma(cf, pfC, radCb, radCr, cw, ch, cand,
                                                 sigmaCb, sigmaCr, epsCb, epsCr,
                                                 outA, outB, scratch);
                ++r.evals;
                if (sc >= curC - 1e-12)
                    break;
                cur = cand;
                curC = sc;
                if (++rides > 1)
                    break;
            }
        }
    }
    r.chromaTuned = curC;

    // ---- hold-out cross-validation (chroma) ----
    {
        const double pc2  = sureEvalChroma(cf, pfC2, radCb2, radCr2, cw, ch, preChroma,
                                           sigmaCb, sigmaCr, epsCb, epsCr,
                                           outA, outB, scratch);
        const double cur2 = sureEvalChroma(cf, pfC2, radCb2, radCr2, cw, ch, cur,
                                           sigmaCb, sigmaCr, epsCb, epsCr,
                                           outA, outB, scratch);
        r.evals += 2;
        if (cur2 >= pc2) {               // did not generalize — revert chroma axes
            for (size_t a = 0; a < sizeof(kChromaAxes) / sizeof(kChromaAxes[0]); ++a)
                sureAxisSet(cur, kChromaAxes[a].id, sureAxisGet(preChroma, kChromaAxes[a].id));
            r.chromaTuned = r.chromaBase;
        }
    }

    r.motionThresh   = cur.motionThresh;
    r.preserveDetail = cur.preserveDetail;
    r.detailRescue   = cur.detailRescue;
    r.eqMedium       = cur.eqMedium;
    r.effnSteer      = cur.effnSteer;
    r.temporalChroma = cur.temporalChroma;
    r.spatialChroma  = cur.spatialChroma;
    r.chromaBlotch   = cur.chromaBlotch;
    r.chromaCorr     = cB;
    r.ran = 1;
    return r;
}

} // namespace nranalyze

#endif // OPENNR_NR_ANALYZE_H
