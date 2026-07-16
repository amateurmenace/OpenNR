// OpenNR — free spatio-temporal noise reduction for DaVinci Resolve (OpenFX).
//
// Runs on the Color page (OpenFX panel), Edit page (Effects > OpenFX) and
// Fusion page, in both the free and Studio editions of Resolve.
//
// GPU paths: Metal (macOS), CUDA and OpenCL (Windows/Linux). A single-threaded
// CPU reference path (plugin/nr_core.h) is used if no GPU rendering is offered
// by the host; it is also the specification the GPU kernels are ported from.
//
// MIT License.

#include "OpenNRPlugin.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <climits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include "ofxsSupportPrivate.h"
#include "ofxDrawSuite.h"

#include "NRParams.h"

// v3.5 R1: the render-boost history is only usable when every parameter that
// shapes the temporal result matches the render that wrote it. FNV-1a over
// the params with the per-frame/display-only fields zeroed — frameIndex
// advances by design, views and scopes never touch the temporal buffer,
// histValid is this test's OUTPUT. hasTemporalDiff stays IN: clip-edge
// frames merge differently and must invalidate the chain. Keep in sync with
// the three GPU hosts. (NRParams is all 4-byte fields, no padding.)
static uint32_t boostParamsHash(const NRParams& p)
{
    NRParams c = p;
    c.frameIndex   = 0;
    c.viewMode     = 0;
    c.scopeMeasure = 0;
    c.scopeMotion  = 0;
    c.scopeEq      = 0;
    c.histValid    = 0;
    c.effnSteer    = 0;   // v3.6 S1: spatial-only, never touches the
                          // temporal buffer the history caches
    c.exportMatteAlpha = 0; // v3.7: output-alpha only — toggling it must not
                            // invalidate the boost history (like viewMode)
    uint32_t h = 2166136261u;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&c);
    for (size_t i = 0; i < sizeof(NRParams); ++i) {
        h ^= b[i];
        h *= 16777619u;
    }
    return h;
}
#include "nr_core.h"
#include "nr_analyze.h"

#define kPluginName "Hush Open NR"
#define kPluginGrouping "Hush"
#define kPluginDescription \
    "Hush Open NR v3.7 — the free noise reduction suite.\n\n" \
    "Click AUTO SETUP and the plugin measures your clip and dials in every " \
    "slider (one undo reverts) — or CLEAN SLATE to zero everything and work " \
    "fully manually. Work top to bottom: 1 measure, 2 temporal, 3 spatial, " \
    "4 refine, 5 inspect.\n\n" \
    "New in 3.6 — texture reconstruction: OPTICAL ACUTANCE adds lens-like " \
    "edge sharpness back (gated to real edges, clamped so there is no halo); " \
    "the FILM GRAIN is now matched to the noise that was removed (loud in " \
    "shadows, faded on edges, with an optional finer blue-noise spectrum); " \
    "LUMA TEXTURE cores out noise before it re-injects micro-detail so it no " \
    "longer re-noises shadows; SHADOW COLOR CLEANUP clears the blotchy color " \
    "speckle deep in shadows; and a CLEAN-CONFIDENCE matte exports where the " \
    "denoiser averaged deepest for downstream grading.\n\n" \
    "Each step has a Scope checkbox that draws a panel right in the viewer. " \
    "Turn scopes off before rendering.\n\n" \
    "MIT-licensed and free forever."
#define kPluginIdentifier "org.opennr.Denoise"
#define kPluginVersionMajor 3
#define kPluginVersionMinor 7

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////
// GPU entry points
////////////////////////////////////////////////////////////////////////////////

#ifdef __APPLE__
extern void RunMetalNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                       const float* const p_Srcs[7], float* p_Dst);
#endif
#ifdef HUSH_ENABLE_CUDA
extern void RunCudaNR(void* p_Stream, int p_Width, int p_Height, const NRParams& p_Params,
                      const float* const p_Srcs[7], float* p_Dst);
#endif
extern void RunOpenCLNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                        const float* const p_Srcs[7], float* p_Dst);

////////////////////////////////////////////////////////////////////////////////
// Processor
////////////////////////////////////////////////////////////////////////////////

class OpenNRProcessor : public OFX::ImageProcessor
{
public:
    explicit OpenNRProcessor(OFX::ImageEffect& p_Instance)
        : OFX::ImageProcessor(p_Instance)
    {
    }

    void setSrcImgs(OFX::Image* p_Imgs[7])
    {
        for (int i = 0; i < 7; ++i)
            _srcImgs[i] = p_Imgs[i];
    }

    void setParams(const NRParams& p) { _params = p; }

    virtual void processImagesMetal()
    {
#ifdef __APPLE__
        const OfxRectI& bounds = _srcImgs[3]->getBounds();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        const float* srcs[7];
        for (int i = 0; i < 7; ++i)
            srcs[i] = static_cast<float*>(_srcImgs[i]->getPixelData());
        RunMetalNR(_pMetalCmdQ, width, height, _params, srcs, static_cast<float*>(_dstImg->getPixelData()));
#endif
    }

    virtual void processImagesCUDA()
    {
#ifdef HUSH_ENABLE_CUDA
        const OfxRectI& bounds = _srcImgs[3]->getBounds();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        const float* srcs[7];
        for (int i = 0; i < 7; ++i)
            srcs[i] = static_cast<float*>(_srcImgs[i]->getPixelData());
        RunCudaNR(_pCudaStream, width, height, _params, srcs, static_cast<float*>(_dstImg->getPixelData()));
#endif
    }

    virtual void processImagesOpenCL()
    {
        const OfxRectI& bounds = _srcImgs[3]->getBounds();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        const float* srcs[7];
        for (int i = 0; i < 7; ++i)
            srcs[i] = static_cast<float*>(_srcImgs[i]->getPixelData());
        RunOpenCLNR(_pOpenCLCmdQ, width, height, _params, srcs, static_cast<float*>(_dstImg->getPixelData()));
    }

    // CPU fallback is handled whole-frame from the render action.
    virtual void multiThreadProcessImages(OfxRectI) {}

private:
    OFX::Image* _srcImgs[7] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    NRParams _params = {};
};

////////////////////////////////////////////////////////////////////////////////
// Plugin instance
////////////////////////////////////////////////////////////////////////////////

class OpenNRPlugin : public OFX::ImageEffect
{
public:
    explicit OpenNRPlugin(OfxImageEffectHandle p_Handle)
        : ImageEffect(p_Handle)
    {
        m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
        m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

        m_Master         = fetchDoubleParam("master");
        m_AutoSetup      = fetchPushButtonParam("autoSetup");
        m_AutoReport     = fetchStringParam("autoReport");
        m_CleanSlate     = fetchPushButtonParam("cleanSlate");
        m_AutoUndo       = fetchStringParam("autoSetupUndo");
        m_ProfileSource  = fetchChoiceParam("profileSource");
        m_ProfileAdjust  = fetchDoubleParam("profileAdjust");
        m_SigmaY         = fetchDoubleParam("sigmaLuma");
        m_SigmaC         = fetchDoubleParam("sigmaChroma");
        m_RegionCX       = fetchDoubleParam("regionCenterX");
        m_RegionCY       = fetchDoubleParam("regionCenterY");
        m_RegionSize     = fetchDoubleParam("regionSize");
        m_LockProfile    = fetchBooleanParam("lockProfile");
        m_LockData       = fetchStringParam("lockedProfileData");
        m_EnableTemporal = fetchBooleanParam("enableTemporal");
        m_TemporalFrames = fetchChoiceParam("temporalFrames");
        m_MotionTracking = fetchBooleanParam("motionTracking");
        m_TemporalLuma   = fetchDoubleParam("temporalLuma");
        m_TemporalChroma = fetchDoubleParam("temporalChroma");
        m_MotionThresh   = fetchDoubleParam("motionThreshold");
        m_FireflyRemoval = fetchBooleanParam("fireflyRemoval");
        m_RenderBoost    = fetchBooleanParam("renderBoost");
        m_EnableSpatial  = fetchBooleanParam("enableSpatial");
        m_SpatialMode    = fetchChoiceParam("spatialMethod");
        m_SpatialRadius  = fetchIntParam("spatialRadius");
        m_SpatialLuma    = fetchDoubleParam("spatialLuma");
        m_SpatialChroma  = fetchDoubleParam("spatialChroma");
        m_PreserveDetail = fetchDoubleParam("preserveDetail");
        m_DetailRescue   = fetchDoubleParam("detailRescue");
        m_EffnSteer      = fetchDoubleParam("adaptiveStrength");
        m_DeepClean      = fetchBooleanParam("deepClean");
        m_ChromaBlotch   = fetchDoubleParam("chromaBlotch");
        m_EqFine         = fetchDoubleParam("eqFine");
        m_EqMedium       = fetchDoubleParam("eqMedium");
        m_EqCoarse       = fetchDoubleParam("eqCoarse");
        m_ScopeMeasure   = fetchBooleanParam("scopeMeasure");
        m_ScopeMotion    = fetchBooleanParam("scopeMotion");
        m_ScopeEq        = fetchBooleanParam("scopeEq");
        m_GhostGuard     = fetchBooleanParam("ghostGuard");
        m_GlobalBlend    = fetchDoubleParam("globalBlend");
        m_EnableRefine   = fetchBooleanParam("enableRefine");
        m_ShadowDesat    = fetchDoubleParam("shadowDesat");
        m_DesatRange     = fetchDoubleParam("desatRange");
        m_LumaTexture    = fetchDoubleParam("lumaTexture");
        m_Deband         = fetchDoubleParam("deband");
        m_GrainAmount    = fetchDoubleParam("grainAmount");
        m_GrainSize      = fetchDoubleParam("grainSize");
        m_GrainChroma    = fetchDoubleParam("grainChroma");
        m_GrainBlue      = fetchDoubleParam("grainBlue");
        m_Acutance       = fetchDoubleParam("acutance");
        m_ChromaSpeckle  = fetchDoubleParam("chromaSpeckle");
        m_ViewMode       = fetchChoiceParam("viewMode");
        m_ExportMatte    = fetchBooleanParam("exportMatteAlpha");

        // restore a locked profile saved with the project
        std::string lockStr;
        m_LockData->getValue(lockStr);
        m_LockValid = nranalyze::parseLockedProfile(lockStr, m_LockAgg);

        updateEnabledness();
    }

    virtual void render(const OFX::RenderArguments& p_Args)
    {
        if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) &&
            (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
        {
            setupAndProcess(p_Args);
        }
        else
        {
            OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
    {
        const double t = p_Args.time;
        int view = 0;
        m_ViewMode->getValueAtTime(t, view);
        if (view != 0)
            return false;
        // v3.1: scope overlays must render even when the filters are neutral
        if (m_ScopeMeasure->getValueAtTime(t) || m_ScopeMotion->getValueAtTime(t) ||
            m_ScopeEq->getValueAtTime(t))
            return false;

        const double master = m_Master->getValueAtTime(t);
        const bool tOn = m_EnableTemporal->getValueAtTime(t) &&
                         (m_TemporalLuma->getValueAtTime(t) > 0.0 || m_TemporalChroma->getValueAtTime(t) > 0.0);
        const bool sOn = m_EnableSpatial->getValueAtTime(t) &&
                         (m_SpatialLuma->getValueAtTime(t) > 0.0 || m_SpatialChroma->getValueAtTime(t) > 0.0 ||
                          m_DeepClean->getValueAtTime(t));   // v3.3: the pre-pass renders on its own
        // v3: refine-only configurations (grain / texture / desat / deband
        // with both NR stages off) must render — v2.1 wrongly skipped them
        const bool rOn = m_EnableRefine->getValueAtTime(t) &&
                         (m_ShadowDesat->getValueAtTime(t) > 0.0 ||
                          m_LumaTexture->getValueAtTime(t) > 0.0 ||
                          m_GrainAmount->getValueAtTime(t) > 0.0 ||
                          m_Deband->getValueAtTime(t) > 0.0);

        if (master <= 0.0 || m_GlobalBlend->getValueAtTime(t) <= 0.0 || (!tOn && !sOn && !rOn))
        {
            p_IdentityClip = m_SrcClip;
            p_IdentityTime = t;
            return true;
        }
        return false;
    }

    virtual void getFramesNeeded(const OFX::FramesNeededArguments& p_Args, OFX::FramesNeededSetter& p_FramesNeededSetter)
    {
        int reach = 0, prevReach = 0;
        fetchReach(p_Args.time, reach, prevReach);
        OfxRangeD range;
        range.min = p_Args.time - prevReach;
        range.max = p_Args.time + reach;
        p_FramesNeededSetter.setFramesNeeded(*m_SrcClip, range);
    }

    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
    {
        // m_InAutoApply guards recursion: Auto Setup / Revert set many params
        // (including lockProfile) from inside this action.
        if (!m_InAutoApply)
        {
            if (p_ParamName == "autoSetup")
                runAutoSetup(p_Args.time);
            else if (p_ParamName == "autoRegion")
                runAutoRegion(p_Args.time);
            else if (p_ParamName == "cleanSlate")
                runCleanSlate();
            else if (p_ParamName == "lockProfile")
                lockProfileToggled(p_Args.time);

            // v3.1: the first manual touch of a Noise EQ control pops the EQ
            // scope so the user can SEE what the bands do. Once per instance —
            // if they close it, it stays closed.
            if (!m_EqScopeShown &&
                (p_ParamName == "eqFine" || p_ParamName == "eqMedium" ||
                 p_ParamName == "eqCoarse" || p_ParamName == "chromaBlotch"))
            {
                m_EqScopeShown = true;
                bool on = false;
                m_ScopeEq->getValue(on);
                if (!on)
                    m_ScopeEq->setValue(true);
            }
        }

        if (p_ParamName == "profileSource" || p_ParamName == "enableTemporal" ||
            p_ParamName == "enableSpatial" || p_ParamName == "enableRefine" ||
            p_ParamName == "lockProfile")
            updateEnabledness();
    }

private:
    // ------------------------------------------------------------------
    // v3 clip analysis (Auto Setup + Lock Profile)
    // ------------------------------------------------------------------

    static void packImage(OFX::Image* p_Img, int& p_W, int& p_H, std::vector<float>& p_Out)
    {
        const OfxRectI& b = p_Img->getBounds();
        p_W = b.x2 - b.x1;
        p_H = b.y2 - b.y1;
        if (p_W <= 0 || p_H <= 0) { p_W = p_H = 0; return; }
        p_Out.resize(static_cast<size_t>(p_W) * p_H * 4);
        for (int y = 0; y < p_H; ++y)
        {
            const float* row = static_cast<float*>(p_Img->getPixelAddress(b.x1, b.y1 + y));
            if (row)
                std::memcpy(&p_Out[static_cast<size_t>(y) * p_W * 4], row,
                            static_cast<size_t>(p_W) * 4 * sizeof(float));
            else
                std::memset(&p_Out[static_cast<size_t>(y) * p_W * 4], 0,
                            static_cast<size_t>(p_W) * 4 * sizeof(float));
        }
    }

    // Fetches the current frame plus up to 4 frames spread across the clip
    // and runs the nrcore estimators on each. fetchImage inside
    // kOfxActionInstanceChanged is not guaranteed by the OFX spec (Resolve
    // honors it, but a busy host may return nothing) — every failure path
    // returns what was gathered so the caller can message the user; this
    // never throws out of the action and never crashes.
    // p_Flat (optional): v3.3 B4 — the flattest-patch scan runs on the
    // first successfully fetched frame (the playhead frame)
    // p_ForceWholeFrame: v3.4 — Auto Setup in region mode measures the
    // spread frames whole-frame (they classify the CLIP: motion energy);
    // a screen-fixed region means nothing at distant times — that was the
    // "lock brings the noise back" bug. The region's own sigmas come from
    // the tracked playhead sweep (analyzeLockSweepAt) instead.
    int analyzeClipFrames(double p_Time, std::vector<nrcore::Stats>& p_Out,
                          nranalyze::FlatPatch* p_Flat = nullptr,
                          bool p_ForceWholeFrame = false)
    {
        try
        {
            if (!m_SrcClip || !m_SrcClip->isConnected())
                return 0;
            const OfxRangeD range = m_SrcClip->getFrameRange();
            std::vector<double> times;
            times.push_back(p_Time);
            const double len = range.max - range.min;
            for (int k = 0; k < 4 && len >= 1.0; ++k)
            {
                double t = std::floor(range.min + (0.5 + k) * len / 4.0 + 0.5);
                if (t < range.min) t = range.min;
                if (t > range.max) t = range.max;
                bool dup = false;
                for (size_t i = 0; i < times.size(); ++i)
                    if (std::fabs(times[i] - t) < 1.0) { dup = true; break; }
                if (!dup)
                    times.push_back(t);
            }
            // v3.2: honor the user's measurement setup — if they placed a
            // sampling region, measure THERE, not the whole frame (this was
            // the "lock brings the noise back" bug: locking silently swapped
            // a region profile for a blander whole-frame one). The lock
            // stores the RAW measurement — Auto Profile Adjust is applied
            // live at use time so the trim keeps working afterwards.
            nrcore::Params ap;
            int src = 0;
            m_ProfileSource->getValue(src);
            if (src == 1 && !p_ForceWholeFrame)
            {
                ap.profileSource = 1;
                ap.regionCX   = static_cast<float>(m_RegionCX->getValue());
                ap.regionCY   = static_cast<float>(m_RegionCY->getValue());
                ap.regionSize = static_cast<float>(m_RegionSize->getValue());
            }
            for (size_t ti = 0; ti < times.size(); ++ti)
            {
                const double t = times[ti];
                std::unique_ptr<OFX::Image> img(m_SrcClip->fetchImage(t));
                if (!img || img->getPixelDepth() != OFX::eBitDepthFloat ||
                    img->getPixelComponents() != OFX::ePixelComponentRGBA)
                    continue;
                int W = 0, H = 0;
                std::vector<float> buf;
                packImage(img.get(), W, H, buf);
                if (W < 16 || H < 16)
                    continue;
                // temporal-diff partner: the next frame (previous at clip end)
                double tp = (t + 1.0 <= range.max) ? t + 1.0 : t - 1.0;
                if (tp < range.min) tp = t;
                std::vector<float> pbuf;
                const float* partner = nullptr;
                if (tp != t)
                {
                    std::unique_ptr<OFX::Image> pimg(m_SrcClip->fetchImage(tp));
                    if (pimg && pimg->getPixelDepth() == OFX::eBitDepthFloat &&
                        pimg->getPixelComponents() == OFX::ePixelComponentRGBA)
                    {
                        int PW = 0, PH = 0;
                        packImage(pimg.get(), PW, PH, pbuf);
                        if (PW == W && PH == H)
                            partner = pbuf.data();
                    }
                }
                nrcore::Stats st;
                nrcore::estimateInput(buf.data(), partner, W, H, ap, st);
                if (p_Flat && !p_Flat->valid)
                    *p_Flat = nranalyze::findFlattestPatch(buf.data(), W, H, st);
                p_Out.push_back(st);
            }
            return static_cast<int>(p_Out.size());
        }
        catch (...)
        {
            return static_cast<int>(p_Out.size());
        }
    }

    // v3.4 — the tracked lock sweep at the playhead (see nr_analyze.h for
    // the semantics). This is a streaming port of nranalyze::lockSweepAnalyze
    // feeding the same helpers — at most 3 frames resident (anchor + a
    // sliding prev/cur pair) instead of 9, because instanceChanged can run
    // on a UHD timeline where nine packed float frames are >1 GB.
    bool analyzeLockSweepAt(double p_Time, nranalyze::LockSweep& p_Out)
    {
        try
        {
            if (!m_SrcClip || !m_SrcClip->isConnected())
                return false;
            const OfxRangeD range = m_SrcClip->getFrameRange();

            nrcore::Params ap;
            int src = 0;
            m_ProfileSource->getValue(src);
            const bool region = (src == 1);
            if (region)
            {
                ap.profileSource = 1;
                ap.regionCX   = static_cast<float>(m_RegionCX->getValue());
                ap.regionCY   = static_cast<float>(m_RegionCY->getValue());
                ap.regionSize = static_cast<float>(m_RegionSize->getValue());
            }

            int W = 0, H = 0;
            auto fetchPacked = [&](double t, std::vector<float>& buf) -> bool {
                std::unique_ptr<OFX::Image> img(m_SrcClip->fetchImage(t));
                if (!img || img->getPixelDepth() != OFX::eBitDepthFloat ||
                    img->getPixelComponents() != OFX::ePixelComponentRGBA)
                    return false;
                int w = 0, h = 0;
                packImage(img.get(), w, h, buf);
                if (w < 16 || h < 16)
                    return false;
                if (W == 0) { W = w; H = h; }
                return (w == W && h == H);
            };

            std::vector<float> anchorBuf;
            if (!fetchPacked(p_Time, anchorBuf))
                return false;

            nranalyze::SweepSample anchor;
            bool anchorMeasured = false;
            std::vector<nranalyze::SweepSample> others;

            for (int dir = +1; dir >= -1; dir -= 2)
            {
                std::vector<float> prev;
                const float* prevPtr = anchorBuf.data();
                nranalyze::RegionTrack tr;
                for (int k = 1; k <= 4; ++k)
                {
                    const double t = p_Time + dir * k;
                    if (t > range.max || t < range.min)
                        break;
                    std::vector<float> cur;
                    if (!fetchPacked(t, cur))
                        break;
                    // the first fetched neighbour doubles as the anchor's
                    // temporal-diff partner (matches the reference: next
                    // frame when the forward pass produced one, else prev)
                    if (!anchorMeasured)
                    {
                        nranalyze::measureAtOffset(anchorBuf.data(), cur.data(), W, H,
                                                   ap, 0, 0, anchor);
                        anchorMeasured = true;
                    }
                    if (region)
                    {
                        tr = nranalyze::trackRegionStep(prevPtr, cur.data(), W, H,
                                                        ap.regionCX, ap.regionCY,
                                                        ap.regionSize, tr);
                        if (!tr.ok)
                            break;
                    }
                    nranalyze::SweepSample s;
                    nranalyze::measureAtOffset(cur.data(), prevPtr, W, H,
                                               ap, tr.dx, tr.dy, s);
                    others.push_back(s);
                    prev = std::move(cur);
                    prevPtr = prev.data();
                }
            }
            if (!anchorMeasured)   // single-frame clip: no partner anywhere
                nranalyze::measureAtOffset(anchorBuf.data(), nullptr, W, H,
                                           ap, 0, 0, anchor);

            p_Out = nranalyze::composeLockAggregate(anchor, others,
                        (region && !others.empty()) ? 1 : 0);
            return p_Out.measured > 0;
        }
        catch (...)
        {
            return false;
        }
    }

    void analysisFailedMessage()
    {
        sendMessage(OFX::Message::eMessageMessage, "",
                    "Hush could not read the clip to analyze it. This can happen while "
                    "the host is busy rendering — pause playback, park the playhead on "
                    "this clip and try again.");
    }

    // one k=v per param, '|'-separated; the lock data string is comma/hex so
    // the separators never collide
    std::string snapshotDenoiseParams()
    {
        char buf[2048];
        int tf = 0, sm = 0;
        m_TemporalFrames->getValue(tf);
        m_SpatialMode->getValue(sm);
        std::string lockStr;
        m_LockData->getValue(lockStr);
        snprintf(buf, sizeof(buf),
                 "v1|et=%d|tf=%d|tl=%.17g|tc=%.17g|mt=%.17g|mtr=%d|ff=%d|gg=%d|rb=%d|es=%d|sm=%d|sr=%d|"
                 "sl=%.17g|sc=%.17g|pd=%.17g|rs=%.17g|est=%.17g|dc=%d|cb=%.17g|eqf=%.17g|eqm=%.17g|eqc=%.17g|pa=%.17g|lp=%d|ld=%s",
                 m_EnableTemporal->getValue() ? 1 : 0, tf,
                 m_TemporalLuma->getValue(), m_TemporalChroma->getValue(), m_MotionThresh->getValue(),
                 m_MotionTracking->getValue() ? 1 : 0, m_FireflyRemoval->getValue() ? 1 : 0,
                 m_GhostGuard->getValue() ? 1 : 0,
                 m_RenderBoost->getValue() ? 1 : 0,
                 m_EnableSpatial->getValue() ? 1 : 0, sm, m_SpatialRadius->getValue(),
                 m_SpatialLuma->getValue(), m_SpatialChroma->getValue(),
                 m_PreserveDetail->getValue(), m_DetailRescue->getValue(),
                 m_EffnSteer->getValue(),
                 m_DeepClean->getValue() ? 1 : 0, m_ChromaBlotch->getValue(),
                 m_EqFine->getValue(), m_EqMedium->getValue(), m_EqCoarse->getValue(),
                 m_ProfileAdjust->getValue(), m_LockProfile->getValue() ? 1 : 0,
                 lockStr.c_str());
        return std::string(buf);
    }

    static bool snapField(const std::string& s, const char* key, std::string& out)
    {
        const std::string k = std::string("|") + key + "=";
        const size_t pos = s.find(k);
        if (pos == std::string::npos)
            return false;
        const size_t v0 = pos + k.size();
        size_t v1 = s.find('|', v0);
        if (v1 == std::string::npos)
            v1 = s.size();
        out = s.substr(v0, v1 - v0);
        return true;
    }

    // v3.5 X1 — MC-SURE self-tune at Auto Setup: fetch the real 7-frame
    // stack at the playhead, centre-crop it (<=512x288 — the grid costs 18
    // CPU denoises, so the crop pays that bill, not the user) and sweep
    // (temporal luma, spatial luma) around the table values. Any fetch
    // failure returns not-ran: Auto Setup must never fail because of the
    // tuner. Deterministic: fixed-seed Rademacher, same playhead = same
    // frames = same answer.
    nranalyze::SureTune tuneAutoSetupAt(double p_Time, const nranalyze::AutoSettings& p_As,
                                        nranalyze::SureDescent& p_Desc)
    {
        nranalyze::SureTune r;
        try
        {
            if (!m_SrcClip || !m_SrcClip->isConnected())
                return r;
            const OfxRangeD range = m_SrcClip->getFrameRange();
            // seven times, clamped at the clip edges; duplicate times ALIAS
            // the same crop pointer so the pipeline's pointer-equality edge
            // logic sees a real clip edge, exactly like a render would
            double times[7];
            for (int k = 0; k < 7; ++k)
            {
                double tk = p_Time + (k - 3);
                if (tk < range.min) tk = range.min;
                if (tk > range.max) tk = range.max;
                times[k] = tk;
            }
            std::vector<float> crops[7];
            const float* fp[7] = { 0, 0, 0, 0, 0, 0, 0 };
            int cw = 0, ch = 0;
            for (int k = 0; k < 7; ++k)
            {
                int prior = -1;
                for (int j = 0; j < k; ++j)
                    if (times[j] == times[k]) { prior = j; break; }
                if (prior >= 0)
                {
                    fp[k] = fp[prior];
                    continue;
                }
                std::unique_ptr<OFX::Image> img(m_SrcClip->fetchImage(times[k]));
                if (!img || img->getPixelDepth() != OFX::eBitDepthFloat ||
                    img->getPixelComponents() != OFX::ePixelComponentRGBA)
                    return r;
                int W = 0, H = 0;
                std::vector<float> buf;
                packImage(img.get(), W, H, buf);
                if (W < 64 || H < 64)
                    return r;
                const int tw = std::min(W, 512) & ~1;
                const int th = std::min(H, 288) & ~1;
                if (cw == 0) { cw = tw; ch = th; }
                else if (tw != cw || th != ch)
                    return r;
                const int x0 = (W - tw) / 2, y0 = (H - th) / 2;
                crops[k].resize(static_cast<size_t>(tw) * th * 4);
                for (int y = 0; y < th; ++y)
                    std::memcpy(&crops[k][static_cast<size_t>(y) * tw * 4],
                                &buf[(static_cast<size_t>(y0 + y) * W + x0) * 4],
                                static_cast<size_t>(tw) * 4 * sizeof(float));
                fp[k] = crops[k].data();
            }
            r = nranalyze::sureTuneGrid(fp, cw, ch, nranalyze::paramsFromAutoSettings(p_As));
            // v3.6 #19: coordinate descent + chroma SURE, seeded from the
            // grid winner, on the same fetched crop (it sub-crops itself
            // to <=384x216 to pay its own denoise bill). Failure leaves
            // p_Desc.ran == 0 and the grid/table values stand.
            if (r.ran)
            {
                nranalyze::AutoSettings asT = p_As;
                asT.temporalLuma = r.temporalLuma * 100.0f;
                asT.spatialLuma  = r.spatialLuma * 100.0f;
                p_Desc = nranalyze::sureTuneDescent(fp, cw, ch,
                                                    nranalyze::paramsFromAutoSettings(asT));
            }
        }
        catch (...)
        {
            r.ran = 0;   // any host hiccup: the table values stand
            p_Desc.ran = 0;
        }
        return r;
    }

    void runAutoSetup(double p_Time)
    {
        int srcNow0 = 0;
        m_ProfileSource->getValue(srcNow0);
        const bool fromRegion = (srcNow0 == 1);

        // spread frames classify the CLIP (motion energy; the noise class in
        // whole-frame mode) — in region mode they are measured whole-frame,
        // and the noise itself comes from the tracked playhead sweep below,
        // exactly what Lock Profile would store (v3.4)
        std::vector<nrcore::Stats> per;
        nranalyze::FlatPatch flat;
        if (analyzeClipFrames(p_Time, per, &flat, fromRegion) <= 0)
        {
            analysisFailedMessage();
            return;
        }
        nranalyze::ClipAggregate agg = nranalyze::aggregateClipStats(per);
        if (fromRegion)
        {
            nranalyze::LockSweep sw;
            if (!analyzeLockSweepAt(p_Time, sw) || sw.measured <= 0)
            {
                analysisFailedMessage();
                return;
            }
            const float clipMotion = agg.motion;   // a property of the clip,
            agg = sw.agg;                          // not of the user's patch
            agg.motion = clipMotion;
        }
        nranalyze::AutoSettings as = nranalyze::mapAnalysisToSettings(agg);

        // v3.5 X1: MC-SURE self-tune on a centre crop of the real stack at
        // the playhead — the two swept sliders are overwritten only when the
        // sweep actually ran (~3-5 s, all CPU, deterministic; the table is
        // always the fallback)
        nranalyze::SureDescent desc;
        const nranalyze::SureTune tuned = tuneAutoSetupAt(p_Time, as, desc);
        if (tuned.ran)
        {
            as.temporalLuma = tuned.temporalLuma * 100.0f;
            as.spatialLuma  = tuned.spatialLuma * 100.0f;
        }
        // v3.6 #19: the descent's winners land in the same sliders the
        // user sees — measurement over table, table as fallback
        if (desc.ran)
        {
            as.motionThresh   = desc.motionThresh * 100.0f;
            as.preserveDetail = desc.preserveDetail * 100.0f;
            as.detailRescue   = desc.detailRescue * 100.0f;
            as.eqMedium       = desc.eqMedium * 100.0f;
            as.effnSteer      = desc.effnSteer * 100.0f;
            as.temporalChroma = desc.temporalChroma * 100.0f;
            as.spatialChroma  = desc.spatialChroma * 100.0f;
            as.chromaBlotch   = desc.chromaBlotch * 100.0f;
        }

        // belt and braces: the whole apply is one edit block (single Cmd+Z),
        // AND the prior values are snapshotted for the Revert button
        m_AutoUndo->setValue(snapshotDenoiseParams());

        m_InAutoApply = true;
        beginEditBlock("Hush Auto Setup");
        m_EnableTemporal->setValue(as.enableTemporal != 0);
        m_TemporalFrames->setValue(as.temporalFrames >= 7 ? 2 : (as.temporalFrames >= 5 ? 1 : 0));
        m_TemporalLuma->setValue(as.temporalLuma);
        m_TemporalChroma->setValue(as.temporalChroma);
        m_MotionThresh->setValue(as.motionThresh);
        m_MotionTracking->setValue(as.motionTracking != 0);
        m_FireflyRemoval->setValue(as.fireflyRemoval != 0);
        m_GhostGuard->setValue(as.ghostGuard != 0);
        m_EnableSpatial->setValue(as.enableSpatial != 0);
        m_SpatialMode->setValue(as.spatialMode);
        m_SpatialRadius->setValue(as.spatialRadius);
        m_SpatialLuma->setValue(as.spatialLuma);
        m_SpatialChroma->setValue(as.spatialChroma);
        m_PreserveDetail->setValue(as.preserveDetail);
        m_DetailRescue->setValue(as.detailRescue);
        m_EffnSteer->setValue(as.effnSteer);
        m_DeepClean->setValue(as.deepClean != 0);
        m_ChromaBlotch->setValue(as.chromaBlotch);
        m_EqFine->setValue(as.eqFine);
        m_EqMedium->setValue(as.eqMedium);
        m_EqCoarse->setValue(as.eqCoarse);
        m_ProfileAdjust->setValue(as.profileAdjust);
        m_LockData->setValue(nranalyze::formatLockedProfile(agg));
        m_LockProfile->setValue(as.lockProfile != 0);
        int srcNow = 0;
        m_ProfileSource->getValue(srcNow);
        std::string report = nranalyze::formatAutoReport(agg, as, srcNow == 1 ? 1 : 0, flat);
        if (desc.ran)
            report += " \xc2\xb7 deep-tuned";  // grid + coordinate descent + chroma SURE
        else if (tuned.ran)
            report += " \xc2\xb7 tuned";   // the SURE sweep confirmed or moved the sliders
        m_AutoReport->setValue(report);
        endEditBlock();
        m_InAutoApply = false;

        m_LockAgg = agg;
        m_LockValid = true;
        updateEnabledness();
    }

    // v3.2: zero every processing control so the node passes the image
    // through untouched — the fully-manual starting point (defaults and
    // Auto Setup both process; this is the "do nothing until I say so"
    // button). One Cmd+Z restores the previous state (single edit block).
    void runCleanSlate()
    {
        m_AutoUndo->setValue(snapshotDenoiseParams());
        m_InAutoApply = true;
        beginEditBlock("Hush Clean Slate");
        m_Master->setValue(1.0);
        m_TemporalLuma->setValue(0.0);
        m_TemporalChroma->setValue(0.0);
        m_MotionThresh->setValue(30.0);
        m_SpatialLuma->setValue(0.0);
        m_SpatialChroma->setValue(0.0);
        m_PreserveDetail->setValue(35.0);
        m_DetailRescue->setValue(0.0);
        m_EffnSteer->setValue(0.0);
        m_DeepClean->setValue(false);
        m_ChromaBlotch->setValue(0.0);
        m_EqFine->setValue(100.0);
        m_EqMedium->setValue(0.0);
        m_EqCoarse->setValue(0.0);
        m_ShadowDesat->setValue(0.0);
        m_LumaTexture->setValue(0.0);
        m_Deband->setValue(0.0);
        m_GrainAmount->setValue(0.0);
        m_GlobalBlend->setValue(100.0);
        m_ProfileAdjust->setValue(1.0);
        m_LockProfile->setValue(false);
        m_AutoReport->setValue("Clean slate — nothing is processing. Raise sliders, or click Auto Setup.");
        endEditBlock();
        m_InAutoApply = false;
        updateEnabledness();
    }

    // v3.4 — Auto Region: find the flattest patch on the playhead frame,
    // switch the profile to From Region and move the yellow box there. The
    // button IS the consent to move the rectangle (Auto Setup still never
    // touches it). Does not lock — the user watches the live measurement at
    // the new spot, then locks (or runs Auto Setup). One undo restores.
    void runAutoRegion(double p_Time)
    {
        try
        {
            if (!m_SrcClip || !m_SrcClip->isConnected())
            {
                analysisFailedMessage();
                return;
            }
            std::unique_ptr<OFX::Image> img(m_SrcClip->fetchImage(p_Time));
            if (!img || img->getPixelDepth() != OFX::eBitDepthFloat ||
                img->getPixelComponents() != OFX::ePixelComponentRGBA)
            {
                analysisFailedMessage();
                return;
            }
            int W = 0, H = 0;
            std::vector<float> buf;
            packImage(img.get(), W, H, buf);
            if (W < 64 || H < 64)
            {
                analysisFailedMessage();
                return;
            }

            // whole-frame estimate first: the scan scores candidates against
            // the expected noise at their brightness (gain curve)
            nrcore::Params ap;
            nrcore::Stats st;
            nrcore::estimateInput(buf.data(), nullptr, W, H, ap, st);
            const float rs = static_cast<float>(m_RegionSize->getValue());
            const nranalyze::FlatPatch fp =
                nranalyze::findFlattestPatch(buf.data(), W, H, st, rs);
            if (!fp.valid)
            {
                analysisFailedMessage();
                return;
            }

            // what the region will read at the new spot (for the status line)
            nrcore::Params rp;
            rp.profileSource = 1;
            rp.regionCX = fp.cx;
            rp.regionCY = fp.cy;
            rp.regionSize = rs;
            nrcore::Stats rst;
            nrcore::estimateInput(buf.data(), nullptr, W, H, rp, rst);

            m_InAutoApply = true;
            beginEditBlock("Hush Auto Region");
            m_ProfileSource->setValue(1);
            m_RegionCX->setValue(fp.cx);
            m_RegionCY->setValue(fp.cy);
            m_LockProfile->setValue(false);   // a new spot measures live again
            m_AutoReport->setValue(nranalyze::formatRegionReport(fp, rst));
            endEditBlock();
            m_InAutoApply = false;
            updateEnabledness();
        }
        catch (...)
        {
            m_InAutoApply = false;
            analysisFailedMessage();
        }
    }

    void lockProfileToggled(double p_Time)
    {
        bool on = false;
        m_LockProfile->getValue(on);
        if (!on)
        {
            // unlock: back to live per-frame measurement (stored data kept —
            // re-ticking without re-analysis restores it via the project)
            m_AutoReport->setValue("Profile unlocked \xe2\x80\x94 measuring live again, "
                                   "every frame.");
            return;
        }
        // v3.4: freeze what the user is LOOKING AT — the playhead measurement,
        // hardened by the tracked +/-4-frame sweep (see nr_analyze.h). The
        // pre-3.4 clip-spread median could lock junk on moving footage: the
        // live render measured the playhead, the lock measured four distant
        // frames, and the NR visibly collapsed the moment you locked.
        int src = 0;
        m_ProfileSource->getValue(src);
        nranalyze::LockSweep sw;
        if (!analyzeLockSweepAt(p_Time, sw) || sw.measured <= 0)
        {
            m_InAutoApply = true;
            m_LockProfile->setValue(false);
            m_InAutoApply = false;
            analysisFailedMessage();
            return;
        }
        m_LockData->setValue(nranalyze::formatLockedProfile(sw.agg));
        m_LockAgg = sw.agg;
        m_LockValid = true;
        m_AutoReport->setValue(nranalyze::formatLockReport(sw, src == 1 ? 1 : 0));
    }

    // reach: frames needed after t for temporal NR; prevReach: frames needed
    // before t (temporal NR, plus one frame for the noise estimator when
    // auto profiling is on).
    void fetchReach(double t, int& reach, int& prevReach)
    {
        reach = 0;
        if (m_EnableTemporal->getValueAtTime(t))
        {
            int choice = 0;
            m_TemporalFrames->getValueAtTime(t, choice);
            // 0 -> 3 frames (+/-1), 1 -> 5 (+/-2), 2 -> 7 (+/-3, v3.3)
            reach = (choice == 2) ? 3 : (choice == 1) ? 2 : 1;
        }
        int source = 0;
        m_ProfileSource->getValueAtTime(t, source);
        prevReach = (source != 2) ? std::max(reach, 1) : reach;
    }

    void updateEnabledness()
    {
        int source = 0;
        m_ProfileSource->getValue(source);
        bool locked = false;
        m_LockProfile->getValue(locked);
        const bool autoOn = (source != 2);
        const bool regionOn = (source == 1);
        // v3.4: a locked profile overrides everything (even Manual), so the
        // inputs it ignores grey out — the region stops sampling (the box
        // dims and its dragging goes inert too), the manual sigmas stop
        // mattering, and only Auto Profile Adjust keeps trimming the lock.
        m_SigmaY->setEnabled(!autoOn && !locked);
        m_SigmaC->setEnabled(!autoOn && !locked);
        m_ProfileAdjust->setEnabled(autoOn || locked);
        m_RegionCX->setEnabled(regionOn && !locked);
        m_RegionCY->setEnabled(regionOn && !locked);
        m_RegionSize->setEnabled(regionOn && !locked);

        const bool tOn = m_EnableTemporal->getValue();
        m_TemporalFrames->setEnabled(tOn);
        m_MotionTracking->setEnabled(tOn);
        m_TemporalLuma->setEnabled(tOn);
        m_TemporalChroma->setEnabled(tOn);
        m_MotionThresh->setEnabled(tOn);
        m_FireflyRemoval->setEnabled(tOn);
        m_RenderBoost->setEnabled(tOn);

        const bool sOn = m_EnableSpatial->getValue();
        m_SpatialMode->setEnabled(sOn);
        m_SpatialRadius->setEnabled(sOn);
        m_SpatialLuma->setEnabled(sOn);
        m_SpatialChroma->setEnabled(sOn);
        m_PreserveDetail->setEnabled(sOn);
        m_DetailRescue->setEnabled(sOn);
        // v3.6 S1: needs the temporal stage's per-pixel sample counts
        m_EffnSteer->setEnabled(sOn && tOn);
        m_DeepClean->setEnabled(sOn);
        m_ChromaBlotch->setEnabled(sOn);
        m_EqFine->setEnabled(sOn);
        m_EqMedium->setEnabled(sOn);
        m_EqCoarse->setEnabled(sOn);

        const bool rOn = m_EnableRefine->getValue();
        m_ShadowDesat->setEnabled(rOn);
        m_DesatRange->setEnabled(rOn);
        m_LumaTexture->setEnabled(rOn);
        m_Deband->setEnabled(rOn);
        m_GrainAmount->setEnabled(rOn);
        m_GrainSize->setEnabled(rOn);
        m_GrainChroma->setEnabled(rOn);
        m_GrainBlue->setEnabled(rOn);
        m_Acutance->setEnabled(rOn);
        m_ChromaSpeckle->setEnabled(rOn);
    }

    NRParams gatherParams(double t)
    {
        NRParams p;
        int source = 0;
        m_ProfileSource->getValueAtTime(t, source);
        p.profileSource   = source;
        p.sigmaY          = static_cast<float>(m_SigmaY->getValueAtTime(t) / 100.0);
        p.sigmaC          = static_cast<float>(m_SigmaC->getValueAtTime(t) / 100.0);
        p.profileAdjust   = static_cast<float>(m_ProfileAdjust->getValueAtTime(t));
        p.regionCX        = static_cast<float>(m_RegionCX->getValueAtTime(t));
        p.regionCY        = static_cast<float>(m_RegionCY->getValueAtTime(t));
        p.regionSize      = static_cast<float>(m_RegionSize->getValueAtTime(t));
        p.hasTemporalDiff = 0; // decided by the GPU/CPU hosts from actual frame pointers

        p.enableTemporal  = m_EnableTemporal->getValueAtTime(t) ? 1 : 0;
        int framesChoice = 0;
        m_TemporalFrames->getValueAtTime(t, framesChoice);
        p.temporalFrames  = (framesChoice == 2) ? 7 : (framesChoice == 1) ? 5 : 3;
        p.temporalLuma    = static_cast<float>(m_TemporalLuma->getValueAtTime(t) / 100.0);
        p.temporalChroma  = static_cast<float>(m_TemporalChroma->getValueAtTime(t) / 100.0);
        p.motionThresh    = static_cast<float>(m_MotionThresh->getValueAtTime(t) / 100.0);

        p.enableSpatial   = m_EnableSpatial->getValueAtTime(t) ? 1 : 0;
        int mode = 1;
        m_SpatialMode->getValueAtTime(t, mode);
        p.spatialMode     = mode;
        p.spatialRadius   = m_SpatialRadius->getValueAtTime(t);
        p.spatialLuma     = static_cast<float>(m_SpatialLuma->getValueAtTime(t) / 100.0);
        p.spatialChroma   = static_cast<float>(m_SpatialChroma->getValueAtTime(t) / 100.0);
        p.preserveDetail  = static_cast<float>(m_PreserveDetail->getValueAtTime(t) / 100.0);
        p.chromaBlotch    = static_cast<float>(m_ChromaBlotch->getValueAtTime(t) / 100.0);

        p.enableRefine    = m_EnableRefine->getValueAtTime(t) ? 1 : 0;
        p.shadowDesat     = static_cast<float>(m_ShadowDesat->getValueAtTime(t) / 100.0);
        p.desatRange      = static_cast<float>(m_DesatRange->getValueAtTime(t));
        p.lumaTexture     = static_cast<float>(m_LumaTexture->getValueAtTime(t) / 100.0);
        p.grainAmount     = static_cast<float>(m_GrainAmount->getValueAtTime(t) / 100.0);
        p.grainSize       = static_cast<float>(m_GrainSize->getValueAtTime(t));
        p.grainChroma     = static_cast<float>(m_GrainChroma->getValueAtTime(t) / 100.0);
        p.grainBlue       = static_cast<float>(m_GrainBlue->getValueAtTime(t) / 100.0);
        p.acutance        = static_cast<float>(m_Acutance->getValueAtTime(t) / 100.0);
        p.chromaSpeckle   = static_cast<float>(m_ChromaSpeckle->getValueAtTime(t) / 100.0);
        p.frameIndex      = static_cast<int>(t);

        p.master          = static_cast<float>(m_Master->getValueAtTime(t));
        int view = 0;
        m_ViewMode->getValueAtTime(t, view);
        p.viewMode        = view;
        p.exportMatteAlpha = m_ExportMatte->getValueAtTime(t) ? 1 : 0;

        // ---- v3 ----
        p.motionTracking  = m_MotionTracking->getValueAtTime(t) ? 1 : 0;
        p.fireflyRemoval  = m_FireflyRemoval->getValueAtTime(t) ? 1 : 0;
        p.eqFine          = static_cast<float>(m_EqFine->getValueAtTime(t) / 100.0);
        p.eqMedium        = static_cast<float>(m_EqMedium->getValueAtTime(t) / 100.0);
        p.eqCoarse        = static_cast<float>(m_EqCoarse->getValueAtTime(t) / 100.0);
        p.deband          = static_cast<float>(m_Deband->getValueAtTime(t) / 100.0);

        // ---- v3.1 ----
        p.detailRescue    = static_cast<float>(m_DetailRescue->getValueAtTime(t) / 100.0);
        p.scopeMeasure    = m_ScopeMeasure->getValueAtTime(t) ? 1 : 0;
        p.scopeMotion     = m_ScopeMotion->getValueAtTime(t) ? 1 : 0;
        p.scopeEq         = m_ScopeEq->getValueAtTime(t) ? 1 : 0;

        // ---- v3.2 ----
        p.ghostGuard      = m_GhostGuard->getValueAtTime(t) ? 1 : 0;
        // ---- v3.3 ----
        p.deepClean       = m_DeepClean->getValueAtTime(t) ? 1 : 0;
        p.globalBlend     = static_cast<float>(m_GlobalBlend->getValueAtTime(t) / 100.0);
        const bool locked = m_LockProfile->getValueAtTime(t) && m_LockValid;
        p.profileLocked   = locked ? 1 : 0;
        p.lockSY = locked ? m_LockAgg.sy : 0.02f;
        p.lockSC = locked ? m_LockAgg.scb : 0.02f;   // v3.3 B5: Cb pair
        p.lockTY = locked ? m_LockAgg.ty : 0.02f;
        p.lockTC = locked ? m_LockAgg.tcb : 0.02f;
        p.lockSCr = locked ? m_LockAgg.scr : 0.02f;  // and the Cr pair
        p.lockTCr = locked ? m_LockAgg.tcr : 0.02f;
        for (int b = 0; b < 16; ++b)
        {
            p.lockGainY[b] = locked ? m_LockAgg.gainY[b] : 1.0f;
            p.lockGainC[b] = locked ? m_LockAgg.gainC[b] : 1.0f;
        }
        // ---- v3.5 ----
        p.renderBoost     = m_RenderBoost->getValueAtTime(t) ? 1 : 0;
        p.histValid       = 0;   // decided per render by the GPU/CPU hosts
        // ---- v3.6 ----
        p.effnSteer       = static_cast<float>(m_EffnSteer->getValueAtTime(t) / 100.0);
        return p;
    }

    void setupAndProcess(const OFX::RenderArguments& p_Args)
    {
        const double t = p_Args.time;

        std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(t));
        if (!dst)
            OFX::throwSuiteStatusException(kOfxStatFailed);

        const NRParams params = gatherParams(t);

        int reach = 0, prevReach = 0;
        fetchReach(t, reach, prevReach);

        const OfxRangeD clipRange = m_SrcClip->getFrameRange();

        std::unique_ptr<OFX::Image> held[7];
        OFX::Image* imgs[7] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };

        held[3].reset(m_SrcClip->fetchImage(t));
        if (!held[3])
            OFX::throwSuiteStatusException(kOfxStatFailed);
        imgs[3] = held[3].get();

        for (int k = -prevReach; k <= reach; ++k)
        {
            if (k == 0)
                continue;
            double tk = t + k;
            if (tk < clipRange.min) tk = clipRange.min;
            if (tk > clipRange.max) tk = clipRange.max;
            held[3 + k].reset(m_SrcClip->fetchImage(tk));
            imgs[3 + k] = held[3 + k] ? held[3 + k].get() : held[3].get();
        }
        for (int i = 0; i < 7; ++i)
            if (!imgs[i])
                imgs[i] = held[3].get();

        if ((imgs[3]->getPixelDepth() != dst->getPixelDepth()) ||
            (imgs[3]->getPixelComponents() != dst->getPixelComponents()))
            OFX::throwSuiteStatusException(kOfxStatErrValue);

        const bool gpu = p_Args.isEnabledMetalRender || p_Args.isEnabledCudaRender || p_Args.isEnabledOpenCLRender;
        if (gpu)
        {
            OpenNRProcessor proc(*this);
            proc.setDstImg(dst.get());
            proc.setSrcImgs(imgs);
            proc.setGPURenderArgs(p_Args);
            proc.setRenderWindow(p_Args.renderWindow);
            proc.setParams(params);
            proc.process();
        }
        else
        {
            renderCPU(imgs, dst.get(), params);
        }
    }

    void renderCPU(OFX::Image* imgs[7], OFX::Image* dst, const NRParams& params)
    {
        const OfxRectI& b = imgs[3]->getBounds();
        const int W = b.x2 - b.x1;
        const int H = b.y2 - b.y1;
        if (W <= 0 || H <= 0)
            return;

        nrcore::Params p;
        p.profileSource  = params.profileSource;
        p.sigmaY         = params.sigmaY;
        p.sigmaC         = params.sigmaC;
        p.profileAdjust  = params.profileAdjust;
        p.regionCX       = params.regionCX;
        p.regionCY       = params.regionCY;
        p.regionSize     = params.regionSize;
        p.enableTemporal = params.enableTemporal;
        p.temporalFrames = params.temporalFrames;
        p.temporalLuma   = params.temporalLuma;
        p.temporalChroma = params.temporalChroma;
        p.motionThresh   = params.motionThresh;
        p.enableSpatial  = params.enableSpatial;
        p.spatialMode    = params.spatialMode;
        p.spatialRadius  = params.spatialRadius;
        p.spatialLuma    = params.spatialLuma;
        p.spatialChroma  = params.spatialChroma;
        p.preserveDetail = params.preserveDetail;
        p.chromaBlotch   = params.chromaBlotch;
        p.enableRefine   = params.enableRefine;
        p.shadowDesat    = params.shadowDesat;
        p.desatRange     = params.desatRange;
        p.lumaTexture    = params.lumaTexture;
        p.grainAmount    = params.grainAmount;
        p.grainSize      = params.grainSize;
        p.grainChroma    = params.grainChroma;
        p.grainBlue      = params.grainBlue;
        p.acutance       = params.acutance;
        p.chromaSpeckle  = params.chromaSpeckle;
        p.frameIndex     = params.frameIndex;
        p.master         = params.master;
        p.viewMode       = params.viewMode;
        p.exportMatteAlpha = params.exportMatteAlpha;
        p.motionTracking = params.motionTracking;
        p.fireflyRemoval = params.fireflyRemoval;
        p.eqFine         = params.eqFine;
        p.eqMedium       = params.eqMedium;
        p.eqCoarse       = params.eqCoarse;
        p.deband         = params.deband;
        p.detailRescue   = params.detailRescue;
        p.scopeMeasure   = params.scopeMeasure;
        p.scopeMotion    = params.scopeMotion;
        p.scopeEq        = params.scopeEq;
        p.ghostGuard     = params.ghostGuard;
        p.globalBlend    = params.globalBlend;
        p.deepClean      = params.deepClean;
        p.lockSCr        = params.lockSCr;
        p.lockTCr        = params.lockTCr;
        p.profileLocked  = params.profileLocked;
        p.lockSY         = params.lockSY;
        p.lockSC         = params.lockSC;
        p.lockTY         = params.lockTY;
        p.lockTC         = params.lockTC;
        for (int b2 = 0; b2 < 16; ++b2)
        {
            p.lockGainY[b2] = params.lockGainY[b2];
            p.lockGainC[b2] = params.lockGainC[b2];
        }
        // ---- v3.5 R1 ----
        p.renderBoost    = params.renderBoost;
        p.histValid      = 0;   // decided against the instance cache below
        // ---- v3.6 S1 ----
        p.effnSteer      = params.effnSteer;

        const size_t n = static_cast<size_t>(W) * H * 4;
        std::vector<std::vector<float>> packed;
        const float* fptr[7];

        for (int i = 0; i < 7; ++i)
        {
            int prior = -1;
            for (int j = 0; j < i; ++j)
                if (imgs[j] == imgs[i]) { prior = j; break; }
            if (prior >= 0)
            {
                fptr[i] = fptr[prior];
                continue;
            }
            packed.emplace_back(n);
            std::vector<float>& buf = packed.back();
            for (int y = 0; y < H; ++y)
            {
                const float* row = static_cast<float*>(imgs[i]->getPixelAddress(b.x1, b.y1 + y));
                if (row)
                    std::memcpy(&buf[static_cast<size_t>(y) * W * 4], row, static_cast<size_t>(W) * 4 * sizeof(float));
            }
            fptr[i] = buf.data();
        }

        // v3.5 R1: sequential-render history — the same validity contract as
        // the GPU hosts (previous committed render was frame-1, same dims,
        // same effective params). hasTemporalDiff is derived here exactly as
        // the GPU hosts derive it, so clip edges invalidate the chain.
        const bool wantBoost = (params.renderBoost != 0) && (params.enableTemporal != 0) &&
                               (params.master > 0.0f);
        NRParams hashSrc = params;
        const float* partnerPtr = fptr[3];
        if (fptr[2] != fptr[3])      partnerPtr = fptr[2];
        else if (fptr[4] != fptr[3]) partnerPtr = fptr[4];
        hashSrc.hasTemporalDiff = (partnerPtr != fptr[3]) ? 1 : 0;
        hashSrc.histValid = 0;
        const uint32_t boostHash = boostParamsHash(hashSrc);

        std::vector<float> histCopy;
        if (wantBoost)
        {
            std::lock_guard<std::mutex> lk(m_CpuHistMutex);
            if (m_CpuHistFrame == params.frameIndex - 1 && m_CpuHistHash == boostHash &&
                m_CpuHistW == W && m_CpuHistH == H && m_CpuHist.size() == n)
            {
                histCopy = m_CpuHist;   // copy out — a concurrent render may
                p.histValid = 1;        // replace the member while we read
            }
        }

        std::vector<float> out(n), scratch;
        nrcore::denoiseFrame(fptr, W, H, p, out.data(), scratch,
                             p.histValid ? histCopy.data() : 0);

        // publish this frame's TRUE temporal result (scratch's first plane —
        // under Deep Clean the second plane holds the cleaned copy) for the
        // next sequential render
        if (wantBoost && scratch.size() >= n)
        {
            std::lock_guard<std::mutex> lk(m_CpuHistMutex);
            m_CpuHist.assign(scratch.begin(), scratch.begin() + n);
            m_CpuHistFrame = params.frameIndex;
            m_CpuHistHash  = boostHash;
            m_CpuHistW = W;
            m_CpuHistH = H;
        }

        const OfxRectI& db = dst->getBounds();
        for (int y = 0; y < H; ++y)
        {
            float* row = static_cast<float*>(dst->getPixelAddress(db.x1, db.y1 + y));
            if (row)
                std::memcpy(row, &out[static_cast<size_t>(y) * W * 4], static_cast<size_t>(W) * 4 * sizeof(float));
        }
    }

    OFX::Clip* m_DstClip = nullptr;
    OFX::Clip* m_SrcClip = nullptr;

    OFX::DoubleParam*  m_Master = nullptr;
    OFX::PushButtonParam* m_AutoSetup = nullptr;
    OFX::StringParam*  m_AutoReport = nullptr;
    OFX::PushButtonParam* m_CleanSlate = nullptr;
    OFX::StringParam*  m_AutoUndo = nullptr;
    OFX::BooleanParam* m_LockProfile = nullptr;
    OFX::StringParam*  m_LockData = nullptr;
    OFX::BooleanParam* m_MotionTracking = nullptr;
    OFX::BooleanParam* m_FireflyRemoval = nullptr;
    OFX::BooleanParam* m_RenderBoost = nullptr;
    OFX::DoubleParam*  m_EqFine = nullptr;
    OFX::DoubleParam*  m_EqMedium = nullptr;
    OFX::DoubleParam*  m_EqCoarse = nullptr;
    OFX::DoubleParam*  m_Deband = nullptr;
    nranalyze::ClipAggregate m_LockAgg;
    bool m_LockValid = false;
    bool m_InAutoApply = false;
    // v3.5 R1: CPU-path render-boost history (same contract as the GPU hosts)
    std::vector<float> m_CpuHist;
    int m_CpuHistFrame = INT_MIN;
    uint32_t m_CpuHistHash = 0;
    int m_CpuHistW = 0, m_CpuHistH = 0;
    std::mutex m_CpuHistMutex;
    OFX::ChoiceParam*  m_ProfileSource = nullptr;
    OFX::DoubleParam*  m_ProfileAdjust = nullptr;
    OFX::DoubleParam*  m_SigmaY = nullptr;
    OFX::DoubleParam*  m_SigmaC = nullptr;
    OFX::DoubleParam*  m_RegionCX = nullptr;
    OFX::DoubleParam*  m_RegionCY = nullptr;
    OFX::DoubleParam*  m_RegionSize = nullptr;
    OFX::BooleanParam* m_EnableTemporal = nullptr;
    OFX::ChoiceParam*  m_TemporalFrames = nullptr;
    OFX::DoubleParam*  m_TemporalLuma = nullptr;
    OFX::DoubleParam*  m_TemporalChroma = nullptr;
    OFX::DoubleParam*  m_MotionThresh = nullptr;
    OFX::BooleanParam* m_EnableSpatial = nullptr;
    OFX::ChoiceParam*  m_SpatialMode = nullptr;
    OFX::IntParam*     m_SpatialRadius = nullptr;
    OFX::DoubleParam*  m_SpatialLuma = nullptr;
    OFX::DoubleParam*  m_SpatialChroma = nullptr;
    OFX::DoubleParam*  m_PreserveDetail = nullptr;
    OFX::DoubleParam*  m_DetailRescue = nullptr;
    OFX::DoubleParam*  m_EffnSteer = nullptr;      // v3.6 S1
    OFX::BooleanParam* m_DeepClean = nullptr;
    OFX::DoubleParam*  m_ChromaBlotch = nullptr;
    OFX::BooleanParam* m_ScopeMeasure = nullptr;
    OFX::BooleanParam* m_ScopeMotion = nullptr;
    OFX::BooleanParam* m_ScopeEq = nullptr;
    OFX::BooleanParam* m_GhostGuard = nullptr;
    OFX::DoubleParam*  m_GlobalBlend = nullptr;
    bool m_EqScopeShown = false;
    OFX::BooleanParam* m_EnableRefine = nullptr;
    OFX::DoubleParam*  m_ShadowDesat = nullptr;
    OFX::DoubleParam*  m_DesatRange = nullptr;
    OFX::DoubleParam*  m_LumaTexture = nullptr;
    OFX::DoubleParam*  m_GrainAmount = nullptr;
    OFX::DoubleParam*  m_GrainSize = nullptr;
    OFX::DoubleParam*  m_GrainChroma = nullptr;
    OFX::DoubleParam*  m_GrainBlue = nullptr;     // v3.6
    OFX::DoubleParam*  m_Acutance = nullptr;      // v3.6
    OFX::DoubleParam*  m_ChromaSpeckle = nullptr; // v3.6
    OFX::ChoiceParam*  m_ViewMode = nullptr;
    OFX::BooleanParam* m_ExportMatte = nullptr;   // v3.7
};


////////////////////////////////////////////////////////////////////////////////
// On-viewer noise-sample region: draggable / resizable overlay
// (visible when Noise Profile = Automatic (From Region); enable the OpenFX
// overlay in the viewer's on-screen-controls dropdown)
////////////////////////////////////////////////////////////////////////////////

class HushRegionInteract : public OFX::OverlayInteract
{
public:
    HushRegionInteract(OfxInteractHandle p_Handle, OFX::ImageEffect* p_Effect)
        : OFX::OverlayInteract(p_Handle)
        , m_Drag(0)
    {
        _effect  = p_Effect;
        m_Source = p_Effect->fetchChoiceParam("profileSource");
        m_CX     = p_Effect->fetchDoubleParam("regionCenterX");
        m_CY     = p_Effect->fetchDoubleParam("regionCenterY");
        m_Size   = p_Effect->fetchDoubleParam("regionSize");
        m_Lock   = p_Effect->fetchBooleanParam("lockProfile");
        addParamToSlaveTo(m_Source);
        addParamToSlaveTo(m_CX);
        addParamToSlaveTo(m_CY);
        addParamToSlaveTo(m_Size);
        addParamToSlaveTo(m_Lock);
    }

    virtual bool draw(const OFX::DrawArgs& p_Args)
    {
        if (!regionActive() || !p_Args.context || !OFX::Private::gDrawSuite)
            return false;
        OfxDrawSuiteV1* ds = OFX::Private::gDrawSuite;

        double cx, cy, half;
        geometry(cx, cy, half);
        const double hs = 5.0 * std::max(p_Args.pixelScale.x, 1e-6);
        const bool locked = lockedNow();

        // v3.4: the box tells you its mode. LIVE = yellow, handles, centre
        // cross — it re-measures every frame, drag it around. LOCKED = dim
        // ice blue, no handles, a padlock — it stopped sampling entirely
        // (the profile is frozen; where the box sits no longer matters).
        const OfxRGBAColourF yellow = { 1.0f, 0.9f, 0.1f, 1.0f };
        const OfxRGBAColourF ice    = { 0.45f, 0.72f, 0.95f, 0.75f };
        ds->setColour(p_Args.context, locked ? &ice : &yellow);
        ds->setLineWidth(p_Args.context, 2.0f);
        ds->setLineStipple(p_Args.context, kOfxDrawLineStipplePatternSolid);

        const OfxPointD rect[4] = { { cx - half, cy - half }, { cx + half, cy - half },
                                    { cx + half, cy + half }, { cx - half, cy + half } };
        ds->draw(p_Args.context, kOfxDrawPrimitiveLineLoop, rect, 4);

        if (!locked) {
            // corner handles (filled)
            for (int i = 0; i < 4; ++i) {
                const OfxPointD h[2] = { { rect[i].x - hs, rect[i].y - hs },
                                         { rect[i].x + hs, rect[i].y + hs } };
                ds->draw(p_Args.context, kOfxDrawPrimitiveRectangle, h, 2);
            }

            // center cross
            const OfxPointD crossH[2] = { { cx - hs * 1.5, cy }, { cx + hs * 1.5, cy } };
            const OfxPointD crossV[2] = { { cx, cy - hs * 1.5 }, { cx, cy + hs * 1.5 } };
            ds->draw(p_Args.context, kOfxDrawPrimitiveLines, crossH, 2);
            ds->draw(p_Args.context, kOfxDrawPrimitiveLines, crossV, 2);
        } else {
            // padlock just inside the top-left corner: filled body + shackle
            const double u  = std::max(p_Args.pixelScale.x, 1e-6) * 3.0;
            const double ax = cx - half + 2.5 * u;          // body left
            const double ay = cy + half - 6.5 * u;          // body base
            const OfxPointD body[2] = { { ax, ay }, { ax + 4.0 * u, ay + 3.0 * u } };
            ds->draw(p_Args.context, kOfxDrawPrimitiveRectangle, body, 2);
            const OfxPointD shackle[6] = {
                { ax + 1.0 * u, ay + 3.0 * u }, { ax + 1.0 * u, ay + 4.5 * u },
                { ax + 1.0 * u, ay + 4.5 * u }, { ax + 3.0 * u, ay + 4.5 * u },
                { ax + 3.0 * u, ay + 4.5 * u }, { ax + 3.0 * u, ay + 3.0 * u }
            };
            ds->draw(p_Args.context, kOfxDrawPrimitiveLines, shackle, 6);
        }
        return true;
    }

    virtual bool penDown(const OFX::PenArgs& p_Args)
    {
        if (!regionActive() || lockedNow())
            return false;
        double cx, cy, half;
        geometry(cx, cy, half);
        const double px = p_Args.penPosition.x, py = p_Args.penPosition.y;
        const double tol = 14.0 * std::max(p_Args.pixelScale.x, 1e-6);

        // corners first (resize)
        const double corners[4][2] = { { cx - half, cy - half }, { cx + half, cy - half },
                                       { cx + half, cy + half }, { cx - half, cy + half } };
        for (int i = 0; i < 4; ++i)
            if (std::fabs(px - corners[i][0]) < tol && std::fabs(py - corners[i][1]) < tol) {
                m_Drag = 2;
                return true;
            }
        // inside (move)
        if (std::fabs(px - cx) <= half && std::fabs(py - cy) <= half) {
            m_Drag = 1;
            m_GrabX = px - cx;
            m_GrabY = py - cy;
            return true;
        }
        return false;
    }

    virtual bool penMotion(const OFX::PenArgs& p_Args)
    {
        if (m_Drag == 0 || !regionActive() || lockedNow())
            return false;
        const OfxPointD ext = _effect->getProjectExtent();
        const double minDim = std::min(ext.x, ext.y);
        const double px = p_Args.penPosition.x, py = p_Args.penPosition.y;

        if (m_Drag == 1) {
            m_CX->setValue(std::min(1.0, std::max(0.0, (px - m_GrabX) / ext.x)));
            m_CY->setValue(std::min(1.0, std::max(0.0, (py - m_GrabY) / ext.y)));
        } else {
            double cx, cy, half;
            geometry(cx, cy, half);
            const double newHalf = std::max(std::fabs(px - cx), std::fabs(py - cy));
            m_Size->setValue(std::min(1.0, std::max(0.05, 2.0 * newHalf / minDim)));
        }
        requestRedraw();
        return true;
    }

    virtual bool penUp(const OFX::PenArgs&)
    {
        const bool was = (m_Drag != 0);
        m_Drag = 0;
        return was;
    }

private:
    bool regionActive()
    {
        int s = 0;
        m_Source->getValue(s);
        return s == 1;
    }

    bool lockedNow()
    {
        bool on = false;
        m_Lock->getValue(on);
        return on;
    }

    // canonical-coordinate geometry (OFX images are bottom-up, so the kernel's
    // normalized y maps directly onto the interact's y-up canonical space)
    void geometry(double& cx, double& cy, double& half)
    {
        const OfxPointD ext = _effect->getProjectExtent();
        cx = m_CX->getValue() * ext.x;
        cy = m_CY->getValue() * ext.y;
        half = 0.5 * m_Size->getValue() * std::min(ext.x, ext.y);
    }

    OFX::ChoiceParam*  m_Source;
    OFX::DoubleParam*  m_CX;
    OFX::DoubleParam*  m_CY;
    OFX::DoubleParam*  m_Size;
    OFX::BooleanParam* m_Lock;
    int m_Drag;
    double m_GrabX = 0.0, m_GrabY = 0.0;
};

class HushRegionInteractDescriptor
    : public OFX::DefaultEffectOverlayDescriptor<HushRegionInteractDescriptor, HushRegionInteract>
{
};

////////////////////////////////////////////////////////////////////////////////
// Factory
////////////////////////////////////////////////////////////////////////////////

OpenNRPluginFactory::OpenNRPluginFactory()
    : OFX::PluginFactoryHelper<OpenNRPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void OpenNRPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    p_Desc.addSupportedContext(OFX::eContextFilter);
    p_Desc.addSupportedContext(OFX::eContextGeneral);
    p_Desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(true);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    p_Desc.setSupportsOpenCLRender(true);
#ifdef HUSH_ENABLE_CUDA
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(true);
#endif
#ifdef __APPLE__
    p_Desc.setSupportsMetalRender(true);
#endif

    p_Desc.setOverlayInteractDescriptor(new HushRegionInteractDescriptor);
}

static OFX::DoubleParamDescriptor* defineDouble(OFX::ImageEffectDescriptor& p_Desc, OFX::PageParamDescriptor* p_Page,
                                                const char* p_Name, const char* p_Label, const char* p_Hint,
                                                double p_Default, double p_Min, double p_Max,
                                                OFX::GroupParamDescriptor* p_Parent)
{
    OFX::DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(p_Default);
    param->setRange(p_Min, p_Max);
    param->setIncrement(0.1);
    param->setDisplayRange(p_Min, p_Max);
    if (p_Parent)
        param->setParent(*p_Parent);
    if (p_Page)
        p_Page->addChild(*param);
    return param;
}

static OFX::BooleanParamDescriptor* defineBool(OFX::ImageEffectDescriptor& p_Desc, OFX::PageParamDescriptor* p_Page,
                                               const char* p_Name, const char* p_Label, const char* p_Hint,
                                               bool p_Default, OFX::GroupParamDescriptor* p_Parent)
{
    OFX::BooleanParamDescriptor* param = p_Desc.defineBooleanParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(p_Default);
    if (p_Parent)
        param->setParent(*p_Parent);
    if (p_Page)
        p_Page->addChild(*param);
    return param;
}

void OpenNRPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    OFX::ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(true);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    OFX::ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    dstClip->setSupportsTiles(kSupportsTiles);

    OFX::PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // ------------------------------------------------------------------ master
    defineDouble(p_Desc, page, "master", "Strength",
                 "Overall amount. Below 1 fades the effect; above 1 the filters widen what "
                 "they treat as noise — up to 3 for very noisy footage. 0 = off.",
                 1.0, 0.0, 3.0, nullptr);

    // ------------------------------------------------------------- auto setup
    {
        OFX::PushButtonParamDescriptor* b = p_Desc.definePushButtonParam("autoSetup");
        b->setLabels("Auto Setup (Analyze Footage)", "Auto Setup (Analyze Footage)", "Auto Setup");
        b->setHint("Measures the clip (noise, chroma, motion) and sets every slider below, "
                   "then you adjust from there. One undo — or the Revert button — restores "
                   "everything; your Step 4 look and the views are never touched.");
        page->addChild(*b);
    }
    {
        OFX::PushButtonParamDescriptor* b = p_Desc.definePushButtonParam("autoRegion");
        b->setLabels("Auto Region (Find Flat Patch)", "Auto Region (Find Flat Patch)", "Auto Region");
        b->setHint("Scans the current frame for the flattest area — where noise measures most "
                   "accurately — switches the profile to From Region and moves the yellow box "
                   "there so you can see exactly what is sampled. Watch it live, then Lock "
                   "Profile to freeze it (or run Auto Setup). One undo restores.");
        page->addChild(*b);
    }
    {
        OFX::StringParamDescriptor* s = p_Desc.defineStringParam("autoReport");
        s->setLabels("Analysis", "Analysis", "Analysis");
        s->setHint("What the last analysis (Auto Setup, Auto Region or Lock) measured and decided.");
        s->setDefault("Click Auto Setup to analyze this clip.");
        s->setEnabled(false);
        s->setEvaluateOnChange(false);
        page->addChild(*s);
    }
    {
        OFX::PushButtonParamDescriptor* b = p_Desc.definePushButtonParam("cleanSlate");
        b->setLabels("Clean Slate (All Off)", "Clean Slate (All Off)", "Clean Slate");
        b->setHint("Zeroes every processing control so the node passes the image through "
                   "untouched — the fully-manual starting point. Raise sliders from nothing, "
                   "or click Auto Setup. One undo restores what you had.");
        page->addChild(*b);
    }
    {
        OFX::StringParamDescriptor* s = p_Desc.defineStringParam("autoSetupUndo");
        s->setLabels("autoSetupUndo", "autoSetupUndo", "autoSetupUndo");
        s->setDefault("");
        s->setIsSecret(true);
        s->setEvaluateOnChange(false);
        page->addChild(*s);
    }

    // ---------------------------------------------------------- step 1: profile
    OFX::GroupParamDescriptor* grpProfile = p_Desc.defineGroupParam("grpProfile");
    grpProfile->setLabels("Step 1 · Measure The Noise", "Step 1 · Measure The Noise", "1 · Measure");
    grpProfile->setOpen(true);
    grpProfile->setHint("Every filter is calibrated from this measurement. Tick the Scope "
                        "checkbox to see it live in the viewer.");

    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("profileSource");
        c->setLabels("Noise Profile", "Noise Profile", "Profile");
        c->setHint("Where the noise level comes from: measured LIVE every frame on the whole "
                   "frame, measured live inside the draggable yellow rectangle (put it on a "
                   "flat area — Auto Region finds one — then Lock Profile to freeze it), or "
                   "typed in manually.");
        c->appendOption("Automatic (Whole Frame)");
        c->appendOption("Automatic (From Region)");
        c->appendOption("Manual");
        c->setDefault(0);
        c->setParent(*grpProfile);
        page->addChild(*c);
    }
    defineDouble(p_Desc, page, "regionCenterX", "Region Center X",
                 "Horizontal center of the measurement region (0 = left, 1 = right). "
                 "Easier: drag the yellow rectangle in the viewer, or click Auto Region. "
                 "The box samples live every frame until you Lock Profile.", 0.5, 0.0, 1.0, grpProfile);
    defineDouble(p_Desc, page, "regionCenterY", "Region Center Y",
                 "Vertical center of the measurement region (0 = bottom, 1 = top).", 0.5, 0.0, 1.0, grpProfile);
    defineDouble(p_Desc, page, "regionSize", "Region Size",
                 "Size of the measurement region relative to the frame.", 0.25, 0.05, 1.0, grpProfile);
    defineDouble(p_Desc, page, "profileAdjust", "Auto Profile Adjust",
                 "Scales the measurement — live and locked profiles alike. Raise if noise is "
                 "left behind, lower if detail is being eaten.",
                 1.0, 0.25, 6.0, grpProfile);
    defineDouble(p_Desc, page, "sigmaLuma", "Manual Luma Noise (%)",
                 "Brightness-noise level in percent (Manual profile only). Clean 0.5–1, "
                 "noisy 2–5, low light 5–10.", 2.0, 0.05, 40.0, grpProfile);
    defineDouble(p_Desc, page, "sigmaChroma", "Manual Chroma Noise (%)",
                 "Color-noise level in percent (Manual profile only).", 2.0, 0.05, 40.0, grpProfile);
    defineBool(p_Desc, page, "lockProfile", "Lock Profile",
               "Freezes the measurement you are looking at RIGHT NOW — the playhead frame, "
               "confirmed against a few tracked neighbouring frames — so the profile stops "
               "changing as footage moves under the box (which turns blue and stops sampling). "
               "Park on a frame where it looks right, then lock. Auto Profile Adjust still "
               "trims a locked profile; un-tick for live per-frame measurement. Saved with "
               "the project.",
               false, grpProfile);
    defineBool(p_Desc, page, "scopeMeasure", "Scope: Measurements",
               "Draws the measurement panel in the viewer: noise levels in and after Step 2, "
               "frames averaged, SNR gain, the noise-vs-brightness curve and the histogram. "
               "Turn off before rendering.",
               false, grpProfile);
    {
        OFX::StringParamDescriptor* s = p_Desc.defineStringParam("lockedProfileData");
        s->setLabels("lockedProfileData", "lockedProfileData", "lockedProfileData");
        s->setDefault("");
        s->setIsSecret(true);
        page->addChild(*s);
    }

    // --------------------------------------------------------- step 2: temporal
    OFX::GroupParamDescriptor* grpTemporal = p_Desc.defineGroupParam("grpTemporal");
    grpTemporal->setLabels("Step 2 · Temporal NR (Across Frames)", "Step 2 · Temporal NR", "2 · Temporal");
    grpTemporal->setOpen(true);
    grpTemporal->setHint("Averages matching pixels across neighboring frames — the most "
                         "detail-safe reduction. Moving areas are protected automatically "
                         "and left to Step 3; tick the Scope to see where.");

    defineBool(p_Desc, page, "enableTemporal", "Enable Temporal NR",
               "Toggle the across-frames stage to see its contribution. Off for stills.",
               true, grpTemporal);
    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("temporalFrames");
        c->setLabels("Number of Frames", "Number of Frames", "Frames");
        c->setHint("More frames remove more noise on static areas but render slower; "
                   "7 is for locked-off shots with heavy noise.");
        c->appendOption("3 Frames");
        c->appendOption("5 Frames");
        c->appendOption("7 Frames");
        c->setDefault(0);
        c->setParent(*grpTemporal);
        page->addChild(*c);
    }
    defineBool(p_Desc, page, "motionTracking", "Motion Tracking",
               "Re-aims each neighbor frame by up to 2 px so slow pans and handheld drift "
               "keep their across-frames averaging. Leave on.",
               true, grpTemporal);
    defineDouble(p_Desc, page, "temporalLuma", "Luma Strength",
                 "Brightness averaging strength. Above 100, matching neighbors outweigh the "
                 "current frame — maximum smoothing on static areas.", 60.0, 0.0, 125.0, grpTemporal);
    defineDouble(p_Desc, page, "temporalChroma", "Chroma Strength",
                 "Color averaging strength; color takes more than brightness without side "
                 "effects.", 80.0, 0.0, 125.0, grpTemporal);
    defineDouble(p_Desc, page, "motionThreshold", "Motion Threshold",
                 "How much change counts as motion — past it a pixel is never blended. "
                 "SEEING SMEAR OR GHOSTING ON MOVEMENT? Lower this first, and keep Ghost "
                 "Guard on. Check Scope: Motion Map — moving things should read red.",
                 30.0, 0.0, 150.0, grpTemporal);
    defineBool(p_Desc, page, "ghostGuard", "Ghost Guard",
               "Second motion test that catches the slow, subtle movement the main gate "
               "cannot see (the classic slow-motion smear): noise differences cancel out "
               "when averaged with their signs, coherent motion does not. Nearly free on "
               "static footage — leave on.",
               true, grpTemporal);
    defineBool(p_Desc, page, "fireflyRemoval", "Firefly Removal",
               "Removes single-frame hot pixels and sparkles. Three tests must agree, so "
               "real moving detail is left alone; turn off only if strobes or glints lose "
               "their one-frame flash.",
               true, grpTemporal);
    defineBool(p_Desc, page, "renderBoost", "Render Boost",
               "Blends against the previous rendered frame too, compounding the averaging "
               "on static areas — up to about twice the frames. Sequential playback and "
               "renders only; scrubbing or parameter changes fall back automatically.",
               false, grpTemporal);
    defineBool(p_Desc, page, "scopeMotion", "Scope: Motion Map",
               "Draws a live mini-map in the viewer: green where frames are being stacked, "
               "red where motion protection kicked in. Turn off before rendering.",
               false, grpTemporal);

    // ---------------------------------------------------------- step 3: spatial
    OFX::GroupParamDescriptor* grpSpatial = p_Desc.defineGroupParam("grpSpatial");
    grpSpatial->setLabels("Step 3 · Spatial NR (Within Frame)", "Step 3 · Spatial NR", "3 · Spatial");
    grpSpatial->setOpen(true);
    grpSpatial->setHint("Cleans what remains after Step 2 — and everything in moving areas — "
                        "by averaging each pixel only with genuinely similar neighborhoods.");

    defineBool(p_Desc, page, "enableSpatial", "Enable Spatial NR",
               "Toggle the within-frame stage to see its contribution.",
               true, grpSpatial);
    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("spatialMethod");
        c->setLabels("Method", "Method", "Method");
        c->setHint("Better compares whole patches and keeps texture; Faster compares single "
                   "pixels — use it if playback is too slow.");
        c->appendOption("Faster (Bilateral)");
        c->appendOption("Better (NLM)");
        c->setDefault(1);
        c->setParent(*grpSpatial);
        page->addChild(*c);
    }
    {
        OFX::IntParamDescriptor* i = p_Desc.defineIntParam("spatialRadius");
        i->setLabels("Search Radius", "Search Radius", "Radius");
        i->setHint("How far to look for similar areas. Bigger = smoother flats, slower "
                   "renders; 6–10 for very noisy sources.");
        i->setDefault(3);
        i->setRange(1, 10);
        i->setDisplayRange(1, 10);
        i->setParent(*grpSpatial);
        page->addChild(*i);
    }
    defineDouble(p_Desc, page, "spatialLuma", "Luma Strength",
                 "Brightness smoothing strength. Above 100 the filter widens what it treats "
                 "as noise — crank it and use Detail Rescue to keep edges.", 60.0, 0.0, 150.0, grpSpatial);
    defineDouble(p_Desc, page, "spatialChroma", "Chroma Strength",
                 "Color smoothing strength; color tolerates far more than brightness.",
                 100.0, 0.0, 150.0, grpSpatial);
    defineDouble(p_Desc, page, "preserveDetail", "Preserve Detail",
                 "Backs the filter off where the image has real structure. Raise if fine "
                 "detail softens; lower if edges stay noisy.", 35.0, 0.0, 100.0, grpSpatial);
    defineDouble(p_Desc, page, "detailRescue", "Detail Rescue",
                 "Puts back anything the smoothing changed by more than a noise-sized amount. "
                 "Crank the strengths for smoothness, then raise this until faces and edges "
                 "come back crisp — smoothing without blur. 0 = off.", 0.0, 0.0, 100.0, grpSpatial);
    defineDouble(p_Desc, page, "adaptiveStrength", "Adaptive Strength",
                 "Spends the cleaning where frame averaging couldn't help — moving subjects "
                 "get the full treatment, well-averaged areas are left alone. Needs the "
                 "Temporal stage on. 0 = off.", 0.0, 0.0, 100.0, grpSpatial);
    defineBool(p_Desc, page, "deepClean", "Deep Clean (2-Pass)",
               "Runs a gentle extra cleaning pass before the main one — its corrections are "
               "capped at noise size, so detail is safe. Helps severe or compressed noise; "
               "costs some render speed.",
               false, grpSpatial);

    // Noise EQ subgroup: the spatial stage split by the SIZE of the noise
    OFX::GroupParamDescriptor* grpEq = p_Desc.defineGroupParam("grpEq");
    grpEq->setLabels("Noise EQ · Cut Noise By Size", "Noise EQ · Cut Noise By Size", "Noise EQ");
    grpEq->setOpen(false);
    grpEq->setParent(*grpSpatial);
    grpEq->setHint("Noise comes in sizes: fine grain, mid-size clumps, large soft stains. "
                   "Each slider cuts one size. Touch any of them and the EQ scope appears "
                   "in the viewer so you can see where your footage's noise lives.");

    defineDouble(p_Desc, page, "eqFine", "Fine Grain (~1 px)",
                 "Pixel-size grain — the main pass. 100 = normal; above 100 it both blends "
                 "more and smooths harder (the old no-op top half now works).",
                 100.0, 0.0, 300.0, grpEq);
    defineDouble(p_Desc, page, "eqMedium", "Clumps (3–8 px)",
                 "Mid-size noise clumps left by compression or in-camera NR. Raise when the "
                 "noise looks like moving blotches instead of grain.", 0.0, 0.0, 150.0, grpEq);
    defineDouble(p_Desc, page, "eqCoarse", "Stains (16 px +)",
                 "Large soft brightness stains in skies and walls from severe compression. "
                 "Corrections are clipped to noise size, so structure is safe.", 0.0, 0.0, 150.0, grpEq);
    defineDouble(p_Desc, page, "chromaBlotch", "Color Blotches",
                 "Large soft COLOR stains that 4:2:0 compression leaves behind (reach up to "
                 "~23 px). Brightness-guided, so color never bleeds across edges.", 25.0, 0.0, 150.0, grpEq);
    defineBool(p_Desc, page, "scopeEq", "Scope: Noise EQ",
               "Draws the EQ panel in the viewer: one lane per band — the bar is how much "
               "you are cutting, the amber line is how much noise was measured at that size. "
               "Appears automatically the first time you touch an EQ slider.",
               false, grpEq);

    // ----------------------------------------------------------- step 4: refine
    OFX::GroupParamDescriptor* grpRefine = p_Desc.defineGroupParam("grpRefine");
    grpRefine->setLabels("Step 4 · Refine The Finish", "Step 4 · Refine", "4 · Refine");
    grpRefine->setOpen(true);
    grpRefine->setHint("Finishing touches: hide leftover shadow color noise, bring back "
                       "texture, smooth banding, or lay down clean film grain.");

    defineBool(p_Desc, page, "enableRefine", "Enable Refinements",
               "Toggle the finishing stage to compare.",
               true, grpRefine);
    defineDouble(p_Desc, page, "shadowDesat", "Shadow Desaturate",
                 "Fades color out of the darkest tones — hides the chroma noise filtering "
                 "can't reach and reads cinematic. Try 20–40 on noisy footage.",
                 0.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "desatRange", "Desaturate Range",
                 "How far up the tonal scale the shadow desaturation reaches.",
                 0.15, 0.02, 0.5, grpRefine);
    defineDouble(p_Desc, page, "lumaTexture", "Luma Texture",
                 "Transfers real micro-texture back from the original — noise-scale detail is "
                 "cored out first, so you get structure back without re-noising the shadows. "
                 "Try 15–30.",
                 0.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "acutance", "Optical Acutance",
                 "Adds lens-like edge sharpness back to the cleaned image. Gated to real "
                 "edges (never sharpens noise) and clamped so it steepens edges without the "
                 "bright halo of a normal sharpen. Try 40–100. 0 = off.",
                 0.0, 0.0, 200.0, grpRefine);
    defineDouble(p_Desc, page, "chromaSpeckle", "Shadow Color Cleanup",
                 "Clears the large blotchy color speckle left in deep shadows — the kind the "
                 "regular chroma controls are too small to reach. Follows the luma, so real "
                 "color edges and lighting stay put. Try 50–100. 0 = off.",
                 0.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "deband", "Deband",
                 "Dissolves the stair-steps in skies and gradients; real edges are rejected "
                 "by the banding-sized tolerance and never touched.",
                 0.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "grainAmount", "Film Grain",
                 "Clean synthesized grain, midtone-weighted like real stock and animated per "
                 "frame — the classic finishing move.",
                 0.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "grainSize", "Grain Size",
                 "Particle size in pixels: 1 = fine 35mm, 3–4 = chunky 16mm. Scale up for UHD.",
                 1.6, 0.5, 6.0, grpRefine);
    defineDouble(p_Desc, page, "grainChroma", "Grain Color",
                 "0 = monochrome film-like grain; higher adds color grain (digital-sensor "
                 "character).",
                 25.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "grainBlue", "Grain Fineness",
                 "Shifts grain toward a finer, sharper spectrum (blue noise) by high-passing "
                 "it. 0 = full-band grain; higher reads crisper and hides the plasticky look "
                 "better at the same strength.",
                 0.0, 0.0, 100.0, grpRefine);

    // ------------------------------------------------------- v3.2: global mix
    defineDouble(p_Desc, page, "globalBlend", "Global Blend",
                 "Final crossfade between the original image (0) and the fully processed "
                 "result (100) — a plain output mix, applied after everything else.",
                 100.0, 0.0, 100.0, nullptr);

    // ---------------------------------------------------------- step 5: inspect
    OFX::GroupParamDescriptor* grpOutput = p_Desc.defineGroupParam("grpOutput");
    grpOutput->setLabels("Step 5 · Inspect & Compare", "Step 5 · Inspect", "5 · Inspect");
    grpOutput->setOpen(true);
    grpOutput->setHint("Full-image views for judging the result (the per-step Scope "
                       "checkboxes draw panels on top of any view). Set back to Result "
                       "before rendering.");

    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("viewMode");
        c->setLabels("View", "View", "View");
        c->setHint("Result / Split / Input for comparing; After Temporal shows Step 2 alone.\n\n"
                   "Noise Removed shows what was taken out, auto-gained to the measured noise "
                   "level with a soft knee — it should look like featureless static; faces or "
                   "edges in it mean detail is being cut (raise Preserve Detail or Detail "
                   "Rescue).\n\n"
                   "Noise Analysis overlays the measurement scope; Temporal Activity and SNR "
                   "Map are full-frame heat maps; the Matte writes the noise-dominance map "
                   "into RGB and alpha for keying downstream.");
        c->appendOption("Result");
        c->appendOption("Split (Input | Result)");
        c->appendOption("Input (Original)");
        c->appendOption("After Temporal (Step 2 Only)");
        c->appendOption("Noise Removed (Auto Gain)");
        c->appendOption("Noise Analysis (Measurements)");
        c->appendOption("Temporal Activity (Green = Averaging)");
        c->appendOption("SNR Map (Magenta = Noisy)");
        c->appendOption("Matte: Noisiness (RGB + Alpha)");
        c->appendOption("Matte: Clean Confidence (RGB + Alpha)");
        c->setDefault(0);
        c->setParent(*grpOutput);
        page->addChild(*c);
    }

    // v3.7: the downstream handoff. Result view only; the inspection views keep
    // the true alpha. The hint states plainly that it REPLACES incoming alpha.
    defineBool(p_Desc, page, "exportMatteAlpha", "Export Clean Matte to Alpha",
               "In the Result view, writes the clean-confidence matte into the output ALPHA "
               "(clamp((effN-1)/6), identical to the Clean Confidence view) so a downstream "
               "node - Speak's grain, or any qualifier - can key on where cleaning succeeded. "
               "Replaces the incoming alpha while on; RGB is untouched.",
               false, grpOutput);
}

OFX::ImageEffect* OpenNRPluginFactory::createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum /*p_Context*/)
{
    return new OpenNRPlugin(p_Handle);
}

// Speak — the film-reconstruction node, registered as a SECOND plugin in the
// same .ofx bundle (org.opennr.Speak). Forward-declared to keep this file
// decoupled from SpeakPlugin.h; defined in SpeakPlugin.cpp.
namespace speakofx { void registerSpeak(OFX::PluginFactoryArray& p_FactoryArray); }

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& p_FactoryArray)
{
    static OpenNRPluginFactory openNRPlugin;
    p_FactoryArray.push_back(&openNRPlugin);
    speakofx::registerSpeak(p_FactoryArray);   // one bundle, two plugins
}
