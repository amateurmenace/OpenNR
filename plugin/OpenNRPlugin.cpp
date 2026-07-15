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
#include <memory>
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
#include "nr_core.h"
#include "nr_analyze.h"

#define kPluginName "Hush Open NR"
#define kPluginGrouping "Hush"
#define kPluginDescription \
    "Hush Open NR v3.0 — the free noise reduction suite.\n\n" \
    "New: click AUTO SETUP and the plugin analyzes your clip (noise level, " \
    "chroma character, motion) and dials in every slider for you — then you " \
    "adjust from there. One Cmd+Z (or the Revert button) undoes it.\n\n" \
    "Work top to bottom: 1) measure the noise (automatic, from a region, or " \
    "manual — and lock the measured profile so it stops re-measuring), " \
    "2) temporal NR averages matching pixels across frames, with motion " \
    "tracking and a firefly zapper, 3) spatial NR cleans what remains with a " \
    "three-band Noise EQ, 4) refine the finish (shadow desaturation, texture, " \
    "debanding, film grain), 5) inspect with the analysis views.\n\n" \
    "Open Step 5 and choose 'Noise Analysis' to SEE the measured noise levels " \
    "on screen, or 'Noise Only' to see exactly what is being removed.\n\n" \
    "MIT-licensed and free forever."
#define kPluginIdentifier "org.opennr.Denoise"
#define kPluginVersionMajor 3
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////
// GPU entry points
////////////////////////////////////////////////////////////////////////////////

#ifdef __APPLE__
extern void RunMetalNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                       const float* const p_Srcs[5], float* p_Dst);
#endif
#ifdef HUSH_ENABLE_CUDA
extern void RunCudaNR(void* p_Stream, int p_Width, int p_Height, const NRParams& p_Params,
                      const float* const p_Srcs[5], float* p_Dst);
#endif
extern void RunOpenCLNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                        const float* const p_Srcs[5], float* p_Dst);

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

    void setSrcImgs(OFX::Image* p_Imgs[5])
    {
        for (int i = 0; i < 5; ++i)
            _srcImgs[i] = p_Imgs[i];
    }

    void setParams(const NRParams& p) { _params = p; }

    virtual void processImagesMetal()
    {
#ifdef __APPLE__
        const OfxRectI& bounds = _srcImgs[2]->getBounds();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        const float* srcs[5];
        for (int i = 0; i < 5; ++i)
            srcs[i] = static_cast<float*>(_srcImgs[i]->getPixelData());
        RunMetalNR(_pMetalCmdQ, width, height, _params, srcs, static_cast<float*>(_dstImg->getPixelData()));
#endif
    }

    virtual void processImagesCUDA()
    {
#ifdef HUSH_ENABLE_CUDA
        const OfxRectI& bounds = _srcImgs[2]->getBounds();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        const float* srcs[5];
        for (int i = 0; i < 5; ++i)
            srcs[i] = static_cast<float*>(_srcImgs[i]->getPixelData());
        RunCudaNR(_pCudaStream, width, height, _params, srcs, static_cast<float*>(_dstImg->getPixelData()));
#endif
    }

    virtual void processImagesOpenCL()
    {
        const OfxRectI& bounds = _srcImgs[2]->getBounds();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        const float* srcs[5];
        for (int i = 0; i < 5; ++i)
            srcs[i] = static_cast<float*>(_srcImgs[i]->getPixelData());
        RunOpenCLNR(_pOpenCLCmdQ, width, height, _params, srcs, static_cast<float*>(_dstImg->getPixelData()));
    }

    // CPU fallback is handled whole-frame from the render action.
    virtual void multiThreadProcessImages(OfxRectI) {}

private:
    OFX::Image* _srcImgs[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
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
        m_RevertAuto     = fetchPushButtonParam("revertAutoSetup");
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
        m_EnableSpatial  = fetchBooleanParam("enableSpatial");
        m_SpatialMode    = fetchChoiceParam("spatialMethod");
        m_SpatialRadius  = fetchIntParam("spatialRadius");
        m_SpatialLuma    = fetchDoubleParam("spatialLuma");
        m_SpatialChroma  = fetchDoubleParam("spatialChroma");
        m_PreserveDetail = fetchDoubleParam("preserveDetail");
        m_ChromaBlotch   = fetchDoubleParam("chromaBlotch");
        m_EqFine         = fetchDoubleParam("eqFine");
        m_EqMedium       = fetchDoubleParam("eqMedium");
        m_EqCoarse       = fetchDoubleParam("eqCoarse");
        m_EnableRefine   = fetchBooleanParam("enableRefine");
        m_ShadowDesat    = fetchDoubleParam("shadowDesat");
        m_DesatRange     = fetchDoubleParam("desatRange");
        m_LumaTexture    = fetchDoubleParam("lumaTexture");
        m_Deband         = fetchDoubleParam("deband");
        m_GrainAmount    = fetchDoubleParam("grainAmount");
        m_GrainSize      = fetchDoubleParam("grainSize");
        m_GrainChroma    = fetchDoubleParam("grainChroma");
        m_ViewMode       = fetchChoiceParam("viewMode");

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

        const double master = m_Master->getValueAtTime(t);
        const bool tOn = m_EnableTemporal->getValueAtTime(t) &&
                         (m_TemporalLuma->getValueAtTime(t) > 0.0 || m_TemporalChroma->getValueAtTime(t) > 0.0);
        const bool sOn = m_EnableSpatial->getValueAtTime(t) &&
                         (m_SpatialLuma->getValueAtTime(t) > 0.0 || m_SpatialChroma->getValueAtTime(t) > 0.0);
        // v3: refine-only configurations (grain / texture / desat / deband
        // with both NR stages off) must render — v2.1 wrongly skipped them
        const bool rOn = m_EnableRefine->getValueAtTime(t) &&
                         (m_ShadowDesat->getValueAtTime(t) > 0.0 ||
                          m_LumaTexture->getValueAtTime(t) > 0.0 ||
                          m_GrainAmount->getValueAtTime(t) > 0.0 ||
                          m_Deband->getValueAtTime(t) > 0.0);

        if (master <= 0.0 || (!tOn && !sOn && !rOn))
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
            else if (p_ParamName == "revertAutoSetup")
                revertAutoSetup();
            else if (p_ParamName == "lockProfile")
                lockProfileToggled(p_Args.time);
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
    int analyzeClipFrames(double p_Time, std::vector<nrcore::Stats>& p_Out)
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
                nrcore::Params ap;   // stock automatic whole-frame profiling
                nrcore::Stats st;
                nrcore::estimateInput(buf.data(), partner, W, H, ap, st);
                p_Out.push_back(st);
            }
            return static_cast<int>(p_Out.size());
        }
        catch (...)
        {
            return static_cast<int>(p_Out.size());
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
        char buf[1024];
        int tf = 0, sm = 0;
        m_TemporalFrames->getValue(tf);
        m_SpatialMode->getValue(sm);
        std::string lockStr;
        m_LockData->getValue(lockStr);
        snprintf(buf, sizeof(buf),
                 "v1|et=%d|tf=%d|tl=%.17g|tc=%.17g|mt=%.17g|mtr=%d|ff=%d|es=%d|sm=%d|sr=%d|"
                 "sl=%.17g|sc=%.17g|pd=%.17g|cb=%.17g|eqf=%.17g|eqm=%.17g|eqc=%.17g|pa=%.17g|lp=%d|ld=%s",
                 m_EnableTemporal->getValue() ? 1 : 0, tf,
                 m_TemporalLuma->getValue(), m_TemporalChroma->getValue(), m_MotionThresh->getValue(),
                 m_MotionTracking->getValue() ? 1 : 0, m_FireflyRemoval->getValue() ? 1 : 0,
                 m_EnableSpatial->getValue() ? 1 : 0, sm, m_SpatialRadius->getValue(),
                 m_SpatialLuma->getValue(), m_SpatialChroma->getValue(),
                 m_PreserveDetail->getValue(), m_ChromaBlotch->getValue(),
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

    void runAutoSetup(double p_Time)
    {
        std::vector<nrcore::Stats> per;
        if (analyzeClipFrames(p_Time, per) <= 0)
        {
            analysisFailedMessage();
            return;
        }
        const nranalyze::ClipAggregate agg = nranalyze::aggregateClipStats(per);
        const nranalyze::AutoSettings as = nranalyze::mapAnalysisToSettings(agg);

        // belt and braces: the whole apply is one edit block (single Cmd+Z),
        // AND the prior values are snapshotted for the Revert button
        m_AutoUndo->setValue(snapshotDenoiseParams());

        m_InAutoApply = true;
        beginEditBlock("Hush Auto Setup");
        m_EnableTemporal->setValue(as.enableTemporal != 0);
        m_TemporalFrames->setValue(as.temporalFrames >= 5 ? 1 : 0);
        m_TemporalLuma->setValue(as.temporalLuma);
        m_TemporalChroma->setValue(as.temporalChroma);
        m_MotionThresh->setValue(as.motionThresh);
        m_MotionTracking->setValue(as.motionTracking != 0);
        m_FireflyRemoval->setValue(as.fireflyRemoval != 0);
        m_EnableSpatial->setValue(as.enableSpatial != 0);
        m_SpatialMode->setValue(as.spatialMode);
        m_SpatialRadius->setValue(as.spatialRadius);
        m_SpatialLuma->setValue(as.spatialLuma);
        m_SpatialChroma->setValue(as.spatialChroma);
        m_PreserveDetail->setValue(as.preserveDetail);
        m_ChromaBlotch->setValue(as.chromaBlotch);
        m_EqFine->setValue(as.eqFine);
        m_EqMedium->setValue(as.eqMedium);
        m_EqCoarse->setValue(as.eqCoarse);
        m_ProfileAdjust->setValue(as.profileAdjust);
        m_LockData->setValue(nranalyze::formatLockedProfile(agg));
        m_LockProfile->setValue(as.lockProfile != 0);
        m_AutoReport->setValue(nranalyze::formatAutoReport(agg, as));
        endEditBlock();
        m_InAutoApply = false;

        m_LockAgg = agg;
        m_LockValid = true;
        updateEnabledness();
    }

    void revertAutoSetup()
    {
        std::string s;
        m_AutoUndo->getValue(s);
        if (s.compare(0, 3, "v1|") != 0)
        {
            sendMessage(OFX::Message::eMessageMessage, "",
                        "Nothing to revert — Auto Setup has not been run on this node.");
            return;
        }
        std::string v;
        m_InAutoApply = true;
        beginEditBlock("Revert Hush Auto Setup");
        if (snapField(s, "et", v))  m_EnableTemporal->setValue(atoi(v.c_str()) != 0);
        if (snapField(s, "tf", v))  m_TemporalFrames->setValue(atoi(v.c_str()));
        if (snapField(s, "tl", v))  m_TemporalLuma->setValue(atof(v.c_str()));
        if (snapField(s, "tc", v))  m_TemporalChroma->setValue(atof(v.c_str()));
        if (snapField(s, "mt", v))  m_MotionThresh->setValue(atof(v.c_str()));
        if (snapField(s, "mtr", v)) m_MotionTracking->setValue(atoi(v.c_str()) != 0);
        if (snapField(s, "ff", v))  m_FireflyRemoval->setValue(atoi(v.c_str()) != 0);
        if (snapField(s, "es", v))  m_EnableSpatial->setValue(atoi(v.c_str()) != 0);
        if (snapField(s, "sm", v))  m_SpatialMode->setValue(atoi(v.c_str()));
        if (snapField(s, "sr", v))  m_SpatialRadius->setValue(atoi(v.c_str()));
        if (snapField(s, "sl", v))  m_SpatialLuma->setValue(atof(v.c_str()));
        if (snapField(s, "sc", v))  m_SpatialChroma->setValue(atof(v.c_str()));
        if (snapField(s, "pd", v))  m_PreserveDetail->setValue(atof(v.c_str()));
        if (snapField(s, "cb", v))  m_ChromaBlotch->setValue(atof(v.c_str()));
        if (snapField(s, "eqf", v)) m_EqFine->setValue(atof(v.c_str()));
        if (snapField(s, "eqm", v)) m_EqMedium->setValue(atof(v.c_str()));
        if (snapField(s, "eqc", v)) m_EqCoarse->setValue(atof(v.c_str()));
        if (snapField(s, "pa", v))  m_ProfileAdjust->setValue(atof(v.c_str()));
        if (snapField(s, "ld", v))  m_LockData->setValue(v);
        if (snapField(s, "lp", v))  m_LockProfile->setValue(atoi(v.c_str()) != 0);
        m_AutoReport->setValue("Reverted. Click Auto Setup to analyze again.");
        m_AutoUndo->setValue("");
        endEditBlock();
        m_InAutoApply = false;

        std::string lockStr;
        m_LockData->getValue(lockStr);
        m_LockValid = nranalyze::parseLockedProfile(lockStr, m_LockAgg);
        updateEnabledness();
    }

    void lockProfileToggled(double p_Time)
    {
        bool on = false;
        m_LockProfile->getValue(on);
        if (!on)
            return;   // unlock: back to live measurement, keep the stored data
        std::vector<nrcore::Stats> per;
        if (analyzeClipFrames(p_Time, per) <= 0)
        {
            m_InAutoApply = true;
            m_LockProfile->setValue(false);
            m_InAutoApply = false;
            analysisFailedMessage();
            return;
        }
        const nranalyze::ClipAggregate agg = nranalyze::aggregateClipStats(per);
        m_LockData->setValue(nranalyze::formatLockedProfile(agg));
        m_LockAgg = agg;
        m_LockValid = true;
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
            reach = (choice == 1) ? 2 : 1;   // 0 -> 3 frames (+/-1), 1 -> 5 frames (+/-2)
        }
        int source = 0;
        m_ProfileSource->getValueAtTime(t, source);
        prevReach = (source != 2) ? std::max(reach, 1) : reach;
    }

    void updateEnabledness()
    {
        int source = 0;
        m_ProfileSource->getValue(source);
        const bool autoOn = (source != 2);
        const bool regionOn = (source == 1);
        m_SigmaY->setEnabled(!autoOn);
        m_SigmaC->setEnabled(!autoOn);
        m_ProfileAdjust->setEnabled(autoOn);
        m_RegionCX->setEnabled(regionOn);
        m_RegionCY->setEnabled(regionOn);
        m_RegionSize->setEnabled(regionOn);

        const bool tOn = m_EnableTemporal->getValue();
        m_TemporalFrames->setEnabled(tOn);
        m_MotionTracking->setEnabled(tOn);
        m_TemporalLuma->setEnabled(tOn);
        m_TemporalChroma->setEnabled(tOn);
        m_MotionThresh->setEnabled(tOn);
        m_FireflyRemoval->setEnabled(tOn);

        const bool sOn = m_EnableSpatial->getValue();
        m_SpatialMode->setEnabled(sOn);
        m_SpatialRadius->setEnabled(sOn);
        m_SpatialLuma->setEnabled(sOn);
        m_SpatialChroma->setEnabled(sOn);
        m_PreserveDetail->setEnabled(sOn);
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
        p.temporalFrames  = (framesChoice == 1) ? 5 : 3;
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
        p.frameIndex      = static_cast<int>(t);

        p.master          = static_cast<float>(m_Master->getValueAtTime(t));
        int view = 0;
        m_ViewMode->getValueAtTime(t, view);
        p.viewMode        = view;

        // ---- v3 ----
        p.motionTracking  = m_MotionTracking->getValueAtTime(t) ? 1 : 0;
        p.fireflyRemoval  = m_FireflyRemoval->getValueAtTime(t) ? 1 : 0;
        p.eqFine          = static_cast<float>(m_EqFine->getValueAtTime(t) / 100.0);
        p.eqMedium        = static_cast<float>(m_EqMedium->getValueAtTime(t) / 100.0);
        p.eqCoarse        = static_cast<float>(m_EqCoarse->getValueAtTime(t) / 100.0);
        p.deband          = static_cast<float>(m_Deband->getValueAtTime(t) / 100.0);
        const bool locked = m_LockProfile->getValueAtTime(t) && m_LockValid;
        p.profileLocked   = locked ? 1 : 0;
        p.lockSY = locked ? m_LockAgg.sy : 0.02f;
        p.lockSC = locked ? m_LockAgg.sc : 0.02f;
        p.lockTY = locked ? m_LockAgg.ty : 0.02f;
        p.lockTC = locked ? m_LockAgg.tc : 0.02f;
        for (int b = 0; b < 16; ++b)
        {
            p.lockGainY[b] = locked ? m_LockAgg.gainY[b] : 1.0f;
            p.lockGainC[b] = locked ? m_LockAgg.gainC[b] : 1.0f;
        }
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

        std::unique_ptr<OFX::Image> held[5];
        OFX::Image* imgs[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };

        held[2].reset(m_SrcClip->fetchImage(t));
        if (!held[2])
            OFX::throwSuiteStatusException(kOfxStatFailed);
        imgs[2] = held[2].get();

        for (int k = -prevReach; k <= reach; ++k)
        {
            if (k == 0)
                continue;
            double tk = t + k;
            if (tk < clipRange.min) tk = clipRange.min;
            if (tk > clipRange.max) tk = clipRange.max;
            held[2 + k].reset(m_SrcClip->fetchImage(tk));
            imgs[2 + k] = held[2 + k] ? held[2 + k].get() : held[2].get();
        }
        for (int i = 0; i < 5; ++i)
            if (!imgs[i])
                imgs[i] = held[2].get();

        if ((imgs[2]->getPixelDepth() != dst->getPixelDepth()) ||
            (imgs[2]->getPixelComponents() != dst->getPixelComponents()))
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

    void renderCPU(OFX::Image* imgs[5], OFX::Image* dst, const NRParams& params)
    {
        const OfxRectI& b = imgs[2]->getBounds();
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
        p.frameIndex     = params.frameIndex;
        p.master         = params.master;
        p.viewMode       = params.viewMode;
        p.motionTracking = params.motionTracking;
        p.fireflyRemoval = params.fireflyRemoval;
        p.eqFine         = params.eqFine;
        p.eqMedium       = params.eqMedium;
        p.eqCoarse       = params.eqCoarse;
        p.deband         = params.deband;
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

        const size_t n = static_cast<size_t>(W) * H * 4;
        std::vector<std::vector<float>> packed;
        const float* fptr[5];

        for (int i = 0; i < 5; ++i)
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

        std::vector<float> out(n), scratch;
        nrcore::denoiseFrame(fptr, W, H, p, out.data(), scratch);

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
    OFX::PushButtonParam* m_RevertAuto = nullptr;
    OFX::StringParam*  m_AutoUndo = nullptr;
    OFX::BooleanParam* m_LockProfile = nullptr;
    OFX::StringParam*  m_LockData = nullptr;
    OFX::BooleanParam* m_MotionTracking = nullptr;
    OFX::BooleanParam* m_FireflyRemoval = nullptr;
    OFX::DoubleParam*  m_EqFine = nullptr;
    OFX::DoubleParam*  m_EqMedium = nullptr;
    OFX::DoubleParam*  m_EqCoarse = nullptr;
    OFX::DoubleParam*  m_Deband = nullptr;
    nranalyze::ClipAggregate m_LockAgg;
    bool m_LockValid = false;
    bool m_InAutoApply = false;
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
    OFX::DoubleParam*  m_ChromaBlotch = nullptr;
    OFX::BooleanParam* m_EnableRefine = nullptr;
    OFX::DoubleParam*  m_ShadowDesat = nullptr;
    OFX::DoubleParam*  m_DesatRange = nullptr;
    OFX::DoubleParam*  m_LumaTexture = nullptr;
    OFX::DoubleParam*  m_GrainAmount = nullptr;
    OFX::DoubleParam*  m_GrainSize = nullptr;
    OFX::DoubleParam*  m_GrainChroma = nullptr;
    OFX::ChoiceParam*  m_ViewMode = nullptr;
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
        addParamToSlaveTo(m_Source);
        addParamToSlaveTo(m_CX);
        addParamToSlaveTo(m_CY);
        addParamToSlaveTo(m_Size);
    }

    virtual bool draw(const OFX::DrawArgs& p_Args)
    {
        if (!regionActive() || !p_Args.context || !OFX::Private::gDrawSuite)
            return false;
        OfxDrawSuiteV1* ds = OFX::Private::gDrawSuite;

        double cx, cy, half;
        geometry(cx, cy, half);
        const double hs = 5.0 * std::max(p_Args.pixelScale.x, 1e-6);

        const OfxRGBAColourF yellow = { 1.0f, 0.9f, 0.1f, 1.0f };
        ds->setColour(p_Args.context, &yellow);
        ds->setLineWidth(p_Args.context, 2.0f);
        ds->setLineStipple(p_Args.context, kOfxDrawLineStipplePatternSolid);

        const OfxPointD rect[4] = { { cx - half, cy - half }, { cx + half, cy - half },
                                    { cx + half, cy + half }, { cx - half, cy + half } };
        ds->draw(p_Args.context, kOfxDrawPrimitiveLineLoop, rect, 4);

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
        return true;
    }

    virtual bool penDown(const OFX::PenArgs& p_Args)
    {
        if (!regionActive())
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
        if (m_Drag == 0 || !regionActive())
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

    // canonical-coordinate geometry (OFX images are bottom-up, so the kernel's
    // normalized y maps directly onto the interact's y-up canonical space)
    void geometry(double& cx, double& cy, double& half)
    {
        const OfxPointD ext = _effect->getProjectExtent();
        cx = m_CX->getValue() * ext.x;
        cy = m_CY->getValue() * ext.y;
        half = 0.5 * m_Size->getValue() * std::min(ext.x, ext.y);
    }

    OFX::ChoiceParam* m_Source;
    OFX::DoubleParam* m_CX;
    OFX::DoubleParam* m_CY;
    OFX::DoubleParam* m_Size;
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
                 "Overall amount of noise reduction. Below 1.0 it fades the effect in and out. "
                 "Above 1.0 it goes further: the filters widen what they treat as noise, so a "
                 "Strength of 2 genuinely removes more than 1 (use for very noisy footage). "
                 "0 = off. Start here.",
                 1.0, 0.0, 2.0, nullptr);

    // ------------------------------------------------------------- auto setup
    {
        OFX::PushButtonParamDescriptor* b = p_Desc.definePushButtonParam("autoSetup");
        b->setLabels("Auto Setup (Analyze Footage)", "Auto Setup (Analyze Footage)", "Auto Setup");
        b->setHint("Analyzes this clip — noise level, chroma character, spatial correlation "
                   "and camera motion, measured on several frames spread across the clip — "
                   "then writes the best settings into the sliders below and locks the "
                   "measured noise profile.\n\n"
                   "This is not a mode: afterwards everything is ordinary manual state, so "
                   "dial any slider in or back from what it chose. One undo (or the Revert "
                   "button) restores everything. Your Step 4 look choices (grain, texture, "
                   "desaturation, debanding) and the View are never touched.");
        page->addChild(*b);
    }
    {
        OFX::StringParamDescriptor* s = p_Desc.defineStringParam("autoReport");
        s->setLabels("Analysis", "Analysis", "Analysis");
        s->setHint("What the last Auto Setup measured and decided.");
        s->setDefault("Click Auto Setup to analyze this clip.");
        s->setEnabled(false);
        s->setEvaluateOnChange(false);
        page->addChild(*s);
    }
    {
        OFX::PushButtonParamDescriptor* b = p_Desc.definePushButtonParam("revertAutoSetup");
        b->setLabels("Revert Auto Setup", "Revert Auto Setup", "Revert Auto");
        b->setHint("Puts every denoise control back to its value from just before the last "
                   "Auto Setup (a plain undo right after Auto Setup does the same thing).");
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
    grpProfile->setHint("Everything else is calibrated from the measured noise level. "
                        "Check the measurement any time with Step 4 > View > Noise Analysis.");

    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("profileSource");
        c->setLabels("Noise Profile", "Noise Profile", "Profile");
        c->setHint("How the noise level is determined.\n\n"
                   "Automatic: measures every frame using frame-to-frame differences plus two "
                   "spatial estimators — works for most footage.\n\n"
                   "Automatic (From Region): same, but measured only inside a rectangle you place "
                   "over a flat area (wall, sky, gray card). Drag the rectangle right in the viewer "
                   "(enable the OpenFX overlay in the viewer's on-screen-controls menu), drag a "
                   "corner to resize — or use the sliders below and the Noise Analysis view.\n\n"
                   "Manual: type the noise levels yourself using the two sliders below.");
        c->appendOption("Automatic (Whole Frame)");
        c->appendOption("Automatic (From Region)");
        c->appendOption("Manual");
        c->setDefault(0);
        c->setParent(*grpProfile);
        page->addChild(*c);
    }
    defineDouble(p_Desc, page, "regionCenterX", "Region Center X",
                 "Horizontal center of the measurement region (0 = left edge, 1 = right edge). "
                 "See the yellow rectangle in the Noise Analysis view.", 0.5, 0.0, 1.0, grpProfile);
    defineDouble(p_Desc, page, "regionCenterY", "Region Center Y",
                 "Vertical center of the measurement region (0 = bottom, 1 = top). Easier: drag the rectangle in the viewer.", 0.5, 0.0, 1.0, grpProfile);
    defineDouble(p_Desc, page, "regionSize", "Region Size",
                 "Size of the measurement region relative to the frame.", 0.25, 0.05, 1.0, grpProfile);
    defineDouble(p_Desc, page, "profileAdjust", "Auto Profile Adjust",
                 "Fine-tunes the automatic measurement. 1.0 = trust it as-is. Raise if noise is left "
                 "behind (the filters think the footage is cleaner than it is); lower if detail is "
                 "getting eaten. Check the result in the Noise Only view.",
                 1.0, 0.25, 4.0, grpProfile);
    defineDouble(p_Desc, page, "sigmaLuma", "Manual Luma Noise (%)",
                 "Noise level of the brightness channel, as a percentage of full signal range. Only "
                 "used when Noise Profile is set to Manual. Typical clean footage: 0.5–1. Visibly "
                 "noisy: 2–5. Very noisy low light: 5–10.", 2.0, 0.05, 25.0, grpProfile);
    defineDouble(p_Desc, page, "sigmaChroma", "Manual Chroma Noise (%)",
                 "Noise level of the color channels (the colored speckle). Only used when Noise "
                 "Profile is set to Manual.", 2.0, 0.05, 25.0, grpProfile);
    defineBool(p_Desc, page, "lockProfile", "Lock Profile",
               "Measures the noise on several frames spread across the clip, then freezes that "
               "profile (both sigma pairs and the brightness curve) so every frame filters "
               "against the same numbers instead of re-measuring per frame.\n\n"
               "Lock when a clip flickers between shots of different content (per-frame "
               "measurement can breathe) or when you want renders to be repeatable. The "
               "analysis HUD shows LOCKED while active; its histogram stays live so you can "
               "compare. Un-tick to go back to per-frame measurement. Saved with the project.",
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
    grpTemporal->setHint("Averages each pixel with the same pixel in neighboring frames — the most "
                         "effective and detail-safe reduction, but only where nothing is moving. "
                         "Motion is detected automatically and those areas are left to Step 3. "
                         "See where it is working with Step 4 > View > Temporal Activity.");

    defineBool(p_Desc, page, "enableTemporal", "Enable Temporal NR",
               "Toggle the across-frames stage on/off to see its contribution. "
               "Turn off for still images or single-frame clips.",
               true, grpTemporal);
    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("temporalFrames");
        c->setLabels("Number of Frames", "Number of Frames", "Frames");
        c->setHint("How many frames are compared and averaged. 5 frames removes more noise in "
                   "static areas but renders slower. 3 frames is a good default.");
        c->appendOption("3 Frames");
        c->appendOption("5 Frames");
        c->setDefault(0);
        c->setParent(*grpTemporal);
        page->addChild(*c);
    }
    defineBool(p_Desc, page, "motionTracking", "Motion Tracking",
               "Before deciding whether a neighboring frame matches, tries shifting its patch "
               "by up to 2 pixels and uses the best alignment. Slow pans and handheld drift "
               "keep their across-frames averaging instead of falling back to spatial-only.\n\n"
               "Leave on for almost everything; turn off for a small speed gain on locked-off "
               "tripod footage (where it has nothing to find). Motion faster than ~2 px/frame "
               "is still protected by the gate exactly as before.",
               true, grpTemporal);
    defineDouble(p_Desc, page, "temporalLuma", "Luma Strength",
                 "How strongly brightness noise is averaged across frames (0–100).", 60.0, 0.0, 100.0, grpTemporal);
    defineDouble(p_Desc, page, "temporalChroma", "Chroma Strength",
                 "How strongly color noise is averaged across frames. Color can take more than "
                 "brightness without visible side effects.", 80.0, 0.0, 100.0, grpTemporal);
    defineDouble(p_Desc, page, "motionThreshold", "Motion Threshold",
                 "How much frame-to-frame change counts as motion, relative to the measured noise. "
                 "The gate closes to exactly zero past this point, so pixels that changed are never "
                 "blended. Low = cautious (less reduction near movement), high = stronger reduction "
                 "on near-static areas. If you see any trailing on motion, lower this first.",
                 30.0, 0.0, 100.0, grpTemporal);
    defineBool(p_Desc, page, "fireflyRemoval", "Firefly Removal",
               "Zaps single-frame impulses — hot pixels, sensor 'fireflies', stray sparkles — "
               "by clipping them to the 3-frame temporal median. A pixel is only zapped when "
               "it spikes hard against BOTH neighboring frames while those frames agree with "
               "each other, and it is also an outlier within its own frame, so moving detail "
               "is left alone.\n\n"
               "Leave on; turn off only if legitimate one-frame flashes (strobes, muzzle "
               "flashes, glints) lose their sparkle.",
               true, grpTemporal);

    // ---------------------------------------------------------- step 3: spatial
    OFX::GroupParamDescriptor* grpSpatial = p_Desc.defineGroupParam("grpSpatial");
    grpSpatial->setLabels("Step 3 · Spatial NR (Within Frame)", "Step 3 · Spatial NR", "3 · Spatial");
    grpSpatial->setOpen(true);
    grpSpatial->setHint("Cleans the noise that remains after Step 2 (and everything in moving areas) "
                        "by averaging each pixel only with genuinely similar neighborhoods, which "
                        "preserves edges and texture.");

    defineBool(p_Desc, page, "enableSpatial", "Enable Spatial NR",
               "Toggle the within-frame stage on/off to see its contribution.",
               true, grpSpatial);
    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("spatialMethod");
        c->setLabels("Method", "Method", "Method");
        c->setHint("Better (NLM) compares whole 3x3 patches — higher quality, keeps texture. "
                   "Faster (Bilateral) compares single pixels — use it if playback is too slow.");
        c->appendOption("Faster (Bilateral)");
        c->appendOption("Better (NLM)");
        c->setDefault(1);
        c->setParent(*grpSpatial);
        page->addChild(*c);
    }
    {
        OFX::IntParamDescriptor* i = p_Desc.defineIntParam("spatialRadius");
        i->setLabels("Search Radius", "Search Radius", "Radius");
        i->setHint("How far (in pixels) to look for similar areas. Larger = smoother flat areas but "
                   "slower rendering. 3 is a good default; go to 6-8 for very soft, very noisy sources.");
        i->setDefault(3);
        i->setRange(1, 8);
        i->setDisplayRange(1, 8);
        i->setParent(*grpSpatial);
        page->addChild(*i);
    }
    defineDouble(p_Desc, page, "spatialLuma", "Luma Strength",
                 "How much brightness noise to remove within the frame (0–100). Higher values both "
                 "blend more of the filtered result AND filter more aggressively. Too high can look "
                 "waxy — pair with Step 4's texture/grain to keep a filmic feel.", 60.0, 0.0, 100.0, grpSpatial);
    defineDouble(p_Desc, page, "spatialChroma", "Chroma Strength",
                 "How much color noise (speckle) to remove. Chroma tolerates maximum strength on "
                 "most footage — 100 is the default for a reason.", 100.0, 0.0, 100.0, grpSpatial);
    defineDouble(p_Desc, page, "preserveDetail", "Preserve Detail",
                 "Protects edges and texture: filtering is automatically reduced where the image has "
                 "real structure above the measured noise floor. Raise if fine detail is softening; "
                 "lower if edges stay noisy.", 35.0, 0.0, 100.0, grpSpatial);
    defineDouble(p_Desc, page, "chromaBlotch", "Chroma Blotch Reduction",
                 "A second, LARGE-radius color pass (up to 16 px) that reaches the big soft color "
                 "stains that 4:2:0 compression leaves behind — the ones the normal search radius "
                 "physically can't span. Guided by brightness so color never bleeds across edges. "
                 "Raise for blotchy phone/low-light footage.", 25.0, 0.0, 100.0, grpSpatial);
    defineDouble(p_Desc, page, "eqFine", "Noise EQ · Fine",
                 "Band strength for pixel-scale noise — scales the main NLM/bilateral pass. "
                 "100 = normal. Lower if fine texture is going waxy while you want the other "
                 "bands to keep working; raise above 100 to lean harder on fine grain when "
                 "the sliders above are already maxed.", 100.0, 0.0, 200.0, grpSpatial);
    defineDouble(p_Desc, page, "eqMedium", "Noise EQ · Medium",
                 "Band strength for mid-size noise clumps (roughly 3–8 px) — the chunky blotch "
                 "left by heavy compression or in-camera NR, which the fine pass can't tell "
                 "from detail. Off by default. Raise when noise looks like moving 'clumps' "
                 "rather than fine grain; Auto Setup raises it when it measures spatially "
                 "correlated noise.", 0.0, 0.0, 100.0, grpSpatial);
    defineDouble(p_Desc, page, "eqCoarse", "Noise EQ · Coarse (Luma)",
                 "Band strength for LARGE soft brightness stains (16 px and up, reaching to "
                 "32 px) — the slow luma mottling severe compression leaves in skies and "
                 "walls. Works on 4x4 block averages with a clipped correction, so real "
                 "structure is untouched by more than a noise-sized amount. Off by default; "
                 "raise on very compressed low-light footage.", 0.0, 0.0, 100.0, grpSpatial);

    // ----------------------------------------------------------- step 4: refine
    OFX::GroupParamDescriptor* grpRefine = p_Desc.defineGroupParam("grpRefine");
    grpRefine->setLabels("Step 4 · Refine The Finish", "Step 4 · Refine", "4 · Refine");
    grpRefine->setOpen(true);
    grpRefine->setHint("Finishing touches after denoising: hide remaining color noise in the "
                       "shadows, bring back natural texture, or lay down clean synthetic film "
                       "grain in place of the ugly noise you removed.");

    defineBool(p_Desc, page, "enableRefine", "Enable Refinements",
               "Toggle the whole finishing stage on/off to compare.",
               true, grpRefine);
    defineDouble(p_Desc, page, "shadowDesat", "Shadow Desaturate",
                 "Fades color saturation toward zero in the darkest tones (a saturation-vs-luma "
                 "curve). Chroma noise lives in shadows — this hides what filtering can't remove, "
                 "and reads as a clean, cinematic shadow rendering. Try 20–40 on noisy footage.",
                 0.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "desatRange", "Desaturate Range",
                 "How far up the tonal scale the shadow desaturation reaches (in luma, 0.02–0.5). "
                 "Default 0.15 affects only true shadows.",
                 0.15, 0.02, 0.5, grpRefine);
    defineDouble(p_Desc, page, "lumaTexture", "Luma Texture",
                 "Re-injects a percentage of the ORIGINAL brightness texture (grain) into the "
                 "denoised image — the color noise stays gone, but the image keeps its natural "
                 "film-like energy. Try 15–30 instead of cranking strengths down.",
                 0.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "deband", "Deband",
                 "Smooths banding — the visible stair-steps in skies and gradients that 8-bit "
                 "sources (and denoising itself) can reveal — by averaging along the gradient "
                 "within a banding-sized tolerance, plus an invisible micro-dither so any "
                 "remaining step decorrelates. Real edges are rejected by the tolerance and "
                 "never touched. 0 = off. Raise until the contours in flat gradients dissolve; "
                 "it will not soften detail.",
                 0.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "grainAmount", "Film Grain",
                 "Adds clean, synthesized grain — soft, organic, animated per frame, strongest in "
                 "the midtones like real film stock. The classic finishing move: remove the ugly "
                 "noise, then lay down grain you chose.",
                 0.0, 0.0, 100.0, grpRefine);
    defineDouble(p_Desc, page, "grainSize", "Grain Size",
                 "Grain particle size in pixels. 1 = fine 35mm-like, 3-4 = chunky 16mm feel. "
                 "Scale up for UHD timelines.",
                 1.6, 0.5, 6.0, grpRefine);
    defineDouble(p_Desc, page, "grainChroma", "Grain Color",
                 "0 = pure monochrome grain (film-like). Higher adds independent color grain "
                 "(digital-sensor character).",
                 25.0, 0.0, 100.0, grpRefine);

    // ---------------------------------------------------------- step 5: inspect
    OFX::GroupParamDescriptor* grpOutput = p_Desc.defineGroupParam("grpOutput");
    grpOutput->setLabels("Step 5 · Inspect & Compare", "Step 5 · Inspect", "5 · Inspect");
    grpOutput->setOpen(true);
    grpOutput->setHint("Views for checking what the plugin is measuring and doing. Set back to "
                       "Result before rendering.");

    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("viewMode");
        c->setLabels("View", "View", "View");
        c->setHint("Follow the image through the pipeline:\n\n"
                   "Result: the finished frame.\n\n"
                   "Split: input on the left, result on the right.\n\n"
                   "Input: the untouched source, for flip-comparisons.\n\n"
                   "After Temporal: the image after Step 2 only — see what the across-frames stage "
                   "contributed before spatial filtering.\n\n"
                   "Noise Removed: what denoising took out (amplified 4x around gray, excludes "
                   "grain/refinements). Should look like pure static — visible faces or edges mean "
                   "detail is being cut: lower strengths or raise Preserve Detail.\n\n"
                   "Noise Analysis: live measurements — input noise (blue) and the residual left "
                   "after temporal NR (amber) for luma and chroma, effective frames averaged, the "
                   "SNR gain so far, the noise-vs-brightness curve, the noise histogram, and the "
                   "measurement region rectangle.\n\n"
                   "Temporal Activity: green = frame-averaging active, red = motion-protected.\n\n"
                   "SNR Map: signal-to-noise per pixel — magenta where noise dominates (NR matters "
                   "most), green where the image wins.\n\n"
                   "Matte - Noisiness: the noise-dominance map written to BOTH the RGB and the "
                   "alpha channel (white/opaque = noise dominates, black/transparent = the image "
                   "wins) — feed it downstream as a key so later nodes treat noisy areas "
                   "differently.");
        c->appendOption("Result");
        c->appendOption("Split (Input | Result)");
        c->appendOption("Input (Original)");
        c->appendOption("After Temporal (Step 2 Only)");
        c->appendOption("Noise Removed (Amplified)");
        c->appendOption("Noise Analysis (Measurements)");
        c->appendOption("Temporal Activity (Green = Averaging)");
        c->appendOption("SNR Map (Magenta = Noisy)");
        c->appendOption("Matte: Noisiness (RGB + Alpha)");
        c->setDefault(0);
        c->setParent(*grpOutput);
        page->addChild(*c);
    }
}

OFX::ImageEffect* OpenNRPluginFactory::createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum /*p_Context*/)
{
    return new OpenNRPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& p_FactoryArray)
{
    static OpenNRPluginFactory openNRPlugin;
    p_FactoryArray.push_back(&openNRPlugin);
}
