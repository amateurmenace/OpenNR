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

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#include "NRParams.h"
#include "nr_core.h"

#define kPluginName "OpenNR Denoise"
#define kPluginGrouping "OpenNR"
#define kPluginDescription \
    "OpenNR v1.2 — free spatio-temporal video noise reduction.\n\n" \
    "Work top to bottom: 1) measure the noise (automatic, from a region, or " \
    "manual), 2) temporal NR averages matching pixels across frames, 3) spatial " \
    "NR cleans what remains, 4) inspect with the analysis views.\n\n" \
    "Open Step 4 and choose 'Noise Analysis' to SEE the measured noise levels " \
    "on screen, or 'Noise Only' to see exactly what is being removed.\n\n" \
    "MIT-licensed and free forever."
#define kPluginIdentifier "org.opennr.Denoise"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 2

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////
// GPU entry points
////////////////////////////////////////////////////////////////////////////////

#ifdef __APPLE__
extern void RunMetalNR(void* p_CmdQ, int p_Width, int p_Height, const NRParams& p_Params,
                       const float* const p_Srcs[5], float* p_Dst);
#else
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
#ifndef __APPLE__
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
        m_ProfileSource  = fetchChoiceParam("profileSource");
        m_ProfileAdjust  = fetchDoubleParam("profileAdjust");
        m_SigmaY         = fetchDoubleParam("sigmaLuma");
        m_SigmaC         = fetchDoubleParam("sigmaChroma");
        m_RegionCX       = fetchDoubleParam("regionCenterX");
        m_RegionCY       = fetchDoubleParam("regionCenterY");
        m_RegionSize     = fetchDoubleParam("regionSize");
        m_EnableTemporal = fetchBooleanParam("enableTemporal");
        m_TemporalFrames = fetchChoiceParam("temporalFrames");
        m_TemporalLuma   = fetchDoubleParam("temporalLuma");
        m_TemporalChroma = fetchDoubleParam("temporalChroma");
        m_MotionThresh   = fetchDoubleParam("motionThreshold");
        m_EnableSpatial  = fetchBooleanParam("enableSpatial");
        m_SpatialMode    = fetchChoiceParam("spatialMethod");
        m_SpatialRadius  = fetchIntParam("spatialRadius");
        m_SpatialLuma    = fetchDoubleParam("spatialLuma");
        m_SpatialChroma  = fetchDoubleParam("spatialChroma");
        m_PreserveDetail = fetchDoubleParam("preserveDetail");
        m_ViewMode       = fetchChoiceParam("viewMode");

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

        if (master <= 0.0 || (!tOn && !sOn))
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

    virtual void changedParam(const OFX::InstanceChangedArgs& /*p_Args*/, const std::string& p_ParamName)
    {
        if (p_ParamName == "profileSource" || p_ParamName == "enableTemporal" || p_ParamName == "enableSpatial")
            updateEnabledness();
    }

private:
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
        m_TemporalLuma->setEnabled(tOn);
        m_TemporalChroma->setEnabled(tOn);
        m_MotionThresh->setEnabled(tOn);

        const bool sOn = m_EnableSpatial->getValue();
        m_SpatialMode->setEnabled(sOn);
        m_SpatialRadius->setEnabled(sOn);
        m_SpatialLuma->setEnabled(sOn);
        m_SpatialChroma->setEnabled(sOn);
        m_PreserveDetail->setEnabled(sOn);
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

        p.master          = static_cast<float>(m_Master->getValueAtTime(t));
        int view = 0;
        m_ViewMode->getValueAtTime(t, view);
        p.viewMode        = view;
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
        p.master         = params.master;
        p.viewMode       = params.viewMode;

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
    OFX::ChoiceParam*  m_ViewMode = nullptr;
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
#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(true);
#endif
#ifdef __APPLE__
    p_Desc.setSupportsMetalRender(true);
#endif
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
                 "Overall amount of noise reduction. Scales every strength below at once. "
                 "0 = off, 1 = normal, up to 2 for very noisy footage. Start here.",
                 1.0, 0.0, 2.0, nullptr);

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
                   "over a flat area (wall, sky, gray card). Use when the frame is full of fine "
                   "texture that could be mistaken for noise. Position it in the Noise Analysis view.\n\n"
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
                 "Vertical center of the measurement region (0 = top, 1 = bottom).", 0.5, 0.0, 1.0, grpProfile);
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
                   "slower rendering. 3 is a good default.");
        i->setDefault(3);
        i->setRange(1, 5);
        i->setDisplayRange(1, 5);
        i->setParent(*grpSpatial);
        page->addChild(*i);
    }
    defineDouble(p_Desc, page, "spatialLuma", "Luma Strength",
                 "How much brightness noise to remove within the frame (0–100). Too high can look "
                 "waxy — keep moderate to retain a filmic texture.", 45.0, 0.0, 100.0, grpSpatial);
    defineDouble(p_Desc, page, "spatialChroma", "Chroma Strength",
                 "How much color noise (blotches, speckle) to remove. Chroma tolerates high values "
                 "— 75–100 is normal for phone or low-light footage.", 75.0, 0.0, 100.0, grpSpatial);
    defineDouble(p_Desc, page, "preserveDetail", "Preserve Detail",
                 "Protects edges and texture: filtering is automatically reduced where the image has "
                 "real structure above the measured noise floor. Raise if fine detail is softening; "
                 "lower if edges stay noisy.", 35.0, 0.0, 100.0, grpSpatial);

    // ---------------------------------------------------------- step 4: inspect
    OFX::GroupParamDescriptor* grpOutput = p_Desc.defineGroupParam("grpOutput");
    grpOutput->setLabels("Step 4 · Inspect & Compare", "Step 4 · Inspect", "4 · Inspect");
    grpOutput->setOpen(true);
    grpOutput->setHint("Views for checking what the plugin is measuring and doing. Set back to "
                       "Result before rendering.");

    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("viewMode");
        c->setLabels("View", "View", "View");
        c->setHint("Result: the denoised image.\n\n"
                   "Split: original on the left, denoised on the right.\n\n"
                   "Noise Only: what is being removed (amplified 4x around gray). It should look "
                   "like pure static — if you can see edges or faces in it, you are removing "
                   "detail: lower Strength or raise Preserve Detail.\n\n"
                   "Noise Analysis: on-screen readout of the measured noise levels (spatial and "
                   "temporal, luma and chroma), meters, the noise histogram with its median marked, "
                   "and the measurement region rectangle.\n\n"
                   "Temporal Activity: where across-frame averaging is active (green) versus "
                   "motion-protected (red). Red everywhere = temporal NR is doing nothing (motion "
                   "or disabled); green flats = it is working.");
        c->appendOption("Result");
        c->appendOption("Split (Before | After)");
        c->appendOption("Noise Only (What Is Removed)");
        c->appendOption("Noise Analysis (Measurements)");
        c->appendOption("Temporal Activity (Green = Averaging)");
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
