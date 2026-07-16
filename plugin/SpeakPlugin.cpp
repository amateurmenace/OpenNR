// Speak — free film-reconstruction OpenFX plugin for DaVinci Resolve.
//
// The film counterpart to Hush: the LAST node in a grade (Hush is the first).
// Registered as a SECOND plugin (org.opennr.Speak) in the same .ofx bundle,
// sharing the vendored ofx/ support and Hush's four-backend / parity discipline.
//
// Phase 1 ships the density spine (color-managed Log-Exposure Spine + closed-
// form negative->printer->print H&D tone scale) and the live H&D curve scope.
// speak_core.h is the single source of truth; the GPU kernels are ports of it.
//
// MIT License.

#include "SpeakPlugin.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#include "SpeakParams.h"
#include "speak_core.h"

#define kSpeakName "Speak Film"
#define kSpeakGrouping "Hush"
#define kSpeakDescription \
    "Speak — the free film-reconstruction node. Drop it on the LAST node of a " \
    "DaVinci Wide Gamut / Intermediate managed timeline; leave the output on " \
    "\"Working space\" and let Resolve Color Management deliver Rec.709.\n\n" \
    "Phase 1: a physically-grounded film tone scale (Hurter-Driffield negative " \
    "and print characteristic curves with printer-light color timing), every " \
    "curve shown on screen. Raise Strength to dial the look; tick Scope: H&D " \
    "Curves to see the exact curve the pixels use.\n\n" \
    "Hush quiets the noise; Speak gives the image its voice. MIT-licensed, free."
#define kSpeakIdentifier "org.opennr.Speak"
#define kSpeakVersionMajor 0
#define kSpeakVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////
// GPU entry points (line-by-line ports of speak_core.h::speakFrame)
////////////////////////////////////////////////////////////////////////////////

#ifdef __APPLE__
extern void RunMetalSpeak(void* p_CmdQ, int p_Width, int p_Height,
                          const SpeakParams& p_Params, const float* p_Src, float* p_Dst);
#endif
#ifdef HUSH_ENABLE_CUDA
extern void RunCudaSpeak(void* p_Stream, int p_Width, int p_Height,
                         const SpeakParams& p_Params, const float* p_Src, float* p_Dst);
#endif
extern void RunOpenCLSpeak(void* p_CmdQ, int p_Width, int p_Height,
                           const SpeakParams& p_Params, const float* p_Src, float* p_Dst);

////////////////////////////////////////////////////////////////////////////////
// Processor
////////////////////////////////////////////////////////////////////////////////

class SpeakProcessor : public OFX::ImageProcessor
{
public:
    explicit SpeakProcessor(OFX::ImageEffect& p_Instance) : OFX::ImageProcessor(p_Instance) {}

    void setSrcImg(OFX::Image* p_Img) { _srcImg = p_Img; }
    void setParams(const SpeakParams& p) { _params = p; }

    virtual void processImagesMetal()
    {
#ifdef __APPLE__
        const OfxRectI& b = _srcImg->getBounds();
        RunMetalSpeak(_pMetalCmdQ, b.x2 - b.x1, b.y2 - b.y1, _params,
                      static_cast<float*>(_srcImg->getPixelData()),
                      static_cast<float*>(_dstImg->getPixelData()));
#endif
    }

    virtual void processImagesCUDA()
    {
#ifdef HUSH_ENABLE_CUDA
        const OfxRectI& b = _srcImg->getBounds();
        RunCudaSpeak(_pCudaStream, b.x2 - b.x1, b.y2 - b.y1, _params,
                     static_cast<float*>(_srcImg->getPixelData()),
                     static_cast<float*>(_dstImg->getPixelData()));
#endif
    }

    virtual void processImagesOpenCL()
    {
        const OfxRectI& b = _srcImg->getBounds();
        RunOpenCLSpeak(_pOpenCLCmdQ, b.x2 - b.x1, b.y2 - b.y1, _params,
                       static_cast<float*>(_srcImg->getPixelData()),
                       static_cast<float*>(_dstImg->getPixelData()));
    }

    virtual void multiThreadProcessImages(OfxRectI) {}   // CPU handled whole-frame

private:
    OFX::Image* _srcImg = nullptr;
    SpeakParams _params = {};
};

////////////////////////////////////////////////////////////////////////////////
// Plugin instance
////////////////////////////////////////////////////////////////////////////////

class SpeakPlugin : public OFX::ImageEffect
{
public:
    explicit SpeakPlugin(OfxImageEffectHandle p_Handle) : ImageEffect(p_Handle)
    {
        m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
        m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

        m_InputCS       = fetchChoiceParam("inputColorSpace");
        m_OutputMode    = fetchChoiceParam("outputMode");
        m_EnableTone    = fetchBooleanParam("enableTone");
        m_Strength      = fetchDoubleParam("strength");
        m_Contrast      = fetchDoubleParam("contrast");
        m_PrintShoulder = fetchDoubleParam("printShoulder");
        m_Toe           = fetchDoubleParam("toe");
        m_PrinterR      = fetchDoubleParam("printerR");
        m_PrinterG      = fetchDoubleParam("printerG");
        m_PrinterB      = fetchDoubleParam("printerB");
        m_PrinterMaster = fetchDoubleParam("printerMaster");
        m_EnableDye     = fetchBooleanParam("enableDye");
        m_SubSat        = fetchDoubleParam("subSat");
        m_DyeCoupler    = fetchDoubleParam("dyeCoupler");
        m_DyeKnee       = fetchDoubleParam("dyeKnee");
        m_EnableSplit   = fetchBooleanParam("enableSplit");
        m_SplitShadowR  = fetchDoubleParam("splitShadowR");
        m_SplitShadowG  = fetchDoubleParam("splitShadowG");
        m_SplitShadowB  = fetchDoubleParam("splitShadowB");
        m_SplitHighR    = fetchDoubleParam("splitHighR");
        m_SplitHighG    = fetchDoubleParam("splitHighG");
        m_SplitHighB    = fetchDoubleParam("splitHighB");
        m_SplitPivot    = fetchDoubleParam("splitPivot");
        m_SplitBalance  = fetchDoubleParam("splitBalance");
        m_ViewMode      = fetchChoiceParam("viewMode");
        m_ScopeHD       = fetchBooleanParam("scopeHD");
        m_ScopeDensity  = fetchBooleanParam("scopeDensity");
        updateEnabledness();
    }

    virtual void render(const OFX::RenderArguments& p_Args)
    {
        if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) &&
            (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
            setupAndProcess(p_Args);
        else
            OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }

    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
    {
        const double t = p_Args.time;
        int vm = 0, om = 0;
        m_ViewMode->getValueAtTime(t, vm);
        m_OutputMode->getValueAtTime(t, om);
        if (vm != 0) return false;
        if (om != SPEAK_OUT_WORKING) return false;        // bake always transforms
        if (m_ScopeHD->getValueAtTime(t) || m_ScopeDensity->getValueAtTime(t)) return false;  // scopes must still draw
        const bool toneOn = m_EnableTone->getValueAtTime(t) && (m_Strength->getValueAtTime(t) > 0.0);
        const bool dyeOn  = m_EnableDye->getValueAtTime(t) &&
                            (m_SubSat->getValueAtTime(t) > 0.0 || m_DyeCoupler->getValueAtTime(t) > 0.0);
        const bool splitOn = m_EnableSplit->getValueAtTime(t) &&
                             (m_SplitShadowR->getValueAtTime(t) != 0.0 || m_SplitShadowG->getValueAtTime(t) != 0.0 ||
                              m_SplitShadowB->getValueAtTime(t) != 0.0 || m_SplitHighR->getValueAtTime(t) != 0.0 ||
                              m_SplitHighG->getValueAtTime(t) != 0.0 || m_SplitHighB->getValueAtTime(t) != 0.0);
        if (!toneOn && !dyeOn && !splitOn) { p_IdentityClip = m_SrcClip; p_IdentityTime = t; return true; }
        return false;
    }

    virtual void changedParam(const OFX::InstanceChangedArgs& /*p_Args*/, const std::string& p_ParamName)
    {
        if (p_ParamName == "enableTone" || p_ParamName == "enableDye" ||
            p_ParamName == "enableSplit") updateEnabledness();
    }

private:
    void updateEnabledness()
    {
        const bool on = m_EnableTone->getValue();
        m_Strength->setEnabled(on);
        m_Contrast->setEnabled(on);
        m_PrintShoulder->setEnabled(on);
        m_Toe->setEnabled(on);
        m_PrinterR->setEnabled(on);
        m_PrinterG->setEnabled(on);
        m_PrinterB->setEnabled(on);
        m_PrinterMaster->setEnabled(on);
        const bool dye = m_EnableDye->getValue();
        m_SubSat->setEnabled(dye);
        m_DyeCoupler->setEnabled(dye);
        m_DyeKnee->setEnabled(dye);
        const bool sp = m_EnableSplit->getValue();
        m_SplitShadowR->setEnabled(sp); m_SplitShadowG->setEnabled(sp); m_SplitShadowB->setEnabled(sp);
        m_SplitHighR->setEnabled(sp); m_SplitHighG->setEnabled(sp); m_SplitHighB->setEnabled(sp);
        m_SplitPivot->setEnabled(sp); m_SplitBalance->setEnabled(sp);
    }

    SpeakParams gatherParams(double t)
    {
        SpeakParams p = {};
        int cs = 0, om = 0, vm = 0;
        m_InputCS->getValueAtTime(t, cs);
        m_OutputMode->getValueAtTime(t, om);
        m_ViewMode->getValueAtTime(t, vm);
        p.inputColorSpace = cs;
        p.outputMode      = om;
        p.grainRef        = 0;
        p.strength        = static_cast<float>(m_Strength->getValueAtTime(t));
        p.frameIndex      = static_cast<int>(std::floor(t + 0.5));
        p.viewMode        = vm;
        p.enableTone      = m_EnableTone->getValueAtTime(t) ? 1 : 0;
        p.enableDye       = m_EnableDye->getValueAtTime(t) ? 1 : 0;
        p.enableSplit     = m_EnableSplit->getValueAtTime(t) ? 1 : 0;
        p.enableOptics    = 0;
        p.scopeHD         = m_ScopeHD->getValueAtTime(t) ? 1 : 0;
        p.scopeDensity    = m_ScopeDensity->getValueAtTime(t) ? 1 : 0;
        p.scopeVector     = 0;

        // Build the look profile: start from the gray-balanced Neutral stock and
        // apply the Phase-1 macro handles. Built-in stock families and Shoot-a-
        // Chart calibration will emit this SAME struct (one kernel path).
        SpeakProfile prof = speakcore::neutralProfile();
        const float contrast = static_cast<float>(m_Contrast->getValueAtTime(t));
        const float shoulder = static_cast<float>(m_PrintShoulder->getValueAtTime(t));
        const float toe      = static_cast<float>(m_Toe->getValueAtTime(t));
        for (int c = 0; c < 3; ++c) {
            prof.prnGamma[c]    *= contrast;
            prof.prnShoulder[c]  = shoulder;
            prof.prnToe[c]       = toe;
        }
        prof.printerLights[0] = static_cast<float>(m_PrinterR->getValueAtTime(t));
        prof.printerLights[1] = static_cast<float>(m_PrinterG->getValueAtTime(t));
        prof.printerLights[2] = static_cast<float>(m_PrinterB->getValueAtTime(t));
        prof.printerMaster    = static_cast<float>(m_PrinterMaster->getValueAtTime(t));

        // Subtractive color (Phase 2): one knob drives the per-dye saturation,
        // one scales the shared dye cross-absorption pattern (speak_core.h).
        const float sat  = static_cast<float>(m_SubSat->getValueAtTime(t));
        const float knee = static_cast<float>(m_DyeKnee->getValueAtTime(t));
        for (int c = 0; c < 3; ++c) { prof.subSat[c] = sat; prof.subSatKnee[c] = knee; }
        speakcore::setDyeCoupler(prof, static_cast<float>(m_DyeCoupler->getValueAtTime(t)));

        // Split toning (Phase 3): per-channel density offsets per tonal zone.
        prof.splitShadow[0] = static_cast<float>(m_SplitShadowR->getValueAtTime(t));
        prof.splitShadow[1] = static_cast<float>(m_SplitShadowG->getValueAtTime(t));
        prof.splitShadow[2] = static_cast<float>(m_SplitShadowB->getValueAtTime(t));
        prof.splitHigh[0]   = static_cast<float>(m_SplitHighR->getValueAtTime(t));
        prof.splitHigh[1]   = static_cast<float>(m_SplitHighG->getValueAtTime(t));
        prof.splitHigh[2]   = static_cast<float>(m_SplitHighB->getValueAtTime(t));
        prof.splitPivot     = static_cast<float>(m_SplitPivot->getValueAtTime(t));
        prof.splitBalance   = static_cast<float>(m_SplitBalance->getValueAtTime(t));

        p.profile = prof;
        return p;
    }

    void setupAndProcess(const OFX::RenderArguments& p_Args)
    {
        const double t = p_Args.time;
        std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(t));
        std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(t));
        if (!dst || !src) OFX::throwSuiteStatusException(kOfxStatFailed);
        if ((src->getPixelDepth() != dst->getPixelDepth()) ||
            (src->getPixelComponents() != dst->getPixelComponents()))
            OFX::throwSuiteStatusException(kOfxStatErrValue);

        const SpeakParams params = gatherParams(t);

        const bool gpu = p_Args.isEnabledMetalRender || p_Args.isEnabledCudaRender || p_Args.isEnabledOpenCLRender;
        if (gpu) {
            SpeakProcessor proc(*this);
            proc.setDstImg(dst.get());
            proc.setSrcImg(src.get());
            proc.setGPURenderArgs(p_Args);
            proc.setRenderWindow(p_Args.renderWindow);
            proc.setParams(params);
            proc.process();
        } else {
            renderCPU(src.get(), dst.get(), params);
        }
    }

    // CPU reference path — packs to a contiguous frame (fetchImage buffers may
    // have padded rowBytes) and runs the SAME whole-frame entry point the tests
    // and the GPU ports are verified against, so the scope's measurement pass
    // is never duplicated. The GPU paths take Resolve's contiguous buffers.
    void renderCPU(OFX::Image* src, OFX::Image* dst, const SpeakParams& params)
    {
        const OfxRectI b = src->getBounds();
        const int W = b.x2 - b.x1, H = b.y2 - b.y1;
        if (W <= 0 || H <= 0) return;
        std::vector<float> in(static_cast<size_t>(W) * H * 4, 0.0f), out(static_cast<size_t>(W) * H * 4, 0.0f);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float* s = static_cast<float*>(src->getPixelAddress(b.x1 + x, b.y1 + y));
                if (!s) continue;
                float* d = &in[(static_cast<size_t>(y) * W + x) * 4];
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
            }
        speakcore::speakFrame(in.data(), W, H, params, out.data());
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float* dp = static_cast<float*>(dst->getPixelAddress(b.x1 + x, b.y1 + y));
                if (!dp) continue;
                const float* o = &out[(static_cast<size_t>(y) * W + x) * 4];
                dp[0] = o[0]; dp[1] = o[1]; dp[2] = o[2]; dp[3] = o[3];
            }
    }

    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    OFX::ChoiceParam*  m_InputCS;
    OFX::ChoiceParam*  m_OutputMode;
    OFX::BooleanParam* m_EnableTone;
    OFX::DoubleParam*  m_Strength;
    OFX::DoubleParam*  m_Contrast;
    OFX::DoubleParam*  m_PrintShoulder;
    OFX::DoubleParam*  m_Toe;
    OFX::DoubleParam*  m_PrinterR;
    OFX::DoubleParam*  m_PrinterG;
    OFX::DoubleParam*  m_PrinterB;
    OFX::DoubleParam*  m_PrinterMaster;
    OFX::BooleanParam* m_EnableDye;
    OFX::DoubleParam*  m_SubSat;
    OFX::DoubleParam*  m_DyeCoupler;
    OFX::DoubleParam*  m_DyeKnee;
    OFX::BooleanParam* m_EnableSplit;
    OFX::DoubleParam*  m_SplitShadowR;
    OFX::DoubleParam*  m_SplitShadowG;
    OFX::DoubleParam*  m_SplitShadowB;
    OFX::DoubleParam*  m_SplitHighR;
    OFX::DoubleParam*  m_SplitHighG;
    OFX::DoubleParam*  m_SplitHighB;
    OFX::DoubleParam*  m_SplitPivot;
    OFX::DoubleParam*  m_SplitBalance;
    OFX::ChoiceParam*  m_ViewMode;
    OFX::BooleanParam* m_ScopeHD;
    OFX::BooleanParam* m_ScopeDensity;
};

////////////////////////////////////////////////////////////////////////////////
// Factory
////////////////////////////////////////////////////////////////////////////////

SpeakPluginFactory::SpeakPluginFactory()
    : OFX::PluginFactoryHelper<SpeakPluginFactory>(kSpeakIdentifier, kSpeakVersionMajor, kSpeakVersionMinor)
{
}

void SpeakPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    p_Desc.setLabels(kSpeakName, kSpeakName, kSpeakName);
    p_Desc.setPluginGrouping(kSpeakGrouping);
    p_Desc.setPluginDescription(kSpeakDescription);

    p_Desc.addSupportedContext(OFX::eContextFilter);
    p_Desc.addSupportedContext(OFX::eContextGeneral);
    p_Desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);   // single-frame effect
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
}

static OFX::DoubleParamDescriptor* sDefDouble(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* pg,
                                              const char* name, const char* label, const char* hint,
                                              double def, double mn, double mx, double inc,
                                              OFX::GroupParamDescriptor* parent)
{
    OFX::DoubleParamDescriptor* p = d.defineDoubleParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setDefault(def);
    p->setRange(mn, mx);
    p->setIncrement(inc);
    p->setDisplayRange(mn, mx);
    if (parent) p->setParent(*parent);
    if (pg) pg->addChild(*p);
    return p;
}

static OFX::BooleanParamDescriptor* sDefBool(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* pg,
                                             const char* name, const char* label, const char* hint,
                                             bool def, OFX::GroupParamDescriptor* parent)
{
    OFX::BooleanParamDescriptor* p = d.defineBooleanParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setDefault(def);
    if (parent) p->setParent(*parent);
    if (pg) pg->addChild(*p);
    return p;
}

void SpeakPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    OFX::ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    OFX::ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    dstClip->setSupportsTiles(kSupportsTiles);

    OFX::PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // ---------------------------------------------------------------- 1 · Color
    OFX::GroupParamDescriptor* grpColor = p_Desc.defineGroupParam("grpColor");
    grpColor->setLabels("1 \xC2\xB7 Color", "1 \xC2\xB7 Color", "1 \xC2\xB7 Color");
    grpColor->setOpen(true);
    grpColor->setHint("Tell Speak what space this node is in (OFX can't detect it) and how "
                      "to hand the image back.");
    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("inputColorSpace");
        c->setLabels("Input Color Space", "Input Color Space", "Input");
        c->setHint("The working space of this node. Default is DaVinci Wide Gamut / "
                   "Intermediate — the managed timeline space. Set it to match a manual-CST "
                   "or ACES node if that's where Speak sits.");
        c->appendOption("DaVinci Wide Gamut / Intermediate");
        c->appendOption("Rec.709 (Gamma 2.4)");
        c->appendOption("DaVinci Wide Gamut (Linear)");
        c->appendOption("ACEScct");
        c->appendOption("Linear");
        c->setDefault(SPEAK_CS_DWG_INTERMEDIATE);
        c->setParent(*grpColor);
        page->addChild(*c);
    }
    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("outputMode");
        c->setLabels("Output", "Output", "Output");
        c->setHint("Working space returns DWG/DI so Resolve Color Management delivers "
                   "Rec.709 / P3 / HDR from the same look (delivery-agnostic default).\n\n"
                   "Bake to Rec.709 makes Speak the literal last node for NON-managed "
                   "projects: it converts DaVinci Wide Gamut -> Rec.709 (Gamma 2.4) itself. "
                   "Use it only when Input is a DaVinci Wide Gamut space; for other inputs it "
                   "applies the Rec.709 transfer without a gamut change.");
        c->appendOption("Working space (let RCM deliver)");
        c->appendOption("Bake to Rec.709");
        c->setDefault(SPEAK_OUT_WORKING);
        c->setParent(*grpColor);
        page->addChild(*c);
    }

    // ------------------------------------------------------------ 2 · Film Tone
    OFX::GroupParamDescriptor* grpTone = p_Desc.defineGroupParam("grpTone");
    grpTone->setLabels("2 \xC2\xB7 Film Tone", "2 \xC2\xB7 Film Tone", "2 \xC2\xB7 Tone");
    grpTone->setOpen(true);
    grpTone->setHint("The density spine: a negative and print characteristic curve with "
                     "printer-light color timing. Tick Scope: H&D Curves to watch it.");

    sDefBool(p_Desc, page, "enableTone", "Enable Film Tone",
             "Toggle the tone scale to compare against the untouched image.", true, grpTone);
    sDefDouble(p_Desc, page, "strength", "Strength",
               "Blends the film tone scale in. 0 = identity (untouched); 1 = full film "
               "response. Raise from 0 to dial the look.", 0.0, 0.0, 1.0, 0.01, grpTone);
    sDefDouble(p_Desc, page, "contrast", "Contrast (Print Gamma)",
               "Scales the print's contrast index. 1 keeps the stock's native ~1.4 system "
               "gamma; higher is punchier, lower is softer.", 1.0, 0.5, 2.0, 0.01, grpTone);
    sDefDouble(p_Desc, page, "printShoulder", "Print Shoulder",
               "Highlight roll-off sharpness of the print curve. Lower = longer, gentler "
               "shoulder (more highlight compression).", 2.2, 0.5, 8.0, 0.05, grpTone);
    sDefDouble(p_Desc, page, "toe", "Print Toe",
               "Shadow foot sharpness of the print curve. Lower = longer, gentler toe "
               "(more shadow lift).", 3.5, 0.5, 8.0, 0.05, grpTone);
    sDefDouble(p_Desc, page, "printerR", "Printer Light R",
               "Red color timing in printer points (1 pt = 0.025 logE), injected between "
               "the negative and print. Neutral-preserving and curve-shaped.", 0.0, -12.0, 12.0, 0.1, grpTone);
    sDefDouble(p_Desc, page, "printerG", "Printer Light G",
               "Green color timing in printer points.", 0.0, -12.0, 12.0, 0.1, grpTone);
    sDefDouble(p_Desc, page, "printerB", "Printer Light B",
               "Blue color timing in printer points.", 0.0, -12.0, 12.0, 0.1, grpTone);
    sDefDouble(p_Desc, page, "printerMaster", "Printer Light Master",
               "Master printing exposure in points — an overall lighter/darker print.",
               0.0, -12.0, 12.0, 0.1, grpTone);

    // ------------------------------------------------- 3 · Subtractive Color
    OFX::GroupParamDescriptor* grpDye = p_Desc.defineGroupParam("grpDye");
    grpDye->setLabels("3 \xC2\xB7 Subtractive Color", "3 \xC2\xB7 Subtractive Color", "3 \xC2\xB7 Color");
    grpDye->setOpen(true);
    grpDye->setHint("Why film looks like film: saturation done in log-density like real dyes, "
                    "not in linear or HSL. Highlights roll toward base white instead of "
                    "blowing out, and hues bend toward the dye axes. Works on any grade — "
                    "you don't need the Film Tone stage on.");

    sDefBool(p_Desc, page, "enableDye", "Enable Subtractive Color",
             "Toggle the density-domain color stage to compare.", true, grpDye);
    sDefDouble(p_Desc, page, "subSat", "Subtractive Saturation",
               "Saturation the film way: each dye's density deviation from neutral is "
               "amplified, so highlight chroma self-compresses toward base white instead of "
               "clipping the way a normal saturation does. Neutral tones are untouched by "
               "construction. 0 = off.", 0.0, 0.0, 1.5, 0.01, grpDye);
    sDefDouble(p_Desc, page, "dyeCoupler", "Color Coupler (Punch)",
               "Inter-image coupling — the dyes' unwanted absorptions. Adds film's colour "
               "separation and bends hues toward the dye axes; zero on the neutral axis by "
               "construction. 0 = off.", 0.0, 0.0, 1.5, 0.01, grpDye);
    sDefDouble(p_Desc, page, "dyeKnee", "Dye Dmax Knee",
               "Soft ceiling on dye density, so deep colours self-limit like a real emulsion "
               "instead of running away. Lower = tighter. 0 = no ceiling.",
               2.2, 0.0, 4.0, 0.05, grpDye);

    // ----------------------------------------------------- 4 · Split Toning
    OFX::GroupParamDescriptor* grpSplit = p_Desc.defineGroupParam("grpSplit");
    grpSplit->setLabels("4 \xC2\xB7 Split Toning", "4 \xC2\xB7 Split Toning", "4 \xC2\xB7 Split");
    grpSplit->setOpen(false);
    grpSplit->setHint("The lift/gamma/gain replacement, done in density like a real emulsion. "
                      "Tints are per-channel DENSITY offsets weighted by tonal zones anchored to "
                      "the working H&D curve — so mid-gray cannot drift (its zone weight is zero) "
                      "and hues stay put where lift/gamma/gain swings them. Opposite shadow and "
                      "highlight tints give film's chromogenic crossover.");

    sDefBool(p_Desc, page, "enableSplit", "Enable Split Toning",
             "Toggle the density-domain split to compare.", true, grpSplit);
    sDefDouble(p_Desc, page, "splitShadowR", "Shadow Tint R (density)",
               "Red density added in the shadows. POSITIVE removes red (cooler); negative adds "
               "red (warmer). Density offsets are multiplicative in light, which is why hue holds.",
               0.0, -0.3, 0.3, 0.005, grpSplit);
    sDefDouble(p_Desc, page, "splitShadowG", "Shadow Tint G (density)",
               "Green density added in the shadows.", 0.0, -0.3, 0.3, 0.005, grpSplit);
    sDefDouble(p_Desc, page, "splitShadowB", "Shadow Tint B (density)",
               "Blue density added in the shadows.", 0.0, -0.3, 0.3, 0.005, grpSplit);
    sDefDouble(p_Desc, page, "splitHighR", "Highlight Tint R (density)",
               "Red density added in the highlights.", 0.0, -0.3, 0.3, 0.005, grpSplit);
    sDefDouble(p_Desc, page, "splitHighG", "Highlight Tint G (density)",
               "Green density added in the highlights.", 0.0, -0.3, 0.3, 0.005, grpSplit);
    sDefDouble(p_Desc, page, "splitHighB", "Highlight Tint B (density)",
               "Blue density added in the highlights.", 0.0, -0.3, 0.3, 0.005, grpSplit);
    sDefDouble(p_Desc, page, "splitPivot", "Split Pivot (stops)",
               "Moves the untouched mid zone up or down the tone scale, in stops from 18% gray. "
               "Whatever sits at the pivot is never tinted.", 0.0, -3.0, 3.0, 0.05, grpSplit);
    sDefDouble(p_Desc, page, "splitBalance", "Zone Width",
               "How wide the untouched mid zone is. Narrow = shadows and highlights tint close to "
               "mid-gray; wide = only the extremes tint.", 0.5, 0.0, 1.0, 0.01, grpSplit);

    // -------------------------------------------------------------- 5 · Inspect
    OFX::GroupParamDescriptor* grpInspect = p_Desc.defineGroupParam("grpInspect");
    grpInspect->setLabels("5 \xC2\xB7 Inspect", "5 \xC2\xB7 Inspect", "5 \xC2\xB7 Inspect");
    grpInspect->setOpen(true);
    grpInspect->setHint("Views and the read-only scopes. Turn scopes off before rendering.");
    {
        OFX::ChoiceParamDescriptor* c = p_Desc.defineChoiceParam("viewMode");
        c->setLabels("View", "View", "View");
        c->setHint("Result / Split (input | result) / Input for comparing.");
        c->appendOption("Result");
        c->appendOption("Split (Input | Result)");
        c->appendOption("Input (Original)");
        c->setDefault(SPEAK_VIEW_RESULT);
        c->setParent(*grpInspect);
        page->addChild(*c);
    }
    sDefBool(p_Desc, page, "scopeHD", "Scope: H&D Curves",
             "Draws the applied per-channel characteristic curves in the viewer (top-left) with "
             "this frame's exposure histogram on the logE axis — the exact curve the pixels use, "
             "sampled from the render kernel. Turn off before rendering.",
             false, grpInspect);
    sDefBool(p_Desc, page, "scopeDensity", "Scope: Density (Status-M)",
             "Draws an RGB parade of the RESULT's film density in the viewer (top-right), with "
             "markers at paper white and 18% gray. Measured from the frame by the same look the "
             "pixels get. Turn off before rendering.",
             false, grpInspect);
}

OFX::ImageEffect* SpeakPluginFactory::createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum /*p_Context*/)
{
    return new SpeakPlugin(p_Handle);
}

namespace speakofx {
void registerSpeak(OFX::PluginFactoryArray& p_FactoryArray)
{
    static SpeakPluginFactory speakFactory;
    p_FactoryArray.push_back(&speakFactory);
}
}
