// OpenNR — free spatio-temporal noise reduction for DaVinci Resolve (OpenFX).
// MIT License.

#ifndef OPENNR_PLUGIN_H
#define OPENNR_PLUGIN_H

#include "ofxsImageEffect.h"

class OpenNRPluginFactory : public OFX::PluginFactoryHelper<OpenNRPluginFactory>
{
public:
    OpenNRPluginFactory();
    virtual void load() {}
    virtual void unload() {}
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc);
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context);
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum p_Context);
};

#endif
