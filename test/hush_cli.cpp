// hush_cli — real-clip A/B harness (roadmap idea #22).
//
// Runs the EXACT CPU reference pipeline (nr_core.h — the thing the GPU
// kernels are parity-tested against) on a PPM image sequence, so every
// algorithm change can be scored on real footage BEFORE a release instead
// of after. Dev tool only: never ships in the bundle, has no parity burden,
// and is deterministic end to end.
//
// Any footage becomes a PPM sequence with one ffmpeg line, and back:
//
//   ffmpeg -i clip.mov -pix_fmt rgb48be seq/f_%04d.ppm
//   ./hush_cli -i seq/f_%04d.ppm -f 1 -l 100 -o out/f_%04d.ppm --auto --boost
//   ffmpeg -framerate 24 -i out/f_%04d.ppm -c:v prores_ks -profile:v 3 out.mov
//
// The pixels are processed in whatever space ffmpeg hands over — exactly the
// plugin's color-space-agnostic contract with its host.
//
// Build: c++ -O2 -std=c++14 -I../plugin hush_cli.cpp -o hush_cli

#include "nr_core.h"
#include "nr_analyze.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// PPM I/O — P6, maxval 255 (8-bit) or 65535 (16-bit big-endian, ffmpeg's
// rgb48be). Values normalize to [0,1] float RGBA in, 16-bit out.
// ---------------------------------------------------------------------------
static bool readPPM(const std::string& path, int& W, int& H, std::vector<float>& out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return false;
    char magic[3] = { 0 };
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fclose(f);
        return false;
    }
    // header ints with comment support
    int vals[3], got = 0;
    while (got < 3) {
        int c = fgetc(f);
        if (c == EOF) { fclose(f); return false; }
        if (isspace(c))
            continue;
        if (c == '#') {
            while (c != '\n' && c != EOF) c = fgetc(f);
            continue;
        }
        ungetc(c, f);
        if (fscanf(f, "%d", &vals[got]) != 1) { fclose(f); return false; }
        ++got;
    }
    fgetc(f);   // the single whitespace after maxval
    W = vals[0];
    H = vals[1];
    const int maxv = vals[2];
    if (W <= 0 || H <= 0 || (maxv != 255 && maxv != 65535)) {
        fclose(f);
        return false;
    }
    const size_t n = static_cast<size_t>(W) * H;
    const size_t bytes = n * 3 * (maxv == 255 ? 1 : 2);
    std::vector<unsigned char> raw(bytes);
    const bool ok = fread(raw.data(), 1, bytes, f) == bytes;
    fclose(f);
    if (!ok)
        return false;
    out.resize(n * 4);
    if (maxv == 255) {
        const float s = 1.0f / 255.0f;
        for (size_t i = 0; i < n; ++i) {
            out[i * 4 + 0] = raw[i * 3 + 0] * s;
            out[i * 4 + 1] = raw[i * 3 + 1] * s;
            out[i * 4 + 2] = raw[i * 3 + 2] * s;
            out[i * 4 + 3] = 1.0f;
        }
    } else {
        const float s = 1.0f / 65535.0f;
        for (size_t i = 0; i < n; ++i) {
            for (int c = 0; c < 3; ++c) {
                const unsigned hi = raw[(i * 3 + c) * 2];
                const unsigned lo = raw[(i * 3 + c) * 2 + 1];
                out[i * 4 + c] = static_cast<float>((hi << 8) | lo) * s;
            }
            out[i * 4 + 3] = 1.0f;
        }
    }
    return true;
}

static bool writePPM16(const std::string& path, int W, int H, const float* rgba)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return false;
    fprintf(f, "P6\n%d %d\n65535\n", W, H);
    const size_t n = static_cast<size_t>(W) * H;
    std::vector<unsigned char> raw(n * 6);
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = rgba[i * 4 + c];
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            const unsigned u = static_cast<unsigned>(v * 65535.0f + 0.5f);
            raw[(i * 3 + c) * 2] = static_cast<unsigned char>(u >> 8);
            raw[(i * 3 + c) * 2 + 1] = static_cast<unsigned char>(u & 0xFF);
        }
    const bool ok = fwrite(raw.data(), 1, raw.size(), f) == raw.size();
    fclose(f);
    return ok;
}

static std::string framePath(const std::string& pattern, int idx)
{
    char buf[1024];
    snprintf(buf, sizeof(buf), pattern.c_str(), idx);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Shared frame cache — each output frame reads a 7-frame stack, so plain
// per-frame loading would read every file up to 7 times. Frames are pruned
// once every in-flight worker is past them.
// ---------------------------------------------------------------------------
struct FrameCache
{
    std::string pattern;
    int first = 0, last = 0;
    int W = 0, H = 0;
    std::mutex mu;
    std::map<int, std::shared_ptr<std::vector<float>>> frames;

    std::shared_ptr<std::vector<float>> get(int idx)
    {
        idx = idx < first ? first : (idx > last ? last : idx);
        std::lock_guard<std::mutex> lk(mu);
        auto it = frames.find(idx);
        if (it != frames.end())
            return it->second;
        auto buf = std::make_shared<std::vector<float>>();
        int w = 0, h = 0;
        if (!readPPM(framePath(pattern, idx), w, h, *buf)) {
            fprintf(stderr, "hush_cli: cannot read %s\n", framePath(pattern, idx).c_str());
            exit(1);
        }
        if (W == 0) { W = w; H = h; }
        else if (w != W || h != H) {
            fprintf(stderr, "hush_cli: frame %d is %dx%d, expected %dx%d\n", idx, w, h, W, H);
            exit(1);
        }
        frames[idx] = buf;
        return buf;
    }

    void pruneBelow(int idx)
    {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = frames.begin(); it != frames.end();)
            it = (it->first < idx) ? frames.erase(it) : ++it;
    }
};

// ---------------------------------------------------------------------------
// Metrics / views
// ---------------------------------------------------------------------------
static double psnrRGB(const float* a, const float* b, int W, int H, int border = 8)
{
    double mse = 0.0;
    size_t n = 0;
    for (int y = border; y < H - border; ++y)
        for (int x = border; x < W - border; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            for (int c = 0; c < 3; ++c) {
                const double d = static_cast<double>(a[i + c]) - b[i + c];
                mse += d * d;
                ++n;
            }
        }
    mse /= static_cast<double>(n ? n : 1);
    return 10.0 * std::log10(1.0 / std::max(mse, 1e-12));
}

static void removedView(const float* in, const float* out, int W, int H,
                        float gain, std::vector<float>& view)
{
    const size_t n = static_cast<size_t>(W) * H;
    view.resize(n * 4);
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c)
            view[i * 4 + c] = 0.5f + gain * (in[i * 4 + c] - out[i * 4 + c]);
        view[i * 4 + 3] = 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Settings — plugin slider units in, gatherParams' mapping to nrcore::Params
// (so numbers here are the numbers in the Inspector).
// ---------------------------------------------------------------------------
struct Cli
{
    std::string in, out, clean, residual, triptych;
    int first = -1, last = -1;
    bool autoSetup = false;
    bool boost = false;
    int jobs = 1;
    float rgain = 4.0f;
    // overrides (slider units; NaN/-1 = untouched)
    float master = -1, tl = -1, tc = -1, mt = -1, sl = -1, sc = -1;
    float preserve = -1, rescue = -1, blotch = -1, eqf = -1, eqm = -1, eqc = -1;
    int frames = -1, mode = -1, radius = -1;
    int tracking = -1, guard = -1, firefly = -1, deep = -1;
};

static void usage()
{
    printf(
        "hush_cli — run the exact Hush CPU pipeline on a PPM sequence (dev harness)\n\n"
        "  ffmpeg -i clip.mov -pix_fmt rgb48be seq/f_%%04d.ppm\n"
        "  ./hush_cli -i seq/f_%%04d.ppm -f 1 -l 100 -o out/f_%%04d.ppm --auto --boost\n"
        "  ffmpeg -framerate 24 -i out/f_%%04d.ppm -c:v prores_ks -profile:v 3 out.mov\n\n"
        "io:\n"
        "  -i PAT -f N -l N     input printf pattern + first/last frame number\n"
        "  -o PAT               output pattern (16-bit P6)\n"
        "  --clean PAT          clean reference -> per-frame PSNR (synthetic tests)\n"
        "  --residual PAT       write amplified removed-noise view\n"
        "  --triptych PAT       write before|after|removed strip\n"
        "  --rgain X            removed-view gain (default 4)\n"
        "setup:\n"
        "  --auto               analyze like Auto Setup v3.5: spread frames -> class\n"
        "                       table -> MC-SURE tune on a centre crop (explicit\n"
        "                       flags below still override the tuned values)\n"
        "  --boost              Render Boost sequential chain (forces --jobs 1)\n"
        "  --jobs N             parallel frames when boost is off (default 1)\n"
        "settings (plugin slider units; defaults = plugin defaults):\n"
        "  --master X --frames 3|5|7 --temporal-luma X --temporal-chroma X\n"
        "  --motion-thresh X --no-tracking --no-ghost-guard --no-firefly\n"
        "  --spatial-mode 0|1 --radius N --spatial-luma X --spatial-chroma X\n"
        "  --preserve X --rescue X --deep-clean --blotch X\n"
        "  --eq-fine X --eq-med X --eq-coarse X\n");
}

static void applyOverrides(const Cli& a, nrcore::Params& p)
{
    if (a.master >= 0)  p.master = a.master;
    if (a.frames > 0)   p.temporalFrames = a.frames;
    if (a.tl >= 0)      p.temporalLuma = a.tl / 100.0f;
    if (a.tc >= 0)      p.temporalChroma = a.tc / 100.0f;
    if (a.mt >= 0)      p.motionThresh = a.mt / 100.0f;
    if (a.sl >= 0)      p.spatialLuma = a.sl / 100.0f;
    if (a.sc >= 0)      p.spatialChroma = a.sc / 100.0f;
    if (a.preserve >= 0) p.preserveDetail = a.preserve / 100.0f;
    if (a.rescue >= 0)  p.detailRescue = a.rescue / 100.0f;
    if (a.blotch >= 0)  p.chromaBlotch = a.blotch / 100.0f;
    if (a.eqf >= 0)     p.eqFine = a.eqf / 100.0f;
    if (a.eqm >= 0)     p.eqMedium = a.eqm / 100.0f;
    if (a.eqc >= 0)     p.eqCoarse = a.eqc / 100.0f;
    if (a.mode >= 0)    p.spatialMode = a.mode;
    if (a.radius > 0)   p.spatialRadius = a.radius;
    if (a.tracking >= 0) p.motionTracking = a.tracking;
    if (a.guard >= 0)   p.ghostGuard = a.guard;
    if (a.firefly >= 0) p.fireflyRemoval = a.firefly;
    if (a.deep >= 0)    p.deepClean = a.deep;
}

// mirror of the plugin's whole-frame Auto Setup at v3.5: playhead (= middle
// frame) + up to 4 spread frames -> aggregate -> table -> SURE tune on a
// centre crop (<=512x288) of the playhead's 7-frame stack
static nrcore::Params runAuto(FrameCache& fc)
{
    const int mid = fc.first + (fc.last - fc.first) / 2;
    std::vector<double> times;
    times.push_back(mid);
    const double len = fc.last - fc.first;
    for (int k = 0; k < 4 && len >= 1.0; ++k) {
        double t = std::floor(fc.first + (0.5 + k) * len / 4.0 + 0.5);
        bool dup = false;
        for (size_t i = 0; i < times.size(); ++i)
            if (std::fabs(times[i] - t) < 1.0) { dup = true; break; }
        if (!dup)
            times.push_back(t);
    }

    std::vector<nrcore::Stats> per;
    for (size_t i = 0; i < times.size(); ++i) {
        const int t = static_cast<int>(times[i]);
        const int tp = (t + 1 <= fc.last) ? t + 1 : t - 1;
        auto cur = fc.get(t);
        std::shared_ptr<std::vector<float>> par;
        if (tp >= fc.first && tp != t)
            par = fc.get(tp);
        nrcore::Params ap;
        nrcore::Stats st;
        nrcore::estimateInput(cur->data(), par ? par->data() : nullptr,
                              fc.W, fc.H, ap, st);
        per.push_back(st);
    }
    const nranalyze::ClipAggregate agg = nranalyze::aggregateClipStats(per);
    const nranalyze::AutoSettings as = nranalyze::mapAnalysisToSettings(agg);
    printf("auto: %s\n", nranalyze::formatAutoReport(agg, as).c_str());

    nrcore::Params base = nranalyze::paramsFromAutoSettings(as);

    // SURE tune on a centre crop of the playhead stack, aliased at edges
    // exactly like a render (tuneAutoSetupAt's cropping, CLI-side)
    std::shared_ptr<std::vector<float>> held[7];
    const float* fp[7];
    for (int k = 0; k < 7; ++k) {
        held[k] = fc.get(mid + k - 3);
        fp[k] = held[k]->data();
    }
    const int tw = std::min(fc.W, 512) & ~1;
    const int th = std::min(fc.H, 288) & ~1;
    if (tw >= 64 && th >= 64) {
        const int x0 = (fc.W - tw) / 2, y0 = (fc.H - th) / 2;
        std::vector<float> crops[7];
        const float* cp[7];
        for (int k = 0; k < 7; ++k) {
            int prior = -1;
            for (int j = 0; j < k; ++j)
                if (fp[j] == fp[k]) { prior = j; break; }
            if (prior >= 0) {
                cp[k] = cp[prior];
                continue;
            }
            crops[k].resize(static_cast<size_t>(tw) * th * 4);
            for (int y = 0; y < th; ++y)
                std::memcpy(&crops[k][static_cast<size_t>(y) * tw * 4],
                            fp[k] + (static_cast<size_t>(y0 + y) * fc.W + x0) * 4,
                            static_cast<size_t>(tw) * 4 * sizeof(float));
            cp[k] = crops[k].data();
        }
        const nranalyze::SureTune tn = nranalyze::sureTuneGrid(cp, tw, th, base);
        if (tn.ran) {
            printf("auto: SURE tuned  TL %.0f -> %.0f   SL %.0f -> %.0f  (sigma %.4f)\n",
                   base.temporalLuma * 100.0f, tn.temporalLuma * 100.0f,
                   base.spatialLuma * 100.0f, tn.spatialLuma * 100.0f, tn.sigma);
            base.temporalLuma = tn.temporalLuma;
            base.spatialLuma = tn.spatialLuma;
        }
    }
    return base;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    Cli a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "hush_cli: %s needs a value\n", flag);
                exit(1);
            }
            return argv[++i];
        };
        if (s == "-i") a.in = next("-i");
        else if (s == "-o") a.out = next("-o");
        else if (s == "-f") a.first = atoi(next("-f"));
        else if (s == "-l") a.last = atoi(next("-l"));
        else if (s == "--clean") a.clean = next("--clean");
        else if (s == "--residual") a.residual = next("--residual");
        else if (s == "--triptych") a.triptych = next("--triptych");
        else if (s == "--rgain") a.rgain = static_cast<float>(atof(next("--rgain")));
        else if (s == "--auto") a.autoSetup = true;
        else if (s == "--boost") a.boost = true;
        else if (s == "--jobs") a.jobs = atoi(next("--jobs"));
        else if (s == "--master") a.master = static_cast<float>(atof(next("--master")));
        else if (s == "--frames") a.frames = atoi(next("--frames"));
        else if (s == "--temporal-luma") a.tl = static_cast<float>(atof(next("--temporal-luma")));
        else if (s == "--temporal-chroma") a.tc = static_cast<float>(atof(next("--temporal-chroma")));
        else if (s == "--motion-thresh") a.mt = static_cast<float>(atof(next("--motion-thresh")));
        else if (s == "--spatial-luma") a.sl = static_cast<float>(atof(next("--spatial-luma")));
        else if (s == "--spatial-chroma") a.sc = static_cast<float>(atof(next("--spatial-chroma")));
        else if (s == "--preserve") a.preserve = static_cast<float>(atof(next("--preserve")));
        else if (s == "--rescue") a.rescue = static_cast<float>(atof(next("--rescue")));
        else if (s == "--blotch") a.blotch = static_cast<float>(atof(next("--blotch")));
        else if (s == "--eq-fine") a.eqf = static_cast<float>(atof(next("--eq-fine")));
        else if (s == "--eq-med") a.eqm = static_cast<float>(atof(next("--eq-med")));
        else if (s == "--eq-coarse") a.eqc = static_cast<float>(atof(next("--eq-coarse")));
        else if (s == "--spatial-mode") a.mode = atoi(next("--spatial-mode"));
        else if (s == "--radius") a.radius = atoi(next("--radius"));
        else if (s == "--no-tracking") a.tracking = 0;
        else if (s == "--no-ghost-guard") a.guard = 0;
        else if (s == "--no-firefly") a.firefly = 0;
        else if (s == "--deep-clean") a.deep = 1;
        else if (s == "--help" || s == "-h") { usage(); return 0; }
        else {
            fprintf(stderr, "hush_cli: unknown flag %s (see --help)\n", s.c_str());
            return 1;
        }
    }
    if (a.in.empty() || a.out.empty() || a.first < 0 || a.last < a.first) {
        usage();
        return 1;
    }
    const int count = a.last - a.first + 1;
    if (count > 1 && a.in.find('%') == std::string::npos) {
        fprintf(stderr, "hush_cli: -i needs a %%d pattern for multi-frame runs\n");
        return 1;
    }
    if (count > 1 && a.out.find('%') == std::string::npos) {
        fprintf(stderr, "hush_cli: -o needs a %%d pattern for multi-frame runs\n");
        return 1;
    }
    if (a.boost && a.jobs != 1) {
        printf("note: --boost is a sequential chain; forcing --jobs 1\n");
        a.jobs = 1;
    }
    if (a.jobs < 1)
        a.jobs = 1;

    FrameCache fc;
    fc.pattern = a.in;
    fc.first = a.first;
    fc.last = a.last;
    fc.get(a.first);   // sizes the cache, validates the first frame

    nrcore::Params base;   // plugin defaults
    if (a.autoSetup)
        base = runAuto(fc);
    applyOverrides(a, base);
    base.renderBoost = a.boost ? 1 : 0;

    printf("run: %dx%d, frames %d..%d, %s, %df stack, TL %.0f TC %.0f mt %.0f, "
           "%s R%d SL %.0f SC %.0f%s%s, jobs %d\n",
           fc.W, fc.H, a.first, a.last,
           a.boost ? "BOOST chain" : "independent frames",
           base.temporalFrames, base.temporalLuma * 100.0f,
           base.temporalChroma * 100.0f, base.motionThresh * 100.0f,
           base.spatialMode ? "NLM" : "bilateral", base.spatialRadius,
           base.spatialLuma * 100.0f, base.spatialChroma * 100.0f,
           base.deepClean ? ", deep clean" : "",
           base.motionTracking ? "" : ", tracking off", a.jobs);

    const size_t plane = static_cast<size_t>(fc.W) * fc.H * 4;
    std::vector<std::string> report(count);
    double sumPsnr = 0.0, sumMs = 0.0, sumRy = 0.0;
    std::mutex sumMu;

    // the per-frame body; hist/histFrame only advance in sequential mode
    std::vector<float> hist;
    int histFrame = INT_MIN;

    auto doFrame = [&](int t, std::vector<float>& out, std::vector<float>& scratch) {
        std::shared_ptr<std::vector<float>> held[7];
        const float* fp[7];
        for (int k = 0; k < 7; ++k) {
            held[k] = fc.get(t + k - 3);
            fp[k] = held[k]->data();
        }
        nrcore::Params p = base;
        p.frameIndex = t;
        p.histValid = (a.boost && histFrame == t - 1 &&
                       hist.size() == plane) ? 1 : 0;
        const auto t0 = std::chrono::steady_clock::now();
        nrcore::Stats st = nrcore::denoiseFrame(fp, fc.W, fc.H, p, out.data(), scratch,
                                                p.histValid ? hist.data() : 0);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (a.boost && scratch.size() >= plane) {
            hist.assign(scratch.begin(), scratch.begin() + plane);
            histFrame = t;
        }

        if (!writePPM16(framePath(a.out, t), fc.W, fc.H, out.data())) {
            fprintf(stderr, "hush_cli: cannot write %s\n", framePath(a.out, t).c_str());
            exit(1);
        }
        std::vector<float> view;
        if (!a.residual.empty()) {
            removedView(fp[3], out.data(), fc.W, fc.H, a.rgain, view);
            writePPM16(framePath(a.residual, t), fc.W, fc.H, view.data());
        }
        if (!a.triptych.empty()) {
            removedView(fp[3], out.data(), fc.W, fc.H, a.rgain, view);
            std::vector<float> strip(static_cast<size_t>(fc.W) * 3 * fc.H * 4);
            for (int y = 0; y < fc.H; ++y) {
                float* row = &strip[static_cast<size_t>(y) * fc.W * 3 * 4];
                std::memcpy(row, fp[3] + static_cast<size_t>(y) * fc.W * 4,
                            static_cast<size_t>(fc.W) * 4 * sizeof(float));
                std::memcpy(row + static_cast<size_t>(fc.W) * 4,
                            out.data() + static_cast<size_t>(y) * fc.W * 4,
                            static_cast<size_t>(fc.W) * 4 * sizeof(float));
                std::memcpy(row + static_cast<size_t>(fc.W) * 8,
                            view.data() + static_cast<size_t>(y) * fc.W * 4,
                            static_cast<size_t>(fc.W) * 4 * sizeof(float));
            }
            writePPM16(framePath(a.triptych, t), fc.W * 3, fc.H, strip.data());
        }

        char line[256];
        if (!a.clean.empty()) {
            int cw = 0, ch = 0;
            std::vector<float> cln;
            if (!readPPM(framePath(a.clean, t), cw, ch, cln) || cw != fc.W || ch != fc.H) {
                fprintf(stderr, "hush_cli: bad clean frame %d\n", t);
                exit(1);
            }
            const double before = psnrRGB(fp[3], cln.data(), fc.W, fc.H);
            const double after = psnrRGB(out.data(), cln.data(), fc.W, fc.H);
            snprintf(line, sizeof(line),
                     "frame %d: %6.2f -> %6.2f dB   sigT %.4f  res %.4f  effN %.1f  %7.1f ms",
                     t, before, after, st.ty, st.ry, st.effNMed, ms);
            std::lock_guard<std::mutex> lk(sumMu);
            sumPsnr += after;
            sumMs += ms;
            sumRy += st.ry;
        } else {
            snprintf(line, sizeof(line),
                     "frame %d: sigS %.4f  sigT %.4f  res %.4f  effN %.1f  %7.1f ms",
                     t, st.sy, st.ty, st.ry, st.effNMed, ms);
            std::lock_guard<std::mutex> lk(sumMu);
            sumMs += ms;
            sumRy += st.ry;
        }
        report[t - a.first] = line;
    };

    if (a.jobs == 1) {
        std::vector<float> out(plane), scratch;
        for (int t = a.first; t <= a.last; ++t) {
            doFrame(t, out, scratch);
            printf("%s\n", report[t - a.first].c_str());
            fc.pruneBelow(t - 3);
        }
    } else {
        std::atomic<int> nextT(a.first);
        std::vector<std::thread> pool;
        for (int j = 0; j < a.jobs; ++j)
            pool.emplace_back([&] {
                std::vector<float> out(plane), scratch;
                for (;;) {
                    const int t = nextT.fetch_add(1);
                    if (t > a.last)
                        return;
                    doFrame(t, out, scratch);
                    // prune conservatively: everything the slowest possible
                    // in-flight frame could still need is kept
                    fc.pruneBelow(t - 3 - a.jobs);
                }
            });
        for (auto& th : pool)
            th.join();
        for (int t = a.first; t <= a.last; ++t)
            printf("%s\n", report[t - a.first].c_str());
    }

    if (!a.clean.empty())
        printf("mean: %6.2f dB   residual %.4f   %7.1f ms/frame\n",
               sumPsnr / count, sumRy / count, sumMs / count);
    else
        printf("mean: residual %.4f   %7.1f ms/frame\n", sumRy / count, sumMs / count);
    return 0;
}
