#!/usr/bin/env python3
# Rebuilds site/index.html from index.template.html + fresh screenshots.
# Screenshots come from the test renders, flipped vertically to display
# orientation (the HUD anchors in display space; PPMs are buffer order):
#   cd test && ./test_denoise
#   sips -s format png out_view5.ppm --out v5.png && sips --flip vertical v5.png ...
import base64, os, sys
here = os.path.dirname(os.path.abspath(__file__))
imgs = {"B64_AFTER": "after.jpg", "B64_BEFORE": "before.jpg",
        "B64_SCM": "scope-measure.jpg", "B64_SCMO": "scope-motion.jpg",
        "B64_SCEQ": "scope-eq.jpg", "B64_SCSNR": "scope-snr.jpg",
        "B64_WM": "wm-logo.png"}
html = open(os.path.join(here, "index.template.html")).read()
for tok, name in imgs.items():
    p = os.path.join(here, "assets", name)
    if not os.path.exists(p):
        sys.exit(f"missing {p} — see comments for how to regenerate")
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    html = html.replace(f"%%{tok}%%", f"data:{mime};base64," + base64.b64encode(open(p, "rb").read()).decode())
open(os.path.join(here, "index.html"), "w").write(html)
docs = os.path.join(here, "..", "docs")
os.makedirs(os.path.join(docs, "assets"), exist_ok=True)
open(os.path.join(docs, "index.html"), "w").write(html)
import shutil
for f in os.listdir(os.path.join(here, "assets")):
    shutil.copy(os.path.join(here, "assets", f), os.path.join(docs, "assets", f))
# whitepaper.html is a self-contained static page (no image tokens) — it is
# its own source AND output; sync it to docs/ so GitHub Pages stays in step.
shutil.copy(os.path.join(here, "whitepaper.html"), os.path.join(docs, "whitepaper.html"))
# NO CNAME HERE. control-z.org moved to the suite repo (amateurmenace/control-z,
# gh-pages branch) on 2026-07-16 — it writes its own CNAME at bake time. Two repos
# claiming one custom domain fight, and this one would win by accident and take
# control-z.org down. This repo now publishes at amateurmenace.github.io/Hush-OpenNR/.
# If a stale docs/CNAME is lying around from before the move, clear it.
_stale_cname = os.path.join(docs, "CNAME")
if os.path.exists(_stale_cname):
    os.remove(_stale_cname)
    print("removed stale docs/CNAME (control-z.org lives in the control-z repo now)")
print("site/index.html + docs/ (GitHub Pages, incl. whitepaper.html) rebuilt — also run build_embed.py")
