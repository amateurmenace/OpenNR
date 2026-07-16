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
# GitHub Pages custom domain: keep the CNAME in docs/ so a rebuild never drops it
# (the apex control-z.org points at Pages' four A records; www is a CNAME to
# amateurmenace.github.io). Change here if the domain ever changes.
open(os.path.join(docs, "CNAME"), "w").write("control-z.org\n")
print("site/index.html + docs/ (GitHub Pages, incl. whitepaper.html + CNAME) rebuilt — also run build_embed.py")
