#!/usr/bin/env python3
# Rebuilds site/index.html from index.template.html + fresh screenshots.
# Screenshots come from the test renders, flipped vertically to display
# orientation (the HUD anchors in display space; PPMs are buffer order):
#   cd test && ./test_denoise
#   sips -s format png out_view5.ppm --out v5.png && sips --flip vertical v5.png ...
import base64, os, sys
here = os.path.dirname(os.path.abspath(__file__))
imgs = {"B64_AFTER": "after.jpg", "B64_BEFORE": "before.jpg",
        "B64_HUD": "hud.jpg", "B64_SNR": "snr.jpg"}
html = open(os.path.join(here, "index.template.html")).read()
for tok, name in imgs.items():
    p = os.path.join(here, "assets", name)
    if not os.path.exists(p):
        sys.exit(f"missing {p} — see comments for how to regenerate")
    html = html.replace(f"%%{tok}%%", "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode())
open(os.path.join(here, "index.html"), "w").write(html)
docs = os.path.join(here, "..", "docs")
os.makedirs(os.path.join(docs, "assets"), exist_ok=True)
open(os.path.join(docs, "index.html"), "w").write(html)
import shutil
for f in os.listdir(os.path.join(here, "assets")):
    shutil.copy(os.path.join(here, "assets", f), os.path.join(docs, "assets", f))
print("site/index.html + docs/ (GitHub Pages) rebuilt — also run build_embed.py")
