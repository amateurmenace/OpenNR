#!/usr/bin/env python3
# Builds site/squarespace-embed.html — a self-scoped fragment of the Hush site
# for pasting into a Squarespace Code Block (or any CMS embed).
#
# Differences from the standalone page:
#   - no <html>/<head>/<body> wrapper (CMS code blocks want fragments)
#   - all CSS scoped under #hush-embed (no bleed in either direction)
#   - all JS-referenced element ids prefixed hush- (no id collisions)
#   - images load from GitHub Pages instead of base64 (small paste size)
import os, re

here = os.path.dirname(os.path.abspath(__file__))
PAGES = "https://amateurmenace.github.io/Hush/assets"

t = open(os.path.join(here, "index.template.html")).read()

# --- images -> hosted URLs
for tok, name in [("B64_AFTER","after.jpg"), ("B64_BEFORE","before.jpg"),
                  ("B64_SCM","scope-measure.jpg"), ("B64_SCMO","scope-motion.jpg"),
                  ("B64_SCEQ","scope-eq.jpg"), ("B64_SCSNR","scope-snr.jpg"),
                  ("B64_WM","wm-logo.png")]:
    t = t.replace(f"%%{tok}%%", f"{PAGES}/{name}")

# --- pull out fonts, style, body
fonts = "\n".join(re.findall(r'<link[^>]+fonts[^>]+>', t))
style = re.search(r'<style>(.*?)</style>', t, re.S).group(1)
body  = re.search(r'<body>(.*?)</body>', t, re.S).group(1)

# --- scope the CSS under #hush-embed
style = style.replace(":root", "#hush-embed")
style = re.sub(r'(?m)^\s*html\s*\{[^}]*\}', '', style)          # drop html{} rule
style = style.replace("body {", "#hush-embed {")                 # body rule -> wrapper

def prefix_selectors(css):
    out, i, n = [], 0, len(css)
    while i < n:
        m = re.match(r'\s*@media[^{]*\{', css[i:])
        if m:
            # find matching closing brace of the media block
            start = i + m.end()
            depth, j = 1, start
            while j < n and depth:
                if css[j] == '{': depth += 1
                elif css[j] == '}': depth -= 1
                j += 1
            out.append(css[i:start] + prefix_selectors(css[start:j-1]) + '}')
            i = j
            continue
        m = re.match(r'([^{}@]+)\{([^{}]*)\}', css[i:], re.S)
        if m:
            sels = []
            for s in m.group(1).split(','):
                s = s.strip()
                if not s: continue
                sels.append(s if s.startswith('#hush-embed') else '#hush-embed ' + s)
            out.append(', '.join(sels) + ' {' + m.group(2) + '}\n')
            i += m.end()
            continue
        out.append(css[i]); i += 1
    return ''.join(out)

style = prefix_selectors(style)

# --- prefix ids referenced from JS (html + scripts together)
ids = ["testcard","noise","hush","true","meas","verdict",
       "wipe","wipeTop","wipeHandle","pipe","pipeText","pipeHint","chart",
       "scopeTabs","scopeImg","scopeCap","scopeChk","scopeStep","scopeHint"]
for x in ids:
    body = body.replace(f'id="{x}"', f'id="hush-{x}"')
    body = body.replace(f"getElementById('{x}')", f"getElementById('hush-{x}')")
    body = body.replace(f"querySelectorAll('#{x} ", f"querySelectorAll('#hush-{x} ")

out = f"""<!-- ============================================================
  HUSH OPEN NR — Squarespace embed fragment (generated; do not hand-edit)
  Regenerate: python3 site/build_embed.py

  HOW TO USE
  1. Squarespace editor -> add a  Code  block where you want the page.
  2. Paste this entire file. Set the block to HTML (default).
  3. JavaScript in Code Blocks requires a Business plan or higher —
     on Personal plans the live demo/interactions are stripped, so use
     the iframe embed instead:
       <iframe src="https://amateurmenace.github.io/Hush/"
               style="width:100%;height:5200px;border:0"
               loading="lazy" title="Hush Open NR"></iframe>
  Images load from GitHub Pages; nothing to upload to Squarespace.
============================================================= -->
{fonts}
<style>
{style}
</style>
<div id="hush-embed">
{body}
</div>
"""
open(os.path.join(here, "squarespace-embed.html"), "w").write(out)
print("squarespace-embed.html:", len(out)//1024, "KB")
