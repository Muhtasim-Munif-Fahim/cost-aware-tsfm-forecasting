r"""Convert paper/sections/*.md to LaTeX fragments via pandoc.

- Strips the top-level '# <Section>' heading (main.tex supplies \section commands).
- Pre-converts unicode math pandoc/inputenc cannot handle (superscript exponents).
- Drops the ledger HTML comments from the .tex output (the canonical, audited source
  stays in markdown; run analysis/number_audit.py against paper/sections/).
- Also emits generated/abstract.tex from the '# Abstract' block of
  00_title_abstract.md, so main.tex never carries a hand-copied abstract
  (a stale inline copy is how the Stage-E abstract revision was silently lost).

Usage: python paper/latex/md2tex.py
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEC = os.path.join(ROOT, "paper", "sections")
OUT = os.path.join(ROOT, "paper", "latex", "generated")

SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
       "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}

FILES = {
    "01_introduction.md": "introduction.tex",
    "02_results.md": "results.tex",
    "03_discussion.md": "discussion.tex",
    "04_methods.md": "methods.tex",
}


def pre(md, strip_heading=True):
    # drop ledger/editor HTML comments; collapse the space they leave behind
    md = re.sub(r"\s*<!--.*?-->", "", md, flags=re.DOTALL)
    md = re.sub(r" +([,.;:)])", r"\1", md)
    # 3.2 × 10⁻⁶  ->  $3.2 \times 10^{-6}$   (also bare 10⁻⁸)
    def sup_repl(m):
        return "$10^{" + "".join(SUP[c] for c in m.group(1)) + "}$"
    md = re.sub(r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", sup_repl, md)
    # join "N × $10^{-k}$" into one math expression
    md = re.sub(r"([\d.]+)\s*×\s*\$10\^", r"$\1 \\times 10^", md)
    # unicode math pandoc would pass through raw
    md = md.replace("×", "$\\times$").replace("·", "$\\cdot$")
    md = md.replace("≤", "$\\le$").replace("≥", "$\\ge$")
    md = md.replace("λ", "$\\lambda$").replace("≈", "$\\approx$")
    md = re.sub(r"−([\d.]+)", r"$-\1$", md)  # U+2212 minus in CI bounds
    md = re.sub(r"~(?=[\d])", r"\\ensuremath{\\sim}", md)  # command form survives raw_tex
    # pandoc mangles a bare "\," (thin space) into a literal comma, which turned
    # "µg m^-3" into "µg,m^-3" in review. Route unit strings through ASCII sentinels
    # and expand them to LaTeX in post() AFTER pandoc has run.
    md = md.replace("µg/m³", "@@UGM3@@")
    md = md.replace("°C", "@@DEGC@@")
    if strip_heading:
        # drop first-level heading (main.tex owns \section)
        md = re.sub(r"^# .*\n", "", md, count=1)
    return md


def post(tex):
    """Expand ASCII unit sentinels to LaTeX (after pandoc, which would mangle \\,)."""
    tex = tex.replace("@@UGM3@@", "\\textmu g\\,m\\textsuperscript{-3}")
    tex = tex.replace("@@DEGC@@", "\\textdegree C")
    return tex


def convert(src, dst):
    md = open(os.path.join(SEC, src), encoding="utf-8").read()
    md = pre(md)
    tex = post(_pandoc(md))
    # pandoc demotes our '## x' to \subsection already since '#' was stripped;
    # any leftover markdown h2 became \section — demote to \subsection*
    tex = tex.replace("\\subsection{", "\\subsection*{")
    tex = tex.replace("\\section{", "\\subsection*{")
    tex = re.sub(r"\\label\{[^}]*\}", "", tex)  # pandoc slugs collide across sections
    with open(os.path.join(OUT, dst), "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    print(f"{src} -> generated/{dst}  ({len(tex)} chars)")


def _pandoc(md):
    tex = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "latex", "--wrap=preserve",
         "--top-level-division=section"],
        input=md.encode("utf-8"), capture_output=True, check=True,
    ).stdout.decode("utf-8").replace("\r\n", "\n")
    return tex


def convert_abstract():
    """Emit generated/abstract.tex from the '# Abstract' block of 00_title_abstract.md."""
    md = open(os.path.join(SEC, "00_title_abstract.md"), encoding="utf-8").read()
    md = re.sub(r"\s*<!--.*?-->", "", md, flags=re.DOTALL)
    m = re.search(r"^# Abstract\s*\n(.*?)(?=^# |\Z)", md, re.MULTILINE | re.DOTALL)
    if not m:
        raise SystemExit("00_title_abstract.md: no '# Abstract' block found")
    tex = post(_pandoc(pre(m.group(1).strip() + "\n", strip_heading=False)))
    tex = re.sub(r"\\label\{[^}]*\}", "", tex)
    dst = os.path.join(OUT, "abstract.tex")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    n_words = len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", tex).split())
    print(f"00_title_abstract.md -> generated/abstract.tex  (~{n_words} words)")


def main():
    os.makedirs(OUT, exist_ok=True)
    for src, dst in FILES.items():
        convert(src, dst)
    convert_abstract()


if __name__ == "__main__":
    main()
