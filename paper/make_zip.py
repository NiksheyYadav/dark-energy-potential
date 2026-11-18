"""Build paper/overleaf.zip: main.tex, refs.bib and every figure main.tex includes (from ../figures)."""
import os, re, zipfile

os.chdir(os.path.dirname(os.path.abspath(__file__)))
tex = open("main.tex", encoding="utf-8").read()
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
with zipfile.ZipFile("overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("main.tex")
    z.write("refs.bib")
    for f in figs:
        z.write(os.path.join("..", "figures", f), "figures/" + f)
print(sorted(zipfile.ZipFile("overleaf.zip").namelist()))
