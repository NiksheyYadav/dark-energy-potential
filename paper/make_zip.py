"""Pack the paper for upload.

  python paper/make_zip.py           ->  paper/overleaf.zip : main.tex, refs.bib, figures
  python paper/make_zip.py --arxiv   ->  paper/arxiv.zip    : main.tex, main.bbl, figures
                                         (arXiv does not run BibTeX; compile first so build/main.bbl exists)
"""
import os, re, sys, zipfile

os.chdir(os.path.dirname(os.path.abspath(__file__)))
arxiv = "--arxiv" in sys.argv
tex = open("main.tex", encoding="utf-8").read()
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
out = "arxiv.zip" if arxiv else "overleaf.zip"
if arxiv:
    todos = re.findall(r"\\todo\{", tex)
    if todos:
        print(f"warning: {len(todos)} \\todo markers are still in main.tex")
    bbl = os.path.join("build", "main.bbl")
    if not os.path.exists(bbl):
        raise SystemExit("build/main.bbl missing: compile with tectonic --keep-intermediates first")
    if os.path.getmtime(bbl) < max(os.path.getmtime("main.tex"), os.path.getmtime("refs.bib")):
        raise SystemExit("build/main.bbl is older than main.tex or refs.bib: recompile with --keep-intermediates")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("main.tex")
    if arxiv:
        z.write(os.path.join("build", "main.bbl"), "main.bbl")
    else:
        z.write("refs.bib")
    for f in figs:
        z.write(os.path.join("..", "figures", f), "figures/" + f)
print(out, sorted(zipfile.ZipFile(out).namelist()))
