"""Static checks for main.tex when no TeX compiler is available."""
import re, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
t = open("main.tex", encoding="utf-8").read()
b = open("refs.bib", encoding="utf-8").read()
s = re.sub(r"\\[{}%]", "", t)
s = re.sub(r"(?<!\\)%.*", "", s)
print("brace balance:", s.count("{") - s.count("}"))
st = []
for k, e in re.findall(r"\\(begin|end)\{(\w+\*?)\}", s):
    if k == "begin":
        st.append(e)
    elif not st or st.pop() != e:
        print("environment mismatch at", e)
print("unclosed environments:", st)
keys = set(re.findall(r"@\w+\{([^,]+),", b))
cites = {c.strip() for g in re.findall(r"\\cite\w*\{([^}]+)\}", t) for c in g.split(",")}
print("missing bib keys:", cites - keys, "| cited:", len(cites))
labels = set(re.findall(r"\\label\{([^}]+)\}", t))
refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", t))
print("refs without label:", refs - labels)
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t)
print("missing figures:", [f for f in figs if not os.path.exists(os.path.join("..", "figures", f))])
print("todo markers:", len(re.findall(r"\\todo\{", t)))
for i, line in enumerate(t.splitlines(), 1):
    if line.count("$") % 2:
        print("odd $ count on line", i)
