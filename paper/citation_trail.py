"""Citation trail for the novelty audit (REPORT.md Sec. 2). Rerun right before submission.

Lists papers that cite, or are cited by, the closest prior work and whose title or abstract
mentions both a scalar-field potential and a machine-learning / differentiable method.
  python paper/citation_trail.py [extra arXiv IDs ...]   ->  results/citation_trail.json
Source: Semantic Scholar Graph API (unauthenticated; backs off on rate limits).
"""
import json, os, re, sys, time, urllib.request, urllib.error

sys.stdout.reconfigure(encoding="utf-8")
ALL = "--all" in sys.argv                     # --all: print every citing paper, unfiltered
EXTRA = [a for a in sys.argv[1:] if not a.startswith("--")]
SEEDS = EXTRA if "--only" in sys.argv else ["2305.15500", "2205.02945", "2605.30139"] + EXTRA
PHYS = re.compile(r"potential|quintessence|scalar[- ]field|scalar[- ]tensor|thawing", re.I)
ML = re.compile(r"neural|network|machine.learning|differentiab|autodiff|physics.informed|PINN|deep.learning", re.I)
API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}/{}?fields=title,year,externalIds,abstract&limit=1000"


def get(url):
    for i in range(6):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(15 * (i + 1))
    raise RuntimeError("still rate limited: " + url)


hits = {}
for s in SEEDS:
    for mode, key in (("citations", "citingPaper"), ("references", "citedPaper")):
        data = get(API.format(s, mode)).get("data") or []
        n = 0
        for d in data:
            p = d.get(key) or {}
            txt = (p.get("title") or "") + " " + (p.get("abstract") or "")
            if (ALL and mode == "citations") or (PHYS.search(txt) and ML.search(txt)):
                ax = (p.get("externalIds") or {}).get("ArXiv") or "-"
                h = hits.setdefault(p.get("title"), dict(arxiv=ax, year=p.get("year"), via=[]))
                h["via"].append(f"{mode[:4]}:{s}")
                n += 1
        print(f"{s} {mode}: {len(data)} papers, {n} match", flush=True)
        time.sleep(4)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(root, "results", "citation_trail.json"), "w", encoding="utf-8") as fh:
    json.dump(hits, fh, indent=1)
for t, h in sorted(hits.items(), key=lambda kv: -(kv[1]["year"] or 0)):
    print(f"{h['arxiv']:>12} {h['year']} {(t or '')[:95]}  [{','.join(h['via'])}]")
