"""Download the Pantheon+ files (not redistributed here) and check their SHA-256.

  python data/fetch_data.py

Source: github.com/PantheonPlusSH0ES/DataRelease, Pantheon+_Data/4_DISTANCES_AND_COVAR.
The other data files are small and are included in this directory (sources in README.md).
"""
import hashlib, os, urllib.request

BASE = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
FILES = {
    "Pantheon+SH0ES.dat": "1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8",
    "Pantheon+SH0ES_STAT+SYS.cov": "abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


for name, digest in FILES.items():
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print("downloading", name)
        urllib.request.urlretrieve(BASE + name.replace("+", "%2B"), path)
    ok = sha256(path) == digest
    print(f"{name}: {'ok' if ok else 'CHECKSUM MISMATCH'}")
    if not ok:
        raise SystemExit("the file differs from the one used in the paper; delete it and rerun")
