"""Clone the Metrica Sports sample data into data/external/metrica-sample, pinned to one commit.

https://github.com/metrica-sports/sample-data -- three anonymised matches with tracking and
human-annotated events. The repository has no formal licence; its README asks for responsible use
and attribution. Pinned so that a rebuild reads exactly the files the submission was built from.

    python -m src.fetch_metrica
"""

import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "external", "metrica-sample")
URL = "https://github.com/metrica-sports/sample-data.git"
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"


def head():
    return subprocess.run(["git", "-C", OUT_DIR, "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def fetch():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
        subprocess.run(["git", "clone", "-q", URL, OUT_DIR], check=True)
    if head() != COMMIT:
        subprocess.run(["git", "-C", OUT_DIR, "fetch", "-q", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", OUT_DIR, "checkout", "-q", COMMIT], check=True)
    if head() != COMMIT:
        raise RuntimeError("could not check out {} in {}".format(COMMIT, OUT_DIR))
    print("ok      metrica-sample at {}".format(COMMIT[:12]))


if __name__ == "__main__":
    fetch()
