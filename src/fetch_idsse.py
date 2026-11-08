"""Download the IDSSE open Bundesliga data (7 games) into data/external/idsse/.

Bassek, Weber, Rein & Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data, doi:10.1038/s41597-025-04505-y. CC BY 4.0. Figshare article
28196177, listed through figshare's public API; each file is checked against its published MD5
and skipped if already present.

Only the match information (meta) and raw positions (tracking) are fetched by default: the DFL
position data marks every frame with BallStatus and BallPossession, which is all the labelling
needs. --events adds the event files.

    python -m src.fetch_idsse              # ~2.6 GB
"""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "external", "idsse")
ARTICLE_URL = "https://api.figshare.com/v2/articles/28196177"
KINDS = {"matchinformation": "meta", "positions_raw": "tracking", "events_raw": "events"}


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kind_of(name):
    return next((kind for key, kind in KINDS.items() if key in name), None)


def fetch(kinds):
    os.makedirs(OUT_DIR, exist_ok=True)
    with urllib.request.urlopen(ARTICLE_URL) as response:
        files = json.load(response)["files"]

    for entry in sorted(files, key=lambda f: f["size"]):
        if kind_of(entry["name"]) not in kinds:
            continue
        path = os.path.join(OUT_DIR, entry["name"])
        if os.path.isfile(path) and md5(path) == entry["computed_md5"]:
            print("ok      {}".format(entry["name"]))
            continue
        print("fetch   {}  ({:.0f} MB)".format(entry["name"], entry["size"] / 1e6), flush=True)
        partial = path + ".part"
        with urllib.request.urlopen(entry["download_url"]) as response, \
                open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)
        if md5(partial) != entry["computed_md5"]:
            os.remove(partial)
            raise IOError("checksum mismatch for {}".format(entry["name"]))
        os.replace(partial, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", action="store_true", help="also fetch the event files")
    args = parser.parse_args()
    fetch({"meta", "tracking"} | ({"events"} if args.events else set()))
