"""Descarga SkillCorner opendata a un commit fijado (ML-48, ADR 0004).

    uv run python tools/fetch_skillcorner.py            # los 10 partidos
    uv run python tools/fetch_skillcorner.py --only 1886347

Deja `datasets/skillcorner/<commit>/` (en .gitignore): el LICENSE, `matches.json` y, por
partido, `match.json`, `dynamic_events.csv`, `phases_of_play.csv` y el tracking
(`tracking_extrapolated.jsonl`, que el repositorio guarda en Git LFS y aquí se baja por
su URL de medios). Comprueba que el LICENSE es MIT y apunta en `PROCEDENCIA.txt` que los
DATOS no traen licencia aparte: un N3 preentrenado con ellos no se vende hasta aclararlo
(ADR 0021, punto abierto). No se vendoriza nada: no se clona el repositorio.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
"""El commit de SkillCorner/opendata fijado (2026-09-14). Otro commit es otro dataset."""

RAW = f"https://raw.githubusercontent.com/SkillCorner/opendata/{COMMIT}"
MEDIA = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{COMMIT}"
ROOT = Path(__file__).resolve().parent.parent / "datasets" / "skillcorner" / COMMIT
TIMEOUT_S = 120
FILES = ("match.json", "dynamic_events.csv", "phases_of_play.csv")
TRACKING = "tracking_extrapolated.jsonl"
MIT_HEADER = "MIT License"
PROVENANCE = """SkillCorner opendata, commit {commit}.
El repositorio es MIT (LICENSE). Los DATOS no traen licencia aparte: sirven para
preentrenar y medir N3, pero un modelo preentrenado con ellos no se vende hasta
aclararlo (ADR 0021, punto abierto; ADR 0004 de este repo).
"""


def download(url: str, dest: Path) -> int:
    """Baja `url` a `dest` si no está ya. Devuelve los bytes del fichero."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, tmp.open("wb") as f:  # noqa: S310 - URL fija de GitHub
        while trozo := r.read(1 << 20):
            f.write(trozo)
    tmp.rename(dest)
    return dest.stat().st_size


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", type=int, help="ids de partido; por defecto, todos")
    o = p.parse_args(argv)

    download(f"{RAW}/LICENSE", ROOT / "LICENSE")
    if MIT_HEADER not in (ROOT / "LICENSE").read_text():
        print("el LICENSE de SkillCorner ya no es MIT: se para", file=sys.stderr)
        return 1
    (ROOT / "PROCEDENCIA.txt").write_text(PROVENANCE.format(commit=COMMIT))
    download(f"{RAW}/data/matches.json", ROOT / "matches.json")
    ids = o.only or [m["id"] for m in json.loads((ROOT / "matches.json").read_text())]
    for mid in ids:
        base = f"data/matches/{mid}/{mid}_"
        for f in FILES:
            download(f"{RAW}/{base}{f}", ROOT / str(mid) / f)
        n = download(f"{MEDIA}/{base}{TRACKING}", ROOT / str(mid) / TRACKING)
        print(f"{mid}: tracking {n / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
