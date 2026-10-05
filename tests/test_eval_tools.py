"""Las herramientas de ML-26 de punta a punta: eval_players y ball_size_table."""

from __future__ import annotations

import json

import numpy as np

from ftrain.labels import Box, FrameRef, LabelSet, to_coco
from tools.ball_size_table import table
from tools.eval_players import main as eval_players


def coco_con_bandas():
    frames = [FrameRef(f"p{p}.jpg", 3840, 2160) for p in range(3)]
    cajas = []
    for p in range(3):
        cajas += [
            Box(f"p{p}.jpg", "player", 100.0, 100.0, 140.0, 180.0),
            Box(f"p{p}.jpg", "player", 500.0, 900.0, 520.0, 940.0),
            Box(f"p{p}.jpg", "ball", 700.0, 700.0, 700.0 + 6 + p, 700.0 + 6 + p),
        ]
    coco = to_coco(LabelSet(frames, cajas))
    for im in coco["images"]:
        im["match"] = im["file_name"][:2]
        im["light"] = "day"
    for an in coco["annotations"]:
        an["band"] = 0 if an["bbox"][1] < 500 else 3
    return coco


def test_eval_players_escribe_json_y_tabla(tmp_path):
    coco = coco_con_bandas()
    (tmp_path / "gt.json").write_text(json.dumps(coco))
    # Detecta el jugador cercano en los tres partidos y el lejano solo en uno.
    np.savez(
        tmp_path / "pred.npz",
        image_id=np.array([0, 1, 2, 0]),
        category=np.array([1, 1, 1, 1]),
        boxes=np.array(
            [[101, 100, 141, 181], [100, 101, 140, 180], [99, 100, 139, 180], [500, 900, 520, 941]],
            float,
        ),
        scores=np.array([0.9, 0.9, 0.9, 0.8]),
    )

    assert (
        eval_players(
            [
                "--truth",
                str(tmp_path / "gt.json"),
                "--pred",
                str(tmp_path / "pred.npz"),
                "--out",
                str(tmp_path / "inf"),
            ]
        )
        == 0
    )

    informe = json.loads((tmp_path / "inf.json").read_text())
    assert informe["bands"]["0-20 m"]["recall"] == 1.0
    assert abs(informe["bands"]["60-80 m"]["recall"] - 1 / 3) < 1e-9
    assert "| 60-80 m | 3 |" in (tmp_path / "inf.md").read_text()


def test_tabla_del_balon_por_banda():
    t = table(coco_con_bandas())
    assert list(t) == [3]
    assert t[3]["n"] == 3
    assert t[3]["p50"] == 7.0
