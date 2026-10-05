"""SkillCorner a la serie de N3 (ML-48): fixture con su forma → rejilla y etiquetas."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from ftrain.constants import N3_CHANNELS, N3_CLASSES, N3_GRID_H, N3_GRID_W
from ftrain.events.features import cell_of, grids, label_timeline
from ftrain.events.skillcorner import Sample, labels_from_events, load_match, resample

CH = {n: k for k, n in enumerate(N3_CHANNELS)}
GK, JUGADOR = 7, 9


def partido(tmp_path, frames=40):
    d = tmp_path / "123"
    d.mkdir()
    (d / "match.json").write_text(
        json.dumps(
            {
                "id": 123,
                "pitch_length": 104,
                "pitch_width": 68,
                "players": [
                    {"id": GK, "player_role": {"acronym": "GK"}},
                    {"id": JUGADOR, "player_role": {"acronym": "CF"}},
                ],
                "match_periods": [{"period": 1, "start_frame": 10}],
            }
        )
    )
    with (d / "tracking_extrapolated.jsonl").open("w") as f:
        f.write(
            json.dumps({"frame": 0, "period": None, "player_data": [], "ball_data": {"x": None}})
            + "\n"
        )
        for k in range(10, 10 + frames):
            f.write(
                json.dumps(
                    {
                        "frame": k,
                        "period": 1,
                        "ball_data": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "player_data": [
                            {"player_id": GK, "x": -50.0, "y": 0.0},
                            # Corre a 10 m/s a lo largo: 1 m por fotograma de 10 Hz.
                            {"player_id": JUGADOR, "x": float(k - 10), "y": 20.0},
                        ],
                    }
                )
                + "\n"
            )
    filas = [
        {
            "event_type": "player_possession",
            "frame_start": "20",
            "frame_end": "25",
            "game_interruption_before": "throw_in_for",
            "game_interruption_after": "",
        },
        # La misma reanudación repetida en otra fila: un solo evento.
        {
            "event_type": "player_possession",
            "frame_start": "21",
            "frame_end": "24",
            "game_interruption_before": "throw_in_against",
            "game_interruption_after": "",
        },
        {
            "event_type": "passing_option",
            "frame_start": "22",
            "frame_end": "23",
            "game_interruption_before": "corner_for",
            "game_interruption_after": "",
        },
        {
            "event_type": "player_possession",
            "frame_start": "26",
            "frame_end": "30",
            "game_interruption_before": "",
            "game_interruption_after": "goal_for",
        },
        {
            "event_type": "player_possession",
            "frame_start": "45",
            "frame_end": "48",
            "game_interruption_before": "goal_against",
            "game_interruption_after": "",
        },
        {
            "event_type": "player_possession",
            "frame_start": "47",
            "frame_end": "49",
            "game_interruption_before": "penalty_for",
            "game_interruption_after": "",
        },
    ]
    with (d / "dynamic_events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return d


def test_etiquetas_de_las_posesiones(tmp_path):
    m = load_match(partido(tmp_path))
    # Las reanudaciones, al poner el balón en juego (el fin de la posesión de quien saca):
    # de las dos filas del mismo saque de banda queda la primera, a 2,4 s.
    assert m.labels == [(1.0, "kickoff"), (2.4, "throw_in"), (3.0, "goal"), (4.8, "kickoff")]
    assert m.goalkeepers == {GK}


def test_posiciones_en_el_campo_de_referencia_y_remuestreo(tmp_path):
    m = load_match(partido(tmp_path))
    # 40 fotogramas a 10 Hz (3,9 s) → 30 muestras a 7,5 Hz.
    assert len(m.samples) == 30
    assert [round(s.t_s, 4) for s in m.samples[:3]] == [
        1.0,
        round(1 + 1 / 7.5, 4),
        round(1 + 2 / 7.5, 4),
    ]
    # x escalada de 104 a 105 m, e interpolada: en t = 1 + 1/7,5 s, el fotograma 11,33.
    x, y = m.samples[1].players[JUGADOR]
    assert x == pytest.approx((1 + 1 / 3) * 105 / 104)
    assert y == pytest.approx(20.0)


def test_rejilla_con_porteros_velocidad_y_balon(tmp_path):
    m = load_match(partido(tmp_path))
    g = grids(m)
    assert g.shape == (30, len(N3_CHANNELS), N3_GRID_H, N3_GRID_W)
    f_gk, c_gk = cell_of(-50.0 * 105 / 104, 0.0)
    assert g[5, CH["goalkeeper"], f_gk, c_gk] == 1.0
    assert g[5, CH["players"]].sum() == 2.0
    f_b, c_b = cell_of(0.0, 0.0)
    assert g[5, CH["ball"], f_b, c_b] == 1.0
    f_j, c_j = cell_of(*m.samples[5].players[JUGADOR])
    assert g[5, CH["vx"], f_j, c_j] == pytest.approx(10.0 * 105 / 104)
    assert g[0, CH["vx"]].sum() == 0.0, "sin muestra anterior no hay velocidad"
    assert g[:, CH["referee"]].sum() == 0.0
    assert g[:, CH["whistle"]].sum() == 0.0


def test_lo_que_pisa_la_linea_va_a_la_celda_del_borde():
    assert cell_of(-60.0, -40.0) == (0, 0)
    assert cell_of(60.0, 40.0) == (N3_GRID_H - 1, N3_GRID_W - 1)


def test_etiqueta_por_paso_dilatada():
    muestras = [Sample(1.0 + k / 7.5, {}, None) for k in range(20)]
    banda, gol = N3_CLASSES.index("throw_in") + 1, N3_CLASSES.index("goal") + 1

    # El saque de banda cae en el paso 8 y el gol en el 10: el 9 empata y se queda con el
    # primero.
    y = label_timeline(muestras, [(2.0, "throw_in"), (1.0 + 10 / 7.5, "goal")])

    assert list(y[6:13]) == [0, banda, banda, banda, gol, gol, 0]
    assert int((y == 0).sum()) == 15


def test_remuestreo_con_un_hueco_toma_el_vecino():
    # Falta el fotograma 1: la muestra de t = 0,133 s (el 1,33) sale del 2.
    frames = {0: ({1: (0.0, 0.0)}, None), 2: ({1: (2.0, 0.0), 2: (5.0, 5.0)}, (3.0, 3.0))}
    s = resample(frames)
    assert len(s) == 2
    assert s[1].players == {1: (2.0, 0.0), 2: (5.0, 5.0)}
    assert s[1].ball == (3.0, 3.0)


def test_sin_eventos_solo_los_saques_de_cada_parte():
    assert labels_from_events([], [10, 27800]) == [(1.0, "kickoff"), (2780.0, "kickoff")]
    assert np.array_equal(label_timeline([], []), np.zeros(0, dtype=np.int64))


def test_el_espejo_invierte_el_campo_y_la_velocidad():
    from ftrain.events.features import mirror  # noqa: PLC0415

    g = np.zeros((len(N3_CHANNELS), N3_GRID_H, N3_GRID_W), dtype=np.float32)
    g[CH["players"], 2, 3] = 1.0
    g[CH["vx"], 2, 3] = 4.0
    g[CH["vy"], 2, 3] = -1.0
    m = mirror(g, along=True, across=True)
    assert m[CH["players"], N3_GRID_H - 1 - 2, N3_GRID_W - 1 - 3] == 1.0
    assert m[CH["vx"], N3_GRID_H - 1 - 2, N3_GRID_W - 1 - 3] == -4.0
    assert m[CH["vy"], N3_GRID_H - 1 - 2, N3_GRID_W - 1 - 3] == 1.0
    assert np.array_equal(mirror(m, along=True, across=True), g)
