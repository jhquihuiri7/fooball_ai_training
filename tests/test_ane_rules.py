"""Cada regla del lint del ANE con listas sintéticas, sin coremltools (ML-10)."""

from __future__ import annotations

import pytest
import yaml

from ftrain.export.ane_rules import LintConfig, LintError, MilOp, lint, load_config


def _op(tipo: str, nombre: str = "x", **kwargs) -> MilOp:
    return MilOp(type=tipo, name=nombre, **kwargs)


def _reglas(violaciones) -> set[str]:
    return {v.rule for v in violaciones}


def test_conv3d_y_rango_5():
    ops = [_op("conv", "c3", shapes=((1, 16, 4, 8, 8),), dtypes=("fp16",))]
    reglas = _reglas(lint(ops, LintConfig()))
    assert "conv3d" in reglas
    assert "rango-5" in reglas  # el rango 5 cae por su propia regla además


def test_recurrentes_y_scatter_nombran_la_op():
    ops = [
        _op("gru", "memoria", shapes=((1, 16, 8, 8),), dtypes=("fp16",)),
        _op("scatter_nd", "disperso", shapes=((1, 16, 8, 8),), dtypes=("fp16",)),
    ]
    violaciones = lint(ops, LintConfig())
    assert {"recurrente", "scatter"} <= _reglas(violaciones)
    assert {v.op for v in violaciones} == {"memoria", "disperso"}


def test_topk_en_medio_falla_y_en_la_cola_declarada_no():
    relu = _op("relu", shapes=((1, 16, 8, 8),), dtypes=("fp16",))
    topk = _op("topk", "eleccion", shapes=((1, 16, 8, 16),), dtypes=("fp16",))

    en_medio = lint([relu, topk, relu], LintConfig())
    assert any(v.rule == "orden-fuera-de-cola" and v.op == "eleccion" for v in en_medio)

    en_cola = lint([relu, topk], LintConfig(cola=frozenset({"topk"})))
    assert en_cola == []


def test_un_permitido_no_paga_la_cola_pero_el_resto_si():
    relu = _op("relu", shapes=((1, 16, 8, 8),), dtypes=("fp16",))
    topk = _op("topk", "queries", shapes=((1, 16, 8, 16),), dtypes=("fp16",))
    argsort = _op("argsort", "orden", shapes=((1, 16, 8, 16),), dtypes=("fp16",))

    config = LintConfig(permitidos=frozenset({"topk"}))
    violaciones = lint([relu, topk, argsort, relu], config)

    assert [v.op for v in violaciones if v.rule == "orden-fuera-de-cola"] == ["orden"]


def test_control_de_flujo():
    ops = [_op("while_loop", "bucle"), _op("cond", "rama")]
    assert _reglas(lint(ops, LintConfig())) == {"control"}


def test_canales_no_multiplo_de_16_y_sus_exenciones():
    malo = _op("conv", "estrecha", shapes=((1, 24, 8, 8),), dtypes=("fp16",))
    assert _reglas(lint([malo], LintConfig())) == {"canales"}

    # La cabeza del modelo está exenta sola; la lista blanca exime por nombre.
    assert lint([malo], LintConfig(), heads=frozenset({"estrecha"})) == []
    assert lint([malo], LintConfig(exentos=frozenset({"estrecha"}))) == []

    # const y cast no son activaciones: los pesos (8,3,3,3) no pagan la regla.
    pesos = _op("const", "w", shapes=((8, 3, 3, 3),), dtypes=("fp16",))
    assert lint([pesos], LintConfig()) == []


def test_el_preproceso_de_la_imagen_de_entrada_esta_exento():
    # Lo que coremltools inyecta para un ImageType con escala: const y mul fp32.
    escala = _op("const", "image__scaled___y_0", shapes=((),), dtypes=("fp32",))
    mul = _op("mul", "image__scaled__", shapes=((1, 3, 32, 32),), dtypes=("fp32",))
    tronco = _op("conv", "tronco", shapes=((1, 16, 32, 32),), dtypes=("fp16",))

    assert lint([escala, mul, tronco], LintConfig(), inputs=frozenset({"image"})) == []
    # Sin declarar la entrada, el mismo programa cae por canales y por fp32.
    assert {"canales", "fp32-fuera-de-cola"} <= _reglas(lint([escala, mul, tronco], LintConfig()))


def test_dimension_simbolica():
    ops = [_op("relu", shapes=((1, 16, "?", 8),), dtypes=("fp16",))]
    assert _reglas(lint(ops, LintConfig())) == {"simbolica"}


def test_fp32_fuera_de_la_cola_y_el_cast_final_no():
    dentro = _op("relu", "r32", shapes=((1, 16, 8, 8),), dtypes=("fp32",))
    salida = _op("cast", "heatmap", shapes=((1, 16, 8, 8),), dtypes=("fp32",))

    violaciones = lint([dentro, salida], LintConfig())
    assert [v.op for v in violaciones if v.rule == "fp32-fuera-de-cola"] == ["r32"]


def test_la_lista_blanca_se_carga_y_valida(tmp_path):
    ruta = tmp_path / "demo.yaml"
    ruta.write_text(yaml.safe_dump({"cola": ["topk"], "exentos": ["salida"]}), encoding="utf-8")
    config = load_config(ruta)
    assert config.cola == frozenset({"topk"})
    assert config.exentos == frozenset({"salida"})

    ruta.write_text(yaml.safe_dump({"cola": "topk"}), encoding="utf-8")
    with pytest.raises(LintError, match="cola"):
        load_config(ruta)
