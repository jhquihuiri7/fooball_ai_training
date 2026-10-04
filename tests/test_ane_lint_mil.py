"""El walker sobre un programa MIL de verdad, construido con mb (ML-10)."""

from __future__ import annotations

import pytest

from ftrain.export.ane_rules import LintConfig, lint, ops_from_spec

ct = pytest.importorskip("coremltools", reason="sin ruedas de coremltools aquí")

from coremltools.converters.mil import Builder as mb  # noqa: E402, N813 — el alias canónico de MIL


def _convertir(programa):
    return ct.convert(
        programa,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )


def test_un_topk_en_medio_falla_nombrando_la_op():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 8, 8))], opset_version=ct.target.iOS18)
    def programa(x):
        valores, _indices = mb.topk(x=x, k=4, axis=-1)
        return mb.relu(x=valores)  # el topk NO cierra el programa

    ops, cabezas, entradas = ops_from_spec(_convertir(programa).get_spec())
    violaciones = lint(ops, LintConfig(cola=frozenset({"topk"})), cabezas, entradas)

    culpables = [v for v in violaciones if v.rule == "orden-fuera-de-cola"]
    assert culpables, [v.to_dict() for v in violaciones]
    assert all(v.type == "topk" for v in culpables)


def test_un_programa_limpio_pasa_con_su_cola_declarada():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 8, 8))], opset_version=ct.target.iOS18)
    def programa(x):
        suave = mb.relu(x=x)
        valores, _indices = mb.topk(x=suave, k=4, axis=-1)
        return valores

    ops, cabezas, entradas = ops_from_spec(_convertir(programa).get_spec())
    violaciones = lint(ops, LintConfig(cola=frozenset({"topk"})), cabezas, entradas)

    assert violaciones == [], [v.to_dict() for v in violaciones]
    assert any(op.type == "topk" for op in ops)  # el walker lo vio de verdad


def test_canales_estrechos_en_mil():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 8, 8))], opset_version=ct.target.iOS18)
    def programa(x):
        trozo = mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[1, 8, 8, 8])
        estrecha = mb.relu(x=trozo)  # (1,8,8,8): activación intermedia estrecha
        return mb.concat(values=[estrecha, estrecha], axis=1)  # la cabeza vuelve a 16

    ops, cabezas, entradas = ops_from_spec(_convertir(programa).get_spec())
    violaciones = lint(ops, LintConfig(), cabezas, entradas)

    assert any(v.rule == "canales" and "8 canales" in v.detail for v in violaciones), [
        v.to_dict() for v in violaciones
    ]
