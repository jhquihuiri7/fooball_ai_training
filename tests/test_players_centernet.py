"""CenterNet-MNv4: objetivo, decodificación, pérdida, mosaico y el entreno de punta a punta."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ftrain.constants import CENTERNET_STRIDE, PLAYER_CLASSES
from ftrain.players.centernet_data import MosaicConfig, draw_targets, letterbox, row_mosaic

if TYPE_CHECKING:
    from pathlib import Path

PLAYER = PLAYER_CLASSES.index("player")


def test_el_objetivo_marca_centro_tamano_y_desplazamiento():
    caja = np.array([[41.0, 22.0, 61.0, 82.0]])  # 20x60 px: centro (51, 52)
    t = draw_targets(caja, np.array([PLAYER]), input_hw=(128, 256), n_classes=3)
    assert t.heatmap.shape == (3, 32, 64)
    iy, ix = 52 // CENTERNET_STRIDE, 51 // CENTERNET_STRIDE
    assert t.heatmap[PLAYER, iy, ix] == 1.0
    assert t.heatmap[[0, 2]].max() == 0.0
    assert t.mask.sum() == 1.0
    assert t.mask[0, iy, ix] == 1.0
    np.testing.assert_allclose(t.size[:, iy, ix], (5.0, 15.0))
    np.testing.assert_allclose(t.offset[:, iy, ix], (51 / 4 - ix, 52 / 4 - iy))
    # Elíptico: una persona alta marca más hacia arriba y abajo que a los lados.
    assert t.heatmap[PLAYER, iy + 2, ix] > t.heatmap[PLAYER, iy, ix + 2]


def test_las_multitudes_se_ignoran_y_los_centros_nunca():
    caja = np.array([[40.0, 40.0, 60.0, 80.0]])
    t = draw_targets(
        caja,
        np.array([PLAYER]),
        input_hw=(128, 256),
        n_classes=3,
        ignore_boxes=np.array([[0.0, 0.0, 128.0, 128.0]]),
        ignore_classes=np.array([PLAYER]),
    )
    assert t.ignore[PLAYER, 0, 0] == 1.0
    assert t.ignore[PLAYER, 15, 12] == 0.0  # el centro de la caja
    assert t.ignore[PLAYER, 0, 40] == 0.0  # fuera de la multitud
    assert t.ignore[[0, 2]].max() == 0.0


def test_letterbox_pega_en_la_esquina_sin_deformar():
    pytest.importorskip("cv2")
    lienzo, escala = letterbox(np.zeros((480, 640, 3), np.uint8), 576, 1920)
    assert lienzo.shape == (576, 1920, 3)
    assert escala == pytest.approx(1.2)
    assert lienzo[0, 0].tolist() == [0, 0, 0]
    assert lienzo[0, 800].tolist() == [114, 114, 114]


def _imagen_con_rectangulo(color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((120, 160, 3), np.uint8)
    img[30:90, 60:90] = color
    return img, np.array([[60.0, 30.0, 30.0, 60.0]])


def test_el_mosaico_llena_el_lienzo_y_las_cajas_caen_sobre_su_objeto():
    pytest.importorskip("cv2")
    colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    config = MosaicConfig(96, 400, min_rel_height=0.5, max_rel_height=1.0)

    def armar(semilla: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = iter(range(100))

        def draw():
            img, cajas = _imagen_con_rectangulo(colores[next(k) % 3])
            return img, cajas, np.zeros((0, 4))

        return row_mosaic(draw, config, np.random.default_rng(semilla))

    lienzo, cajas, _ = armar(7)
    assert lienzo.shape == (96, 400, 3)
    assert len(cajas) >= 2
    for x1, y1, x2, y2 in cajas:
        dentro = lienzo[int(y1) + 2 : int(y2) - 2, int(x1) + 2 : int(x2) - 2]
        assert dentro.size
        assert (dentro.max(axis=2) > 200).mean() > 0.95  # el rectángulo de color, entero
    otra = armar(7)
    assert np.array_equal(lienzo, otra[0])
    assert np.array_equal(cajas, otra[1])


torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ftrain.players.centernet_train import (  # noqa: E402
    CocoPersonMosaic,
    Ema,
    decode,
    focal_loss,
    init_for_training,
    load_imagenet_backbone,
)
from ftrain.players.plan_b import CENTERNET_BACKBONE, CenterNetMnv4, build_centernet  # noqa: E402


def test_decode_recupera_las_cajas_de_un_objetivo_perfecto():
    cajas = np.array(
        [[10.0, 20.0, 30.0, 80.0], [100.0, 8.0, 112.0, 44.0], [200.0, 60.0, 250.0, 120.0]]
    )
    t = draw_targets(cajas, np.full(3, PLAYER), input_hw=(128, 256), n_classes=3)
    salida = decode(
        torch.from_numpy(t.heatmap)[None],
        torch.from_numpy(t.size)[None],
        torch.from_numpy(t.offset)[None],
    )
    puntos, scores, clases = (v[0].numpy() for v in salida)
    arriba = scores == 1.0
    assert arriba.sum() == 3
    assert set(clases[arriba].tolist()) == {PLAYER}
    encontradas = sorted(puntos[arriba].tolist())
    np.testing.assert_allclose(encontradas, sorted(cajas.tolist()), atol=1e-4)


def test_la_focal_baja_al_acertar_y_lo_ignorado_no_cuenta():
    t = draw_targets(
        np.array([[40.0, 40.0, 60.0, 80.0]]), np.array([PLAYER]), input_hw=(128, 256), n_classes=3
    )
    gt = torch.from_numpy(t.heatmap)[None]
    sin_ignorar = torch.zeros_like(gt)
    bien = focal_loss(gt.clamp(0.01, 0.99), gt, sin_ignorar)
    mal = focal_loss(torch.full_like(gt, 0.5), gt, sin_ignorar)
    assert float(bien) < float(mal)
    todo_ignorado = torch.ones_like(gt)
    solo_positivos = focal_loss(torch.full_like(gt, 0.5), gt, todo_ignorado)
    assert float(solo_positivos) < float(mal)
    # Con todo ignorado solo queda el centro: log(0,5) * 0,5^2.
    assert float(solo_positivos) == pytest.approx(-np.log(0.5) * 0.25, rel=1e-4)


def test_el_heatmap_arranca_en_la_probabilidad_inicial():
    m = CenterNetMnv4()
    init_for_training(m)
    assert torch.sigmoid(m.heatmap[-1].bias).tolist() == pytest.approx([0.1] * 3)


def test_la_ema_sigue_al_modelo_y_copia_los_contadores():
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    ema = Ema(m, decay=0.5, ramp_steps=1e-9)
    with torch.no_grad():
        m[0].weight.add_(1.0)
    m[1].num_batches_tracked += 3
    ema.update(m)
    antes = ema.module[0].weight.clone()
    assert torch.allclose(antes, m[0].weight - 0.5)
    assert int(ema.module[1].num_batches_tracked) == 3


def test_el_tronco_de_imagenet_entra_entero(tmp_path):
    import timm  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    torch.manual_seed(1)
    red = timm.create_model(CENTERNET_BACKBONE, pretrained=False)
    save_file(red.state_dict(), str(tmp_path / "w.safetensors"))
    m = CenterNetMnv4()
    load_imagenet_backbone(m, tmp_path / "w.safetensors")
    assert torch.equal(m.stem[0].weight, red.conv_stem.weight)
    assert torch.equal(
        m.stages[3].state_dict()[next(iter(m.stages[3].state_dict()))],
        red.blocks[3].state_dict()[next(iter(red.blocks[3].state_dict()))],
    )


def _coco_falso(raiz: Path, n: int) -> None:
    cv2 = pytest.importorskip("cv2")
    (raiz / "train2017").mkdir(parents=True)
    (raiz / "val2017").mkdir()
    (raiz / "annotations").mkdir()
    indice, imagenes, anotaciones = [], [], []
    for i in range(1, n + 1):
        img, cajas = _imagen_con_rectangulo((200, 50, 50))
        nombre = f"{i:012d}.jpg"
        cv2.imwrite(str(raiz / "train2017" / nombre), img)
        cv2.imwrite(str(raiz / "val2017" / nombre), img)
        indice.append(
            {
                "id": i,
                "file": nombre,
                "width": 160,
                "height": 120,
                "license": 4,
                "boxes": cajas.tolist(),
                "crowd": [],
            }
        )
        imagenes.append({"id": i, "file_name": nombre, "width": 160, "height": 120})
        anotaciones.append(
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": cajas[0].tolist(),
                "area": 1800.0,
                "iscrowd": 0,
            }
        )
    (raiz / "person_train2017.json").write_text(json.dumps(indice))
    (raiz / "annotations" / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": imagenes,
                "annotations": anotaciones,
                "categories": [{"id": 1, "name": "person"}],
            }
        )
    )


def test_el_dataset_da_imagen_y_mapas_con_la_forma_del_lienzo(tmp_path):
    _coco_falso(tmp_path, 3)
    indice = json.loads((tmp_path / "person_train2017.json").read_text())
    ds = CocoPersonMosaic(indice, tmp_path / "train2017", MosaicConfig(64, 192))
    imagen, objetivos = ds[0]
    assert imagen.dtype == torch.uint8
    assert tuple(imagen.shape) == (3, 64, 192)
    assert tuple(objetivos["heatmap"].shape) == (3, 16, 48)
    assert float(objetivos["mask"].sum()) >= 1


def test_el_entreno_guarda_evalua_y_reanuda(tmp_path):
    pytest.importorskip("faster_coco_eval")
    from tools.train_centernet import main  # noqa: PLC0415

    _coco_falso(tmp_path / "coco", 4)
    salida = tmp_path / "run"
    comun = [
        "--data",
        str(tmp_path / "coco"),
        "--out",
        str(salida),
        "--hours",
        "1",
        "--batch",
        "2",
        "--canvas",
        "64x192",
        "--workers",
        "0",
        "--eval-every",
        "1",
        "--eval-n",
        "2",
        "--no-imagenet",
        "--device",
        "cpu",
    ]
    assert main([*comun, "--max-steps", "2"]) == 0
    assert torch.load(salida / "last.pt")["step"] == 2
    lineas = (salida / "metrics.jsonl").read_text().splitlines()
    assert len(lineas) >= 2
    assert {"step", "ap", "ap50"} <= set(json.loads(lineas[0]))
    assert build_centernet(salida / "best.pt") is not None  # lo carga el builder del export

    assert main([*comun, "--max-steps", "3"]) == 0
    assert torch.load(salida / "last.pt")["step"] == 3
