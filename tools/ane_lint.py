"""Lint estático del programa MIL de un .mlpackage (ML-10).

    uv sync --group apple
    uv run python tools/ane_lint.py dist/demo.mlpackage \\
        --config configs/ane_lint/demo.yaml

Imprime un JSON con las violaciones y termina con código 1 si hay alguna. La
lista blanca del modelo (los tipos de op de su cola de postproceso y las salidas
exentas del múltiplo de 16) vive en configs/ane_lint/<modelo>.yaml; sin
`--config`, no hay cola y todo topk/argsort/NMS o fp32 es violación.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.export.ane_rules import LintConfig, lint, load_config, ops_from_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="el .mlpackage a revisar")
    parser.add_argument("--config", type=Path, help="la lista blanca del modelo")
    opciones = parser.parse_args(argv)

    config = load_config(opciones.config) if opciones.config else LintConfig()
    ops, cabezas, entradas = ops_from_package(opciones.package)
    violaciones = lint(ops, config, cabezas, entradas)

    print(
        json.dumps(
            {
                "package": str(opciones.package),
                "ops": len(ops),
                "violations": [v.to_dict() for v in violaciones],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if violaciones else 0


if __name__ == "__main__":
    raise SystemExit(main())
