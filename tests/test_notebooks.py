"""Guard against stale notebook imports and invalid notebook source."""
import ast
import importlib
import json
from pathlib import Path

import pytest


NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_json_code_and_package_imports_are_current(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])
        tree = ast.parse(source, filename=f"{notebook_path.name}:cell-{index}")
        compile(tree, f"{notebook_path.name}:cell-{index}", "exec")

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("diabetes_ml"):
                continue

            module = importlib.import_module(node.module)
            for imported in node.names:
                if imported.name != "*":
                    assert hasattr(module, imported.name), (
                        f"{notebook_path.name}, cell {index}: "
                        f"{node.module} does not expose {imported.name}"
                    )
