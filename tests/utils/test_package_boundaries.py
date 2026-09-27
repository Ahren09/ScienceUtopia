"""Guard package ownership without importing optional model or plotting libraries."""

import ast
import subprocess
import sys

from utopia.utils.paths import project_root


ROOT = project_root()


def test_generic_utilities_do_not_import_domain_packages():
    violations = []
    for path in (ROOT / "utopia/utils").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module.startswith("utopia.") and not module.startswith(
                    "utopia.utils"
                ):
                    violations.append((str(path), node.lineno, module))
    assert not violations


def test_visualization_libraries_and_chart_construction_have_one_owner():
    violations = []
    libraries = {"matplotlib", "seaborn", "plotly", "pacmap"}
    for path in (ROOT / "utopia").rglob("*.py"):
        if "visual" in path.relative_to(ROOT).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                modules = []
            if any(module.split(".")[0] in libraries for module in modules):
                violations.append((str(path), node.lineno))
            if isinstance(node, ast.Call) and (
                ast.unparse(node.func).startswith("wandb.plot.")
                or ast.unparse(node.func) in {"wandb.Histogram", "wandb.Image"}
            ):
                violations.append((str(path), node.lineno))
    assert not violations


def test_removed_modules_have_no_remaining_source_imports():
    removed = {
        "utopia.utils.general_utils",
        "utopia.utils.visual_utils",
        "utopia.metrics.keyword_extractor",
        "utopia.utils.citation_extractor",
        "utopia.prompts",
    }
    for name in removed:
        assert not (ROOT / (name.replace(".", "/") + ".py")).exists()
    for path in (ROOT / "utopia").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in removed, path
            elif isinstance(node, ast.Import):
                assert not removed.intersection(alias.name for alias in node.names), (
                    path
                )


def test_generic_and_keyword_imports_do_not_load_models_or_start_runtime():
    code = """
import argparse, subprocess, sys
from unittest.mock import patch
with patch.object(subprocess, "Popen", side_effect=AssertionError("process launch")), \\
     patch.object(argparse.ArgumentParser, "parse_args", side_effect=AssertionError("CLI parsing")):
    import utopia.utils.seeding
    import utopia.utils.logging
    import utopia.utils.data_utils
    import utopia.runtime.setup
    import utopia.runtime.provenance
    import utopia.data.keyword_extractor
assert not {"torch", "vllm", "transformers", "openai", "datasets",
            "matplotlib", "seaborn", "langchain_core"} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)
