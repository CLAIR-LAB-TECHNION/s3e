"""Import hygiene: bare installs import cleanly; optional features fail with
the exact extra named; heavy modules are never imported eagerly.

Each test runs a fresh subprocess so module caches can't mask problems. A
meta-path blocker simulates missing optional packages even though the dev
environment has them installed.
"""

import subprocess
import sys

# NOTE: contains f-string braces, so it is filled via .replace(), not .format().
BLOCKER = """
import sys

class _Blocker:
    def __init__(self, blocked):
        self.blocked = set(blocked)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.blocked:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _Blocker(BLOCKED_LIST))
"""

HEAVY = ["torch", "torchvision", "transformers", "accelerate",
         "unified_planning", "vllm", "openai", "sklearn"]


def run_python(body: str, blocked=()) -> subprocess.CompletedProcess:
    script = BLOCKER.replace("BLOCKED_LIST", repr(list(blocked))) + body
    return subprocess.run(
        [sys.executable, "-W", "error::Warning", "-c", script],
        capture_output=True, text=True, timeout=120,
    )


class TestBareImport:
    def test_import_s3e_with_all_heavy_deps_blocked(self):
        result = run_python("import s3e", blocked=HEAVY)
        assert result.returncode == 0, result.stderr

    def test_import_s3e_pulls_no_heavy_modules(self):
        body = (
            "import s3e, sys\n"
            "heavy = " + repr(HEAVY) + "\n"
            "loaded = [m for m in heavy if m in sys.modules]\n"
            "assert not loaded, f'eagerly imported: {loaded}'"
        )
        result = run_python(body)
        assert result.returncode == 0, result.stderr

    def test_core_symbols_usable_without_heavy_deps(self):
        body = (
            "from s3e import BinaryAnswers, PredictionSet, QueryEngine, "
            "SemanticStateEstimator, TemplateTranslator\n"
            "space = BinaryAnswers('true', 'false')\n"
            "assert space.true_label == 'true'"
        )
        result = run_python(body, blocked=HEAVY)
        assert result.returncode == 0, result.stderr


class TestMissingDependencyErrors:
    def check(self, body: str, blocked: list, expected_extra: str):
        script = (
            "try:\n"
            + "".join("    " + line + "\n" for line in body.splitlines())
            + "except ImportError as e:\n"
            "    assert 's3e[" + expected_extra + "]' in str(e), str(e)\n"
            "else:\n"
            "    raise SystemExit('expected ImportError')\n"
        )
        result = run_python(script, blocked=blocked)
        assert result.returncode == 0, result.stderr or result.stdout

    def test_huggingface_names_hf_extra(self):
        self.check(
            "from s3e import HuggingFaceVLM", ["torch"], "hf"
        )

    def test_huggingface_names_hf_extra_when_only_transformers_missing(self):
        self.check(
            "from s3e import HuggingFaceVLM", ["transformers"], "hf"
        )

    def test_vllm_names_vllm_extra(self):
        self.check("from s3e import VLLMBackend", ["vllm"], "vllm")

    def test_openai_names_openai_extra(self):
        self.check("from s3e import OpenAIVLM", ["openai"], "openai")

    def test_pddl_names_pddl_extra(self):
        self.check(
            "import s3e.pddl", ["unified_planning"], "pddl"
        )

    def test_platt_fit_names_calibration_extra(self):
        self.check(
            "from s3e.calibration.platt import fit_platt_parameters\n"
            "fit_platt_parameters([1.0, -1.0], [True, False])",
            ["sklearn"],
            "calibration",
        )

    def test_llm_translator_hf_path_names_hf_extra(self):
        self.check(
            "from s3e import LLMTranslator\n"
            "LLMTranslator('not-openai/some-model')",
            ["transformers", "torch"],
            "hf",
        )
