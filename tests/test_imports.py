"""Import hygiene: bare installs import cleanly; optional features fail with
the exact extra named; heavy modules are never imported eagerly.

Each test runs a fresh subprocess so module caches can't mask problems.
Meta-path hooks simulate optional packages that are missing (``_Blocker``) or
installed-but-broken (``_Breaker``) even though the dev environment has working
copies of them.
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

# NOTE: contains f-string braces, so it is filled via .replace(), not .format().
BREAKER = """
import sys
from importlib.machinery import ModuleSpec

class _BrokenLoader:
    '''Loader for a package that is installed but fails on import.'''

    def __init__(self, name):
        self.name = name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        dep = self.name + "_dependency"
        raise ModuleNotFoundError(f"No module named {dep!r}", name=dep)

class _Breaker:
    def __init__(self, broken):
        self.broken = set(broken)

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in self.broken:
            return ModuleSpec(name, _BrokenLoader(root))
        return None

sys.meta_path.insert(0, _Breaker(BROKEN_LIST))
"""

HEAVY = ["torch", "torchvision", "transformers", "accelerate",
         "unified_planning", "vllm", "openai", "sklearn"]


def run_python(body: str, blocked=(), broken=()) -> subprocess.CompletedProcess:
    script = (
        BLOCKER.replace("BLOCKED_LIST", repr(list(blocked)))
        + BREAKER.replace("BROKEN_LIST", repr(list(broken)))
        + body
    )
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


class TestBrokenDependencyErrors:
    """A dependency that is installed but broken must surface its own error.

    ``require`` checks presence with ``find_spec`` rather than importing, so a
    package whose *own* imports fail raises that failure with its real
    traceback. Rebuilding ``require`` around ``try: import x`` would instead
    report every such breakage as a missing extra, sending users to reinstall
    something they already have; these tests fail if that happens.

    The counterpart to ``TestMissingDependencyErrors``: same import sites, the
    other side of the presence check.
    """

    def check(self, body: str, broken: str):
        script = (
            "try:\n"
            + "".join("    " + line + "\n" for line in body.splitlines())
            + "except ModuleNotFoundError as e:\n"
            "    assert e.name == '" + broken + "_dependency', repr(e)\n"
            "except ImportError as e:\n"
            "    raise SystemExit('masked broken dependency: ' + str(e))\n"
            "else:\n"
            "    raise SystemExit('expected the broken dependency to surface')\n"
        )
        result = run_python(script, broken=[broken])
        assert result.returncode == 0, result.stderr or result.stdout

    def test_broken_torch_surfaces_from_huggingface(self):
        self.check("from s3e import HuggingFaceVLM", "torch")

    def test_broken_openai_surfaces_from_openai(self):
        self.check("from s3e import OpenAIVLM", "openai")

    def test_broken_vllm_surfaces_from_vllm(self):
        self.check("from s3e import VLLMBackend", "vllm")
