"""Tests for the optional-dependency helper."""

import pytest

from s3e._deps import require


class TestRequire:
    def test_noop_when_module_present(self):
        require("json", "someextra", "SomeFeature")  # stdlib always present

    def test_raises_naming_extra_when_missing(self):
        with pytest.raises(ImportError) as excinfo:
            require("s3e_definitely_not_a_module", "hf", "HuggingFaceVLM")
        message = str(excinfo.value)
        assert "HuggingFaceVLM" in message
        assert 'pip install "s3e[hf]"' in message

    def test_survives_meta_path_finders_that_raise(self):
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "blocked_module_xyz":
                    raise ModuleNotFoundError(name=name)
                return None

        sys.meta_path.insert(0, Blocker())
        try:
            with pytest.raises(ImportError, match=r"s3e\[vllm\]"):
                require("blocked_module_xyz", "vllm", "VLLMBackend")
        finally:
            sys.meta_path.pop(0)
