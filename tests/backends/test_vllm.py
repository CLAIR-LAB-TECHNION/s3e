"""Tests for the vLLM backend: mocked-engine units, import behavior,
slow GPU integration, and the shared backend contract.

Importing ``s3e.backends.vllm`` needs a working ``vllm`` package (the
``dev-gpu`` extra); tests guard themselves so a CPU ``dev`` install
skips them cleanly instead of failing.
"""

import importlib.util
import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="torch not installed (s3e[hf])")

from s3e.backends.backend import VLMOutput

from backends.test_contract import BackendContract


import math as _math  # noqa: E402  (module-level math for vLLM helpers)


def _make_logprob(token, logprob, rank=None):
    """Build a mock vLLM Logprob (has .decoded_token, .logprob and .rank)."""
    item = MagicMock()
    item.decoded_token = token
    item.logprob = logprob
    item.rank = rank
    return item


def _make_logprobs_output(token_logprobs):
    """Build a mock vLLM RequestOutput for logprobs mode.

    token_logprobs: list of (token_str, logprob_float). Keyed by fake ids.
    """
    completion = MagicMock()
    completion.logprobs = [
        {idx: _make_logprob(tok, lp) for idx, (tok, lp) in enumerate(token_logprobs)}
    ]
    output = MagicMock()
    output.outputs = [completion]
    return output


def _make_text_output(text):
    """Build a mock vLLM RequestOutput for text mode."""
    completion = MagicMock()
    completion.text = text
    output = MagicMock()
    output.outputs = [completion]
    return output


def _make_id_logprobs_output(id_logprobs):
    """Build a mock vLLM RequestOutput keyed by explicit token ids.

    Mirrors ``detokenize=False`` output: ``decoded_token`` is None, the
    logprobs dict keys are real token ids, and every entry carries its
    vocab ``rank`` (1 = highest probability), as real vLLM entries do.
    """
    ranked_ids = sorted(id_logprobs, key=id_logprobs.get, reverse=True)
    completion = MagicMock()
    completion.logprobs = [
        {
            token_id: _make_logprob(
                None, logprob, rank=ranked_ids.index(token_id) + 1
            )
            for token_id, logprob in id_logprobs.items()
        }
    ]
    output = MagicMock()
    output.outputs = [completion]
    return output


def _make_id_tokenizer(vocab):
    """Build a mock tokenizer whose id ``i`` decodes to ``vocab[i]``."""
    tokenizer = MagicMock()
    tokenizer.__len__.return_value = len(vocab)
    tokenizer.batch_decode.side_effect = lambda sequences, **kwargs: [
        vocab[sequence[0]] for sequence in sequences
    ]
    tokenizer.decode.side_effect = lambda ids, **kwargs: vocab[
        ids[0] if isinstance(ids, list) else ids
    ]
    return tokenizer


class TestVLLMBackendMocked:
    """Unit tests for VLLMBackend with vllm mocked out.

    The engine is mocked, but importing ``s3e.backends.vllm`` still needs a
    working ``vllm`` package (SamplingParams etc.), which the ``dev`` extra
    deliberately omits — install ``s3e[dev-gpu]`` for these.
    """

    @pytest.fixture(autouse=True)
    def _requires_vllm(self):
        pytest.importorskip("vllm", reason="vllm not installed (s3e[dev-gpu])")

    def _make_backend(self, mock_llm_cls, num_logprobs=None):
        """Construct a VLLMBackend whose engine is the mocked LLM instance."""
        from s3e.backends.vllm import VLLMBackend

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        with patch("torch.cuda.device_count", return_value=1):
            backend = VLLMBackend("test/model", num_logprobs=num_logprobs)
        return backend, mock_llm

    @patch("torch.cuda.device_count", return_value=3)
    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_tensor_parallel_defaults_to_all_local_gpus(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.backends.vllm import VLLMBackend

        VLLMBackend("test/model")

        kwargs = mock_llm_cls.call_args.kwargs
        assert kwargs["model"] == "test/model"
        assert kwargs["tensor_parallel_size"] == 3

    @patch("torch.cuda.device_count", return_value=8)
    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_tensor_parallel_explicit_override(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.backends.vllm import VLLMBackend

        VLLMBackend("test/model", tensor_parallel_size=2)

        assert mock_llm_cls.call_args.kwargs["tensor_parallel_size"] == 2

    @pytest.mark.parametrize("tensor_parallel_size", [0, -1, 1.5, True])
    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_invalid_tensor_parallel_size_rejected_before_engine_load(
        self, mock_llm_cls, mock_sp_cls, tensor_parallel_size
    ):
        from s3e.backends.vllm import VLLMBackend

        with pytest.raises(ValueError, match="tensor_parallel_size"):
            VLLMBackend("test/model", tensor_parallel_size=tensor_parallel_size)

        mock_llm_cls.assert_not_called()

    @patch("torch.cuda.device_count", return_value=1)
    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_max_logprobs_full_vocab_by_default(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.backends.vllm import VLLMBackend

        VLLMBackend("test/model")

        assert mock_llm_cls.call_args.kwargs["max_logprobs"] == -1

    @patch("torch.cuda.device_count", return_value=1)
    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_max_logprobs_finite_when_num_logprobs_set(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.backends.vllm import VLLMBackend

        VLLMBackend("test/model", num_logprobs=5)

        assert mock_llm_cls.call_args.kwargs["max_logprobs"] == 5

    @pytest.mark.parametrize("num_logprobs", [0, -1, 1.5, True])
    @patch("torch.cuda.device_count", return_value=1)
    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_invalid_num_logprobs_rejected_before_engine_load(
        self, mock_llm_cls, mock_sp_cls, mock_device_count, num_logprobs
    ):
        from s3e.backends.vllm import VLLMBackend

        with pytest.raises(ValueError, match="num_logprobs"):
            VLLMBackend("test/model", num_logprobs=num_logprobs)

        mock_llm_cls.assert_not_called()

    @patch("torch.cuda.device_count", return_value=1)
    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_engine_kwargs_forwarded(
        self, mock_llm_cls, mock_sp_cls, mock_device_count
    ):
        from s3e.backends.vllm import VLLMBackend

        VLLMBackend("test/model", gpu_memory_utilization=0.5, max_model_len=2048)

        kwargs = mock_llm_cls.call_args.kwargs
        assert kwargs["gpu_memory_utilization"] == 0.5
        assert kwargs["max_model_len"] == 2048

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_query_batch_builds_one_chat_call_with_conversation_structure(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [
            _make_logprobs_output([("yes", _math.log(0.8))]),
            _make_logprobs_output([("no", _math.log(0.6))]),
        ]
        img = Image.new("RGB", (64, 64))

        results = backend.query_batch([img], ["q1", "q2"], system_prompt="sys")

        assert mock_llm.chat.call_count == 1
        conversations = mock_llm.chat.call_args.args[0]
        assert len(conversations) == 2
        first = conversations[0]
        assert first[0] == {"role": "system", "content": "sys"}
        assert first[1]["role"] == "user"
        assert first[1]["content"][0]["type"] == "image_pil"
        assert first[1]["content"][0]["image_pil"] is img
        assert first[1]["content"][-1] == {"type": "text", "text": "q1"}
        assert len(results) == 2

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_query_batch_omits_system_message_when_absent(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query_batch([], ["q1"])

        conversation = mock_llm.chat.call_args.args[0][0]
        assert all(m["role"] != "system" for m in conversation)

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_empty_prompts_returns_empty_list_without_engine_call(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)

        assert backend.query_batch([Image.new("RGB", (8, 8))], []) == []
        mock_llm.chat.assert_not_called()

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_missing_logprobs_raises_informative_error(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        completion = MagicMock()
        completion.logprobs = None  # vLLM returned no logprobs
        bad_output = MagicMock()
        bad_output.outputs = [completion]
        mock_llm.chat.return_value = [bad_output]

        with pytest.raises(RuntimeError, match="no logprobs"):
            backend.query([], "q1")

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_logprobs_mode_sampling_params_defaults(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls, num_logprobs=None)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query_batch([], ["q1"])

        kwargs = mock_sp_cls.call_args.kwargs
        assert kwargs["max_tokens"] == 1
        assert kwargs["temperature"] == 0.0
        assert kwargs["logprobs"] == -1

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_logprobs_value_follows_num_logprobs(self, mock_llm_cls, mock_sp_cls):
        backend, mock_llm = self._make_backend(mock_llm_cls, num_logprobs=4)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query_batch([], ["q1"])

        assert mock_sp_cls.call_args.kwargs["logprobs"] == 4

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_max_tokens_is_overridable_in_logprobs_mode(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query_batch([], ["q1"], max_tokens=7)

        assert mock_sp_cls.call_args.kwargs["max_tokens"] == 7

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_logprobs_extraction_converts_to_probabilities(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [
            _make_logprobs_output([("yes", _math.log(0.7)), ("no", _math.log(0.3))])
        ]

        result = backend.query([], "q1")

        assert result.text is None
        assert result.token_probs["yes"] == pytest.approx(0.7)
        assert result.token_probs["no"] == pytest.approx(0.3)

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_logprobs_extraction_sums_duplicate_tokens(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [
            _make_logprobs_output([("same", _math.log(0.4)), ("same", _math.log(0.25))])
        ]

        result = backend.query([], "q1")

        assert set(result.token_probs) == {"same"}
        assert result.token_probs["same"] == pytest.approx(0.65)

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_text_mode_returns_text_and_no_token_probs(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [
            _make_text_output("yes"),
            _make_text_output("no"),
        ]

        results = backend.query_batch([], ["q1", "q2"], generate=True)

        assert [r.text for r in results] == ["yes", "no"]
        assert all(r.token_probs is None for r in results)

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_text_mode_does_not_bound_or_request_logprobs(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_text_output("yes")]

        backend.query_batch([], ["q1"], generate=True)

        kwargs = mock_sp_cls.call_args.kwargs
        assert "max_tokens" not in kwargs  # model may reason freely
        assert "logprobs" not in kwargs
        assert kwargs["temperature"] == 0.0

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_interest_mode_requests_detokenize_false(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(["yes", "no"])
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.8), 1: _math.log(0.2)})
        ]

        backend.query([], "q1", interest_tokens=["yes", "no"])

        kwargs = mock_sp_cls.call_args.kwargs
        assert kwargs["detokenize"] is False
        assert kwargs["logprobs"] == -1
        assert kwargs["max_tokens"] == 1

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_no_interest_mode_does_not_touch_detokenize(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_logprobs_output([("yes", _math.log(0.5))])]

        backend.query([], "q1")

        assert "detokenize" not in mock_sp_cls.call_args.kwargs

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_interest_matches_by_token_id_and_backfills(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(
            ["yes", "no", "maybe"]
        )
        mock_llm.chat.return_value = [
            _make_id_logprobs_output(
                {0: _math.log(0.6), 1: _math.log(0.3), 2: _math.log(0.1)}
            )
        ]

        result = backend.query([], "q1", interest_tokens=["yes", "no", "null"])

        assert set(result.token_probs) == {"yes", "no", "null"}
        assert result.token_probs["yes"] == pytest.approx(0.6)
        assert result.token_probs["no"] == pytest.approx(0.3)
        assert result.token_probs["null"] == 0.0
        assert result.argmax_in_interest is True
        assert result.text is None

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_interest_sums_duplicate_ids_decoding_to_same_string(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(["yes", "yes"])
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.4), 1: _math.log(0.2)})
        ]

        result = backend.query([], "q1", interest_tokens=["yes"])

        assert result.token_probs["yes"] == pytest.approx(0.6)

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_interest_argmax_false_when_top_id_outside_interest(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(["maybe", "yes"])
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.5), 1: _math.log(0.4)})
        ]

        result = backend.query([], "q1", interest_tokens=["yes"])

        assert result.argmax_in_interest is False
        assert result.token_probs["yes"] == pytest.approx(0.4)

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_interest_reverse_index_is_built_once(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        tokenizer = _make_id_tokenizer(["yes", "no"])
        mock_llm.get_tokenizer.return_value = tokenizer
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.8), 1: _math.log(0.2)})
        ]

        backend.query([], "q1", interest_tokens=["yes"])
        backend.query([], "q2", interest_tokens=["yes"])

        assert mock_llm.get_tokenizer.call_count == 1
        assert tokenizer.batch_decode.call_count == 1

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_unsupported_interest_tokens_reports_multi_token_strings(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(
            ["yes", "no", "dark", "blue"]
        )

        unsupported = backend.unsupported_interest_tokens(["yes", "dark blue"])

        assert unsupported == ["dark blue"]

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_interest_with_stop_strings_keeps_detokenization(
        self, mock_llm_cls, mock_sp_cls
    ):
        """detokenize=False forbids stop strings, so stop wins."""
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.get_tokenizer.return_value = _make_id_tokenizer(["yes"])
        mock_llm.chat.return_value = [
            _make_id_logprobs_output({0: _math.log(0.8)})
        ]

        backend.query([], "q1", interest_tokens=["yes"], stop=["\n"])

        assert "detokenize" not in mock_sp_cls.call_args.kwargs

    @patch("s3e.backends.vllm.SamplingParams")
    @patch("s3e.backends.vllm.LLM")
    def test_generate_mode_ignores_interest_tokens(
        self, mock_llm_cls, mock_sp_cls
    ):
        backend, mock_llm = self._make_backend(mock_llm_cls)
        mock_llm.chat.return_value = [_make_text_output("yes")]

        result = backend.query([], "q1", generate=True, interest_tokens=["yes"])

        assert result.text == "yes"
        assert result.token_probs is None
        assert result.argmax_in_interest is None
        kwargs = mock_sp_cls.call_args.kwargs
        assert "detokenize" not in kwargs
        assert "logprobs" not in kwargs
        mock_llm.get_tokenizer.assert_not_called()

    def test_missing_vllm_raises_install_guidance(self):
        """Importing the module when vllm is absent raises via require()."""
        import importlib
        import importlib.util
        import sys

        module_name = "s3e.backends.vllm"
        original_module = sys.modules.pop(module_name, None)
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "vllm":
                return None
            return real_find_spec(name, *args, **kwargs)

        try:
            with patch("importlib.util.find_spec", side_effect=fake_find_spec):
                with pytest.raises(
                    ImportError, match=r'pip install "s3e\[vllm\]"'
                ):
                    importlib.import_module(module_name)
        finally:
            sys.modules.pop(module_name, None)
            if original_module is not None:
                sys.modules[module_name] = original_module

    def test_installed_vllm_import_failure_is_not_masked(self):
        import builtins
        import importlib
        import sys

        module_name = "s3e.backends.vllm"
        parent_module = sys.modules.get("s3e.backends")
        original_module = sys.modules.pop(module_name, None)
        had_parent_attr = parent_module is not None and hasattr(parent_module, "vllm")
        original_parent_attr = (
            getattr(parent_module, "vllm", None) if had_parent_attr else None
        )
        if had_parent_attr:
            delattr(parent_module, "vllm")

        real_import = builtins.__import__

        def import_with_broken_vllm(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "vllm":
                raise ModuleNotFoundError(
                    "No module named 'vllm_dependency'", name="vllm_dependency"
                )
            return real_import(name, globals, locals, fromlist, level)

        try:
            with patch("builtins.__import__", side_effect=import_with_broken_vllm):
                with pytest.raises(ModuleNotFoundError, match="vllm_dependency"):
                    importlib.import_module(module_name)
        finally:
            sys.modules.pop(module_name, None)
            if original_module is not None:
                sys.modules[module_name] = original_module
            if parent_module is not None:
                if had_parent_attr:
                    setattr(parent_module, "vllm", original_parent_attr)
                elif hasattr(parent_module, "vllm"):
                    delattr(parent_module, "vllm")


def test_vllm_backend_is_exported():
    pytest.importorskip("vllm", reason="vllm not installed (s3e[dev-gpu])")
    import s3e
    from s3e.backends import VLLMBackend as FromVlm
    from s3e import VLLMBackend as FromTop

    assert FromVlm is FromTop
    assert "VLLMBackend" in s3e.__all__
    assert "VLLMBackend" in s3e.backends.__all__


def test_import_s3e_does_not_touch_broken_vllm_dependency():
    pytest.importorskip("vllm", reason="vllm not installed (s3e[dev-gpu])")
    import subprocess
    import sys

    script = """
import builtins

real_import = builtins.__import__


def import_with_broken_vllm(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "vllm":
        raise ModuleNotFoundError(
            "No module named 'vllm_dependency'", name="vllm_dependency"
        )
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = import_with_broken_vllm

import s3e

assert "VLLMBackend" in s3e.__all__

try:
    from s3e import VLLMBackend  # noqa: F401
except ModuleNotFoundError as exc:
    assert exc.name == "vllm_dependency", exc.name
else:
    raise AssertionError("Explicit VLLMBackend access should surface broken vLLM")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None,
    reason="vLLM requires CUDA and an installed vllm package",
)
class TestVLLMBackendIntegration:
    """Integration test with a small real VLM via vLLM.

    Requires a GPU and an installed ``vllm``; skipped otherwise. The host must
    also expose a CUDA dev toolchain (``nvcc`` + CUDA headers) because vLLM's
    default logprobs sampler (FlashInfer) JIT-compiles a kernel on first use.
    Run with: ``pytest -m slow``.

    The model is the small ``SmolVLM-256M-Instruct`` rather than a degenerate
    ``tiny-random`` stub: such stubs use a head dim (e.g. 4) below the minimum
    that CUDA attention kernels accept, so they cannot actually run on a GPU.
    ``enforce_eager=True`` skips torch.compile / CUDA-graph capture, keeping the
    smoke test fast and free of compile-time backend surprises.
    """

    SMALL_VLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

    @pytest.fixture(scope="class")
    def backend(self):
        # Evaluating this class's skipif guard (torch.cuda.is_available()) at
        # collection time initializes CUDA in the pytest process. vLLM's engine
        # core subprocess uses the fork start method by default, and CUDA
        # cannot re-initialize in a forked child ("Cannot re-initialize CUDA in
        # forked subprocess"), so force spawn. setdefault keeps any explicit
        # user choice.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from s3e.backends.vllm import VLLMBackend

        # num_logprobs=None exercises the full-vocab path (max_logprobs=-1 /
        # logprobs=-1, the vLLM >= 0.11.0 feature the pyproject pin exists for)
        # and makes the token-string assertions below deterministic: the full
        # vocabulary necessarily contains the tokens they look for.
        # Prefix caching is off so repeated queries with the same prompt (the
        # parity test) recompute their logits identically.
        return VLLMBackend(
            self.SMALL_VLM_ID,
            tensor_parallel_size=1,
            num_logprobs=None,
            gpu_memory_utilization=0.3,
            max_model_len=4096,
            enforce_eager=True,
            enable_prefix_caching=False,
        )

    def test_loads_and_queries_logprobs(self, backend):
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = backend.query([img], "Is this a test?")

        assert isinstance(result, VLMOutput)
        assert isinstance(result.token_probs, dict)
        assert len(result.token_probs) > 2  # full-vocab distribution
        assert all(p >= 0 for p in result.token_probs.values())

        # Token-string parity with HuggingFaceVLM: keys must be decoded text
        # (e.g. "Yes"), not raw tokenizer symbols ("▁Yes" / "ĠYes"). The
        # estimator matches token_probs keys against plain strings like
        # "Yes"/"true", so a raw-symbol format would silently zero out the
        # true/false token masses instead of failing loudly — catch it here.
        assert "Yes" in result.token_probs
        assert not any(key.startswith(("▁", "Ġ")) for key in result.token_probs)

    def test_interest_tokens_parity_with_full_vocab(self, backend):
        """Interest-mode masses must equal the full-vocabulary path's."""
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        prompt = "Is this a test?"
        interest = ["Yes", "No", "yes", "no"]

        full = backend.query([img], prompt)
        gathered = backend.query([img], prompt, interest_tokens=interest)

        assert set(gathered.token_probs) == set(interest)
        for token in interest:
            assert gathered.token_probs[token] == pytest.approx(
                full.token_probs.get(token, 0.0), abs=1e-9
            )
        assert gathered.argmax_in_interest == (
            max(full.token_probs, key=full.token_probs.get) in set(interest)
        )



@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None,
    reason="vLLM requires CUDA and an installed vllm package",
)
class TestVLLMBackendContract(BackendContract):
    """Applies the shared backend contract to a real, small vLLM model."""

    @pytest.fixture(scope="class")
    def make_backend(self):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from s3e.backends.vllm import VLLMBackend

        backend = VLLMBackend(
            TestVLLMBackendIntegration.SMALL_VLM_ID,
            tensor_parallel_size=1,
            num_logprobs=None,
            gpu_memory_utilization=0.3,
            max_model_len=4096,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        return lambda: backend
