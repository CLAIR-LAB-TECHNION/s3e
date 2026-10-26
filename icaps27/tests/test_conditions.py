# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e5_conditions_map_to_backend_kwargs():
    e5 = load("e5_reasoning")
    assert e5.condition_kwargs("as-is", "enable_thinking") == {"thinking_delimiters": None}
    assert e5.condition_kwargs("as-is-on", "enable_thinking") == {
        "thinking_delimiters": None, "chat_template_kwargs": {"enable_thinking": True}}
    assert e5.condition_kwargs("switch-off", None) == {"thinking": False}
    assert e5.condition_kwargs("closed", "enable_thinking") == {
        "thinking": False, "chat_template_kwargs": {"enable_thinking": True}}
    assert e5.condition_kwargs("closed", None) == {"thinking": False}
    assert e5.condition_kwargs("think-256", None) == {"thinking": True, "max_thinking_tokens": 256}


def test_e5_refuses_without_thinking_support():
    e5 = load("e5_reasoning")
    assert e5.thinking_supported() is False  # the released 0.5.0


def test_regex_baseline_finds_answers_anywhere():
    from s3e import BinaryAnswers, Prediction, PredictionSet

    from s3e_exp.analysis import regex_results

    answers = BinaryAnswers()
    texts = {"a": "I think the answer is yes.", "b": "No.", "c": "Maybe."}
    results = PredictionSet({k: Prediction(k, {"yes": 0.0, "no": 0.0}, 0.0, 1.0, answers, text=t)
                             for k, t in texts.items()})
    parsed = regex_results(results)
    assert parsed["a"].probability == 1.0
    assert parsed["b"].probability == 0.0
    assert not parsed["c"].has_answer and parsed["c"].probability == 0.5
