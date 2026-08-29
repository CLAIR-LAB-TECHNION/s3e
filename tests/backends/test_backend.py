"""Tests for the VLMBackend base contract and the VLMOutput record.

Dependency-free: runs without torch, openai, or vllm installed.
"""

from PIL import Image

from s3e.backends.backend import VLMBackend, VLMOutput

from fakes import FakeVLM


class TestVLMOutput:
    def test_creation(self):
        output = VLMOutput(token_probs={"yes": 0.9, "no": 0.1})
        assert output.token_probs["yes"] == 0.9
        assert output.text is None

    def test_with_text(self):
        output = VLMOutput(token_probs={"yes": 0.9}, text="yes")
        assert output.text == "yes"

    def test_argmax_in_interest_defaults_to_none(self):
        output = VLMOutput(token_probs={"yes": 0.9})
        assert output.argmax_in_interest is None

    def test_argmax_in_interest_is_settable(self):
        output = VLMOutput(token_probs={"yes": 0.9}, argmax_in_interest=True)
        assert output.argmax_in_interest is True

    def test_token_probs_type_admits_none(self):
        """Every backend sets token_probs=None in generate mode; the declared
        field type must say so."""
        import types

        field = VLMOutput.__dataclass_fields__["token_probs"]
        assert isinstance(field.type, types.UnionType)
        assert type(None) in field.type.__args__


class TestVLMBackend:
    def test_query_returns_vlm_output(self):
        vlm = FakeVLM()
        img = Image.new("RGB", (64, 64))
        result = vlm.query([img], "Is block A on block B?")
        assert isinstance(result, VLMOutput)
        assert "yes" in result.token_probs

    def test_query_batch_default_loops(self):
        # FakeVLM overrides query_batch (to mirror real backends' batching),
        # so a minimal subclass defining only query() exercises the base
        # class's default sequential query_batch implementation instead.
        class SequentialOnlyVLM(VLMBackend):
            def __init__(self):
                self.calls: list[str] = []

            def query(
                self,
                images,
                prompt,
                system_prompt=None,
                generate=False,
                interest_tokens=None,
                **inference_kwargs,
            ):
                del images, system_prompt, generate, interest_tokens, inference_kwargs
                self.calls.append(prompt)
                return VLMOutput(token_probs={"yes": 0.7, "no": 0.2})

        vlm = SequentialOnlyVLM()
        img = Image.new("RGB", (64, 64))
        results = vlm.query_batch([img], ["q1", "q2", "q3"])
        assert len(results) == 3
        assert len(vlm.calls) == 3
        assert all(isinstance(r, VLMOutput) for r in results)

    def test_query_batch_passes_system_prompt(self):
        class TrackingVLM(VLMBackend):
            def __init__(self):
                self.received_system_prompts = []

            def query(
                self,
                images,
                prompt,
                system_prompt=None,
                generate=False,
                **inference_kwargs,
            ):
                del generate
                del inference_kwargs
                self.received_system_prompts.append(system_prompt)
                return VLMOutput(token_probs={"yes": 0.5})

        vlm = TrackingVLM()
        img = Image.new("RGB", (64, 64))
        vlm.query_batch([img], ["q1", "q2"], system_prompt="Be helpful.")
        assert vlm.received_system_prompts == ["Be helpful.", "Be helpful."]

    def test_query_batch_forwards_interest_tokens(self):
        class TrackingVLM(VLMBackend):
            def __init__(self):
                self.received_interest_tokens = []

            def query(
                self,
                images,
                prompt,
                system_prompt=None,
                generate=False,
                interest_tokens=None,
                **inference_kwargs,
            ):
                del generate
                del inference_kwargs
                self.received_interest_tokens.append(interest_tokens)
                return VLMOutput(token_probs={"yes": 0.5})

        vlm = TrackingVLM()
        img = Image.new("RGB", (64, 64))
        vlm.query_batch([img], ["q1", "q2"], interest_tokens=["yes", "no"])
        assert vlm.received_interest_tokens == [["yes", "no"], ["yes", "no"]]

