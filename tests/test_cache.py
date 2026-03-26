"""Tests for the NL translation cache module."""

import json
import pytest

from s3e.cache import make_cache_key, load_cache, save_cache


class TestMakeCacheKey:
    def test_basic_key(self):
        key = make_cache_key("meta-llama/Llama-3", "blocksworld")
        assert "meta-llama__Llama-3" in key
        assert "blocksworld" in key
        assert key.endswith(".json")

    def test_key_with_kwargs(self):
        key = make_cache_key("my-model", "prob1", temperature=0.5, top_p=0.9)
        assert "temperature=0.5" in key
        assert "top_p=0.9" in key
        assert key.endswith(".json")

    def test_key_strips_slashes(self):
        key = make_cache_key("org/model-name", "problem")
        assert "/" not in key.replace(".json", "")

    def test_same_inputs_same_key(self):
        key1 = make_cache_key("model", "prob", x=1)
        key2 = make_cache_key("model", "prob", x=1)
        assert key1 == key2


class TestSaveAndLoadCache:
    def test_round_trip(self, tmp_path):
        queries = {"on(a,b)": "Is a on b?", "clear(a)": "Is a clear?"}
        save_cache(str(tmp_path), "test_cache.json", queries)
        loaded = load_cache(str(tmp_path), "test_cache.json")
        assert loaded == queries

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cache(str(tmp_path), "nonexistent.json")

    def test_save_merges_with_existing(self, tmp_path):
        cache_dir = str(tmp_path)
        save_cache(cache_dir, "merge_test.json", {"on(a,b)": "Is a on b?"})
        save_cache(cache_dir, "merge_test.json", {"clear(a)": "Is a clear?"})
        loaded = load_cache(cache_dir, "merge_test.json")
        assert loaded == {"on(a,b)": "Is a on b?", "clear(a)": "Is a clear?"}

    def test_save_updates_existing_keys(self, tmp_path):
        cache_dir = str(tmp_path)
        save_cache(cache_dir, "update_test.json", {"on(a,b)": "old query"})
        save_cache(cache_dir, "update_test.json", {"on(a,b)": "new query"})
        loaded = load_cache(cache_dir, "update_test.json")
        assert loaded == {"on(a,b)": "new query"}
