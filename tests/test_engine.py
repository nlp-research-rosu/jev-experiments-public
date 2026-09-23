import copy
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig


def tiny_model(hybrid):
    torch.manual_seed(7)
    if hybrid:
        config = Qwen3_5TextConfig(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
        )
        return Qwen3_5ForCausalLM(config).eval()
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).eval()


class EngineTests(unittest.TestCase):
    def test_selected_head_matches_full_logits_and_preserves_candidate_order(self):
        from openjev.engine import score_token_prompts

        with torch.inference_mode():
            model = tiny_model(True)
            prompts = [[1, 2, 3, 4], [1, 2, 3, 5]]
            full, _ = score_token_prompts(model, prompts, parallel=False)
            selected, _ = score_token_prompts(model, prompts, parallel=True, candidate_ids=[9, 4, 20])
            torch.testing.assert_close(selected, full[:, [9, 4, 20]], rtol=1e-4, atol=2e-6)

    def test_parallel_matches_full_reference_with_variable_lengths_and_chunking(self):
        from openjev.engine import score_token_prompts

        prompts = [[1, 2, 3, 4, 5], [1, 2, 3, 7, 8], [1, 2, 3, 9, 10, 11], [1, 2, 3, 12, 13]]
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid), torch.inference_mode():
                model = tiny_model(hybrid)
                expected = torch.cat([model(torch.tensor([p]), logits_to_keep=1).logits[:, -1] for p in prompts])
                calls = []
                hook = model.model.register_forward_pre_hook(
                    lambda module, args, kwargs: calls.append((args[0] if args else kwargs["input_ids"]).shape),
                    with_kwargs=True,
                )
                try:
                    actual, stats = score_token_prompts(model, prompts, parallel=True, branch_batch_size=2)
                finally:
                    hook.remove()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
                self.assertEqual(stats["prefix_tokens"], 3)
                self.assertEqual(stats["forward_calls"], 3)  # prefill and two mixed-length batches
                self.assertEqual(len(calls), 3)
                self.assertEqual([shape[0] for shape in calls], [1, 2, 2])
                again, _ = score_token_prompts(model, list(reversed(prompts)), parallel=True, branch_batch_size=3)
                torch.testing.assert_close(again.flip(0), expected, rtol=1e-4, atol=2e-6)

    def test_padded_endpoints_match_independent_scores_with_and_without_prefix(self):
        from openjev.engine import score_token_prompts

        # One-token and long suffixes exercise endpoint gathering and DeltaNet's
        # internal chunk boundary; zero is a real token, not necessarily padding.
        prompt_sets = (
            [[1, 2, 0], [1, 2, 4, 5, 6], [1, 2] + list(range(3, 68))],
            [[0], [2, 3, 4], list(range(3, 68))],
            [[1, 2], [1, 2, 3, 4], [1, 2, 8]],
        )
        for hybrid in (False, True):
            with torch.inference_mode():
                model = tiny_model(hybrid)
                for prompts in prompt_sets:
                    expected = torch.cat([model(torch.tensor([p]), logits_to_keep=1).logits[:, -1] for p in prompts])
                    for candidates in (None, [20, 4, 9]):
                        with self.subTest(hybrid=hybrid, lengths=list(map(len, prompts)), candidates=candidates):
                            actual, _ = score_token_prompts(model, prompts, candidate_ids=candidates)
                            reference = expected if candidates is None else expected[:, candidates]
                            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=2e-6)

    def test_cache_fork_does_not_mutate_original_or_share_recurrent_rows(self):
        from openjev.engine import fork_cache

        with torch.inference_mode():
            model = tiny_model(True)
            cache = model(torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
            saved = copy.deepcopy(cache)
            branches = fork_cache(cache, 2)
            model(torch.tensor([[4], [8]]), past_key_values=branches, use_cache=True)
            for a, b in zip(cache.layers, saved.layers):
                if hasattr(a, "keys"):
                    torch.testing.assert_close(a.keys, b.keys)
                else:
                    torch.testing.assert_close(a.conv_states[0], b.conv_states[0])
                    torch.testing.assert_close(a.recurrent_states[0], b.recurrent_states[0])
            self.assertEqual(cache.get_seq_length(), 3)
            self.assertEqual(branches.get_seq_length(), 4)
            self.assertFalse(
                torch.equal(branches.layers[0].recurrent_states[0][0], branches.layers[0].recurrent_states[0][1])
            )

    def test_extreme_length_outlier_does_not_multiply_padding_work(self):
        from openjev.engine import score_token_prompts

        prompts = [[1, 2, 3] + list(range(4, 69)), [1, 2, 3, 9], [1, 2, 3, 8, 7], [1, 2, 3, 6, 5, 4]]
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid), torch.inference_mode():
                model = tiny_model(hybrid)
                expected = torch.cat([model(torch.tensor([p]), logits_to_keep=1).logits[:, -1] for p in prompts])
                calls = []
                hook = model.model.register_forward_pre_hook(
                    lambda module, args, kwargs: calls.append((args[0] if args else kwargs["input_ids"]).shape),
                    with_kwargs=True,
                )
                try:
                    actual, stats = score_token_prompts(model, prompts, candidate_ids=[20, 4, 9])
                finally:
                    hook.remove()
                torch.testing.assert_close(actual, expected[:, [20, 4, 9]], rtol=1e-4, atol=2e-6)
                # 71 actual suffix tokens must not become 260 padded tokens.
                self.assertLessEqual(sum(b * s for b, s in calls[1:]), 142)
                self.assertEqual(stats["forward_calls"], len(calls))
                self.assertTrue(any(b > 1 for b, _ in calls[1:]))

    def test_single_field_and_identical_prompts(self):
        from openjev.engine import score_token_prompts

        with torch.inference_mode():
            model = tiny_model(True)
            for prompts in ([[1, 2, 3]], [[1, 2, 3], [1, 2, 3]]):
                expected, _ = score_token_prompts(model, prompts, parallel=False)
                actual, _ = score_token_prompts(model, prompts, parallel=True)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)

    def test_strict_json_parsing_does_not_accept_duplicate_keys_or_numeric_booleans(self):
        from openjev.engine import parse_json_answer
        from openjev.schema import parse_schema

        fields = parse_schema({"ok": {"type": "boolean"}})
        self.assertEqual(parse_json_answer('{"ok":true}', fields), {"ok": True})
        for text in ('{"ok":true,"ok":false}', '{"ok":1}', '```json\n{"ok":true}\n```', '{"ok":true} extra'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_json_answer(text, fields)


if __name__ == "__main__":
    unittest.main()
