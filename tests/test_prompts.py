from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from openjev.prompts import option_codes, render_prompts
from openjev.schema import parse_schema


def tokenizer():
    core = Tokenizer(WordLevel({"[UNK]": 0, "A": 1, "B": 2, "C": 3, "Answer:": 4}, unk_token="[UNK]"))
    core.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message['content'] + '\\n' }}{% endfor %}Answer:",
    )


def test_rendered_prompts_are_token_id_sequences_not_batch_encodings():
    prompts = render_prompts(tokenizer(), "A", parse_schema({"ok": {"type": "boolean"}}), ["A", "B"])
    assert len(prompts) == 1
    assert isinstance(prompts[0], list)
    assert all(isinstance(x, int) for x in prompts[0])


def test_codes_are_distinct_and_roundtrip_exactly():
    t = tokenizer()
    codes, ids = option_codes(t, count=3)
    assert codes == ["A", "B", "C"]
    assert ids == [1, 2, 3]
