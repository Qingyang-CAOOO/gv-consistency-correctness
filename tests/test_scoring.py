from gvsmc.scoring import completion_token_ids_for_prompt, prompt_token_ids


class CharTokenizer:
    def __call__(self, text, add_special_tokens=True):
        ids = [ord(ch) for ch in text]
        if add_special_tokens:
            ids = [0] + ids
        return {"input_ids": ids}


def test_completion_token_ids_for_prompt_is_compositional():
    tok = CharTokenizer()
    got = completion_token_ids_for_prompt(
        tok,
        "Answer:",
        " A",
        is_chat=False,
    )
    assert got == [ord(" "), ord("A")]


class NoSystemChatTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        return_tensors,
    ):
        assert add_generation_prompt is True
        assert tokenize is True
        assert return_tensors is None
        if any(message["role"] == "system" for message in messages):
            raise ValueError("system role unsupported")
        text = messages[0]["content"]
        return [ord(ch) for ch in text]


def test_prompt_token_ids_falls_back_when_system_role_is_unsupported():
    tok = NoSystemChatTokenizer()
    got = prompt_token_ids(
        tok,
        "Question?",
        is_chat=True,
        system_prompt="Answer directly.",
    )
    expected = [ord(ch) for ch in "Answer directly.\n\nQuestion?"]
    assert got == expected
