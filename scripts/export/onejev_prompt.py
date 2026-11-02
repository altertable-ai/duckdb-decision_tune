"""The prompt OneJev1.6 reads: a port of flide_onejev 0.1.7's `prompt.py` (PROMPT_VERSION onejev-labels-v2, default
style, no permutation debiasing) and of its `OneJev._render_prompt`.

One prompt is one question. The chat template wraps a fixed system turn and a user turn of

    <state>\n{state}\n</state>\n\nQuestion: {instructions}\n\nOptions:\nA. key: description\n...\n\nAnswer with one letter: A, B.

and the answer is read from the output-head logits of the letters A..Z at the last token of the prompt. Everything
up to the end of the state is the same for every question about one state.
"""
import json

SYSTEM_PROMPT = ("Apply the question to the state. Choose exactly one of the listed options. "
                 "Respond with only its uppercase letter, with no explanation or reasoning.")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NOUL_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_TRUE = "the statement is true / the answer is yes"
NOUL_FALSE = "the statement is false / the answer is no"


def render_entry(entry) -> str:
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def user_turn(state, question) -> tuple[str, list[str]]:
    """The user turn for one question ({type: choice | noul, instructions, criteria}) and the labels its letters
    stand for. Choices of more than 26 options are not handled (flide splits them into chunks)."""
    if question["type"] == "noul":
        instructions = render_entry(question.get("instructions")) or NOUL_INSTRUCTIONS
        labels, descriptions = ["yes", "no"], [NOUL_TRUE, NOUL_FALSE]
    elif question["type"] == "choice":
        instructions = render_entry(question.get("instructions")) or "Which option applies to the state?"
        labels = list(question["criteria"])
        descriptions = [question["criteria"][k] for k in labels]
    else:
        raise ValueError(f"unsupported question type {question['type']!r}")
    if len(labels) > len(LETTERS):
        raise ValueError(f"{len(labels)} options: at most {len(LETTERS)}")
    lines = []
    for letter, name, description in zip(LETTERS, labels, descriptions):
        text = render_entry(description)
        lines.append(f"{letter}. {name}: {text}" if text else f"{letter}. {name}")
    letters = ", ".join(LETTERS[:len(labels)])
    return (f"<state>\n{render_state(state)}\n</state>\n\n"
            f"Question: {instructions}\n\nOptions:\n" + "\n".join(lines) + "\n\n"
            f"Answer with one letter: {letters}."), labels


def render(tokenizer, state, question) -> tuple[str, list[str]]:
    """The whole prompt text, as `apply_chat_template` renders it with thinking disabled."""
    user, labels = user_turn(state, question)
    text = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text, labels


def render_parts(tokenizer, state, question) -> tuple[tuple[str, str, str], list[str]]:
    """`render`'s prompt cut where a cache can end: the system turn up to the user turn's content (the same for every
    prompt), the state block (the same for every question about one state), and the question with its options and the
    assistant turn's opening. The three tokenize to the whole prompt's tokens: each cut follows a newline, which the
    tokenizer does not merge with the text after it."""
    text, labels = render(tokenizer, state, question)
    user, _ = user_turn(state, question)
    state_block = f"<state>\n{render_state(state)}\n</state>\n\n"
    start = text.index(user)
    cut = start + len(state_block)
    return (text[:start], text[start:cut], text[cut:]), labels
