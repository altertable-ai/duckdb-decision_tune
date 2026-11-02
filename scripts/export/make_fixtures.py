"""Reference outputs for test/sql/parity_*.test, from the published vela3-fast code (PyTorch, CPU, fp32).

    uv run python make_fixtures.py

The decisions go through `vela3_encoder.DecisionEncoder.predict`, the code of the model repository, so the
fixtures cover the prompt rendering (ub_d3_common.to_row) as well as the network. The tokenizer cases go through
the same tokenizer the reference uses.
"""
import csv
import json
import random

from export_onnx import REPO_ROOT, load_reference

OUT = REPO_ROOT / "test" / "data" / "parity"
SEP = "\u241e"  # joins keys and descriptions in one CSV cell

TOKENIZER_TEXTS = [
    "", " ", "  ", "   ", " " * 24, " " * 25, " " * 60, "\t", "\n", "\r\n", "\n\n\n", " \n ", "a\n\nb",
    "The order arrived broken.", "The order arrived broken. I want my money back.",
    "hello  world", "hello   world", "hello    world  ", "  leading", "trailing  ", "a \tb", "a\t\tb",
    "don't", "I'll", "we're", "they've", "I'm", "he'd", "it's", "IT'S", "'s", "''", "rock'n'roll", "o'clock",
    "12345", "3.14159", "1,000,000", "v2.0.1", "2026-10-26T09:15:00Z", "$1,299.99", "50%", "No. 7",
    "café", "cafe\u0301", "naïve", "Ångström", "Straße", "ñandú", "Привет, мир!", "Γειά σου", "שלום עולם",
    "مرحبا بالعالم", "नमस्ते दुनिया", "你好，世界", "日本語のテキスト", "한국어 텍스트", "ไทย", "😀👍🏽🇫🇷",
    "e-mail: someone@example.com", "https://example.com/docs?x=1&y=2#top", "C:\\Users\\me\\file.txt",
    "def f(x):\n    return x ** 2\n", "<html><body>Hi</body></html>", '{"a": 1, "b": [true, null]}',
    "SELECT * FROM t WHERE a = 'b';", "!!!???...", "---", "a—b–c", "«quote» “quote” ‘q’", "\u00a0nbsp\u00a0",
    "ideographic\u3000space", "zero\u200bwidth", "line\u2028separator", "tab\there", "ＦＵＬＬＷＩＤＴＨ",
    "<|endoftext|>", "x<|endoftext|>y", "<|im_start|>user<|im_end|>", "<|sep|>", "a <|sep|> b",
    "<|mask|>", "a <|mask|>", "a   <|mask|> b", "a\n\t<|mask|>", "<|mask|><|mask|>", "x\u3000<|mask|>",
    "[Q] [O] [SEP_SCHEMA] [SEG_user]", "[MASK] [CLS] [SEP]", "<filename>x.py<file_sep>",
    "Billing: charges, refunds, invoices", "Shipping: delivery, lost or damaged packages",
    '{"subject":"Broken mug","message":"The order arrived broken."}',
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "ab" * 200,
]


def random_texts(n, seed=7):
    pool = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + list(" " * 12) +
            list("\t\n\r'\".,;:!?-_()[]{}<>/\\|@#$%^&*+=~`") + list("éèàüßçñøåÉ中文日本語한국어€£¥½²³") +
            ["\u0301", "\u00a0", "\u3000", "\u200b", "😀", "👍🏽", "<|mask|>", "<|endoftext|>", "<|sep|>", "'s",
             "'ll", "  ", "   "])
    rng = random.Random(seed)
    return ["".join(rng.choice(pool) for _ in range(rng.randint(1, 80))) for _ in range(n)]


TEAMS = {"billing": "Billing: charges, refunds, invoices", "shipping": "Shipping: delivery, lost or damaged packages",
         "tech": "Tech support: bugs, crashes, login problems"}
TICKET = {"subject": "Broken mug", "message": "The order arrived broken. I want my money back."}

# (state, question, criteria or None for a yes/no question). A dict state is passed to the extension as JSON.
DECISIONS = [
    ("The order arrived broken.", "Which team should handle this customer message?", TEAMS),
    ("The order arrived broken. I want my money back.", "Is the customer asking for a refund?", None),
    ("What is the weather tomorrow in Paris?", "Which tool should be called?",
     {"get_weather": "Get the weather forecast for a city and date", "send_email": "Send an email to a person",
      "create_event": "Create a meeting or event in the calendar", "none": "No tool: answer the user directly"}),
    (TICKET, "Is the customer asking for a refund or an exchange?", None),
    (TICKET, "What is the tone of the customer?",
     {"negative": "the customer is unhappy or upset", "neutral": "a plain request or question",
      "positive": "the customer is happy or thankful"}),
    (TICKET, "Which team should handle this customer message?", TEAMS),
    ("I was charged twice this month.", "Which team handles this?",
     {"billing": None, "shipping": None, "tech support": None}),
    ("Thanks a lot, everything works great now!", "Does this message ask to delete the account?", None),
    ("", "Is the state empty?", None),
    ("Le colis est arrivé cassé, je veux être remboursé.", "Which team should handle this customer message?",
     {**TEAMS, "none": "None of these: spam, praise, or a message for no team"}),
    ("My app crashes [MASK] on login <|mask|> since the update <|endoftext|>", "Which team?",
     {"billing": None, "tech": None}),
    ("a  b   c    d", "Which letter comes first?", {"a": None, "b": None, "c": None, "d": None}),
    ({"order": 1234, "paid": True, "items": ["mug", "plate"], "note": None}, "Was the order paid?", None),
    ("Login loop: the app keeps asking for my password.", "Which team should handle this customer message?", TEAMS),
]


def main():
    reference, _ = load_reference()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tokenizer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "text", "ids"])
        for i, text in enumerate(TOKENIZER_TEXTS + random_texts(300)):
            w.writerow([i, text, str(reference._ids(text))])
    with open(OUT / "decisions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "kind", "state", "state_is_json", "question", "keys", "descriptions", "probabilities",
                    "prompt_tokens"])
        for i, (state, question, criteria) in enumerate(DECISIONS):
            if criteria is None:
                q = {"type": "noul", "instructions": question}
                keys, descriptions = ["no", "yes"], ["", ""]
            else:
                q = {"type": "choice", "instructions": question, "criteria": criteria}
                keys, descriptions = list(criteria), [d or "" for d in criteria.values()]
            assert not any(SEP in s for s in keys + descriptions)
            answers, tokens = reference.predict(state, {"q": q})
            a = answers["q"]
            probabilities = [1 - a["noul"], a["noul"]] if criteria is None else [a["probabilities"][k] for k in keys]
            is_json = not isinstance(state, str)
            w.writerow([i, "yes_no" if criteria is None else "choose",
                        json.dumps(state, ensure_ascii=False) if is_json else state, str(is_json).lower(), question,
                        SEP.join(keys), SEP.join(descriptions),
                        "[" + ", ".join(repr(float(p)) for p in probabilities) + "]", tokens])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
