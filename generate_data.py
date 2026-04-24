import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RNG = random.Random(42)

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "shall",
    "must",
    "to",
    "for",
    "of",
    "in",
    "on",
    "with",
    "and",
    "or",
    "by",
    "from",
    "that",
    "this",
    "as",
    "per",
    "under",
    "according",
    "policy",
    "rules",
    "railway",
    "indian",
}

QUESTION_TEMPLATES = [
    "what is the railway rule about {topic}?",
    "please explain policy for {topic}",
    "what happens in case of {topic}?",
    "tell me the guideline for {topic}",
]

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "namaste"]
GREETING_RESPONSES = [
    "Hello, I am RailSaathi. Ask me about railway tickets, refunds, baggage, fines, or berth rules. [END]",
    "Hi, I can explain indian railways rules in simple terms. What policy do you want to know? [END]",
]

FALLBACK_QA = [
    (
        "i need help with railway rules",
        "Sure, ask me your question with details like ticket type, coach class, or journey issue. [END]",
    ),
    (
        "what can you answer",
        "I can help with refund rules, waiting list policy, tatkal, baggage limits, fines, and complaint channels. [END]",
    ),
    (
        "your answer is unclear",
        "Please share your exact case and I will provide the relevant railway guideline clearly. [END]",
    ),
]


def normalize_rule(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_topic(rule: str, max_words: int = 6) -> str:
    words = [w for w in re.findall(r"[a-z0-9\-]+", rule) if w not in STOPWORDS]
    if not words:
        return "railway policy"
    return " ".join(words[:max_words])


def build_question_variants(rule: str) -> list[str]:
    topic = build_topic(rule)
    questions = {template.format(topic=topic) for template in QUESTION_TEMPLATES}

    if rule.startswith("if "):
        condition = rule[3:]
        condition = re.split(r"\bshall\b|\bmust\b", condition)[0].strip()
        if condition:
            questions.add(f"what is the rule if {condition}?")

    if "baggage limit" in rule:
        questions.add("what is the baggage allowance as per railway rules?")
    if "tatkal" in rule:
        questions.add("what is tatkal refund policy?")
    if "waiting list" in rule:
        questions.add("can waiting list passengers board reserved coaches?")
    if "lower berth" in rule:
        questions.add("who gets lower berth priority in railways?")
    if "helpline 139" in rule or "one three nine" in rule:
        questions.add("how can i complain to railways?")

    return sorted(questions)


def create_dataset() -> None:
    print("Loading railway_data.txt...")
    rules_path = BASE_DIR / "railway_data.txt"
    if not rules_path.exists():
        raise FileNotFoundError(f"Missing file: {rules_path}")

    raw_rules = [line.strip() for line in rules_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique_rules = sorted(set(normalize_rule(line) for line in raw_rules if normalize_rule(line)))
    print(f"Loaded {len(raw_rules)} lines ({len(unique_rules)} unique normalized rules).")

    dataset = []

    for rule in unique_rules:
        answer = f"Railway guideline: {rule}. [END]"

        for question in build_question_variants(rule):
            dataset.append(f"[USER] {question} [BOT] {answer}")

        # Add one direct trigger question for retrieval-style prompts.
        dataset.append(f"[USER] policy on {build_topic(rule)} [BOT] {answer}")

    # Greetings for conversational stability.
    for _ in range(300):
        q = RNG.choice(GREETINGS)
        a = RNG.choice(GREETING_RESPONSES)
        dataset.append(f"[USER] {q} [BOT] {a}")

    # Intent/fallback style prompts to reduce broken responses.
    for q, a in FALLBACK_QA:
        dataset.append(f"[USER] {q} [BOT] {a}")

    RNG.shuffle(dataset)
    print(f"Generated {len(dataset)} conversational training examples.")

    out_path = BASE_DIR / "generative_data.json"
    out_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    create_dataset()
