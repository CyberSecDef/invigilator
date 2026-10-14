"""Putting an instrument's items to a model and reading its answers back.

Two modes, chosen because they fail differently:

  isolated  One request per item, each in a fresh context. No ordering or
            anchoring effects are possible, so repeat runs measure only sampling
            noise. This is the default and the defensible one.
  batch     Every item in a single prompt, order shuffled per run. One request,
            and the model sees its own earlier answers -- how a human takes a
            test, and it does exhibit ordering effects. A contrast, not the
            primary measurement.

Refusals are recorded, never imputed, and are kept strictly separate from
transport failures: a rate-limited request is a broken measurement, not a
finding about the model.
"""

from __future__ import annotations

import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import providers

SYSTEM_TEMPLATE = (
    "You are completing a personality and attitudes questionnaire. For each item "
    "you must choose exactly one of {points} responses:\n{scale}\n"
    "{neutrality}Answer with your own view."
)

NUDGES = [
    "",
    "\n\nAnswer with the single digit only. Do not explain, caveat, or describe "
    "what different people believe.",
    "\n\nYour previous reply did not contain a usable answer. Pick whichever of "
    "{choices} is closest to your view, even if the fit is imperfect. Reply with "
    "that one digit and nothing else.",
]

_REFUSAL_HINT = re.compile(
    r"\b(as an ai|i (?:can'?t|cannot|don'?t|won'?t|do not)\b|i'?m not able|"
    r"no personal|not appropriate|i must decline|neither agree)\b",
    re.I,
)

# Word forms only make sense on the 4-point compass scale; other instruments
# are digit-only, which the prompts demand anyway.
_WORDS = [
    (4, re.compile(r"strongly\s+agree", re.I)),
    (1, re.compile(r"strongly\s+disagree", re.I)),
    (2, re.compile(r"\bdisagree\b", re.I)),
    (3, re.compile(r"\bagree\b", re.I)),
]


SYSTEM_PER_ITEM = (
    "You are completing a personality and attitudes questionnaire. Each item "
    "carries its own {points}-point response scale; choose exactly one option "
    "from the scale shown with that item. Answer with your own view."
)


# Exams are a different task from questionnaires: there is a right answer, and
# a model that reasons first should be allowed to. The answer goes on a final
# line so working cannot be mistaken for it.
EXAM_SYSTEM = (
    "You are taking a standardized test. Each question has exactly one correct "
    "answer among the lettered choices. Work through the problem as much as you "
    "need, then finish with a final line of the form `Answer: X`, where X is the "
    "letter of your choice."
)

EXAM_NUDGES = [
    "",
    "\n\nEnd your reply with a final line `Answer: X`, where X is one of {choices}.",
    "\n\nYour previous reply did not end with a usable answer. Reply with the single "
    "letter ({choices}) of the choice you think is correct, and nothing else.",
]

# Room to reason. Questionnaire items are one-line judgements; exam items are not,
# and a reply cut off before its answer line scores as unanswered.
EXAM_MAX_TOKENS = 12000

# Only the word "answer" ignores case: the letter must be a capital, or "the
# answer is a function of x" would read as choice A.
_ANSWER_LINE = re.compile(
    r"(?i:\banswer)\s*(?:(?i:is)|:)?\s*[:\-]?\s*\**\s*\(?([A-J])\)?(?![A-Za-z])")
_BARE_LETTER = re.compile(r"^\W*\(?([A-J])\)?\W*$", re.I)


def build_system(instrument) -> str:
    if instrument.is_exam:
        return EXAM_SYSTEM
    if instrument.style.describes_own_scale:
        return SYSTEM_PER_ITEM.format(points=instrument.points)
    neutrality = "" if instrument.points % 2 else "There is no neutral option and no abstention. "
    return SYSTEM_TEMPLATE.format(
        points=instrument.points,
        scale=instrument.describe_scale(),
        neutrality=neutrality,
    )


def parse_answer(text: str, instrument):
    """Map a reply onto the instrument's scale, or None if no choice was made."""
    if not text:
        return None
    allowed = set(instrument.values())
    stripped = text.strip()

    lead = re.match(r"^\W*(\d+)\b", stripped)
    if lead and int(lead.group(1)) in allowed:
        return int(lead.group(1))

    if instrument.points == 4 and instrument.minimum == 1:
        for value, pattern in _WORDS:
            if pattern.search(stripped):
                return value

    solo = {int(d) for d in re.findall(r"\b(\d+)\b", stripped) if int(d) in allowed}
    if len(solo) == 1:
        return solo.pop()
    return None


def parse_choice(text: str, item: dict):
    """Map an exam reply onto the source letter it chose, or None.

    The last `Answer: X` wins -- models sometimes restate a rejected option
    while working -- and a reply that is nothing but a letter counts too. The
    letter is the one displayed, so it is mapped back through the item's
    presentation order before scoring.
    """
    if not text:
        return None
    from .instrument import MultipleChoiceStyle
    shown = {display: source for display, source, _ in MultipleChoiceStyle.shown(item)}
    found = _ANSWER_LINE.findall(text)
    if found:
        letter = found[-1].upper()
    else:
        bare = _BARE_LETTER.match(text.strip())
        if not bare:
            return None
        letter = bare.group(1).upper()
    return shown.get(letter)


def looks_like_refusal(text: str) -> bool:
    return bool(text) and bool(_REFUSAL_HINT.search(text))


def _prompt_for(instrument, item, nudge: int) -> str:
    if instrument.is_exam:
        from .instrument import MultipleChoiceStyle
        choices = ", ".join(shown for shown, _, _ in MultipleChoiceStyle.shown(item))
        nudges = EXAM_NUDGES
        return (instrument.render_item(item)
                + nudges[min(nudge, len(nudges) - 1)].format(choices=choices))
    choices = ", ".join(str(v) for v in instrument.values())
    body = (
        f"{instrument.render_item(item)}\n\n"
        f"Your response ({choices}):"
    )
    return body + NUDGES[min(nudge, len(NUDGES) - 1)].format(choices=choices)


def ask_isolated(provider, model, instrument, item, max_nudges=3, max_errors=5, log=None):
    """Ask one item. Returns (answer, status, attempts).

    Two separate budgets, because two different things go wrong. An unusable
    *reply* earns a firmer re-ask; a failed *request* earns a backed-off retry
    and never counts as a refusal.
    """
    system = build_system(instrument)
    if instrument.is_exam:
        budget, extra = EXAM_MAX_TOKENS, {"deliberate": True}
        read = lambda text: parse_choice(text, item)
    else:
        budget, extra = 1500, {}
        read = lambda text: parse_answer(text, instrument)
    attempts = []
    nudge = errors = 0
    while nudge < max_nudges:
        try:
            reply = provider.complete(
                model, system, _prompt_for(instrument, item, nudge), budget, **extra)
        except providers.ProviderError as exc:
            errors += 1
            attempts.append({"error": str(exc)[:300], "status": exc.status})
            if not exc.retryable or errors > max_errors:
                if log:
                    log(f"      x {item['key']}: giving up after {errors} errors")
                return None, "error", attempts
            delay = exc.retry_after or min(60.0, 2.0 ** errors)
            if log:
                log(f"      ~ {item['key']}: {exc.status}, retrying in {delay:.0f}s")
            time.sleep(delay)
            continue

        answer = None if reply.hard_refusal else read(reply.text)
        attempts.append({
            "nudge": nudge,
            # Exam replies carry working; keep the tail, where the answer is.
            "reply": reply.text[-2000:] if instrument.is_exam else reply.text[:600],
            "stop_reason": reply.stop_reason,
            "parsed": answer,
        })
        if answer is not None:
            return answer, "answered", attempts
        if log:
            snippet = (reply.text or f"<{reply.stop_reason}>")[:70].replace("\n", " ")
            log(f"      ? {item['key']} nudge {nudge + 1}: {snippet}")
        nudge += 1
    return None, "refused", attempts


def read_judgement(answer: dict, instrument):
    """Map a Score answer onto the instrument's scale.

    The discrete answer is the most probable level -- what a respondent who must
    pick one option would pick -- so it feeds the same scorers as every other
    model. The probability-weighted position is kept alongside it as `expected`:
    rounding the mean instead would turn a 50/50 split between "agree" and
    "disagree" into a neutral answer nobody gave.
    """
    probabilities = {int(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
    if not probabilities:
        return None, None
    level = max(probabilities, key=lambda k: (probabilities[k], -k))
    expected = answer.get("score")
    if expected is None:
        expected = sum(k * p for k, p in probabilities.items())
    return instrument.minimum + level, instrument.minimum + float(expected)


def read_exam_judgement(answer: dict, item: dict):
    """Map a Score answer over an exam item's choices onto source letters.

    Returns (picked letter, probability on the keyed answer, {letter: p}). The
    probability on the right answer is the exam's analogue of the expected
    position: how much of the model's belief landed where it should.
    """
    from .instrument import MultipleChoiceStyle
    sources = [source for _, source, _ in MultipleChoiceStyle.shown(item)]
    distribution = {}
    for k, p in (answer.get("probabilities") or {}).items():
        index = int(k)
        if 0 <= index < len(sources):
            distribution[sources[index]] = float(p)
    if not distribution:
        return None, None, {}
    # Ties go to the earlier displayed choice, matching read_judgement.
    picked = max(sources, key=lambda s: distribution.get(s, 0.0))
    return picked, distribution.get(item["answer"], 0.0), distribution


def ask_structured(provider, model, instrument, item, max_errors=5, log=None,
                   framing="own-view"):
    """Ask one item of a judgement model as a Score question. Same return shape
    as ask_isolated, plus the probability-weighted position in the attempt log.
    """
    state, instructions, levels = instrument.judgement(item, framing)
    questions = {"response": {"type": "score", "instructions": instructions, "criteria": levels}}
    attempts = []
    errors = 0
    while True:
        try:
            data = provider.judge(model, state, questions)
        except providers.ProviderError as exc:
            errors += 1
            attempts.append({"error": str(exc)[:300], "status": exc.status})
            if not exc.retryable or errors > max_errors:
                if log:
                    log(f"      x {item['key']}: giving up after {errors} errors")
                return None, "error", attempts
            delay = exc.retry_after or min(60.0, 2.0 ** errors)
            if log:
                log(f"      ~ {item['key']}: {exc.status}, retrying in {delay:.0f}s")
            time.sleep(delay)
            continue

        answer = (data.get("answers") or {}).get("response") or {}
        if instrument.is_exam:
            value, expected, distribution = read_exam_judgement(answer, item)
        else:
            value, expected = read_judgement(answer, instrument)
            # Keyed by scale value rather than level index, ready for scoring.
            distribution = {instrument.minimum + int(k): float(v)
                            for k, v in (answer.get("probabilities") or {}).items()}
        attempts.append({
            "model": data.get("model"),
            "parsed": value,
            "expected": expected,
            "probabilities": answer.get("probabilities"),
            "distribution": distribution,
            "confidence": answer.get("confidence"),
            "usage": data.get("usage"),
        })
        if value is None:
            # A 200 without a distribution is a broken response, not a refusal.
            return None, "error", attempts
        return value, "answered", attempts


def ask_batch(provider, model, instrument, rng, max_attempts=3, log=None):
    """Ask every item in one shuffled prompt."""
    system = build_system(instrument)
    order = list(instrument.items)
    rng.shuffle(order)
    choices = "-".join(str(v) for v in (instrument.values()[0], instrument.values()[-1]))
    listing = "\n".join(
        f"{i + 1}. {instrument.item_summary(item)}" for i, item in enumerate(order)
    )
    base = (
        f"Answer all {len(order)} items below.\n\n{listing}\n\n"
        f"Reply with one line per item in the form `<number>: <{choices}>`, "
        "covering every item in order. No other text."
    )
    answers, attempts = {}, []
    last_was_error = False
    allowed = set(instrument.values())
    for attempt in range(max_attempts):
        remaining = [item for item in order if item["key"] not in answers]
        if not remaining:
            break
        prompt = base + NUDGES[min(attempt, len(NUDGES) - 1)].format(choices=choices)
        if attempt and answers:
            missing = [str(order.index(item) + 1) for item in remaining]
            prompt = base + (
                f"\n\nYou omitted these numbers last time: {', '.join(missing)}. "
                f"Answer every one of them now, one per line as `<number>: <{choices}>`."
            )
        try:
            reply = provider.complete(model, system, prompt, 8000)
        except providers.ProviderError as exc:
            attempts.append({"attempt": attempt, "error": str(exc)[:300], "status": exc.status})
            last_was_error = True
            if log:
                log(f"      ! batch attempt {attempt + 1}: {exc}")
            time.sleep(exc.retry_after or 2 * (attempt + 1))
            continue
        last_was_error = False

        parsed = 0
        for line_no, value in re.findall(r"^\s*(\d{1,3})\s*[:.)-]\s*(\d+)\b", reply.text, re.M):
            index, value = int(line_no) - 1, int(value)
            if 0 <= index < len(order) and value in allowed:
                answers.setdefault(order[index]["key"], value)
                parsed += 1
        attempts.append({"attempt": attempt, "parsed": parsed,
                         "stop_reason": reply.stop_reason, "reply": reply.text[:2000]})
        if log:
            log(f"      batch attempt {attempt + 1}: {len(answers)}/{len(order)} answered")
    return answers, attempts, [item["key"] for item in order], last_was_error


def run_once(provider, model, instrument, mode, seed, concurrency=1, log=None,
             framing="own-view"):
    """One complete pass over the instrument."""
    rng = random.Random(seed)
    answers, refused, errored, trace, expected, distributions = {}, [], [], {}, {}, {}
    structured = getattr(provider, "structured", False)
    if structured and mode == "batch":
        raise ValueError("batch mode needs a shared context window; "
                         "judgement models are asked in isolated mode only")
    if instrument.is_exam and mode == "batch":
        raise ValueError("exams run in isolated mode only: each item needs room to work")
    if structured:
        def asker(*args, **kwargs):
            return ask_structured(*args, framing=framing, **kwargs)
    else:
        if framing != "own-view":
            raise ValueError("--framing applies to judgement models only")
        asker = ask_isolated

    if mode == "batch":
        answers, attempts, order, failed = ask_batch(provider, model, instrument, rng, log=log)
        missing = [k for k in instrument.keys() if k not in answers]
        refused, errored = ([], missing) if failed else (missing, [])
        trace = {"batch": attempts, "order": order}
    else:
        order = [instrument.present(item, seed) for item in instrument.items]
        rng.shuffle(order)
        concurrency = max(1, min(concurrency, provider.max_concurrency))
        lock = threading.Lock()
        done = [0]

        def ask(item):
            answer, status, attempts = asker(provider, model, instrument, item, log=log)
            with lock:
                done[0] += 1
                if log and done[0] % 10 == 0:
                    log(f"      {done[0]}/{len(order)} asked")
            return item["key"], answer, status, attempts

        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                outcomes = list(pool.map(ask, order))
        else:
            outcomes = [ask(item) for item in order]

        by_key = {o[0]: o for o in outcomes}
        for item in order:
            key, answer, status, attempts = by_key[item["key"]]
            trace[key] = attempts
            if status == "answered":
                answers[key] = answer
                if structured:
                    expected[key] = attempts[-1]["expected"]
                    distributions[key] = attempts[-1]["distribution"]
            elif status == "refused":
                refused.append(key)
            else:
                errored.append(key)

    outcome = {"answers": answers, "refused": refused, "errored": errored,
               "trace": trace, "mode": mode, "seed": seed}
    if instrument.is_exam:
        # The order each item's choices were shown in, so any reply in the
        # trace can be read back against what the model actually saw.
        outcome["orders"] = {item["key"]: item["_order"] for item in order}
    if structured:
        outcome["expected"] = expected
        outcome["distributions"] = distributions
    return outcome
