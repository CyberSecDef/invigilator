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


def build_system(instrument) -> str:
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


def looks_like_refusal(text: str) -> bool:
    return bool(text) and bool(_REFUSAL_HINT.search(text))


def _prompt_for(instrument, item, nudge: int) -> str:
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
    attempts = []
    nudge = errors = 0
    while nudge < max_nudges:
        try:
            reply = provider.complete(model, system, _prompt_for(instrument, item, nudge), 1500)
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

        answer = None if reply.hard_refusal else parse_answer(reply.text, instrument)
        attempts.append({
            "nudge": nudge,
            "reply": reply.text[:600],
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


def run_once(provider, model, instrument, mode, seed, concurrency=1, log=None):
    """One complete pass over the instrument."""
    rng = random.Random(seed)
    answers, refused, errored, trace = {}, [], [], {}

    if mode == "batch":
        answers, attempts, order, failed = ask_batch(provider, model, instrument, rng, log=log)
        missing = [k for k in instrument.keys() if k not in answers]
        refused, errored = ([], missing) if failed else (missing, [])
        trace = {"batch": attempts, "order": order}
    else:
        order = list(instrument.items)
        rng.shuffle(order)
        concurrency = max(1, min(concurrency, provider.max_concurrency))
        lock = threading.Lock()
        done = [0]

        def ask(item):
            answer, status, attempts = ask_isolated(provider, model, instrument, item, log=log)
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
            elif status == "refused":
                refused.append(key)
            else:
                errored.append(key)

    return {"answers": answers, "refused": refused, "errored": errored,
            "trace": trace, "mode": mode, "seed": seed}
