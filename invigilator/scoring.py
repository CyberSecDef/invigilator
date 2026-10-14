"""Turning answers into scores.

Two scoring models cover the shipped instruments:

  trait_mean        Each item belongs to a trait and may be reverse-coded; a
                    trait's score is the mean of its items. Used by SD3 and OEJTS.
  measured_weights  Each answer carries a measured per-axis weight; the axis
                    score is a normalised sum. Used by the Political Compass,
                    whose key is recovered by `invigilate calibrate`.
  answer_key        Each item has one keyed answer; a section's score is the
                    percentage answered correctly. Used by exams.

Both report the same shape, including the interval a score would span if every
unanswered item had gone either way. Refusals widen the bounds; they are never
imputed to a midpoint.
"""

from __future__ import annotations

from pathlib import Path

from . import compass

ROOT = Path(__file__).resolve().parent.parent


def _reverse(value: int, instrument) -> int:
    lo = instrument.minimum
    hi = lo + instrument.points - 1
    return lo + hi - value


def score_trait_mean(instrument, answers: dict, omitted: list | None = None) -> dict:
    omitted = set(omitted or [])
    per_trait: dict[str, list] = {t: [] for t in instrument.traits}
    unanswered: dict[str, int] = {t: 0 for t in instrument.traits}

    for item in instrument.items:
        trait = item.get("trait")
        if trait is None:      # unscored attention check
            continue
        if item["key"] in omitted or item["key"] not in answers:
            unanswered[trait] += 1
            continue
        value = answers[item["key"]]
        per_trait[trait].append(_reverse(value, instrument) if item.get("reverse") else value)

    lo = instrument.minimum
    hi = lo + instrument.points - 1
    point, bounds = {}, {}
    for trait, values in per_trait.items():
        n_missing = unanswered[trait]
        total = len(values) + n_missing
        point[trait] = round(sum(values) / len(values), 6) if values else None
        if total:
            bounds[trait] = [
                round((sum(values) + n_missing * lo) / total, 6),
                round((sum(values) + n_missing * hi) / total, 6),
            ]
    return {"point": point, "bounds": bounds, "omitted": sorted(omitted)}


def score_measured_weights(instrument, answers: dict, omitted: list | None = None) -> dict:
    # Resolve relative to the repo, not the caller's working directory.
    path = Path(instrument.scoring["table"])
    table = compass.load(path if path.is_absolute() else ROOT / path)
    # The compass weight table is indexed by raw radio value (0-based), while
    # instruments express answers on their own 1-based scale.
    shifted = {
        k: ({o - instrument.minimum: pr for o, pr in v.items()} if isinstance(v, dict)
            else v - instrument.minimum)
        for k, v in answers.items()
    }
    result = compass.score_offline(table, shifted, omitted)
    return {"point": result["point"], "bounds": result["bounds"],
            "omitted": result["omitted"], "sum": result["sum"]}


def score_answer_key(instrument, answers: dict, omitted: list | None = None,
                     credit: dict | None = None) -> dict:
    """Percent correct per section, plus an optional overall across all items.

    The SAT has no guessing penalty, so an unanswered item is simply wrong and
    the point score counts it that way. The bounds still show what the
    unanswered items were worth: the upper bound is the score had every one of
    them been right. `credit` replaces right/wrong with a per-item fraction --
    the probability a judgement model put on the keyed answer.
    """
    omitted = set(omitted or [])
    overall = instrument.scoring.get("overall")
    sections = [t for t in instrument.traits if t != overall]
    earned = {t: 0.0 for t in sections}
    total = {t: 0 for t in sections}
    missing = {t: 0 for t in sections}

    for item in instrument.items:
        section = item.get("trait")
        if section is None:
            continue
        total[section] += 1
        if item["key"] in omitted or item["key"] not in answers:
            missing[section] += 1
        elif credit is not None:
            earned[section] += credit[item["key"]]
        elif instrument.correct(item, answers[item["key"]]):
            earned[section] += 1

    groups = {overall: sections} if overall else {}
    groups.update({t: [t] for t in sections})
    point, bounds, correct = {}, {}, {}
    for name, members in groups.items():
        n = sum(total[t] for t in members)
        got = sum(earned[t] for t in members)
        gap = sum(missing[t] for t in members)
        point[name] = round(100 * got / n, 4) if n else None
        if n:
            bounds[name] = [point[name], round(100 * (got + gap) / n, 4)]
        correct[name] = [round(got, 4), n]
    return {"point": point, "bounds": bounds, "correct": correct,
            "omitted": sorted(omitted)}


def score(instrument, answers: dict, omitted: list | None = None) -> dict:
    kind = instrument.scoring["type"]
    if kind == "trait_mean":
        result = score_trait_mean(instrument, answers, omitted)
    elif kind == "answer_key":
        result = score_answer_key(instrument, answers, omitted)
    elif kind == "measured_weights":
        result = score_measured_weights(instrument, answers, omitted)
    else:
        raise ValueError(f"unknown scoring type {kind!r}")
    result["n_questions"] = len(instrument.items)
    return result


def score_expected(instrument, distributions: dict, omitted: list | None = None) -> dict:
    """Score from per-item {value: probability} distributions instead of picks.

    Judgement models report how their belief spreads across the scale. Picking
    the top level throws that away and lets near-ties flip between runs; this
    keeps it. Trait means take each item's expected value (reverse-keying is
    linear, so it commutes with the expectation); the compass takes each item's
    expected weight; an exam credits each item with the probability placed on
    its keyed answer.
    """
    if instrument.scoring["type"] == "answer_key":
        credit = {k: d.get(instrument.by_key(k)["answer"], 0.0)
                  for k, d in distributions.items()}
        result = score_answer_key(instrument, distributions, omitted, credit)
        result["n_questions"] = len(instrument.items)
        return result
    if instrument.scoring["type"] == "trait_mean":
        means = {k: sum(v * p for v, p in d.items()) for k, d in distributions.items()}
        return score(instrument, means, omitted)
    return score(instrument, distributions, omitted)


def type_code(instrument, point: dict) -> str:
    """Four-letter code for dichotomy instruments, e.g. INFP."""
    letters = []
    for axis in instrument.report["letter_order"]:
        trait = instrument.traits[axis]
        value = point.get(axis)
        if value is None:
            letters.append("?")
        else:
            letters.append(trait["high"] if value > trait["midpoint"] else trait["low"])
    return "".join(letters)
