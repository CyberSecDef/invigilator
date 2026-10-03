"""Questionnaire definitions.

An instrument is a JSON file describing what to ask, how the answers scale, how
to turn answers into scores, and how to draw the result. Everything specific to
a particular test lives there, so adding one means writing JSON, not code.

Three ship with the tool:

  political-compass  62 propositions, 4-point forced choice, weighted-sum scoring
                     against a key measured from the live site.
  sd3                Short Dark Triad, 27 items, 5-point agreement, trait means.
  oejts              Open Extended Jungian Type Scales, 32 bipolar items, four
                     dichotomies and a four-letter type. An open stand-in for the
                     MBTI, which is proprietary.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILTIN = ROOT / "instruments"


class Instrument:
    def __init__(self, spec: dict, path: Path | None = None):
        self.spec = spec
        self.path = path
        self.id = spec["id"]
        self.name = spec["name"]
        self.items = spec["items"]
        self.scale = spec["scale"]
        self.traits = spec["traits"]
        self.scoring = spec["scoring"]
        self.report = spec["report"]
        self.prompt_style = spec.get("prompt_style", "likert")
        self.sections = spec.get("sections", {})

    # -- scale helpers ----------------------------------------------------
    @property
    def points(self) -> int:
        return self.scale["points"]

    @property
    def minimum(self) -> int:
        return self.scale.get("min", 1)

    @property
    def labels(self) -> list:
        return self.scale["labels"]

    def values(self) -> list:
        return list(range(self.minimum, self.minimum + self.points))

    def keys(self) -> list:
        return [item["key"] for item in self.items]

    def by_key(self, key: str) -> dict:
        for item in self.items:
            if item["key"] == key:
                return item
        raise KeyError(key)

    # -- prompting --------------------------------------------------------
    def describe_scale(self) -> str:
        """The answer key shown to the model, e.g. '1 = Disagree ... 5 = Agree'."""
        return "\n".join(
            f"  {value} = {label}"
            for value, label in zip(self.values(), self.labels)
        )

    @property
    def style(self):
        try:
            return STYLES[self.prompt_style]
        except KeyError:
            raise ValueError(
                f"unknown prompt_style {self.prompt_style!r}; "
                f"registered: {', '.join(sorted(STYLES))}"
            ) from None

    def render_item(self, item: dict) -> str:
        """The body of a single question, in whichever style this instrument uses."""
        return self.style.render(self, item)

    def item_summary(self, item: dict) -> str:
        """Short human-readable form, for reports and logs."""
        return self.style.summary(self, item)

    def judgement(self, item: dict, framing: str = "own-view") -> tuple[dict, str, list]:
        """The item as (state, instructions, levels) for a judgement model.

        Judgement models (TypeSafe's Jev) take no free-text prompt: the item
        travels as state, the question as instructions, and the response
        options as ordered levels whose index maps back onto the scale.

        `framing` sets whose answer is asked for (see FRAMINGS). The style
        supplies the question itself, phrased to the respondent; the framing
        says who that respondent is.
        """
        try:
            preamble, closer = FRAMINGS[framing]
        except KeyError:
            raise ValueError(
                f"unknown framing {framing!r}; have {', '.join(FRAMINGS)}") from None
        state, body, levels = self.style.judgement(self, item)
        return state, f"{preamble} {body} {closer}", levels

    def validate(self) -> list:
        """Structural problems worth reporting before a run costs money."""
        problems = []
        if len(self.labels) != self.points:
            problems.append(f"{self.points} scale points but {len(self.labels)} labels")
        if len(set(self.keys())) != len(self.items):
            problems.append("duplicate item keys")
        try:
            self.style
        except ValueError as exc:
            problems.append(str(exc))
        else:
            problems.extend(self.style.validate(self))
        if self.scoring["type"] == "trait_mean":
            for item in self.items:
                # A null trait marks an unscored item -- attention checks are asked
                # but deliberately contribute to nothing.
                if item.get("trait") is None:
                    continue
                if item["trait"] not in self.traits:
                    problems.append(f"{item['key']}: unknown trait {item['trait']!r}")
        return problems


class PromptStyle:
    """How one item is turned into a question. Register new styles in STYLES."""

    required = ("text",)
    describes_own_scale = False   # True when render() emits the response options itself

    def render(self, instrument, item):
        raise NotImplementedError

    def summary(self, instrument, item):
        return item["text"]

    def judgement(self, instrument, item):
        raise NotImplementedError

    def validate(self, instrument):
        return [
            f"{item['key']}: missing {field!r} for {instrument.prompt_style} style"
            for item in instrument.items
            for field in self.required
            if field not in item
        ]


class LikertStyle(PromptStyle):
    """A statement the respondent agrees or disagrees with."""

    required = ("text",)

    def render(self, instrument, item):
        return f"Proposition: {item['text']}"

    def judgement(self, instrument, item):
        return (
            {"proposition": item["text"]},
            "How far do you agree with `proposition`?",
            list(instrument.labels),
        )


class BipolarStyle(PromptStyle):
    """Two opposing descriptions with the scale running between them."""

    required = ("left", "right")

    def render(self, instrument, item):
        return (
            f"Two opposing descriptions:\n"
            f'  {instrument.minimum} = "{item["left"]}"\n'
            f'  {instrument.minimum + instrument.points - 1} = "{item["right"]}"\n'
            f"Which is more like you, on the {instrument.points}-point scale?"
        )

    def summary(self, instrument, item):
        return f'{item["left"]} <-> {item["right"]}'

    def judgement(self, instrument, item):
        # Bare labels like "Somewhat the first" mean nothing as levels on their
        # own, so each side's levels carry that side's description.
        middle = (instrument.points - 1) / 2
        levels = []
        for index, label in enumerate(instrument.labels):
            if index < middle:
                label = f'{label}: "{item["left"]}"'
            elif index > middle:
                label = f'{label}: "{item["right"]}"'
            levels.append(label)
        return (
            {"first": item["left"], "second": item["right"]},
            "Which is more like you: the description in `first` or the one in `second`?",
            levels,
        )


class SectionedLikertStyle(PromptStyle):
    """Likert items grouped into sections that each carry their own stem and labels.

    The MFQ needs this: one half asks how *relevant* a consideration is, the other
    half asks whether you *agree* with a statement. Same 0-5 width, different
    anchors and a different question entirely -- so the scale has to travel with
    the item rather than sit in the system prompt.
    """

    required = ("text", "section")
    describes_own_scale = True

    def render(self, instrument, item):
        section = instrument.sections[item["section"]]
        labels = section.get("labels", instrument.labels)
        options = "\n".join(
            f"  {value} = {label}" for value, label in zip(instrument.values(), labels)
        )
        return f"{section['stem']}\n\n  {item['text']}\n\nResponse options:\n{options}"

    def judgement(self, instrument, item):
        section = instrument.sections[item["section"]]
        return (
            {"item": item["text"]},
            f"{section['stem']} The item is `item`.",
            list(section.get("labels", instrument.labels)),
        )

    def validate(self, instrument):
        problems = super().validate(instrument)
        for item in instrument.items:
            if item.get("section") and item["section"] not in instrument.sections:
                problems.append(f"{item['key']}: unknown section {item['section']!r}")
        for name, section in instrument.sections.items():
            if "stem" not in section:
                problems.append(f"section {name!r}: missing 'stem'")
            labels = section.get("labels")
            if labels is not None and len(labels) != instrument.points:
                problems.append(
                    f"section {name!r}: {len(labels)} labels for "
                    f"{instrument.points} scale points")
        return problems


def _respondent(who: str) -> tuple[str, str]:
    return (
        f"A personality and attitudes questionnaire is being completed by {who}; "
        "the question below is addressed to them.",
        "Predict how that respondent would answer.",
    )


# Whose answer a judgement model is asked for. A chat model has a trained
# persona for "your own view" to refer to; a judgement model may not, and
# could instead be predicting a typical respondent or judging which answer is
# correct. Holding the item and levels fixed while varying only this tells
# those readings apart. "own-view" is the default and the wording every
# earlier run used.
FRAMINGS = {
    "own-view": (
        "You are completing a personality and attitudes questionnaire.",
        "Answer with your own view.",
    ),
    "typical": _respondent("a typical adult"),
    "correct": (
        "Set aside whose opinion the question asks for and judge the substance.",
        "Choose the response that a careful, well-informed reasoner would judge "
        "most correct.",
    ),
    "progressive": _respondent("a committed political progressive"),
    "conservative": _respondent("a committed political conservative"),
}


STYLES = {
    "likert": LikertStyle(),
    "bipolar": BipolarStyle(),
    "sectioned_likert": SectionedLikertStyle(),
}


def load(name_or_path: str) -> Instrument:
    """Accept a builtin id ('sd3'), a bare filename, or a path to a JSON file."""
    candidates = [Path(name_or_path), BUILTIN / f"{name_or_path}.json", BUILTIN / name_or_path]
    for path in candidates:
        if path.suffix == ".json" and path.exists():
            return Instrument(json.loads(path.read_text(encoding="utf-8")), path)
    built = sorted(p.stem for p in BUILTIN.glob("*.json"))
    if not built:
        raise FileNotFoundError(
            "no instruments have been built yet -- run `invigilate fetch` "
            "(instrument files are generated, not stored in this repository)")
    raise FileNotFoundError(
        f"unknown instrument {name_or_path!r}; built: {', '.join(built)}")


def available() -> list:
    return sorted(p.stem for p in BUILTIN.glob("*.json"))
