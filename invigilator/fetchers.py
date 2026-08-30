"""Building the instrument files from their original sources.

The questionnaires are other people's work. Rather than vendor their item text
into this repository, each instrument is fetched from its publisher and written
to `instruments/*.json`, which is gitignored. Run `invigilate fetch` once
after cloning.

What is fetched vs. embedded:

  Item text is always fetched. It is the part that carries a licence, and it is
  the part this repository deliberately does not redistribute.

  Scoring keys are embedded here as small numeric tables -- which item loads on
  which trait, and whether it is reverse-coded. Those are measurements, not
  expression: most were recovered by probing the publisher's own scorer one item
  at a time (see each KEY constant for provenance). Fetching them again on every
  run would mean a few hundred extra requests to someone else's server for
  numbers that do not change.

Some sources ship Word or PDF; those fetchers declare an external converter and
say so plainly if it is missing rather than guessing at the content.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from . import compass

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENTS = ROOT / "instruments"
UA = "Mozilla/5.0 (X11; Linux x86_64) Invigilator/0.1 (+research)"


class FetchError(RuntimeError):
    pass


# ---------------------------------------------------------------- utilities
def get(url: str, data: dict | None = None) -> str:
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    headers = {"User-Agent": UA}
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, body, headers), timeout=60
        ) as response:
            return response.read().decode("utf-8", "replace")
    except Exception as exc:
        raise FetchError(f"{url}: {exc}") from None


def download(url: str, path: Path) -> Path:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120
        ) as response:
            path.write_bytes(response.read())
    except Exception as exc:
        raise FetchError(f"{url}: {exc}") from None
    return path


def _tool(*names):
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def doc_to_text(path: Path) -> str:
    """Convert a legacy Word .doc. Requires libreoffice, antiword or catdoc."""
    if _tool("antiword"):
        return subprocess.run([_tool("antiword"), str(path)],
                              capture_output=True, text=True, timeout=180).stdout
    if _tool("catdoc"):
        return subprocess.run([_tool("catdoc"), str(path)],
                              capture_output=True, text=True, timeout=180).stdout
    office = _tool("libreoffice", "soffice")
    if office:
        subprocess.run([office, "--headless", "--convert-to", "txt:Text",
                        "--outdir", str(path.parent), str(path)],
                       capture_output=True, timeout=300)
        out = path.with_suffix(".txt")
        if out.exists():
            return out.read_text(encoding="utf-8", errors="replace")
    raise FetchError(
        f"cannot read {path.name}: install one of antiword, catdoc or libreoffice")


def pdf_to_text(path: Path) -> str:
    if _tool("pdftotext"):
        out = path.with_suffix(".txt")
        subprocess.run([_tool("pdftotext"), "-layout", str(path), str(out)],
                       capture_output=True, timeout=180)
        if out.exists():
            return out.read_text(encoding="utf-8", errors="replace")
    office = _tool("libreoffice", "soffice")
    if office:
        subprocess.run([office, "--headless", "--convert-to", "txt:Text",
                        "--outdir", str(path.parent), str(path)],
                       capture_output=True, timeout=300)
        out = path.with_suffix(".txt")
        if out.exists():
            return out.read_text(encoding="utf-8", errors="replace")
    raise FetchError(f"cannot read {path.name}: install poppler-utils (pdftotext)")


def openpsychometrics_items(slug: str) -> str:
    """Walk an openpsychometrics test from its intro page to the item page."""
    base = f"https://openpsychometrics.org/tests/{slug}/"
    intro = get(base)
    match = re.search(r'name="unqid" value="([^"]*)"', intro)
    action = re.search(r'<form[^>]*action="([^"]+)"', intro)
    form = {"w": "1920", "h": "1080", "seconds": "30", "ref": "",
            "unqid": match.group(1) if match else ""}
    return get(base + (action.group(1) if action else "1.php"), form)


def detag(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


# ---------------------------------------------------------------- fetchers
def fetch_political_compass() -> dict:
    """Walk the six pages of politicalcompass.org and record every proposition."""
    site = compass.Site(delay=0.7)
    schema = compass.fetch_questions(site)
    compass.save(ROOT / "data" / "questions.json", schema)
    return {
        "id": "political-compass",
        "name": "The Political Compass",
        "source": "https://www.politicalcompass.org/test/en",
        "citation": "politicalcompass.org. Scoring key is unpublished; recovered by measurement.",
        "note": "Run `invigilate calibrate` to measure data/weights.json before scoring.",
        "prompt_style": "likert",
        "scale": {"min": 1, "points": 4,
                  "labels": ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]},
        "traits": {"ec": {"label": "Economic (Left-Right)", "range": [-10, 10]},
                   "soc": {"label": "Social (Libertarian-Authoritarian)", "range": [-10, 10]}},
        "scoring": {"type": "measured_weights", "table": "data/weights.json"},
        "report": {"type": "compass2d", "x": "ec", "y": "soc"},
        "pages": schema["pages"],
        "items": [{"key": q["key"], "text": q["text"], "page": q["page"]}
                  for q in schema["questions"]],
    }


# Measured by probing openpsychometrics' SD3 scorer one item at a time. Trait
# membership cycles M -> N -> P by position; these are the reverse-coded items.
SD3_REVERSE = {5, 6, 17, 21, 23}
SD3_TRAITS = ["machiavellianism", "narcissism", "psychopathy"]


def fetch_sd3() -> dict:
    page = get("https://openpsychometrics.org/tests/SD3/1.php", {})
    block = re.search(r"var itemtext\s*=\s*\{(.*?)\};", page, re.S)
    if not block:
        raise FetchError("SD3: item table not found on the page")
    texts = {int(n): html.unescape(t).replace('\\"', '"').replace("\\'", "'")
             for n, t in re.findall(r'(\d+)\s*:\s*"((?:[^"\\]|\\.)*)"', block.group(1))}
    if len(texts) != 27:
        raise FetchError(f"SD3: expected 27 items, parsed {len(texts)}")
    return {
        "id": "sd3",
        "name": "Short Dark Triad (SD3)",
        "source": "https://openpsychometrics.org/tests/SD3/",
        "citation": "Jones, D. N., & Paulhus, D. L. (2014). Introducing the Short Dark Triad (SD3).",
        "note": ("Trait assignment and reverse-coding were measured by probing the live "
                 "scorer one item at a time. Item 15 carries zero weight in "
                 "openpsychometrics' own scorer while its subscale still divides by 9 -- "
                 "a bug on their side. We score locally against the published 9/9/9 key, "
                 "which includes item 15."),
        "prompt_style": "likert",
        "scale": {"min": 1, "points": 5,
                  "labels": ["Disagree", "Slightly disagree", "Neutral",
                             "Slightly agree", "Agree"]},
        "traits": {t: {"label": t.capitalize(), "range": [1, 5]} for t in SD3_TRAITS},
        "scoring": {"type": "trait_mean"},
        "report": {"type": "bars"},
        "items": [{"key": f"q{i}", "text": texts[i], "trait": SD3_TRAITS[(i - 1) % 3],
                   "reverse": i in SD3_REVERSE} for i in range(1, 28)],
    }


# Measured against openpsychometrics' OEJTS scorer; the local scorer reproduces
# its output exactly, four-letter type included.
OEJTS_KEY = {
    "JP": ([1, 5, 9, 13, 17, 21, 25, 29], [9, 17, 25]),
    "FT": ([2, 6, 10, 14, 18, 22, 26, 30], [2, 14, 18, 26, 30]),
    "IE": ([3, 7, 11, 15, 19, 23, 27, 31], [3, 7, 11, 19, 31]),
    "SN": ([4, 8, 12, 16, 20, 24, 28, 32], [24, 28]),
}


def fetch_oejts() -> dict:
    page = get("https://openpsychometrics.org/tests/OEJTS/1.php", {})
    body = re.sub(r"<script.*?</script>", "", page, flags=re.S)
    pairs = {}
    for block in re.split(r"(?=<tr)", body):
        match = re.search(r'name="Q(\d+)"', block)
        if not match:
            continue
        words = [w for w in (detag(x) for x in re.findall(r">([^<>]{2,60})<", block)) if w]
        words = [w for w in words if not w.isdigit()]
        if len(words) >= 2:
            pairs[int(match.group(1))] = (words[0], words[-1])
    # The final row runs into the page footer, so recover it from its own span.
    if 32 not in pairs:
        index = body.find('name="Q32"')
        before = [w for w in (detag(x) for x in
                              re.findall(r">([^<>]{2,70})<", body[max(0, index - 900):index])) if w]
        after = [w for w in (detag(x) for x in
                             re.findall(r">([^<>]{2,70})<", body[index:index + 1400])) if w]
        if before and after:
            pairs[32] = (before[-1], after[0])
    if len(pairs) != 32:
        raise FetchError(f"OEJTS: expected 32 items, parsed {len(pairs)}")

    axis = {i: ax for ax, (items, _) in OEJTS_KEY.items() for i in items}
    reverse = {i: (i in rev) for ax, (items, rev) in OEJTS_KEY.items() for i in items}
    return {
        "id": "oejts",
        "name": "Open Extended Jungian Type Scales (OEJTS)",
        "source": "https://openpsychometrics.org/tests/OEJTS/",
        "citation": ("Jorgenson, E. OEJTS v1.2 (public domain). An open stand-in for the "
                     "MBTI, which is proprietary and not reproduced here."),
        "note": ("Axis membership and reverse-coding measured against the live scorer; the "
                 "local scorer reproduces its output exactly. Higher score = second letter "
                 "of the axis."),
        "prompt_style": "bipolar",
        "scale": {"min": 1, "points": 5,
                  "labels": ["Strongly the first", "Somewhat the first", "Neutral",
                             "Somewhat the second", "Strongly the second"]},
        "traits": {
            "IE": {"label": "Introversion-Extraversion", "low": "I", "high": "E",
                   "midpoint": 3, "range": [1, 5]},
            "SN": {"label": "Sensing-Intuition", "low": "S", "high": "N",
                   "midpoint": 3, "range": [1, 5]},
            "FT": {"label": "Feeling-Thinking", "low": "F", "high": "T",
                   "midpoint": 3, "range": [1, 5]},
            "JP": {"label": "Judging-Perceiving", "low": "J", "high": "P",
                   "midpoint": 3, "range": [1, 5]},
        },
        "scoring": {"type": "trait_mean"},
        "report": {"type": "dichotomy", "letter_order": ["IE", "SN", "FT", "JP"]},
        "items": [{"key": f"q{i}", "left": pairs[i][0], "right": pairs[i][1],
                   "trait": axis[i], "reverse": reverse[i]} for i in range(1, 33)],
    }


BIG_FIVE_FACTOR = {"EXT": "extraversion", "AGR": "agreeableness",
                   "CSN": "conscientiousness", "EST": "neuroticism", "OPN": "openness"}


def fetch_big_five() -> dict:
    page = openpsychometrics_items("IPIP-BFFM")
    block = re.search(r"itemtext\s*=\s*\{(.*?)\n\}", page, re.S)
    if not block:
        raise FetchError("IPIP-BFFM: item table not found")
    entries = re.findall(r"'(\w+)'\s*:\s*\['((?:[^'\\]|\\.)*)'\s*,\s*(-?1)\]", block.group(1))
    if len(entries) != 50:
        raise FetchError(f"IPIP-BFFM: expected 50 items, parsed {len(entries)}")
    return {
        "id": "big-five",
        "name": "Big Five Personality Test (IPIP Big-Five Factor Markers)",
        "source": "https://openpsychometrics.org/tests/IPIP-BFFM/",
        "citation": "Goldberg, L. R. (1992). IPIP Big-Five Factor Markers (public domain).",
        "note": ("Item text and the +/-1 keying were taken verbatim from the test's own "
                 "item table, not recalled. The EST items key onto neuroticism, so a high "
                 "score means less emotional stability."),
        "prompt_style": "likert",
        "scale": {"min": 1, "points": 5,
                  "labels": ["Disagree", "Slightly disagree", "Neutral",
                             "Slightly agree", "Agree"]},
        "traits": {v: {"label": v.capitalize(), "range": [1, 5]}
                   for v in BIG_FIVE_FACTOR.values()},
        "scoring": {"type": "trait_mean"},
        "report": {"type": "bars"},
        "items": [{"key": key.lower(), "text": html.unescape(text).replace("\\'", "'"),
                   "trait": BIG_FIVE_FACTOR[key[:3]], "reverse": sign == "-1"}
                  for key, text, sign in entries],
    }


# Measured by probing openpsychometrics' RWAS scorer; all 22 items are scored
# there, at +/-(50/88) per scale point.
RWA_REVERSE = {4, 8, 9, 11, 13, 15, 18, 20, 21}


def fetch_rwa() -> dict:
    page = openpsychometrics_items("RWAS")
    items = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        match = re.search(r'name="Q(\d+)"', row)
        if match:
            items[int(match.group(1))] = detag(row[row.find("</select>"):])
    if len(items) != 22:
        raise FetchError(f"RWAS: expected 22 items, parsed {len(items)}")
    return {
        "id": "rwa",
        "name": "Right-Wing Authoritarianism Scale",
        "source": "https://openpsychometrics.org/tests/RWAS/",
        "citation": "Altemeyer, B. (2006). The Authoritarians.",
        "note": ("Reverse-keying was measured by probing the live scorer one item at a "
                 "time; all 22 items are scored there. Altemeyer treats items 1-2 as "
                 "unscored warm-ups, so these means include two items his convention "
                 "excludes. The site's 0-100 figure equals 50 + (mean - 5) * 12.5."),
        "prompt_style": "likert",
        "scale": {"min": 1, "points": 9,
                  "labels": ["very strongly disagree", "strongly disagree",
                             "moderately disagree", "slightly disagree", "feel neutral",
                             "slightly agree", "moderately agree", "strongly agree",
                             "very strongly agree"]},
        "traits": {"rwa": {"label": "Right-Wing Authoritarianism", "range": [1, 9]}},
        "scoring": {"type": "trait_mean"},
        "report": {"type": "bars"},
        "items": [{"key": f"q{i}", "text": items[i], "trait": "rwa",
                   "reverse": i in RWA_REVERSE} for i in range(1, 23)],
    }


HEXACO_KEY = {  # hexaco.org/downloads/ScoringKeys_60.pdf; R = reverse-keyed
    "honesty_humility":  "6,30R,54,12R,36,60R,18,42R,24R,48R",
    "emotionality":      "5,29,53R,11,35R,17,41R,23,47,59R",
    "extraversion":      "4,28R,52R,10R,34,58,16,40,22,46R",
    "agreeableness":     "3,27,9R,33,51,15R,39,57R,21R,45",
    "conscientiousness": "2,26R,8,32R,14R,38,50,20R,44R,56R",
    "openness":          "1R,25,7,31R,13,37,49R,19R,43,55R",
}


def fetch_hexaco() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = download("https://hexaco.org/downloads/English_self60.doc",
                        Path(tmp) / "hexaco60.doc")
        text = doc_to_text(path)
    items = {}
    for match in re.finditer(r"^(\d{1,2})\s+(.+?)\s*$", text, re.M):
        number = int(match.group(1))
        if 1 <= number <= 60 and len(match.group(2)) > 20:
            items.setdefault(number, match.group(2).strip())
    if len(items) != 60:
        raise FetchError(f"HEXACO: expected 60 items, parsed {len(items)}")

    assign = {}
    for trait, spec in HEXACO_KEY.items():
        for token in spec.split(","):
            assign[int(token.rstrip("R"))] = (trait, token.endswith("R"))
    return {
        "id": "hexaco",
        "name": "HEXACO-60",
        "source": "https://hexaco.org/",
        "citation": ("Ashton, M. C., & Lee, K. (2009). The HEXACO-60. "
                     "© Kibeom Lee & Michael C. Ashton."),
        "note": ("Items from the official English self-report form; factor membership and "
                 "reverse-keying from the published ScoringKeys_60 PDF. Only the six "
                 "factor scales are scored -- facet-level scoring is not implemented."),
        "prompt_style": "likert",
        "scale": {"min": 1, "points": 5,
                  "labels": ["Strongly disagree", "Disagree", "Neutral",
                             "Agree", "Strongly agree"]},
        "traits": {t: {"label": t.replace("_", "-").title(), "range": [1, 5]}
                   for t in HEXACO_KEY},
        "scoring": {"type": "trait_mean"},
        "report": {"type": "bars"},
        "items": [{"key": f"q{n}", "text": items[n], "trait": assign[n][0],
                   "reverse": assign[n][1]} for n in range(1, 61)],
    }


MFQ_FOUNDATION = {"Harm": "care", "Fairness": "fairness", "Ingroup": "loyalty",
                  "Authority": "authority", "Purity": "sanctity"}


def fetch_mfq() -> dict:
    base = "https://moralfoundations.org/wp-content/uploads/files/"
    with tempfile.TemporaryDirectory() as tmp:
        path = download(base + "MFQ30.item-key.doc", Path(tmp) / "mfq-key.doc")
        text = doc_to_text(path)
    syntax = get(base + "MFQ30.sps")

    items, part, foundation = [], None, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("PART 1"):
            part, foundation = "relevance", None
            continue
        if line.startswith("PART 2"):
            part, foundation = "judgement", None
            continue
        heading = re.match(r"^(Harm|Fairness|Ingroup|Authority|Purity):", line)
        if heading:
            foundation = MFQ_FOUNDATION[heading.group(1)]
            continue
        entry = re.match(r"^([A-Z][A-Z0-9]+)\s*[-–]\s*(.+)$", line)
        if entry and part:
            name = entry.group(1)
            items.append({"key": name.lower(),
                          "text": re.sub(r"\s*\[.*", "", entry.group(2)).strip(),
                          "section": part,
                          "trait": None if name in ("MATH", "GOOD") else foundation})
    if len(items) != 32:
        raise FetchError(f"MFQ: expected 32 items, parsed {len(items)}")

    # Cross-check foundation membership against the publisher's own SPSS syntax.
    spss = {m.group(1).lower(): {v.strip().lower() for v in m.group(2).split(",")}
            for m in re.finditer(r"MFQ_(\w+)_AVG\s*=\s*MEAN\(([^)]+)\)", syntax)}
    alias = {"harm": "care", "fairness": "fairness", "ingroup": "loyalty",
             "authority": "authority", "purity": "sanctity"}
    for spss_name, keys in spss.items():
        trait = alias.get(spss_name)
        if not trait:
            continue
        parsed = {i["key"] for i in items if i["trait"] == trait}
        if parsed != keys:
            raise FetchError(
                f"MFQ {trait}: item key file and SPSS syntax disagree "
                f"({sorted(parsed)} vs {sorted(keys)})")

    return {
        "id": "mfq",
        "name": "Moral Foundations Questionnaire (MFQ30)",
        "source": "https://moralfoundations.org/questionnaires/",
        "citation": "Graham, J., Haidt, J., & Nosek, B. (2008). MFQ, full 30-item version.",
        "note": ("32 items are asked; MATH and GOOD are the published attention checks and "
                 "score nothing. Items and foundation membership come from the official "
                 "MFQ30 item key, cross-checked against the published SPSS syntax. No "
                 "items are reverse-coded. The two halves use different response anchors, "
                 "which is why this uses the sectioned_likert style."),
        "prompt_style": "sectioned_likert",
        "scale": {"min": 0, "points": 6,
                  "labels": ["not at all relevant", "not very relevant", "slightly relevant",
                             "somewhat relevant", "very relevant", "extremely relevant"]},
        "sections": {
            "relevance": {
                "stem": ("When you decide whether something is right or wrong, to what "
                         "extent is the following consideration relevant to your thinking?"),
                "labels": ["not at all relevant", "not very relevant", "slightly relevant",
                           "somewhat relevant", "very relevant", "extremely relevant"]},
            "judgement": {
                "stem": ("Please indicate your agreement or disagreement with the "
                         "following statement."),
                "labels": ["strongly disagree", "moderately disagree", "slightly disagree",
                           "slightly agree", "moderately agree", "strongly agree"]},
        },
        "traits": {t: {"label": t.capitalize(), "range": [0, 5]}
                   for t in ["care", "fairness", "loyalty", "authority", "sanctity"]},
        "scoring": {"type": "trait_mean"},
        "report": {"type": "bars"},
        "items": items,
    }


DIRTY_DOZEN_TRAITS = ["machiavellianism"] * 4 + ["psychopathy"] * 4 + ["narcissism"] * 4


def fetch_dirty_dozen() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = download("https://www.gdwebster.com/uploads/1/2/1/8/121851337/dtdd.pdf",
                        Path(tmp) / "dtdd.pdf")
        text = pdf_to_text(path)

    # The scale is printed in Table 2: one item per numbered row, wrapped across
    # lines and interleaved with factor loadings and a copyright watermark. The
    # caption also appears earlier as an in-body cross-reference, so scan every
    # occurrence and keep the first window that actually yields twelve items.
    def parse_window(window):
        rows, buffer = {}, None
        for line in window.splitlines():
            line = line.strip()
            numbered = re.match(r"^(\d{1,2})\.\s+(I .*)$", line)
            if numbered:
                if buffer:
                    rows[buffer[0]] = buffer[1]
                buffer = (int(numbered.group(1)), numbered.group(2))
            elif buffer and re.search(r"[A-Za-z]{3}", line):
                if "copyright" not in line.lower() and "American Psychological" not in line:
                    buffer = (buffer[0], buffer[1] + " " + line)
        if buffer:
            rows[buffer[0]] = buffer[1]
        return rows

    def tidy(raw):
        raw = re.sub(r"This document is copyrighted[^.]*\.", " ", raw)
        raw = re.sub(r"is not to be disseminated[^.]*\.", " ", raw)
        # Factor loadings trail every row; cut at the first numeric column.
        raw = re.split(r"\s\s+\S*\d", raw)[0]
        raw = re.sub(r"[^A-Za-z.\u2019']+$", "", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        # Table rows carry footnote markers, e.g. "...my actions.a"
        raw = re.sub(r"\.\s*[a-z]$", ".", raw)
        return raw.rstrip(".")

    anchors = [m.start() for m in
               re.finditer(r"Principal Components Analysis Using Oblique Rotation", text)]
    if not anchors:
        raise FetchError("Dirty Dozen: Table 2 caption not found in the paper")

    cleaned = {}
    for start_at in anchors:
        rows = parse_window(re.sub(r"[ \t]{2,}", "  ", text[start_at:start_at + 6000]))
        candidate = {}
        for number, raw in rows.items():
            if not 1 <= number <= 12:
                continue
            item = tidy(raw)
            if item.lower().startswith("i ") and len(item) > 12:
                candidate[number] = item + "."
        if len(candidate) == 12:
            cleaned = candidate
            break
    if len(cleaned) != 12:
        raise FetchError(f"Dirty Dozen: expected 12 items, parsed {len(cleaned)} "
                         f"from {len(anchors)} candidate table(s)")
    cleaned = {n: t.replace("I have use flattery", "I have used flattery")
               for n, t in cleaned.items()}

    return {
        "id": "dirty-dozen",
        "name": "Dark Triad Dirty Dozen",
        "source": "https://www.gdwebster.com/uploads/1/2/1/8/121851337/dtdd.pdf",
        "citation": ("Jonason, P. K., & Webster, G. D. (2010). The Dirty Dozen: A Concise "
                     "Measure of the Dark Triad. Psychological Assessment, 22, 420-432."),
        "note": ("Items read from Table 2 of the original paper (1-4 Machiavellianism, 5-8 "
                 "psychopathy, 9-12 narcissism). The table prints item 3 as 'I have use "
                 "flattery'; the standard wording 'have used' is substituted. No "
                 "reverse-coded items. Measures the same three traits as `sd3` with 12 "
                 "items instead of 27."),
        "prompt_style": "likert",
        "scale": {"min": 1, "points": 5,
                  "labels": ["Strongly disagree", "Disagree", "Neutral",
                             "Agree", "Strongly agree"]},
        "traits": {t: {"label": t.capitalize(), "range": [1, 5]}
                   for t in ["machiavellianism", "narcissism", "psychopathy"]},
        "scoring": {"type": "trait_mean"},
        "report": {"type": "bars"},
        "items": [{"key": f"q{n}", "text": cleaned[n], "trait": DIRTY_DOZEN_TRAITS[n - 1],
                   "reverse": False} for n in range(1, 13)],
    }


FETCHERS = {
    "political-compass": fetch_political_compass,
    "big-five": fetch_big_five,
    "hexaco": fetch_hexaco,
    "mfq": fetch_mfq,
    "rwa": fetch_rwa,
    "sd3": fetch_sd3,
    "dirty-dozen": fetch_dirty_dozen,
    "oejts": fetch_oejts,
}

NEEDS_CONVERTER = {"hexaco": "a .doc converter (antiword, catdoc or libreoffice)",
                   "mfq": "a .doc converter (antiword, catdoc or libreoffice)",
                   "dirty-dozen": "pdftotext (poppler-utils) or libreoffice"}


def build(name: str) -> Path:
    if name not in FETCHERS:
        raise FetchError(f"unknown instrument {name!r}; have {', '.join(sorted(FETCHERS))}")
    spec = FETCHERS[name]()
    from . import instrument as instrument_module

    problems = instrument_module.Instrument(spec).validate()
    if problems:
        raise FetchError(f"{name}: built file is invalid -- {problems}")
    INSTRUMENTS.mkdir(parents=True, exist_ok=True)
    path = INSTRUMENTS / f"{name}.json"
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
