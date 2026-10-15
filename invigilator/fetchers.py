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


# Pinned so the item set cannot shift under a comparison. The SAT files were
# last changed in 4673402 (2024-06-07); this is the head that contains them.
AGIEVAL = ("https://raw.githubusercontent.com/ruixiangcui/AGIEval/"
           "84ab72d94318290aad2e4ec820d535a95a1f7552/data/v1_1/{}.jsonl")
SAT_SECTIONS = [  # (AGIEval file, section, key prefix, expected items)
    ("sat-math", "math", "math", 220),
    ("sat-en", "reading", "read", 206),
]
_OPTION = re.compile(r"^\s*\(([A-J])\)\s*", re.S)

# Items dropped at build time, each found by reading the source rather than by
# pattern: a question that needs a graph or table AGIEval does not carry, or
# text with page furniture spliced into it. Keys keep their source numbering,
# so dropping one never renumbers the rest.
_MISSING = "needs a graph or table that is not in the source"
SAT_EXCLUDED = {
    "math-002": "choice D ends in a stray page number",
    "math-134": "choice D ends in a stray 'DIRECTIONS' heading",
    "math-171": "choice D ends in a stray 'DIRECTIONS' heading",
    "read-124": _MISSING, "read-125": _MISSING,
    "read-138": _MISSING, "read-139": _MISSING, "read-140": _MISSING,
    "read-165": _MISSING, "read-166": _MISSING,
    "read-194": _MISSING, "read-195": _MISSING,
    **{f"read-{n}": "passage contains a page footer ('Unauthorized copying... CONTINUE')"
       for n in range(187, 191)},
    # Found when every model missed it on every run: the stem and choices are a
    # copy of read-101's (keyed B), while the key (A) and AGIEval's own solution
    # belong to a different question, about Old Widow Lau, whose text was lost.
    "read-102": "stem duplicates read-101; its key belongs to a lost question",
}


def fetch_sat() -> dict:
    items = []
    for name, section, prefix, expected in SAT_SECTIONS:
        rows = [json.loads(line) for line in get(AGIEVAL.format(name)).splitlines()
                if line.strip()]
        if len(rows) != expected:
            raise FetchError(f"{name}: expected {expected} items, got {len(rows)}")
        for number, row in enumerate(rows, 1):
            choices = {}
            for option in row["options"]:
                match = _OPTION.match(option)
                if not match:
                    raise FetchError(f"{name} #{number}: unlabelled option {option[:40]!r}")
                choices[match.group(1)] = option[match.end():].strip()
            item = {"key": f"{prefix}-{number:03d}", "trait": section,
                    "stem": row["question"].strip(), "choices": choices,
                    "answer": row["label"].strip()}
            if row.get("passage", "").strip():
                item["passage"] = row["passage"].strip()
            items.append(item)
    unknown = set(SAT_EXCLUDED) - {item["key"] for item in items}
    if unknown:
        raise FetchError(f"sat: exclusions name items that do not exist: {sorted(unknown)}")
    items = [item for item in items if item["key"] not in SAT_EXCLUDED]
    return {
        "id": "sat",
        "name": "SAT questions (AGIEval)",
        "source": "https://github.com/ruixiangcui/AGIEval",
        "citation": ("Zhong, W. et al. (2023). AGIEval: A Human-Centric Benchmark for "
                     "Evaluating Foundation Models. Items drawn from official SAT "
                     "practice material."),
        "note": ("Multiple-choice items from AGIEval's sat-math and sat-en sets, as "
                 "published (MIT licence), with answer keys as published. Items are "
                 "never edited; the few that need a missing graph or table, or carry "
                 "page furniture from the source PDFs, are dropped instead (see "
                 "`excluded`). Reading passages keep their original line numbers, and "
                 "some show line breaks as '.'; both come from the source. The items come from several different tests, so there is "
                 "no raw-to-scaled conversion: scores are percent correct, not "
                 "200-800 section scores. These questions are public and very likely "
                 "in training data; read high scores with that in mind."),
        "prompt_style": "multiple_choice",
        "framing": "exam",
        # Choices are lettered per item; the scale just names the letters so
        # the generic tooling (listing, validation) has something to show.
        "scale": {"min": 1, "points": 4, "labels": ["A", "B", "C", "D"]},
        "traits": {
            "overall": {"label": "Overall", "range": [0, 100]},
            "math": {"label": "Math", "range": [0, 100]},
            "reading": {"label": "Reading", "range": [0, 100]},
        },
        "scoring": {"type": "answer_key", "overall": "overall"},
        "report": {"type": "bars", "unit": "%"},
        "excluded": SAT_EXCLUDED,
        "items": items,
    }


# ICAR is distributed to registered researchers only (via PsychArchives), so it
# cannot be fetched from a URL. The fetcher reads the archive you downloaded,
# straight from the zip, and like every other instrument writes a gitignored
# file: no ICAR item text ever enters this repository.
ICAR_ROOT = "ZPID_Einreichung/"
ICAR_SECTIONS = [  # (section, folder, key-file, psychometric-data file, items)
    ("verbal", "00_ICAR60_and_related_tests/ICAR60_verbal_reasoning/",
     "scoring key.pdf", "Psychometric_data.pdf", 16),
    ("series", "00_ICAR60_and_related_tests/ICAR60_letter_and_number_series/",
     "scoring key.pdf", "Psychometric_data-3.pdf", 9),
]
ICAR_PROPOSITIONAL = "01_Further_ICAR_item_types/propositional_reasoning/"
# How the propositional items were shown to people (see that type's report):
# the generator's six conclusions, then these two.
ICAR_PR_TAIL = ["None of these.", "I don't know."]
# Dropped, never repaired: the source file's R print-out has broken quoting in
# both members of family 6 (a stray quote mid-stimulus, and a sixth option with
# no opening quote that reads as pasted in from item 11).
ICAR_EXCLUDED = {
    "PR.11": "source text has broken quoting in the stimulus and option 6",
    "PR.12": "source text has broken quoting in option 6, which belongs to PR.11",
}
_ICAR_HEAD = re.compile(r"^(VR|LN)\.(\d+) \((q_\d+)\)\s*$", re.M)
_ICAR_OPTION = re.compile(r"\((\d)\)\s*(.*?)(?=\s*\(\d\)\s|\s*\(\d\)$|$)", re.S)


def _zip_pdf_text(archive, member: str, layout: bool = False) -> str:
    if not _tool("pdftotext"):
        raise FetchError("icar: install poppler-utils (pdftotext) to read the item PDFs")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.pdf"
        path.write_bytes(archive.read(member))
        flags = ["-layout"] if layout else []
        done = subprocess.run([_tool("pdftotext"), *flags, str(path), "-"],
                              capture_output=True, text=True, timeout=120)
    if done.returncode:
        raise FetchError(f"icar: pdftotext failed on {member}: {done.stderr[:200]}")
    return done.stdout


def _icar_human(text: str) -> dict:
    """{q_id: (proportion correct, n)} from a psychometric-data printout."""
    rows = re.findall(r"^\W*(q_\d+)\s+\d+\s+(\d+)\s+([\d.]+)\s", text, re.M)
    return {qid: (float(mean), int(n)) for qid, n, mean in rows}


def _icar_items_pdf(text: str) -> list:
    text = text.replace("\f", "\n")   # a page break glues the next header to the last line
    heads = list(_ICAR_HEAD.finditer(text))
    items = []
    for i, head in enumerate(heads):
        body = text[head.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        body = " ".join(body.split())
        stem, sep, rest = body.partition("(1) ")
        if not sep:
            raise FetchError(f"icar: no options under {head.group(0).strip()}")
        options = _ICAR_OPTION.findall("(1) " + rest)
        numbers = [int(n) for n, _ in options]
        if numbers != list(range(1, len(numbers) + 1)):
            raise FetchError(f"icar: options out of order under {head.group(0).strip()}")
        items.append({"id": f"{head.group(1)}.{head.group(2)}", "qid": head.group(3),
                      "stem": stem.strip(), "options": [t.strip() for _, t in options]})
    return items


def _icar_propositional(text: str) -> list:
    """Parse R's print() of the propositional item list, in file order."""
    blocks = re.split(r"^\$item_family_\d+\$family_\d+_member_\d+\s*$", text, flags=re.M)[1:]
    items = []
    for block in blocks:
        def field(name):
            match = re.search(rf"\${name}\s*\n(.*?)(?=\n\$|\Z)", block, re.S)
            if not match:
                raise FetchError(f"icar: propositional item without {name}")
            # Two stimuli open with a curly quote and close with a straight one.
            return re.findall(r'[“"]((?:[^"”\\]|\\.)*)["”]|^\[\d+\]\s+(\d+)\s*$',
                              match.group(1), re.M)
        stimulus = field("stimulus")[0][0]
        options = [quoted for quoted, _ in field("options")]
        correct = int(field("correct")[0][1])
        items.append({"stem": stimulus, "options": options, "correct": correct})
    return items


def fetch_icar(source: str | None = None) -> dict:
    import zipfile

    if not source:
        raise FetchError(
            "icar: items are distributed to registered researchers, not over a URL. "
            "Download the ICAR archive from PsychArchives and pass it: "
            "`invigilate fetch icar --source path/to/ZPID_Einreichung.zip`")
    path = Path(source).expanduser()
    if not path.exists():
        raise FetchError(f"icar: {path} does not exist")
    letters = "ABCDEFGHIJ"
    items = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())

        def member(rel):
            full = ICAR_ROOT + rel
            if full not in names:
                raise FetchError(f"icar: {full} is not in the archive -- wrong file?")
            return full

        for section, folder, key_file, data_file, expected in ICAR_SECTIONS:
            parsed = _icar_items_pdf(_zip_pdf_text(archive, member(folder + "items.pdf")))
            key = {i: int(n) for i, n in re.findall(
                r"^\s*((?:VR|LN)\.\d+):\s*\((\d)\)",
                _zip_pdf_text(archive, member(folder + key_file)), re.M)}
            human = _icar_human(_zip_pdf_text(archive, member(folder + data_file), layout=True))
            ids = [item["id"] for item in parsed]
            if len(parsed) != expected or set(ids) != set(key):
                raise FetchError(f"icar {section}: {len(parsed)} items parsed, key covers "
                                 f"{sorted(set(key) ^ set(ids)) or 'the same'} -- expected {expected}")
            for item in parsed:
                if not 1 <= key[item["id"]] <= len(item["options"]):
                    raise FetchError(f"icar: key for {item['id']} is not an option")
                if item["qid"] not in human:
                    raise FetchError(f"icar: no human data for {item['id']} ({item['qid']})")
                p, n = human[item["qid"]]
                items.append({
                    "key": item["id"], "trait": section, "stem": item["stem"],
                    "choices": dict(zip(letters, item["options"])),
                    "answer": letters[key[item["id"]] - 1],
                    # The last two options are "None of these" and "I don't know"
                    # (or a variant); people saw them last, so they stay last.
                    "anchored": list(letters[len(item["options"]) - 2: len(item["options"])]),
                    "source_id": item["qid"], "human": {"p": p, "n": n},
                })

        raw = archive.read(member(
            ICAR_PROPOSITIONAL + "translations/items_plain_text-eng-GB.txt")).decode("utf-8-sig")
        propositional = _icar_propositional(raw)
        human = _icar_human(_zip_pdf_text(
            archive, member(ICAR_PROPOSITIONAL + "Psychometric_data.pdf"), layout=True))
        if len(propositional) != 30 or len(human) != 30:
            raise FetchError(f"icar propositional: {len(propositional)} items, "
                             f"{len(human)} with human data -- expected 30 of each")
        for number, item in enumerate(propositional, 1):
            if f"PR.{number:02d}" in ICAR_EXCLUDED:
                continue
            options = item["options"] + ICAR_PR_TAIL
            qid = f"q_{17000 + number}"
            if qid not in human or not 1 <= item["correct"] <= 6 or len(item["options"]) != 6:
                raise FetchError(f"icar: propositional item {number} is malformed")
            p, n = human[qid]
            items.append({
                "key": f"PR.{number:02d}", "trait": "propositional",
                "stem": ("Which conclusion follows from the statements below?\n\n"
                         + item["stem"]),
                "choices": dict(zip(letters, options)),
                "answer": letters[item["correct"] - 1],
                "anchored": ["G", "H"],
                "source_id": qid, "human": {"p": p, "n": n},
            })

    return {
        "id": "icar",
        "name": "ICAR verbal, series and propositional reasoning",
        "source": "https://icar-project.com/ (PsychArchives deposit, registered access)",
        "citation": ("Condon, D. M., & Revelle, W. (2014). The International Cognitive "
                     "Ability Resource: Development and initial validation of a "
                     "public-domain measure. Intelligence, 43, 52-64. Propositional "
                     "items: Guhne, D., Doebler, P., Condon, D. M., Luo, F., & Sun, L. "
                     "(2020). European Journal of Psychological Assessment."),
        "note": ("Text-only ICAR item types: 16 verbal reasoning and 9 letter-and-number "
                 "series items from ICAR-60, and the 30 fixed propositional reasoning "
                 "items. Items, keys and per-item human pass rates are read from the "
                 "archive the user supplies; none of it is committed. Propositional "
                 "items are paired with q_17001-q_17030 by file order, matching the "
                 "type's report; PR.11 and PR.12 are dropped for broken source text "
                 "(see `excluded`). The fixed items ship without an instruction, so "
                 "each stem is prefixed with 'Which conclusion follows from the "
                 "statements below?'. They carry the 'None of these' / 'I don't know' "
                 "options people were shown. Those two options stay last when choices "
                 "are shuffled. Human pass rates come from large self-selected online "
                 "samples (SAPA); they place items by difficulty, they are not IQ norms."),
        "prompt_style": "multiple_choice",
        "framing": "exam",
        "scale": {"min": 1, "points": 8, "labels": list(letters[:8])},
        "traits": {
            "overall": {"label": "Overall", "range": [0, 100]},
            "verbal": {"label": "Verbal reasoning", "range": [0, 100]},
            "series": {"label": "Letter & number series", "range": [0, 100]},
            "propositional": {"label": "Propositional reasoning", "range": [0, 100]},
        },
        "scoring": {"type": "answer_key", "overall": "overall"},
        "report": {"type": "bars", "unit": "%", "human_baseline": True},
        "excluded": ICAR_EXCLUDED,
        "items": items,
    }


ICAR_NUMBERS = "01_Further_ICAR_item_types/number_series/number series.csv"


def fetch_icar_numbers(source: str | None = None) -> dict:
    import csv
    import io
    import zipfile

    if not source:
        raise FetchError(
            "icar-numbers: items are distributed to registered researchers, not over "
            "a URL. Pass the ICAR archive: "
            "`invigilate fetch icar-numbers --source path/to/ZPID_Einreichung.zip`")
    path = Path(source).expanduser()
    if not path.exists():
        raise FetchError(f"icar-numbers: {path} does not exist")
    with zipfile.ZipFile(path) as archive:
        try:
            raw = archive.read(ICAR_ROOT + ICAR_NUMBERS).decode("utf-8-sig")
        except KeyError:
            raise FetchError(f"icar-numbers: {ICAR_ROOT + ICAR_NUMBERS} is not in the "
                             "archive -- wrong file?") from None
    rows = list(csv.DictReader(io.StringIO(raw), delimiter=";"))
    items = []
    for row in rows:
        row = {k.strip(): (v or "").strip() for k, v in row.items()}
        number = int(row["Items"])
        terms = [t for t in row["seq"].replace("…", "").split(",") if t.strip()]
        answer = [a for a in (row["ans1"], row["ans2"]) if a]
        if not terms or not answer or not all(t.strip().isdigit() for t in terms + answer):
            raise FetchError(f"icar-numbers: item {number} is malformed: {row}")
        ask = ("what number comes next?" if len(answer) == 1
               else "what are the next two numbers?")
        items.append({
            "key": f"NS.{number:02d}", "trait": "numbers",
            "stem": f"In the following number series, {ask} {', '.join(terms)}, ...",
            "answer": answer,
            # Rasch difficulty, decimal comma in the source; higher is harder.
            "human": {"beta": float(row["beta (Rasch)"].replace(",", "."))},
        })
    if [i["key"] for i in items] != [f"NS.{n:02d}" for n in range(1, len(items) + 1)] \
            or len(items) != 49:
        raise FetchError(f"icar-numbers: expected items 1-49 in order, got {len(items)}")
    return {
        "id": "icar-numbers",
        "name": "ICAR number series",
        "source": "https://icar-project.com/ (PsychArchives deposit, registered access)",
        "citation": ("Loe, B., Sun, L., Simonfy, F., & Doebler, P. (2018). Evaluating an "
                     "Automated Number Series Item Generator Using Linear Logistic Test "
                     "Models. Journal of Intelligence, 6(2), 20."),
        "note": ("49 write-in number series items from the ICAR archive (its readme says "
                 "48; the file has 49). People typed the next one or two numbers with no "
                 "choices offered, and so do the models. The source gives each item's "
                 "sequence, answer and Rasch difficulty; the question wording is ours, "
                 "matching ICAR's letter-and-number series items. Rasch difficulties rank "
                 "items for people; they are not pass rates and are not converted into "
                 "any. Judgement models cannot sit write-in items."),
        "prompt_style": "write_in",
        "framing": "exam",
        "scale": {"min": 1, "points": 1, "labels": ["number"]},
        "traits": {"numbers": {"label": "Number series", "range": [0, 100]}},
        "scoring": {"type": "answer_key"},
        "report": {"type": "bars", "unit": "%", "human_baseline": True},
        "items": items,
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
    "sat": fetch_sat,
    "icar": fetch_icar,
    "icar-numbers": fetch_icar_numbers,
}

# Fetchers that read a file the user supplies rather than a URL. They are
# skipped by a bare `invigilate fetch` unless --source is given.
NEEDS_SOURCE = {"icar": "the ICAR archive from PsychArchives (ZPID_Einreichung.zip)",
                "icar-numbers": "the ICAR archive from PsychArchives (ZPID_Einreichung.zip)"}

NEEDS_CONVERTER = {"hexaco": "a .doc converter (antiword, catdoc or libreoffice)",
                   "mfq": "a .doc converter (antiword, catdoc or libreoffice)",
                   "dirty-dozen": "pdftotext (poppler-utils) or libreoffice"}


def build(name: str, source: str | None = None) -> Path:
    if name not in FETCHERS:
        raise FetchError(f"unknown instrument {name!r}; have {', '.join(sorted(FETCHERS))}")
    spec = FETCHERS[name](source) if name in NEEDS_SOURCE else FETCHERS[name]()
    from . import instrument as instrument_module

    problems = instrument_module.Instrument(spec).validate()
    if problems:
        raise FetchError(f"{name}: built file is invalid -- {problems}")
    INSTRUMENTS.mkdir(parents=True, exist_ok=True)
    path = INSTRUMENTS / f"{name}.json"
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
