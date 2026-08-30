"""Interface to politicalcompass.org.

The site scores server-side and has never published its scoring key, so this
module derives one empirically: `calibrate()` submits answer sets, observes the
scores that come back, and solves for the per-question weights and the final
affine transform.

Nothing here is guessed. `calibrate()` measures the weights, measures the
transform, and refuses to write a table it could not reproduce against live
results. Requests are throttled and the whole procedure runs once -- afterwards
scoring is pure arithmetic and the models under test never touch their server.
"""

from __future__ import annotations

import http.cookiejar
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.politicalcompass.org"
TEST_URL = f"{BASE}/test/en"
UA = "Mozilla/5.0 (X11; Linux x86_64) Invigilator/0.1 (+research)"

# Radio values 0..3, in the order the site renders them.
CHOICES = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]
N_PAGES = 6

_RE_RADIO = re.compile(r'<input id="([A-Za-z0-9_]+)_0" name="([A-Za-z0-9_]+)" type="radio"')
_RE_LEGEND = re.compile(r"<legend[^>]*>\s*(.*?)\s*</legend>", re.S)
_RE_HIDDEN = re.compile(r'<input name="([a-z_]+)" type="hidden" value="([^"]*)"')
_RE_COORDS = re.compile(r"[?&]ec=(-?[\d.]+)&(?:amp;)?soc=(-?[\d.]+)")


def _detag(html: str) -> str:
    text = re.sub(r"<[^>]+>", "", html)
    for entity, char in (
        ("&rsquo;", "’"), ("&lsquo;", "‘"),
        ("&ldquo;", "“"), ("&rdquo;", "”"),
        ("&mdash;", "—"), ("&ndash;", "–"),
        ("&amp;", "&"), ("&nbsp;", " "), ("&quot;", '"'),
    ):
        text = text.replace(entity, char)
    return re.sub(r"\s+", " ", text).strip()


class Site:
    """Throttled HTTP client for the test form."""

    def __init__(self, delay: float = 0.7):
        jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        self._delay = delay
        self._last = 0.0
        self.requests = 0

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self._delay:
            time.sleep(self._delay - gap)
        self._last = time.monotonic()

    def get(self, url: str) -> str:
        self._throttle()
        self.requests += 1
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with self._opener.open(req, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace")

    def post(self, form: dict) -> tuple[str, str]:
        """POST one page. Returns (html, final_url)."""
        self._throttle()
        self.requests += 1
        req = urllib.request.Request(
            TEST_URL,
            data=urllib.parse.urlencode(form).encode(),
            headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"},
        )
        with self._opener.open(req, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace"), resp.geturl()


def _parse_page(html: str) -> tuple[list[str], list[str], dict]:
    keys = [m.group(2) for m in _RE_RADIO.finditer(html)]
    legends = [_detag(x) for x in _RE_LEGEND.findall(html)]
    return keys, legends, dict(_RE_HIDDEN.findall(html))


def fetch_questions(site: Site) -> dict:
    """Walk the six pages and record every proposition, in site order.

    Answers "Strongly disagree" throughout purely to advance the form; the
    resulting score is discarded.
    """
    questions, pages = [], []
    html = site.get(f"{TEST_URL}?page=1")
    for page in range(1, N_PAGES + 1):
        keys, legends, hidden = _parse_page(html)
        if len(keys) != len(legends):
            raise RuntimeError(f"page {page}: {len(keys)} inputs vs {len(legends)} legends")
        pages.append(keys)
        for index, (key, text) in enumerate(zip(keys, legends)):
            questions.append({"key": key, "page": page, "index": index, "text": text})
        form = dict(hidden, page=str(page), **{k: "0" for k in keys})
        html, _ = site.post(form)
    return {
        "source": TEST_URL,
        "n_questions": len(questions),
        "pages": pages,
        "questions": questions,
        "choices": CHOICES,
    }


def score_live(site: Site, schema: dict, answers: dict) -> tuple[float, float]:
    """Submit a full answer set page by page and read the site's own verdict.

    Every question must be answered -- the form silently re-serves a page that
    has any radio group untouched.
    """
    carried_ec, carried_soc = "", ""
    final_url = ""
    for page_no, keys in enumerate(schema["pages"], start=1):
        missing = [k for k in keys if k not in answers]
        if missing:
            raise ValueError(f"unanswered on page {page_no}: {missing}")
        form = {
            "page": str(page_no),
            "carried_ec": carried_ec,
            "carried_soc": carried_soc,
            "populated": "",
        }
        form.update({k: str(answers[k]) for k in keys})
        html, final_url = site.post(form)
        hidden = dict(_RE_HIDDEN.findall(html))
        if hidden.get("page") == str(page_no):
            raise RuntimeError(f"site rejected page {page_no} (re-served it)")
        carried_ec = hidden.get("carried_ec", "")
        carried_soc = hidden.get("carried_soc", "")
    match = _RE_COORDS.search(final_url) or _RE_COORDS.search(html)
    if not match:
        raise RuntimeError(f"no coordinates in result: {final_url}")
    return float(match.group(1)), float(match.group(2))


def _probe_mid(site: Site, page_no: int, keys: list, overrides: dict,
               carried: tuple = (0, 0)) -> tuple[int, int]:
    """POST one of pages 1-5 in isolation; read the raw running totals it returns."""
    form = {
        "page": str(page_no),
        "carried_ec": str(carried[0]),
        "carried_soc": str(carried[1]),
        "populated": "",
    }
    form.update({k: "0" for k in keys})
    form.update({k: str(v) for k, v in overrides.items()})
    html, _ = site.post(form)
    hidden = dict(_RE_HIDDEN.findall(html))
    if hidden.get("page") == str(page_no):
        raise RuntimeError(f"site rejected probe of page {page_no}")
    return int(hidden["carried_ec"]), int(hidden["carried_soc"])


def _probe_final(site: Site, keys: list, overrides: dict,
                 carried: tuple = (0, 0)) -> tuple[float, float]:
    """POST the last page, which redirects to the result; read the scored coordinates.

    Page 6 has no successor to carry raw totals, so its weights are measured in
    final units and converted back with the divisor.
    """
    form = {
        "page": str(N_PAGES),
        "carried_ec": str(carried[0]),
        "carried_soc": str(carried[1]),
        "populated": "",
    }
    form.update({k: "0" for k in keys})
    form.update({k: str(v) for k, v in overrides.items()})
    html, final_url = site.post(form)
    match = _RE_COORDS.search(final_url) or _RE_COORDS.search(html)
    if not match:
        raise RuntimeError(f"page {N_PAGES} probe did not reach a result: {final_url}")
    return float(match.group(1)), float(match.group(2))


def calibrate(site: Site, schema: dict, log=print) -> dict:
    """Recover the per-question weight table plus the axis divisors.

    Weights are measured relative to answering 0 ("Strongly disagree") on every
    proposition, so a scored vector is `origin + sum(weight) / divisor` where
    `origin` is the live score of the all-zeros vector. Nothing is assumed: the
    divisors are measured, and `cmd_calibrate` refuses to save a table that does
    not reproduce live results.
    """
    mid_pages = schema["pages"][:-1]
    last_page = schema["pages"][-1]

    # Divisors first -- page 6 weights are expressed in final units and need them.
    # Probing symmetrically around zero cancels the page's own baseline, and a
    # large offset keeps the site's 2-decimal rounding well below the noise floor.
    spread = {"ec": 200, "soc": 400}
    divisor = {}
    for axis, magnitude in spread.items():
        carried_hi = (magnitude, 0) if axis == "ec" else (0, magnitude)
        carried_lo = (-magnitude, 0) if axis == "ec" else (0, -magnitude)
        index = 0 if axis == "ec" else 1
        high = _probe_final(site, last_page, {}, carried_hi)[index]
        low = _probe_final(site, last_page, {}, carried_lo)[index]
        if high == low:
            raise RuntimeError(f"{axis} axis did not respond to carried totals")
        divisor[axis] = round(2 * magnitude / (high - low), 6)
    log(f"  divisors: {divisor}")

    weights: dict[str, dict] = {}
    base_ec = base_soc = 0

    for page_no, keys in enumerate(mid_pages, start=1):
        page_ec, page_soc = _probe_mid(site, page_no, keys, {})
        base_ec += page_ec
        base_soc += page_soc
        log(f"  page {page_no}: baseline ec={page_ec:+d} soc={page_soc:+d}")
        for key in keys:
            per_answer = {"0": [0, 0]}
            for value in (1, 2, 3):
                ec, soc = _probe_mid(site, page_no, keys, {key: value})
                per_answer[str(value)] = [ec - page_ec, soc - page_soc]
            weights[key] = per_answer
            log(f"    {key:<28} {per_answer['1']} {per_answer['2']} {per_answer['3']}")

    log(f"  page {N_PAGES}: measured through the result page")
    final_base = _probe_final(site, last_page, {})
    for key in last_page:
        per_answer = {"0": [0, 0]}
        for value in (1, 2, 3):
            ec, soc = _probe_final(site, last_page, {key: value})
            per_answer[str(value)] = [
                round((ec - final_base[0]) * divisor["ec"]),
                round((soc - final_base[1]) * divisor["soc"]),
            ]
        weights[key] = per_answer
        log(f"    {key:<28} {per_answer['1']} {per_answer['2']} {per_answer['3']}")

    # The all-zeros vector scored live: page 6 answered 0 while carrying the
    # summed baselines of pages 1-5 is exactly a completed all-zeros test.
    origin_ec, origin_soc = _probe_final(site, last_page, {}, (base_ec, base_soc))
    log(f"  origin (all 'Strongly disagree'): ({origin_ec:+.2f}, {origin_soc:+.2f})")

    return {
        "origin": {"ec": origin_ec, "soc": origin_soc},
        "divisor": divisor,
        "weights": weights,
        "verified": [],
    }


def weight_sum(table: dict, answers: dict) -> dict:
    """Sum the per-question raw contributions, relative to all-'Strongly disagree'."""
    totals = {"ec": 0, "soc": 0}
    for key, value in answers.items():
        weight = table["weights"].get(key)
        if weight is None:
            raise KeyError(f"no calibrated weight for {key!r}")
        d_ec, d_soc = weight[str(value)]
        totals["ec"] += d_ec
        totals["soc"] += d_soc
    return totals


def score_offline(table: dict, answers: dict, omitted: list | None = None) -> dict:
    """Score locally. Omitted (refused) questions contribute nothing to the sum.

    Because a refusal is not a neutral answer -- there is no neutral answer on
    this test -- we also report the interval the score would span if every
    refused item had been answered at either extreme. A model with many
    refusals gets a wide, honest error bar rather than a false point estimate.
    """
    omitted = omitted or []
    answered = {k: v for k, v in answers.items() if k not in omitted}
    totals = weight_sum(table, answered)

    def project(axis: str, value: float) -> float:
        return round(table["origin"][axis] + value / table["divisor"][axis], 2)

    point = {axis: project(axis, totals[axis]) for axis in ("ec", "soc")}
    bounds = {}
    for axis, index in (("ec", 0), ("soc", 1)):
        low = high = totals[axis]
        for key in omitted:
            deltas = [table["weights"][key][str(v)][index] for v in range(4)]
            low += min(deltas)
            high += max(deltas)
        bounds[axis] = sorted((project(axis, low), project(axis, high)))
    return {"sum": totals, "point": point, "bounds": bounds, "omitted": omitted}


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
