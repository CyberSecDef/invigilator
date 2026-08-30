"""Rendering results: a standalone SVG plus a markdown table, per instrument."""

from __future__ import annotations

import html
import statistics
from pathlib import Path

from . import scoring

MIN_ANSWERED = 0.9  # a score built from fewer answers than this is not a score

QUADRANTS = [
    (-10, 0, 0, 10, "#d94a4a"), (0, 0, 10, 10, "#3f7fd0"),
    (-10, -10, 0, 0, "#43a047"), (0, -10, 10, 0, "#c9a227"),
]
SERIES = ["#111111", "#e8590c", "#1971c2", "#2f9e44", "#ae3ec9",
          "#c92a2a", "#0c8599", "#f08c00", "#5f3dc4", "#495057"]


def partition(results: list) -> tuple[list, list]:
    usable, unusable = [], []
    for run in results:
        total = run["score"].get("n_questions", 62)
        (usable if len(run["answers"]) >= MIN_ANSWERED * total else unusable).append(run)
    return usable, unusable


def aggregate(results: list, instrument=None) -> list:
    """Collapse repeat runs of the same model into a mean per trait plus spread."""
    grouped: dict[str, list] = {}
    for run in partition(results)[0]:
        grouped.setdefault(run["label"], []).append(run)

    traits = list(instrument.traits) if instrument else ["ec", "soc"]
    rows = []
    for label, runs in grouped.items():
        row = {
            "label": label, "provider": runs[0]["provider"], "model": runs[0]["model"],
            "mode": runs[0]["mode"], "n": len(runs),
            "refused_mean": round(statistics.fmean(len(r["refused"]) for r in runs), 1),
            "errored_mean": round(statistics.fmean(len(r.get("errored", [])) for r in runs), 1),
            "n_questions": runs[0]["score"].get("n_questions", 62),
            "traits": {},
        }
        for trait in traits:
            values = [r["score"]["point"].get(trait) for r in runs]
            values = [v for v in values if v is not None]
            if not values:
                continue
            row["traits"][trait] = {
                "mean": round(statistics.fmean(values), 2),
                "sd": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
                "range": [min(values), max(values)],
            }
        if instrument and instrument.report["type"] == "dichotomy":
            row["type"] = scoring.type_code(
                instrument, {t: row["traits"][t]["mean"] for t in row["traits"]})
        rows.append(row)

    first = traits[0]
    return sorted(rows, key=lambda r: r["traits"].get(first, {}).get("mean", 0))


# --------------------------------------------------------------------------
def _svg_open(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="Helvetica,Arial,sans-serif">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="18" '
        f'font-weight="600" fill="#111">{html.escape(title)}</text>',
    ]


def render_compass2d(rows, instrument, title):
    x_axis, y_axis = instrument.report["x"], instrument.report["y"]
    size, pad = 620, 70
    span = size - 2 * pad
    px = lambda v: pad + (v + 10) / 20 * span
    py = lambda v: pad + (10 - v) / 20 * span

    out = _svg_open(size, size + 30 * len(rows) + 60, title)
    for x0, y0, x1, y1, colour in QUADRANTS:
        out.append(f'<rect x="{px(x0):.1f}" y="{py(y1):.1f}" width="{px(x1)-px(x0):.1f}" '
                   f'height="{py(y0)-py(y1):.1f}" fill="{colour}" opacity="0.16"/>')
    for step in range(-10, 11, 2):
        w = 1.4 if step == 0 else 0.5
        c = "#555" if step == 0 else "#c8c8c8"
        out.append(f'<line x1="{px(step):.1f}" y1="{pad}" x2="{px(step):.1f}" '
                   f'y2="{pad+span}" stroke="{c}" stroke-width="{w}"/>')
        out.append(f'<line x1="{pad}" y1="{py(step):.1f}" x2="{pad+span}" '
                   f'y2="{py(step):.1f}" stroke="{c}" stroke-width="{w}"/>')
    for x, y, anchor, text in [
        (size / 2, pad - 14, "middle", "Authoritarian"),
        (size / 2, pad + span + 26, "middle", "Libertarian"),
        (pad - 10, pad + span / 2, "end", "Left"),
        (pad + span + 10, pad + span / 2, "start", "Right"),
    ]:
        out.append(f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}" font-size="13" '
                   f'fill="#444" dominant-baseline="middle">{text}</text>')

    for index, row in enumerate(rows):
        colour = SERIES[index % len(SERIES)]
        xs, ys = row["traits"][x_axis], row["traits"][y_axis]
        cx, cy = px(xs["mean"]), py(ys["mean"])
        if xs["range"][0] != xs["range"][1]:
            out.append(f'<line x1="{px(xs["range"][0]):.1f}" y1="{cy:.1f}" '
                       f'x2="{px(xs["range"][1]):.1f}" y2="{cy:.1f}" stroke="{colour}" '
                       f'stroke-width="1.6" opacity="0.55"/>')
        if ys["range"][0] != ys["range"][1]:
            out.append(f'<line x1="{cx:.1f}" y1="{py(ys["range"][0]):.1f}" x2="{cx:.1f}" '
                       f'y2="{py(ys["range"][1]):.1f}" stroke="{colour}" '
                       f'stroke-width="1.6" opacity="0.55"/>')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{colour}" '
                   f'stroke="#fff" stroke-width="1.6"/>')
        ly = size + 24 + index * 30
        out.append(f'<circle cx="{pad}" cy="{ly-4:.0f}" r="6" fill="{colour}"/>'
                   f'<text x="{pad+16}" y="{ly:.0f}" font-size="13" fill="#222">'
                   f'{html.escape(row["label"])} ({xs["mean"]:+.2f}, {ys["mean"]:+.2f})</text>')
    out.append("</svg>")
    return "\n".join(out)


def render_bars(rows, instrument, title):
    traits = list(instrument.traits)
    lo, hi = instrument.traits[traits[0]].get("range", [1, 5])
    left, width = 210, 320
    row_h, group_h = 26, 26 * len(traits) + 26
    height = 70 + group_h * len(rows)
    out = _svg_open(left + width + 90, height, title)

    for index, row in enumerate(rows):
        top = 62 + index * group_h
        out.append(f'<text x="12" y="{top:.0f}" font-size="13" font-weight="600" '
                   f'fill="#111">{html.escape(row["label"])}</text>')
        for t_index, trait in enumerate(traits):
            if trait not in row["traits"]:
                continue
            value = row["traits"][trait]["mean"]
            spread = row["traits"][trait]["range"]
            y = top + 14 + t_index * row_h
            frac = (value - lo) / (hi - lo)
            colour = SERIES[t_index % len(SERIES)]
            out.append(f'<text x="{left-8}" y="{y+11:.0f}" text-anchor="end" font-size="12" '
                       f'fill="#444">{html.escape(instrument.traits[trait]["label"])}</text>')
            out.append(f'<rect x="{left}" y="{y:.0f}" width="{width}" height="16" '
                       f'fill="#eee" rx="3"/>')
            out.append(f'<rect x="{left}" y="{y:.0f}" width="{max(0, frac*width):.1f}" '
                       f'height="16" fill="{colour}" opacity="0.85" rx="3"/>')
            if spread[0] != spread[1]:
                x0 = left + (spread[0] - lo) / (hi - lo) * width
                x1 = left + (spread[1] - lo) / (hi - lo) * width
                out.append(f'<line x1="{x0:.1f}" y1="{y+8:.0f}" x2="{x1:.1f}" y2="{y+8:.0f}" '
                           f'stroke="#111" stroke-width="1.4" opacity="0.6"/>')
            out.append(f'<text x="{left+width+8}" y="{y+12:.0f}" font-size="12" '
                       f'fill="#222">{value:.2f}</text>')
    out.append(f'<text x="{left}" y="{height-8}" font-size="11" fill="#777">'
               f'scale {lo}-{hi}; black line spans repeat runs</text>')
    out.append("</svg>")
    return "\n".join(out)


def render_dichotomy(rows, instrument, title):
    axes = instrument.report["letter_order"]
    left, width = 150, 340
    row_h, group_h = 24, 24 * len(axes) + 30
    height = 70 + group_h * len(rows)
    out = _svg_open(left + width + 110, height, title)

    for index, row in enumerate(rows):
        top = 62 + index * group_h
        code = row.get("type", "")
        out.append(f'<text x="12" y="{top:.0f}" font-size="13" font-weight="600" fill="#111">'
                   f'{html.escape(row["label"])} &#8212; <tspan font-size="15">{code}</tspan></text>')
        for a_index, axis in enumerate(axes):
            if axis not in row["traits"]:
                continue
            trait = instrument.traits[axis]
            value = row["traits"][axis]["mean"]
            mid = trait["midpoint"]
            lo, hi = trait.get("range", [1, 5])
            y = top + 14 + a_index * row_h
            cx = left + width / 2
            out.append(f'<text x="{left-10}" y="{y+11:.0f}" text-anchor="end" font-size="12" '
                       f'fill="#444">{trait["low"]}</text>')
            out.append(f'<text x="{left+width+10}" y="{y+11:.0f}" font-size="12" '
                       f'fill="#444">{trait["high"]}</text>')
            out.append(f'<rect x="{left}" y="{y:.0f}" width="{width}" height="15" '
                       f'fill="#eee" rx="3"/>')
            out.append(f'<line x1="{cx}" y1="{y-2:.0f}" x2="{cx}" y2="{y+17:.0f}" '
                       f'stroke="#999" stroke-width="1"/>')
            frac = (value - lo) / (hi - lo)
            x = left + frac * width
            bar_from, bar_to = (cx, x) if x >= cx else (x, cx)
            colour = SERIES[a_index % len(SERIES)]
            out.append(f'<rect x="{bar_from:.1f}" y="{y:.0f}" width="{abs(bar_to-bar_from):.1f}" '
                       f'height="15" fill="{colour}" opacity="0.8" rx="2"/>')
            letter = trait["high"] if value > mid else trait["low"]
            out.append(f'<text x="{left+width+28}" y="{y+12:.0f}" font-size="11" fill="#222">'
                       f'{letter} {value:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


RENDERERS = {"compass2d": render_compass2d, "bars": render_bars, "dichotomy": render_dichotomy}


def render_svg(rows, instrument, title="Results"):
    return RENDERERS[instrument.report["type"]](rows, instrument, title)


def render_table(rows, instrument):
    traits = list(instrument.traits)
    heads = [instrument.traits[t]["label"] for t in traits]
    extra = " Type |" if instrument.report["type"] == "dichotomy" else ""
    extra_sep = "---|" if extra else ""
    header = (
        "| Model | Mode | Runs |" + extra + "".join(f" {h} |" for h in heads)
        + " Spread | Refused | Errors |\n"
        + "|---|---|---:|" + extra_sep + "---:|" * len(traits) + "---|---:|---:|\n"
    )
    lines = []
    for row in rows:
        cells = []
        for trait in traits:
            value = row["traits"].get(trait)
            cells.append(f" {value['mean']:+.2f} |" if value else " — |")
        spread = ", ".join(
            f"±{row['traits'][t]['sd']:.2f}" for t in traits if t in row["traits"]
        ) if row["n"] > 1 else "—"
        refused = (f"{row['refused_mean']:g}/{row['n_questions']}"
                   if row["refused_mean"] else "0")
        type_cell = f" {row.get('type','')} |" if extra else ""
        lines.append(
            f"| {row['label']} | {row['mode']} | {row['n']} |" + type_cell + "".join(cells)
            + f" {spread} | {refused} | {row.get('errored_mean', 0):g} |"
        )
    return header + "\n".join(lines)


def write(rows, instrument, stem: Path, title: str):
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path, md_path = stem.with_suffix(".svg"), stem.with_suffix(".md")
    svg_path.write_text(render_svg(rows, instrument, title), encoding="utf-8")
    md_path.write_text(f"# {title}\n\n{render_table(rows, instrument)}\n", encoding="utf-8")
    return svg_path, md_path
