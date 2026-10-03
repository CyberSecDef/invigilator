"""Invigilate: administer psychometric instruments to language models.

Questionnaires are fetched from their publishers rather than stored here, so a
fresh checkout starts with `invigilate fetch`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path

from . import compass, fetchers, instrument as instruments, providers, report, scoring, survey

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
QUESTIONS = DATA / "questions.json"
WEIGHTS = DATA / "weights.json"
# Keys live in a gitignored .env beside the code. Override with --env or
# $INVIGILATOR_ENV; never hardcode a path into somebody else's project.
DEFAULT_ENV = os.environ.get("INVIGILATOR_ENV") or str(ROOT / ".env")


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _need(path: Path, hint: str) -> dict:
    if not path.exists():
        sys.exit(f"missing {path.name} — run `{hint}` first")
    return compass.load(path)


def cmd_fetch(args):
    """Build instrument files from their publishers. They are not vendored here."""
    names = args.instruments or sorted(fetchers.FETCHERS)
    failed = []
    for name in names:
        need = fetchers.NEEDS_CONVERTER.get(name)
        suffix = f"  (needs {need})" if need else ""
        print(f"fetching {name}...{suffix}")
        try:
            path = fetchers.build(name)
        except fetchers.FetchError as exc:
            failed.append((name, str(exc)))
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        spec = compass.load(path)
        print(f"  {len(spec['items'])} items -> {path.relative_to(ROOT)}")
    if failed:
        print(f"\n{len(failed)} of {len(names)} failed:", file=sys.stderr)
        for name, error in failed:
            print(f"  {name}: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"\n{len(names)} instrument(s) built")


def cmd_calibrate(args):
    schema = _need(QUESTIONS, "invigilate fetch")
    site = compass.Site(delay=args.delay)
    total = 2 + len(schema["pages"]) + 3 * schema["n_questions"] + 2
    total += args.checks * len(schema["pages"])
    print(f"~{total} throttled page submissions at {args.delay}s apart "
          f"(~{total * args.delay / 60:.1f} min)...")
    table = compass.calibrate(site, schema, log=print if args.verbose else lambda *_: None)

    print("verifying against live scoring...")
    rng = random.Random(20260829)
    checks = []
    for trial in range(args.checks):
        answers = {q["key"]: rng.randrange(4) for q in schema["questions"]}
        live_ec, live_soc = compass.score_live(site, schema, answers)
        local = compass.score_offline(table, answers)["point"]
        delta = (abs(local["ec"] - live_ec), abs(local["soc"] - live_soc))
        ok = max(delta) <= 0.01
        checks.append({"live": [live_ec, live_soc], "offline": local, "ok": ok})
        print(f"  check {trial + 1}: live=({live_ec:+.2f},{live_soc:+.2f}) "
              f"offline=({local['ec']:+.2f},{local['soc']:+.2f}) {'OK' if ok else 'MISMATCH'}")
    if not all(c["ok"] for c in checks):
        sys.exit("calibration failed verification — not writing weights.json")

    table["verified"] = checks
    compass.save(WEIGHTS, table)
    print(f"weights verified and written -> {WEIGHTS}  ({site.requests} requests)")


def cmd_providers(args):
    providers.load_dotenv(args.env)
    for name, provider in providers.registry().items():
        status = "ready " if provider.available() else "no key"
        line = f"{status}  {name:<10} default={provider.default_model}"
        if args.list and provider.available():
            try:
                models = provider.list_models()
                line += f"\n           {len(models)} models: {', '.join(models[:12])}"
                if len(models) > 12:
                    line += ", ..."
            except Exception as exc:
                line += f"\n           (listing failed: {str(exc)[:120]})"
        print(line)


def cmd_instruments(args):
    for name in instruments.available():
        ins = instruments.load(name)
        needs = ""
        if ins.scoring["type"] == "measured_weights":
            needs = "  [needs `calibrate`]" if not (ROOT / ins.scoring["table"]).exists() else ""
        print(f"{name:<20} {len(ins.items):>3} items  {ins.points}-point {ins.prompt_style:<8} "
              f"-> {', '.join(ins.traits)}{needs}")
        print(f"{'':<20} {ins.name}")


def cmd_run(args):
    providers.load_dotenv(args.env)
    try:
        instrument = instruments.load(args.instrument)
    except FileNotFoundError as exc:
        sys.exit(f"{exc}\nRun `invigilate fetch {args.instrument}` to build it.")
    if instrument.scoring["type"] == "measured_weights":
        _need(ROOT / instrument.scoring["table"], "invigilate calibrate")

    targets = []
    for spec in args.models:
        provider, model = providers.resolve(spec)
        if not provider.available():
            sys.exit(f"{spec}: {provider.key_env} is not set")
        if provider.structured and args.mode == "batch":
            sys.exit(f"{spec}: judgement models run in isolated mode only")
        if args.framing != "own-view" and not provider.structured:
            sys.exit(f"{spec}: --framing applies to judgement models only")
        # A non-default framing is a different measurement; label it so
        # reports keep it apart from the plain run.
        label = spec if args.framing == "own-view" else f"{spec}/{args.framing}"
        targets.append((label, provider, model))

    results = []
    out_path = RESULTS / f"{instrument.id}-{_stamp()}.json"
    print(f"instrument: {instrument.name} ({len(instrument.items)} items, "
          f"{instrument.points}-point {instrument.prompt_style})")
    for spec, provider, model in targets:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            print(f"[{spec}] repeat {repeat + 1}/{args.repeats} ({args.mode}, seed {seed})")
            outcome = survey.run_once(
                provider, model, instrument, args.mode, seed,
                concurrency=args.concurrency,
                log=print if args.verbose else None,
                framing=args.framing,
            )
            score = scoring.score(
                instrument, outcome["answers"], outcome["refused"] + outcome["errored"])
            results.append({
                "instrument": instrument.id,
                "label": spec, "provider": provider.name, "model": model,
                "framing": args.framing,
                "repeat": repeat, "mode": outcome["mode"], "seed": seed,
                "answers": outcome["answers"], "refused": outcome["refused"],
                "errored": outcome["errored"],
                "score": score,
                "trace": outcome["trace"] if args.trace else None,
            })
            if "distributions" in outcome:
                # Judgement models report a distribution per item; score that
                # too, alongside the top-level picks every other model gets.
                results[-1]["expected"] = outcome["expected"]
                results[-1]["distributions"] = outcome["distributions"]
                results[-1]["score_expected"] = scoring.score_expected(
                    instrument, outcome["distributions"],
                    outcome["refused"] + outcome["errored"])
            point = score["point"]
            flags = []
            if outcome["refused"]:
                flags.append(f"{len(outcome['refused'])} refused")
            if outcome["errored"]:
                flags.append(f"{len(outcome['errored'])} ERRORED")
            note = f"  ({', '.join(flags)})" if flags else ""
            summary = ", ".join(
                f"{instrument.traits[t]['label']} {v:+.2f}"
                for t, v in point.items() if v is not None)
            if instrument.report["type"] == "dichotomy":
                summary = f"{scoring.type_code(instrument, point)} — {summary}"
            print(f"    -> {summary}{note}")
            if "score_expected" in results[-1]:
                weighted = results[-1]["score_expected"]["point"]
                print("       probability-weighted: " + ", ".join(
                    f"{instrument.traits[t]['label']} {v:+.2f}"
                    for t, v in weighted.items() if v is not None))
            compass.save(out_path, {"generated": _stamp(), "results": results})

    print(f"\n{len(results)} runs -> {out_path}")
    _warn_unusable(results)
    rows = report.aggregate(results, instrument)
    print("\n" + report.render_table(rows, instrument))
    title = args.title or f"{instrument.name} — {args.mode} mode, {args.repeats} runs each"
    svg, md = report.write(rows, instrument, out_path.with_suffix(""), title)
    print(f"\nchart -> {svg}\ntable -> {md}")


def _warn_unusable(results):
    """A position derived from a handful of answers is noise; say so out loud."""
    _, dropped = report.partition(results)
    for run in dropped:
        total = run["score"].get("n_questions", 62)
        print(
            f"  ! dropped {run['label']} repeat {run['repeat'] + 1}: only "
            f"{len(run['answers'])}/{total} answered "
            f"({len(run.get('errored', []))} errored, {len(run['refused'])} refused)",
            file=sys.stderr,
        )


def cmd_report(args):
    paths = [Path(p) for p in args.results] if args.results else [
        max(RESULTS.glob("*.json"), key=lambda p: p.stat().st_mtime, default=None)
    ]
    paths = [p for p in paths if p and p.exists()]
    if not paths:
        sys.exit("no results file found")
    results, ids = [], set()
    for path in paths:
        payload = compass.load(path)
        results.extend(payload["results"])
        ids.add(payload.get("instrument") or
                (payload["results"][0].get("instrument") if payload["results"] else None))
    ids.discard(None)
    if len(ids) > 1:
        sys.exit(f"cannot merge different instruments: {', '.join(sorted(ids))}")
    instrument = instruments.load(args.instrument or (ids.pop() if ids else "political-compass"))
    _warn_unusable(results)
    rows = report.aggregate(results, instrument)
    print(report.render_table(rows, instrument))
    stem = Path(args.out) if args.out else paths[-1].with_suffix("")
    title = args.title or instrument.name
    svg, md = report.write(rows, instrument, stem, title)
    print(f"\nchart -> {svg}\ntable -> {md}")


def build_parser():
    parser = argparse.ArgumentParser(prog="invigilate", description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV,
                        help="dotenv file with API keys (default: ./.env)")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch", help="build instrument files from their publishers (not vendored here)")
    fetch.add_argument("instruments", nargs="*", metavar="NAME",
                       help="instrument ids to build; default is all of them")
    fetch.set_defaults(func=cmd_fetch)

    cal = sub.add_parser("calibrate", help="measure the scoring weights, then verify them")
    cal.add_argument("--delay", type=float, default=0.7)
    cal.add_argument("--checks", type=int, default=3, help="random vectors to verify against")
    cal.add_argument("-v", "--verbose", action="store_true")
    cal.set_defaults(func=cmd_calibrate)

    prov = sub.add_parser("providers", help="show which providers have credentials")
    prov.add_argument("--list", action="store_true", help="also query available model ids")
    prov.set_defaults(func=cmd_providers)

    inst = sub.add_parser("instruments", help="list available questionnaires")
    inst.set_defaults(func=cmd_instruments)

    run = sub.add_parser("run", help="administer the test to one or more models")
    run.add_argument("models", nargs="+", metavar="PROVIDER[:MODEL]")
    run.add_argument("-i", "--instrument", default="political-compass",
                     help="builtin id or path to an instrument JSON file")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--mode", choices=["isolated", "batch"], default="isolated")
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--concurrency", type=int, default=1,
                     help="parallel requests per model (isolated mode only)")
    run.add_argument("--framing", choices=list(instruments.FRAMINGS), default="own-view",
                     help="whose answer a judgement model is asked for (typesafe only)")
    run.add_argument("--trace", action="store_true", help="store every raw reply")
    run.add_argument("--title", default=None)
    run.add_argument("-v", "--verbose", action="store_true")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="re-render a chart from saved results")
    rep.add_argument("results", nargs="*", help="one or more run-*.json files")
    rep.add_argument("--out", help="output stem for the chart and table")
    rep.add_argument("--title", default=None)
    rep.add_argument("-i", "--instrument", default=None)
    rep.set_defaults(func=cmd_report)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
