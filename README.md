# Invigilator

Administers psychometric questionnaires to language models and plots where they
land. Ships with three instruments; adding another means writing a JSON file,
not code.

| Instrument | Items | Scale | Output |
|---|---:|---|---|
| `political-compass` | 62 | 4-point forced choice | 2-axis compass chart |
| `big-five` | 50 | 5-point agreement | OCEAN factors (IPIP markers) |
| `hexaco` | 60 | 5-point agreement | Six HEXACO-60 factors |
| `mfq` | 32 | 6-point, two sections | Five moral foundations |
| `rwa` | 22 | 9-point agreement | Right-wing authoritarianism |
| `sd3` | 27 | 5-point agreement | Machiavellianism / Narcissism / Psychopathy |
| `dirty-dozen` | 12 | 5-point agreement | The same three traits, 12 items |
| `oejts` | 32 | 5-point bipolar | Four dichotomies + a four-letter type |

```bash
invigilate instruments                               # what's available
invigilate run xai -i oejts                          # MBTI-style, one model
invigilate run openai -i sd3 --repeats 3             # Dark Triad, one model
invigilate run anthropic -i hexaco                   # HEXACO-60
invigilate run anthropic openai xai                  # political compass, three models
```

`sd3` and `dirty-dozen` measure the same three traits at very different lengths,
which makes them a convenient built-in convergent-validity check: if a model's
Dark Triad profile does not survive the switch from 27 items to 12, that tells
you something about the measurement rather than the model.

Model targets are `provider` or `provider:model`, so a single model is just a
single argument — `run xai:grok-4.6`.

### A note on the two personality instruments

**The MBTI is proprietary** (The Myers-Briggs Company) and is not reproduced
here. `oejts` is the Open Extended Jungian Type Scales, a public-domain
instrument measuring the same four dichotomies and yielding the same style of
four-letter code. It is not the MBTI and should not be reported as one.

`sd3` is the Short Dark Triad (Jones & Paulhus, 2014), the standard 27-item
research instrument.

Every instrument's items and scoring key come from an authoritative source or
were measured against a live scorer — never written from memory. Provenance
travels with the data in each generated file's `source`, `citation` and `note`
fields.

Item *text* is always fetched, because that is the part that carries a licence.
Scoring *keys* are embedded in `fetchers.py` as small numeric tables — which item
loads on which trait, and whether it is reverse-coded. Those are measurements
rather than expression, and re-deriving them would mean a few hundred extra
requests to someone else's server for numbers that do not change. The MFQ is the
exception: its key is fetched and cross-checked against the publisher's own SPSS
syntax, and the fetcher fails if the two disagree.

## What it actually does

The Political Compass is a six-page form scored on the site's own server, and
that scoring key has never been published. Every "political compass scorer" on
GitHub is therefore working from weights of unknown provenance. This tool does
not guess and does not copy someone else's guess.

Instead, `invigilate calibrate` derives the weight table empirically by
submitting answer sets to the site and observing the scores it returns, then
**verifies** the result: it scores random answer vectors both locally and on the
site, and refuses to save a table unless the two agree exactly. On this repo they
agreed to the last decimal place across every check.

Calibration runs once, throttled to one request at a time. After that, scoring is
local arithmetic — **the models under test never touch politicalcompass.org**,
and repeat runs cost their servers nothing. That is the point of doing it this
way: measure once, carefully, then stop.

If you would rather not calibrate at all, the other seven instruments publish
their keys and need none of this.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/invigilate fetch       # pull the 62 propositions   (~7 requests)
./.venv/bin/invigilate calibrate   # measure + verify weights   (~215 requests, ~3 min)
```

Both write to `data/`. You only ever do this once.

API keys are read from `.env` beside the code — copy `.env.example` and fill it
in. `.env` is gitignored. Override the location with `--env path/to/.env` or
`$INVIGILATOR_ENV`. Check what is wired up:

```bash
./.venv/bin/invigilate providers --list
```

## Running

```bash
./.venv/bin/invigilate run anthropic openai deepseek xai ollama \
    --instrument political-compass --repeats 3 --concurrency 8 --trace
```

`--instrument` (`-i`) takes a builtin id or a path to your own JSON file.
Results, an SVG chart and a markdown table land in `results/`, named for the
instrument. Re-render or merge saved runs with
`invigilate report file1.json file2.json` (merging across different instruments
is refused).

| Provider | Adapter | Default model |
|---|---|---|
| `anthropic` | official `anthropic` SDK | `claude-opus-5` |
| `openai` | `/v1/chat/completions` | `gpt-5.5` |
| `deepseek` | OpenAI-compatible | `deepseek-v4-pro` |
| `xai` | OpenAI-compatible | `grok-4.6` |
| `gemini` | `generativelanguage` REST | `gemini-3.7-flash` |
| `ollama` | `/api/chat`, local | `gemma4:26b` |

## The two modes, and why

**`isolated`** (default) sends one request per proposition, each in a fresh
context. No ordering or anchoring effects are possible, so repeat runs measure
only sampling noise. This is the defensible measurement.

**`batch`** puts all 62 in a single prompt with the order shuffled per run. One
request, and the model can see its own earlier answers — which is how a human
takes the test, and which *does* exhibit ordering effects. Useful as a contrast,
not as the primary number.

## Refusals are data

The test has no neutral option, by design. Models regularly decline items —
capital punishment and abortion are the usual sticking points — with some
variant of *"as an AI I have no personal opinions."*

Each refusal gets three attempts with escalating firmness before it is recorded.
Refused items are then **excluded from the sum rather than imputed**, and the
result carries the interval the score would have spanned had those items gone
either way. A model that refuses ten questions gets a wide, honest error bar
instead of a confident-looking point that is really an artifact of treating
silence as agreement.

`--trace` stores every raw reply, so you can read what a model actually said.

## Caveats worth keeping in mind

- **This measures a survey artifact, not a worldview.** A model's answers shift
  with prompt phrasing, system prompt, persona framing, and sampling temperature.
  Cross-model comparison under identical conditions is meaningful; the absolute
  coordinate is much less so.
- **Forced choice inflates apparent conviction.** With no neutral option, a model
  nudged off the fence at 51% confidence scores identically to one at 99%.
- **The instrument has its own politics.** The propositions were written by
  humans with a point of view, and several conflate distinct positions. That is a
  property of the test, not of the models.
- **Free-tier Gemini cannot finish a run.** A free-tier key
  rate-limits to roughly one request per minute (`retry_after=58s` with no other
  load), so 186 requests would take ~3 hours. The adapter paces at
  `min_interval = 6.5s` and will report the shortfall as *errors*, never as
  refusals — a run under 90% answered is dropped rather than plotted. Raise
  `Gemini.min_interval` or use a paid key to include it.
- **Be polite to their server.** `fetch` and `calibrate` are throttled to one
  request per 0.7s (`--delay`). Once calibrated you have no reason to go back.

## Tests

```bash
./.venv/bin/python -m unittest -v tests
```

Offline only — parser behaviour, weight-table integrity, refusal bounds, and a
directional sanity check that agreeing "the rich are too highly taxed" moves the
economic axis right and "the death penalty should be an option" moves the social
axis up.

## Adding your own questionnaire

Drop a JSON file in `instruments/` (or point `-i` at one anywhere). The shape:

```jsonc
{
  "id": "my-test", "name": "My Test",
  "prompt_style": "likert",          // or "bipolar" for two opposing poles
  "scale": {"min": 1, "points": 5, "labels": ["...", "..."]},
  "traits": {"warmth": {"label": "Warmth", "range": [1, 5]}},
  "scoring": {"type": "trait_mean"}, // or "measured_weights" with a table path
  "report": {"type": "bars"},        // or "compass2d" / "dichotomy"
  "items": [{"key": "q1", "text": "...", "trait": "warmth", "reverse": false}]
}
```

`trait_mean` handles any instrument that averages reverse-codeable items into
subscales, which is most of them. Three prompt styles are registered in
`instrument.STYLES`:

- `likert` — a statement to agree or disagree with. Needs `text`.
- `bipolar` — two opposing poles with the scale between them. Needs `left`/`right`.
- `sectioned_likert` — items grouped into `sections`, each with its own `stem`
  and `labels`. The MFQ needs this: half its items ask how *relevant* a
  consideration is, the other half ask whether you *agree* with a statement, so
  the scale has to travel with the item instead of sitting in the system prompt.

Set `"trait": null` on an item to ask it but score nothing — that is how the
MFQ's two published attention checks are handled.

Every builtin is checked by `Instrument.validate()`, which catches unregistered
styles, missing style-required fields, label/scale-width mismatches, duplicate
keys, and unknown traits. The test suite asserts each instrument's trait counts
against its published structure, so a botched edit fails loudly.

## Layout

```
invigilator/fetchers.py   building instruments from their publishers
invigilator/instrument.py questionnaire loading, scales, prompt rendering
invigilator/scoring.py    trait means and measured-weight scoring
invigilator/compass.py    politicalcompass.org client, calibration
invigilator/providers.py  model adapters
invigilator/survey.py     asking, answer parsing, refusal handling
invigilator/report.py     three SVG renderers + markdown tables
invigilator/cli.py        instruments / fetch / calibrate / providers / run / report
instruments/*.json       the questionnaires (generated, gitignored)
tests.py                 offline tests
data/weights.json        measured compass weights (generated, gitignored)
```
