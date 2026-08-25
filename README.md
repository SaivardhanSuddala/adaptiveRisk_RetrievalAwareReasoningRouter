# Adaptive Risk & Retrieval-Aware Reasoning Router (ARRR)

A retrieval-augmented question answering system that routes each query to one
of five reasoning strategies based on two signals estimated before
generation: how likely the query is to induce a hallucination, and how
sufficient the retrieved context is to answer it.

## Pipeline

1. Load `.txt` / `.md` documents from `docs/`, chunk them, embed them with
   `all-MiniLM-L6-v2`, and index them in ChromaDB.
2. Retrieve top-k context for a query and score it (embedding similarity,
   margin, term coverage, redundancy, source diversity).
3. Score the query's hallucination risk from surface features (length,
   risk-term presence, constraint count, multi-step signal, entity count,
   technical-term density).
4. Combine both signals into a routing decision across five strategies:
   **Direct**, **Chain-of-Thought**, **ReAct**, **Reflection**, and
   **Self-Consistency**.
5. Generate an answer through Groq using the selected strategy.
6. Score correctness against a benchmark's ground truth and check the
   answer's claims for retrieval groundedness.

## Reasoning strategies

| Strategy | Behavior |
|---|---|
| `direct` | Single pass, no reasoning trace. |
| `cot` | Single pass with an explicit step-by-step trace, parsed separately from the final answer. |
| `reflection` | Draft answer, self-critique, and revised answer in one structured completion. |
| `self_consistency` | N independent samples at temperature 0.7; final answer by majority vote, with agreement ratio reported as answer stability. |
| `react` | A Thought/Action/Observation loop with one tool — `retrieve[query]` — that calls back into the vector store mid-generation. |

## Setup

```bash
pip install -r requirements.txt
```

```powershell
$env:GROQ_API_KEY="your_key_here"
```

Optional — a comma-separated model list for a multi-model sweep (defaults to
a single model read from `GROQ_MODEL`):

```powershell
$env:GROQ_MODELS="model-a,model-b"
```

Optional — fully offline embeddings once the model is cached:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

## Usage

Run the demo (indexes `docs/`, answers one query):

```bash
python run.py
```

Route a benchmark without generating answers:

```bash
python evaluate.py --benchmark TechReasonBench-100.csv --output outputs/routing_decisions.csv
```

Generate answers for a subset, scored against ground truth:

```bash
python evaluate.py --benchmark TechReasonBench-100.csv --output outputs/generated.csv --limit 5 --generate
```

Force a single strategy across a subset (a "no router" baseline):

```bash
python evaluate.py --benchmark TechReasonBench-100.csv --output outputs/direct_only.csv --limit 5 --generate --strategy direct
```

Sweep multiple models:

```bash
python evaluate.py --benchmark TechReasonBench-100.csv --output outputs/sweep.csv --limit 5 --generate --models "model-a,model-b"
```

Compare routing conditions (random / risk-only / retrieval-only / full ARRR),
routing decisions only, no generation:

```bash
python evaluate.py --benchmark TechReasonBench-100.csv --output outputs/ablation.csv --ablation
```

## Tests

```bash
python -m pytest
```

## Module map

| File | Responsibility |
|---|---|
| `chunking.py` | Sentence-boundary-aware text chunking |
| `embeddings.py` | Query/document embedding via SentenceTransformer |
| `loader.py` | Indexes `docs/` into ChromaDB |
| `vector_store.py` | ChromaDB wrapper |
| `retrieval.py` | Retrieval + the multi-feature retrieval quality estimator |
| `router.py` | Hallucination risk estimator + the ARRR routing rule |
| `generation.py` | The five strategy implementations |
| `detector.py` | Post-hoc retrieval-groundedness check, per claim |
| `scoring.py` | Automatic correctness scoring against ground truth |
| `ablation.py` | Routing decisions under each ablation condition |
| `evaluate.py` | CLI over a benchmark CSV: route, generate, sweep, or ablate |

## Scope and known limitations

- The risk and retrieval-quality estimators are hand-tuned heuristics, not
  trained classifiers — they're a deliberate rule-based first pass, not a
  finished model.
- The ARRR routing weights and thresholds are the handbook's defaults,
  un-tuned against a labeled validation split.
- `score_answer()` is a normalized substring/token-overlap check, not exact
  grading — adequate for short mechanically-checkable answers, approximate
  for open-ended ones.
- `check_groundedness()` flags claims by lexical token overlap with retrieved
  context, not semantic span alignment.
- ReAct's only tool is the local retrieval index; it does not execute code.
- Latency and token usage are logged per generation call; GPU/memory metrics
  are not applicable to API-served models and are not tracked.
- `TechReasonBench-100.csv` is a 100-question subset across the five target
  domains; the research handbook's full design targets 500.

## Reference documents

`Research_Proposal_ARRR_Updated.pdf` and
`Agentic_Reasoning_Strategies_Research_Handbook.pdf` describe the full
research design this codebase implements a working subset of. They are not
indexed by the document loader.
