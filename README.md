# ForwardCheck

Fact-check desk for forwarded messages. See [DESIGN.md](DESIGN.md) for the full design.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env     # then fill in the Nexus key, base URL and model names
```

Without a key (or with `NEXUS_MOCK=1`) the app runs on an offline mock. The whole workflow works,
but the results are placeholders.

## Run

```powershell
# terminal 1 - backend
.venv\Scripts\python -m uvicorn backend.main:app --reload
# terminal 2 - volunteer dashboard
.venv\Scripts\python -m streamlit run frontend/app.py
```

## Test and evaluate

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python eval/run_eval.py --model A --reply
.venv\Scripts\python eval/run_eval.py --model B --reply
```

Results are saved to `eval/results/`. They include verdict accuracy, invented verdicts, and tokens and latency per task, for the Model A vs B comparison.

## Layout

| Path | What |
|---|---|
| `backend/pipeline.py` | Workflow: intake → analyse → review → reply |
| `backend/verify.py` | Code check that a quote appears word for word in the cited source (safety core) |
| `backend/source_pack.py` | Source pack loader and BM25 search |
| `backend/textutil.py` | Normalisation, link defanging, injection scan |
| `backend/nexus.py` | Nexus client (OpenAI-compatible endpoints assumed) and offline mock |
| `backend/prompts.py` | Prompts; forward text is wrapped in delimiters as data |
| `backend/main.py` | FastAPI endpoints |
| `frontend/app.py` | Streamlit volunteer dashboard |
| `source_pack/` | Source documents (the `SAMPLE-*` files are fictional; replace them) |
| `eval/cases/` | Labelled test cases (5 samples; target is 15+) |

## Still to do

- Confirm the Nexus endpoint paths and model names, and update `.env` and `backend/nexus.py` if they differ.
- Replace the `SAMPLE-*` source docs with real saved PIB, WHO and state advisory texts, with licences.
- Add 10+ more eval cases, including mock screenshots and real voice notes in `eval/files/`.
