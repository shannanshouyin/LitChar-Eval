# Collecting new model responses

New collections use the modules in `src/` and write to `results/`; the supplied article results remain in `published_data/`.

## Setup

Download and prepare the corpus:

```bash
python scripts/download_data.py
python -m src.prepare_formal
```

Copy `.env.example` to `.env` and enter credentials for GLM and DeepSeek. Provider and model identifiers are configured in `config/api_config.json`. The experimental settings are in the three protocol files. Review their estimated-cost settings against current provider prices before running a new collection.

## Main experiment

```bash
python -m src.run_formal
python -m src.run_formal --execute
python -m src.analyze_formal
```

The command without `--execute` prepares the request manifest and reports the planned count. Adding `--execute` sends billable requests.

## Matched stability experiment

```bash
python -m src.prepare_matched
python -m src.run_matched --execute
python -m src.analyze_matched
```

## Quotation-deletion experiment

```bash
python -m src.prepare_ablation --phase base
python -m src.run_ablation --phase base --execute
python -m src.prepare_ablation --phase edit
python -m src.run_ablation --phase edit --execute
python -m src.analyze_ablation
```

Run each analysis after its collection completes. The deletion edits are constructed from the new complete-text answers, so the number of eligible pairs may differ from the article. For a fresh independent experiment, use a separate checkout.

The corpus, credentials, and new request logs stay in the ignored local directories. Published numerical outcomes are provided separately from raw response text.
