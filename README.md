# LitChar-Eval

Code and numerical results for *Textual Evidence and Judgment Stability in Large Language Model Assisted Literary Character Analysis*, by Zhu Li, Yanjun Lin, Yuehui Liu, and Rongkai Zhang.

The study examines how quotations relate to character judgments made by GLM-4.6 and DeepSeek flash on PersoNet. A six-condition experiment on 600 source groups is followed by matched repetition and option rotation on 120 groups, then deletion of cited words with equal-length controls. The analyses connect quotation traceability, agreement with reference labels, and changes in judgment.

## Reproduce the reported analyses

Use Python 3.12 or later. From this directory:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_results.py
python scripts/make_figures.py
```

The first script recalculates book-level contrasts, confidence intervals, sign-flip tests, Holm adjustments, and author-level sensitivity from the included numerical results. It also exports the manuscript tables. The second generates Figures 1–4 and A1 in PNG, PDF, and SVG formats.

Results are written to `outputs/`. This workflow uses the included data and requires no model credentials. The original experiment used Python 3.13.13; `requirements-original.txt` records its environment.

## Repository contents

| Directory | Contents |
| --- | --- |
| `src/` | Sample preparation, prompt construction, model collection, scoring, and statistical analysis |
| `config/` | Model settings, experimental protocols, and book–author mapping |
| `published_data/` | Numerical results, paired outcomes, figure source data, tables, and exact prompts |
| `scripts/` | Published-result analysis, figure generation, and corpus download |
| `docs/STUDY_PROTOCOL.md` | Sampling, model access, collection dates, response handling, and case-selection details |
| `docs/COLLECTION.md` | Commands for a new model collection |

The main study comprises 7,200 primary and 1,680 auxiliary tasks. The matched follow-up comprises 1,440 responses; the deletion follow-up comprises 240 fresh anchors and 224 edited-text responses.

## Data and models

The original corpus is [PersoNet](https://github.com/Gorov/personet_acl23), introduced by Yu et al. (ACL 2023), [Personality Understanding of Fictional Characters during Book Reading](https://doi.org/10.18653/v1/2023.acl-long.826). The download script retrieves the study revision.

The reported responses were collected on 22–23 October 2026 using GLM-4.6 and DeepSeek flash (documented at collection as DeepSeek-V4.1-Flash-1017). Provider updates may change the outputs of a new collection. The `openai` Python package supplies the compatible client used to access these two providers.

## Citation

Use `CITATION.cff` for the software attribution. The article title and author list are given above; the publication citation can be added when available.
