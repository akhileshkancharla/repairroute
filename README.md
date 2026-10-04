# RepairRoute

RepairRoute ranks already-faulty trucks for Air Pressure System (APS) inspection. A manager chooses an inspection capacity and sees how that decision would have performed on held-out historical cases. The tool supports a mechanic's inspection decision; it does not diagnose a failed component or predict a future breakdown.

## Working product

- A trained model and saved scores for a held-out historical test set.
- Cost-aware capacity and threshold calculations.
- A backend API for queue, policy evaluation, and scoring new records.
- A responsive three-view website based on the RepairRoute brand kit: inspection planner, batch scoring, and fixed model evidence.

The dataset defines a cost of **10 units** for an unnecessary APS check and **500 units** for a missed APS-related fault. These are benchmark units, not verified money savings.

On the separate 16,000-record test set, the validation-selected policy achieved 93.3% APS recall, 58.6% precision, 72.0% F1, and 98.3% accuracy. It incurred 14,970 benchmark cost units, versus 156,250 for inspecting every record and 187,500 for inspecting none. The full confusion matrix and ranking metrics are saved in `artifacts/metrics.json`.

## Run locally

Use Python 3.12:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn repairroute.api:app --reload
```

The checked-in `artifacts/` directory contains the trained model and a compact scored historical test set. The raw Scania CSV files are not included in this repository.

Open `http://127.0.0.1:8000/` for the website. The frontend calls the API on the same origin. Its capacity chart uses actual held-out test outcomes, while the model-evidence view shows the fixed validation-selected operating point.

Interactive API documentation is available at `http://127.0.0.1:8000/docs` while the server runs.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Check that saved model and historical artifacts load. |
| `GET /v1/metadata` | Fetch dataset size, fixed test result, and benchmark cost definitions. |
| `GET /v1/historical/decision?capacity=597` | Simulate a chosen inspection capacity on held-out historical records. |
| `GET /v1/historical/queue?capacity=597&limit=100&offset=0` | Fetch a page of selected records ranked by priority score; outcome labels are withheld. |
| `GET /v1/historical/curve?points=201` | Fetch sampled cost and caught/missed fault counts across capacity levels. |
| `POST /v1/score` | Score up to 1,000 new records with exactly the 170 anonymised Scania feature columns. |

The scoring request is JSON: `{"records": [{"aa_000": 123, "ab_000": null, "...": 0}]}` with every official feature key present in each record. Use `artifacts/sample_batch.csv` as a complete schema example. Null represents a missing value. Scoring returns input indices, ranks, and model priority scores. It does not return a diagnosis or ground-truth label.

The API reads saved artifacts at startup or on first request. Training does not run when the server receives requests. Set `REPAIRROUTE_CORS_ORIGINS` to a comma-separated list of allowed frontend origins when the frontend is hosted separately.

## Deploy on Render

The root `render.yaml` defines one free Python web service for the website and API, with `/health` as its health check. Connect this GitHub repository to Render as a Blueprint and deploy the `main` branch. The `.python-version` and pinned dependencies match the tested model artifact. No database or scheduled training job is required; the committed artifacts are read by the service. Render's free web service can sleep after inactivity, so the first visit may take longer.

To run the backend checks, use `python -m unittest discover -s tests`. The CSV parser check uses Node.js: `node tests/frontend_smoke.cjs`.

## Reproduce training and evaluation

```powershell
python download_data.py
python train.py
```

`train.py` splits the official 60,000-row training file into stratified model-training and validation portions. It fits a histogram gradient boosting classifier, chooses the default cost-minimising decision threshold on validation data, and evaluates that fixed threshold on the separate 16,000-row test file. Exploratory capacity values on the test set are **retrospective simulations**; they are not used to claim a newly selected test result. Model scores are called *priority scores* because probability calibration has not been established.

## Source and limitations

Dataset: [APS Failure at Scania Trucks, Scania CV AB, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), DOI [10.24432/C51S51](https://doi.org/10.24432/C51S51). The dataset's features are anonymised and may contain missing values. RepairRoute does not assign mechanical meanings to them. The source page lists the dataset under CC BY 4.0; attribution is retained here and in the app.

This prototype demonstrates a decision workflow on historical data. Before operational use, the model would need validation on current fleet data, monitoring for drift, an integration with workshop records, and an accountable human inspection process.

## Model improvement experiments

The current model remains the deployed choice because it had the lowest validation benchmark cost among the tested LightGBM variants: 5,050 units versus 5,240–6,160. A 200-iteration version of the same histogram boosting model achieved 4,990 on the original validation split, but a five-split stability check found a mean change of only −8 units, with two wins, one loss, and two ties. That gain is too small and inconsistent to justify replacing the tested model during this build.

The experiments use only the official training file and are recorded in `experiments/validation_comparison.json` and `experiments/stability_results.json`. To rerun the optional LightGBM comparison, install `lightgbm` in addition to the main requirements, then run `python -m experiments.compare_models`. The stability check uses the main requirements: `python -m experiments.check_stability`.
