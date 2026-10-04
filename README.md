# RepairRoute

RepairRoute ranks already-faulty trucks for Air Pressure System (APS) inspection. A manager chooses an inspection capacity and sees how that decision would have performed on held-out historical cases. The tool supports a mechanic's inspection decision; it does not diagnose a failed component or predict a future breakdown.

## Current scope

- A trained model and saved scores for a held-out historical test set.
- Cost-aware capacity and threshold calculations.
- A backend API for queue, policy evaluation, and scoring new records (in progress).

The dashboard will be designed separately.

The dataset defines a cost of **10 units** for an unnecessary APS check and **500 units** for a missed APS-related fault. These are benchmark units, not verified money savings.

## Run locally

Use Python 3.11 or newer:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn repairroute.api:app --reload
```

The checked-in `artifacts/` directory contains the trained model and a compact scored historical test set. The raw Scania CSV files are not included in this repository.

## Reproduce training and evaluation

```powershell
python download_data.py
python train.py
```

`train.py` splits the official 60,000-row training file into stratified model-training and validation portions. It fits a histogram gradient boosting classifier, chooses the default cost-minimising decision threshold on validation data, and evaluates that fixed threshold on the separate 16,000-row test file. Exploratory capacity values on the test set are **retrospective simulations**; they are not used to claim a newly selected test result. Model scores are called *priority scores* because probability calibration has not been established.

## Source and limitations

Dataset: [APS Failure at Scania Trucks, Scania CV AB, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), DOI [10.24432/C51S51](https://doi.org/10.24432/C51S51). The dataset's features are anonymised and may contain missing values. RepairRoute does not assign mechanical meanings to them. The source page lists the dataset under CC BY 4.0; attribution is retained here and in the app.

This prototype demonstrates a decision workflow on historical data. Before operational use, the model would need validation on current fleet data, monitoring for drift, an integration with workshop records, and an accountable human inspection process.
