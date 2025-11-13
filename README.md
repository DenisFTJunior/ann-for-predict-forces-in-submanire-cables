# IC2 – Regression and ANN models on tabular data

This project loads a tabular Excel dataset and provides:

- Classical regressors (Linear, KNN, Random Forest, SVR, XGBoost)
- A TensorFlow/Keras ANN with hyperparameter tuning (Keras Tuner)
- Metrics, plotting, and an end‑to‑end prediction pipeline that writes results to CSV

Example dataset: `data/Resultados_Patricia-Rodada-03.xlsx`.

## What changed (summary)

- The ANN lives in two variants:
  - `model/ann_keras.py` – used by `main.py`; `optimize()` runs Keras Tuner.
  - `model/ann.py` – same structure but exposes multiple optimization backends via `utils/optimize`.
- A runnable pipeline was added: `pipelines/run_ann.py` trains/loads a saved ANN and writes a prediction row to `outputs/data/output.csv` for a given Excel ID.

## Quick start (Windows PowerShell)

```powershell
py -m pip install --upgrade pip
py -m pip install numpy pandas scikit-learn xgboost matplotlib tensorflow keras-tuner

# Train/tune ANN and save artifacts/plots
py c:\projects\ic2\main.py

# Run the prediction pipeline (writes to outputs/data/output.csv)
py c:\projects\ic2\pipelines\run_ann.py
```

Notes:

- TensorFlow and XGBoost are optional depending on which models you run.
- If you only need classical regressors: `py -m pip install numpy pandas scikit-learn matplotlib`.

## Folder structure

- `data/`
  - `DataProcessor.py` – preprocessing, normalization and train/val/test split
  - `DataTransformer.py` – chooses a loader for csv/json/excel
  - `Resultados_Patricia-Rodada-03.xlsx` – example dataset
- `model/`
  - `regression.py` – scikit‑learn/XGBoost regressors (common API)
  - `ann.py` – TensorFlow ANN with random/evolutionary/Keras‑Tuner optimization hooks
  - `ann_keras.py` – TensorFlow ANN with Keras‑Tuner‑only `optimize()` (used by `main.py`)
- `utils/`
  - `evaluate.py` – regression metrics (`MAE`, `RMSE`, `R2`)
  - `optimize.py` – random search, evolutionary GA, and Keras Tuner helpers
  - `plot.py` – predictions vs. actuals + residuals
  - `data.py` – helpers to read a row by Excel ID and upsert predictions into a CSV
  - `columns_enum.py` – enum of output column names
- `pipelines/run_ann.py` – end‑to‑end prediction (train/load → predict by ID → CSV)
- `outputs/` – generated models, logs, and plots
  - `neural_network/ANNRegressionModel-<col>.keras` – saved ANN in Keras v3 format
  - `data/output.csv` – pipeline output (original row + prediction row)
- `main.py` – example flow (uses `model/ann_keras.py`)

## Data pipeline

Implemented in `data/DataProcessor.py` and `data/DataTransformer.py`.

- Loader: `DataTransformer.get_transformer(type)` → `pd.read_csv` | `pd.read_json` | `pd.read_excel`.
- Preprocessing (`DataProcessor.preprocess()`):
  - Base feature columns are those starting with `R` or `P` (excluding `Peso do sistema`).
  - Adds `Delta yaw neutro` and the target column; filters `Delta yaw neutro < 3` and target `< 100`.
  - Standardizes features via `StandardScaler`.
- Target:
  - Default is `Tensão % nas linhas` unless a column is passed in.
  - Regressors in `model/regression.py` are built for `'% de carga nas ancoras'` by default.
- Split: train/val/test using `train_test_split` with `random_state=4` (test 20%, val 20% of remainder).

## Classical regressors – `model/regression.py`

All follow the same API: `train()`, `train_with_grid()`, `predict()`, `predict_with_grid()`.

- `LinearRegressionModel` – baseline `LinearRegression()` with small grid.
- `KNNRegressionModel` – baseline `KNeighborsRegressor(n_neighbors=5, weights='distance')`.
- `RandomForestRegressionModel` – baseline `RandomForestRegressor(n_estimators=200, random_state=4)`.
- `SVRRegressionModel` – baseline `SVR(kernel='rbf', C=1.0, epsilon=0.1)`.
- `XGBoostRegressionModel` – baseline `XGBRegressor(tree_method='hist', n_estimators=300, ...)`.

Example:

```python
from model.regression import RandomForestRegressionModel

rf = RandomForestRegressionModel('data/Resultados_Patricia-Rodada-03.xlsx')
rf.train()
pred = rf.predict()
```

## ANN models

### `model/ann_keras.py` (used by `main.py`)

- `ANNRegressionModel.train()` compiles Adam(1e‑3) with `mse`/`mae` and trains 50 epochs.
- `optimize()` uses Keras Tuner (Bayesian Optimization) via `utils.optimize.keras_tuner_optimize_ann`.
- `save(filename=None)` writes to `outputs/neural_network/ANNRegressionModel-<col>.keras` by default.
- `load(path)` loads a saved Keras model file or SavedModel directory.

Minimal usage:

```python
from model.ann_keras import ANNRegressionModel

ann = ANNRegressionModel('data/Resultados_Patricia-Rodada-03.xlsx')
ann.train()
ann.optimize()  # Keras Tuner under the hood
ann.save()
preds = ann.predict()
```

### `model/ann.py`

Same interface, but `optimize(method=...)` supports:

- `random` → `utils.optimize.random_search_ann`
- `evolutionary` → `utils.optimize.evolutionary_optimize_ann`
- `keras_tuner` → `utils.optimize.keras_tuner_optimize_ann`

## Prediction pipeline – `pipelines/run_ann.py`

End‑to‑end flow that trains or loads the ANN for a chosen output column and writes a prediction row to a CSV matching the Excel schema.

- Input file: `data/Resultados_Patricia-Rodada-03.xlsx`
- Default column predicted: `utils.columns_enum.OutputColumn.TENSAO`
- Default ID used: `6869`
- Output CSV: `outputs/data/output.csv` (original row + prediction row)

Run:

```powershell
py c:\projects\ic2\pipelines\run_ann.py
```

To change the ID or the target column, edit `col_to_predict` or the `predict_with_ann(id_value=...)` call inside `pipelines/run_ann.py`.

## Plots and metrics

- `utils/plot.py` → `plot_predictions(y_true, y_pred, title, save_path)`; supports single or multi‑output.
- `utils/evaluate.py` → `EvaluateModel.regression_metrics` returns `MAE`, `RMSE`, `R2`.

## Logs and artifacts

- Keras Tuner: `kt_logs/ann_opt/*` (oracle, trials, checkpoints).
- Saved ANN: `outputs/neural_network/ANNRegressionModel-<col>.keras`.
- Plots: saved where specified in `plot_predictions(..., save_path=...)`.

## Troubleshooting

- If TensorFlow is missing CUDA support on your machine, install the CPU packages only (as above). For GPU, follow TensorFlow’s official install guide.
- If Keras Tuner isn’t found: `py -m pip install keras-tuner`.
