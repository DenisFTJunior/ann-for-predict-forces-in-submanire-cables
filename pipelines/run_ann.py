import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.ann import ANNRegressionModel
from sklearn.preprocessing import StandardScaler
from utils.data import read_row_by_id, _read_table, upsert_prediction_row
import numpy as np
from utils.columns_enum import OutputColumn

data_path = os.path.join('data', 'Resultados_Patricia-Rodada-03.xlsx')
output_csv_path = os.path.join('outputs','data', 'output.csv')
col_to_predict = OutputColumn.TENSAO.value

def get_model_filename(col_name: OutputColumn | str) -> str:
    """Generate a model filename based on the output column."""
    if isinstance(col_name, OutputColumn):
        col_str = col_name.value
    else:
        col_str = col_name
    return f"ANNRegressionModel-{col_str}.keras"


def create_or_load_ann(): 
    file = get_model_filename(col_to_predict)
    model_dir = os.path.join('outputs', 'neural_network')
    model_path = os.path.join(model_dir, file)
    if os.path.exists(model_path):
        print(f"Loading ANN model from {model_path} ...")
        return ANNRegressionModel.load(model_path)
    # Train a new model and save with consistent filename
    print("Training new ANN model...")
    ann = ANNRegressionModel(data_path, col=col_to_predict)
    ann.train()
    os.makedirs(model_dir, exist_ok=True)
    ann.save(filename=file)
    return ann
    
def prepare_row(id_value):
    data = _read_table(data_path)
    delta_col = 'Delta yaw neutro'
    base_cols = [c for c in data.columns if (c.startswith('R') or c.startswith('P')) and c != 'Peso do sistema']
    row = read_row_by_id(data_path, id_value)
    if row is None:
        raise ValueError(f"ID {id_value} not found.")
    # Build feature vector in training order
    feature_cols = base_cols + [delta_col]              # matches DataProcessor (base + delta)
    print("row", row)
    print("feature_cols", feature_cols)
    try:
        input_values = [row[c] for c in feature_cols]
    except KeyError as e:
        raise KeyError(f"Missing column in row: {e}")
    print("input_values before scaling:", input_values)

    x_raw = np.array(input_values, dtype=float).reshape(1, -1)
    print("x_raw reshaped:", x_raw)

 
    try:
        data = data[data[delta_col] < 3]
        data =  data[data[col_to_predict] < 100]
        x_all = data[feature_cols].copy()
        print("x_all before scaling:", x_all)
    except KeyError as e:
        raise KeyError(f"While fitting scaler missing column: {e}")
    scaler = StandardScaler().fit(x_all)

    print("x_raw before scaling:", x_raw)
    print("Scaler mean:", scaler.mean_)
    print("Scaler scale:", scaler.scale_)
    print("x_all", x_all)
    x_scaled = scaler.transform(x_raw)
    print("x_scaled after scaling:", x_scaled)
    return x_scaled

def predict_with_ann(id_value=6869): 
    input_data = prepare_row(id_value)
    print(input_data)
    ann_model = create_or_load_ann()
    # input_data is already scaled 2D array
    prediction = ann_model.predict(input_data)[0][0]
    print("Prediction:", prediction)
    
    upsert_prediction_row(
        data_path,
        output_csv_path,
        id_value,
        col_to_predict,
        prediction
    )
    return prediction.ravel() if hasattr(prediction, 'ravel') else prediction

predict_with_ann()