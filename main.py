from model.regression import (
    LinearRegressionModel,
    KNNRegressionModel,
    RandomForestRegressionModel,
    SVRRegressionModel,
    XGBoostRegressionModel,
)
import os
from utils.evaluate import EvaluateModel
from utils.plot import plot_predictions

# sample_data could be a DataFrame or path compatible with your DataProcessor
data = os.path.join('data', 'Resultados_Patricia-Rodada-03.xlsx')

# # Linear
# lin = LinearRegressionModel(data)
# lin.train()
# y_pred_lin = lin.predict()
# metrics_lin = EvaluateModel.regression_metrics(lin.y_test, y_pred_lin)
# print("Linear Regression Metrics:", metrics_lin)

# # KNN
# knn = KNNRegressionModel(data)
# knn.train()
# y_pred_knn = knn.predict()
# metrics_knn = EvaluateModel.regression_metrics(knn.y_test, y_pred_knn)
# print("KNN Regression Metrics:", metrics_knn)

# # Random Forest
# rf = RandomForestRegressionModel(data)
# rf.train()
# y_pred_rf = rf.predict()
# metrics_rf = EvaluateModel.regression_metrics(rf.y_test, y_pred_rf)
# print("test", rf.y_test)
# print("pred", y_pred_rf)
# print("Random Forest Regression Metrics:", metrics_rf)
# plot_predictions(rf.y_test, y_pred_rf, title="Random Forest: Actual vs Predicted", save_path=os.path.join("outputs", "rf_predictions.png"))

# # SVR
# svr = SVRRegressionModel(data)
# svr.train()
# y_pred_svr = svr.predict()
# metrics_svr = EvaluateModel.regression_metrics(svr.y_test, y_pred_svr)
# print("SVR Regression Metrics:", metrics_svr)


# # XGBoost (requires xgboost installed)
# xgb = XGBoostRegressionModel(data)
# xgb.train()
# y_pred_xgb = xgb.predict()
# metrics_xgb = EvaluateModel.regression_metrics(xgb.y_test, y_pred_xgb)
# print("XGBoost Regression Metrics:", metrics_xgb)
# plot_predictions(xgb.y_test, y_pred_xgb, title="XGBoost: Actual vs Predicted", save_path=os.path.join("outputs", "xgb_predictions.png"))

# ANN
#from model.ann import ANNRegressionModel
from model.ann_keras import ANNRegressionModel
ann = ANNRegressionModel(data)
ann.train()
ann.optimize()
ann.save()
y_pred_ann = ann.predict()
metrics_ann = EvaluateModel.regression_metrics(ann.y_test, y_pred_ann)
print("ANN Regression Metrics:", metrics_ann)
plot_predictions(ann.y_test, y_pred_ann, title="ANN: Actual vs Predicted", save_path=os.path.join("outputs", "ann_predictions.png"))
