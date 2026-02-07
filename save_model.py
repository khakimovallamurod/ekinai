import os
import joblib
from utils import load_and_preprocess, train_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ekin7.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("▶ Model train qilinyapti...")

df, le_dict, le_crop, crop_averages = load_and_preprocess(DATA_PATH)
rf, feature_names, metrics, feature_importances = train_model(df)

joblib.dump(rf, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(le_dict, os.path.join(MODEL_DIR, "le_dict.pkl"))
joblib.dump(le_crop, os.path.join(MODEL_DIR, "le_crop.pkl"))
joblib.dump(crop_averages, os.path.join(MODEL_DIR, "crop_averages.pkl"))
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics.pkl"))
joblib.dump(feature_importances, os.path.join(MODEL_DIR, "feature_importances.pkl"))

print("✅ Model va barcha obyektlar saqlandi:", MODEL_DIR)
