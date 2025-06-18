import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import haversine_distances
import torch
from torchvision import models, transforms
from PIL import Image
import os

BASE_PATH = os.path.dirname(__file__)
model_path = os.path.join(BASE_PATH, "../", "xgb_model.pkl")
scaler_path = os.path.join(BASE_PATH, "../", "scaler.pkl")
image_model_path = os.path.join(BASE_PATH, "../", "vgg16_weights.pth")
db_path = os.path.join(BASE_PATH, "../", "predicted_risk.csv")

geo_model = joblib.load(model_path)
geo_scaler = joblib.load(scaler_path)
geo_db = pd.read_csv(db_path)

image_model = models.vgg16_bn(pretrained=False)
image_model.classifier[6] = torch.nn.Linear(4096, 1)
image_model.load_state_dict(torch.load(image_model_path, map_location="cpu"))
image_model.eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_image(img: Image.Image) -> float:
    img_tensor = img_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = image_model(img_tensor)
        prob = torch.sigmoid(output).item()
    return prob

def predict_geo(lat: float, lon: float) -> float:
    coords_rad = np.radians(geo_db[["latitude", "longitude"]])
    input_rad = np.radians([[lat, lon]])
    distances = haversine_distances(coords_rad, input_rad).flatten()
    idx = np.argmin(distances)
    nearest = geo_db.iloc[idx]
    features = nearest[geo_scaler.feature_names_in_].values.reshape(1, -1)
    scaled = geo_scaler.transform(features)
    probs = geo_model.predict_proba(scaled)[0]
    weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    return float(np.dot(probs, weights))

def ensemble(img_prob: float, geo_prob: float, w_img=0.5, w_geo=0.5) -> float:
    if img_prob is None: return geo_prob
    if geo_prob is None: return img_prob
    return w_img * img_prob + w_geo * geo_prob

def predict_from_features(features: dict) -> float:
    complete = {f: 0 for f in geo_scaler.feature_names_in_}
    complete.update(features)
    df = pd.DataFrame([complete])
    df_scaled = geo_scaler.transform(df[geo_scaler.feature_names_in_])
    probs = geo_model.predict_proba(df_scaled)[0]
    
    category = np.argmax(probs)
    return category / 4  


def predict_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    features = geo_scaler.feature_names_in_
    df_scaled = geo_scaler.transform(df[features])
    probs = geo_model.predict_proba(df_scaled)
    weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    df["probability"] = probs @ weights
    df["risk_category"] = np.argmax(probs, axis=1)
    return df[["latitude", "longitude", "probability", "risk_category"] + [col for col in df.columns if col not in ["latitude", "longitude"]]]

def get_feature_importance() -> pd.DataFrame:
    importance = geo_model.feature_importances_
    features = geo_scaler.feature_names_in_
    return pd.DataFrame({"Feature": features, "Importance": importance}).sort_values("Importance", ascending=False)
