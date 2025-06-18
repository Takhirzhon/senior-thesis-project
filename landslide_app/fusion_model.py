import torch
import pickle
import numpy as np
from PIL import Image
from xgboost import XGBClassifier
from torchvision import transforms
from torchvision.models import vgg16

def load_cnn_model(weights_path="vgg16.pth", device="cpu"):
    model = vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_xgb_model(model_path="xgb_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

def fusion_predict(image_tensor, tabular_features, cnn_model, xgb_model, device='cpu'):
    cnn_model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        cnn_prob = torch.sigmoid(cnn_model(image_tensor)).cpu().item()
        cnn_class = int(cnn_prob > 0.5) * 4 
         
    xgb_probs = xgb_model.predict_proba([tabular_features])[0]
    xgb_class = int(np.argmax(xgb_probs))

    final_class = round(0.5 * cnn_class + 0.5 * xgb_class)
    return final_class, cnn_class, xgb_class

def run_demo(image_path, feature_list, cnn_weights="vgg16.pth", xgb_model_path="xgb_model.pkl"):
    print("\n🚀 Starting Fusion Prediction")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnn_model = load_cnn_model(cnn_weights, device)
    xgb_model = load_xgb_model(xgb_model_path)
    image_tensor = preprocess_image(image_path)

    final_class, cnn_class, xgb_class = fusion_predict(image_tensor, feature_list, cnn_model, xgb_model, device)

    label = ["🟢 Very Low", "🟡 Low", "🟠 Moderate", "🔴 High", "🚨 Very High"]

    print(f"\n🧠 CNN Estimated Class:     {cnn_class} - {label[cnn_class]}")
    print(f"📊 XGBoost Predicted Class: {xgb_class} - {label[xgb_class]}")
    print(f"⚠️ Final Fused Class:       {final_class} - {label[final_class]}")
    print("✅ Prediction complete!")

