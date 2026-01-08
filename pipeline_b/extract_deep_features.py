import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

# =======================================
# CONFIG
# =======================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_DIM = 128

# =======================================
# MODEL
# =======================================
weights = MobileNet_V2_Weights.DEFAULT
_mobilenet = models.mobilenet_v2(weights=weights)
_mobilenet.classifier = nn.Identity()
_mobilenet.eval()
_mobilenet.to(DEVICE)

# reduce dimension to 128
_projector = nn.Linear(1280, FEATURE_DIM).to(DEVICE)
_projector.eval()

# =======================================
# TRANSFORM
# =======================================
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =======================================
# EXTRACT FEATURES
# =======================================
def extract_deep_features(img_path):
    img = Image.open(img_path).convert("RGB")
    x = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = _mobilenet(x)     # (1, 1280)
        feats = _projector(feats) # (1, 128)

    return feats.cpu().numpy().flatten()
