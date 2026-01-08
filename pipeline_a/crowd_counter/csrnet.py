import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
from collections import OrderedDict

# =======================================
# CSRNet (CPU version)
# =======================================
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        from torchvision.models import vgg16

        # load VGG16 frontend
        vgg = vgg16(weights="DEFAULT")
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # dilated convolution backend
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=2, dilation=2), nn.ReLU(inplace=True)
        )

        # output layer: 1-channel density
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# =======================================
# LOAD MODEL + PREPROCESS
# =======================================
_device = torch.device("cpu")

_model = CSRNet().to(_device)
_model.eval()

def _load_weights():
    raw = torch.load("pipeline_a/crowd_counter/model.pth", map_location="cpu")

    # case A: checkpoint with state_dict
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    clean = OrderedDict()
    for k, v in state.items():
        clean[k.replace("module.", "")] = v

    _model.load_state_dict(clean, strict=False)

_load_weights()

# disable gradients for speed
for p in _model.parameters():
    p.requires_grad = False

_preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =======================================
# DENSITY ESTIMATION
# =======================================
@torch.inference_mode()
def estimate_density(img_bgr):
    """
    Input: img_bgr (OpenCV image)
    Output: relative density value (float)
    """
    # resize for CPU speedup: 1280x800 → 256x192
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 192))

    tens = _preprocess(img).unsqueeze(0).to(_device)
    density_map = _model(tens)[0, 0].cpu().numpy()

    return float(density_map.sum())
