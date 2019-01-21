import string
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale

from lotusland.models.modules import WideSEBasicBlock, ResNet


class CaptchaNet:
    def __init__(self, model_path):
        self.letters = string.digits + string.ascii_letters
        self.transforms = Compose([
            Resize((48, 128)),
            Grayscale(),
            ToTensor(),
        ])
        self.net = ResNet(WideSEBasicBlock, [2, 1, 1], k=2,
                          num_targets=4, num_classes=62, with_se=True)
        d = torch.load(model_path, map_location='cpu')
        self.net.load_state_dict(d)
        self.net.eval()

    def predict(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        with torch.no_grad():
            y_pred = self.net(self.transforms(img).unsqueeze(0))
        y_pred = y_pred.argmax(dim=1).squeeze().tolist()
        chars = "".join([self.letters[ind] for ind in y_pred])
        return chars
