import torch
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F
from decouple import config

from nima.model import NIMA
from nima.common import Transform, get_mean_score, get_std_score
from nima.common import download_file
from nima.inference.utils import format_output

class InferenceModel:
    def __init__(self,device):
        self.transform = Transform().eval_transform
        self.model = NIMA(pretrained_base_model=True)
        self.model = self.model.to(device)
        self.model.eval()

    def predict_from_file(self, image_path):
        image = default_loader(image_path)
        return self.predict(image)

    def predict_from_pil_image(self, image):
        image = image.convert('RGB')
        return self.predict(image)

    def predict(self, image):
        image = image*0.5 + 0.5 #rescale from [-1,1]-->[0,1]
        image = F.interpolate(image,size=(224,224),mode='bilinear')
        with torch.no_grad():
          prob = self.model(image).data.cpu().numpy()[0]

          mean_score = get_mean_score(prob)
          std_score = get_std_score(prob)
          return mean_score+std_score
