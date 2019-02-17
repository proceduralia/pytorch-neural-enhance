from gaussian import GaussianSmoothing
from models import VGG
import torch
import torch.nn as nn
from nima.inference.inference_model import InferenceModel
from ssim import SSIM
class ColorContentLoss(nn.Module):
    def __init__(self,device):
        super(ColorContentLoss, self).__init__()
        #self.vgg = VGG(device)
        #self.vgg = self.vgg.to(device)
        self.smoothing = GaussianSmoothing(3,10,5)
        self.smoothing = self.smoothing.to(device)
        self.ssim = SSIM()
        self.w1 = 0.00001
        self.fidelity = nn.L1Loss().to(device)
        
    def __call__(self,original_img,target_img):
        #return self.w1*self.color_loss(original_img,target_img) + (1-self.ssim(original_img,target_img)) #+ self.tv_loss(original_img,target_img)
        return self.fidelity(original_img,target_img) + (1-self.ssim(original_img,target_img))
        
    def content_loss(self,original_img,target_img):
        _, c1, h1, w1 = original_img.size()
        chw1 = c1 * h1 * w1
        vgg_original = self.vgg(original_img).detach()
        vgg_enhanched = self.vgg(target_img)
        content_loss = 1.0/chw1 * nn.MSELoss()(vgg_enhanched,vgg_original)
        #print(content_loss)
        return content_loss

    def color_loss(self,original_img,target_img):
        batch_size = original_img.size()[0]
        original_blur = self.smoothing(original_img)
        target_blur = self.smoothing(target_img)
        color_loss = torch.sum(torch.pow(target_blur - original_blur,2))/(2*batch_size)
        #print(color_loss)
        return color_loss


    def tv_loss(self, enhanced_img, target_img):
        bsize, chan, height, width = target_img.size()
        errors = []
        dt = torch.abs(target_img[:,:,1:,:] - target_img[:,:,:-1,:])
        de = torch.abs(enhanced_img[:,:,1:,:] - enhanced_img[:,:,:-1,:])
        error = torch.norm(dt - de, 1)
        return error / height

class NimaLoss(nn.Module):
    def __init__(self,device,gamma):
        super(NimaLoss,self).__init__()
        self.model = InferenceModel(device)
        self.fidelity = nn.MSELoss()
        self.fidelity = self.fidelity.to(device)
        self.gamma = gamma

    def forward(self,x,y): 
        score = self.model.predict(x)
        return self.fidelity(x,y) + self.gamma*(10 - score)
