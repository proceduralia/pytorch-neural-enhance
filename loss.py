import torch
import torch.nn as nn
from torch.nn import functional as F
from nima.inference.inference_model import InferenceModel
from ssim import SSIM
import math
import numbers

class ColorSSIM(nn.Module):
    def __init__(self,device,fidelity=None):
        super().__init__()
        self.smoothing = GaussianSmoothing(3,10,5)
        self.smoothing = self.smoothing.to(device)
        self.ssim = SSIM()
        self.w1 = 1
        if fidelity=='l1':
            self.fidelity = nn.L1Loss().to(device)
        else:
            self.w1 = 0.00001
            self.fidelity = self.color_loss


    def forward(self,original_img,target_img):
        return self.w1*self.fidelity(original_img,target_img) + (1-self.ssim(original_img,target_img))

    def color_loss(self,original_img,target_img):
        batch_size = original_img.size()[0]
        original_blur = self.smoothing(original_img)
        target_blur = self.smoothing(target_img)
        color_loss = torch.sum(torch.pow(target_blur - original_blur,2))/(2*batch_size)
        return color_loss

class NimaLoss(nn.Module):
    def __init__(self,device,gamma,fidelity):
        super().__init__()
        self.model = InferenceModel(device)
        self.fidelity = fidelity
        self.fidelity = self.fidelity.to(device)
        self.gamma = gamma

    def forward(self,x,y):
        score = self.model.predict(x)
        return self.fidelity(x,y) + self.gamma*(10 - score)


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    Example:
    smoothing = GaussianSmoothing(3, 5, 1)
    input = torch.rand(1, 3, 100, 100)
    input = F.pad(input, (2, 2, 2, 2), mode='reflect')
    output = smoothing(input)
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        return self.conv(x, weight=self.weight, groups=self.groups)
