import torch
import torch.nn as nn
from torch_utils import conv_out_shape, same_padding
from torchvision.models import vgg16_bn

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, n_channels, num_classes):
        super().__init__()
        self.n_channels = n_channels
        #BatchNorm with affine=False is just normalization without params
        self.bn = nn.BatchNorm2d(n_channels, affine=False)
        self.embed = nn.Embedding(num_classes, n_channels * 2)
        #First half of embedding is for gamma (scale parameter)
        self.embed.weight.data[:, :n_channels].normal_(1, 0.02)
        #Second half of the embedding is for beta (bias parameter)
        self.embed.weight.data[:, n_channels:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.n_channels, 1, 1) * out + beta.view(-1, self.n_channels, 1, 1)
        return out

class MultipleConditionalBatchNorm2d(nn.Module):
    """Conditional BatchNorm to be used in the case of multiple categorical input features.
    Args:
      - n_channels (int): number of feature maps of the convolutional layer to be conditioned
      - num_classes (iterable of ints): list of the number of classes for the different feature embeddings
    """
    def __init__(self, n_channels, nums_classes, hidden_dim=32):
        super().__init__()
        self.n_channels = n_channels
        self.nums_classes = nums_classes
        #BatchNorm with affine=False is just normalization without params
        self.bn = nn.BatchNorm2d(n_channels, affine=False)
        #An embedding for each different categorical feature
        self.embeddings = nn.ModuleList([nn.Embedding(num_classes, hidden_dim) for num_classes in nums_classes])
        self.linear = nn.Linear(len(nums_classes) * hidden_dim, n_channels * 2)

    def forward(self, x, class_idxs):
        out = self.bn(x)
        concatenated_embeddings = [self.embeddings[i](class_idxs[i]) for i in range(len(class_idxs))]
        concatenated_embeddings = torch.cat(concatenated_embeddings, dim=1)
        gamma, beta = self.linear(concatenated_embeddings).chunk(2, 1)
        out = gamma.view(-1, self.n_channels, 1, 1) * out + beta.view(-1, self.n_channels, 1, 1)
        return out

class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)

class MLP(nn.Module):
    """
    A one-hidden-layer MLP.
    Takes (batch_size, 1, imsize, imsize) tensors as input, and outputs tensor of same shape.
    Outputs in range [-1,1]
    """
    def __init__(self, imsize=32, n_channels=1, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.imsize = imsize
        self.net = nn.Sequential(
            nn.Linear(imsize*imsize*n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, imsize*imsize*n_channels),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, n_channels = x.size(0), x.size(1)
        x = x.view(batch_size, -1)
        x = self.net(x)
        x = x.view(batch_size, n_channels, self.imsize, self.imsize)
        return x

class CAN(nn.Module):
    """Context Aggregation Network based on Table 3 of
       "Fast Image Processing with Fully-Convolutiona Nets".
       In the original paper: n_channels=32, n_middle_blocks=7
    """
    def __init__(self, n_channels=32, n_middle_blocks=5):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=3, padding=same_padding(3, 1)),
            AdaptiveBatchNorm2d(n_channels),
            nn.LeakyReLU(0.2),
        )
        
        #Layers from 2 to 8
        blocks = []
        for i in range(1, n_middle_blocks+1):
            d = 2**i
            blocks.append(nn.Sequential( 
                nn.Conv2d(n_channels, n_channels, kernel_size=3, dilation=d, padding=same_padding(3, d)),
                AdaptiveBatchNorm2d(n_channels),
                nn.LeakyReLU(0.2)
            ))
        self.middle_blocks = nn.Sequential(*blocks)

        self.last_blocks = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=same_padding(3, 1)),
            AdaptiveBatchNorm2d(n_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels, 3, kernel_size=1)
        )

    def forward(self, x):
        x = self.first_block(x)
        x = self.middle_blocks(x)
        x = self.last_blocks(x)
        return x

class ConditionalCAN(nn.Module):
    """Context Aggregation Network that can be conditioned by multiple categorical classes.
    Conditioning is done by conditional batch normalization based on 
    
    Expected input: (image, (class_idx1,class_idx2,...)).
    """
    def __init__(self, nums_classes, n_channels=32, n_middle_blocks=5, embedding_dim=32):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=3, padding=same_padding(3, 1)),
            AdaptiveBatchNorm2d(n_channels),
            nn.LeakyReLU(0.2),
        )
        
        #Layers from 2 to 8
        self.middle_convs = nn.ModuleList([
            nn.Conv2d(n_channels, n_channels, kernel_size=3, dilation=2**i, padding=same_padding(3, 2**i))
            for i in range(1, n_middle_blocks+1)
        ])
        self.middle_cbns = nn.ModuleList([
            MultipleConditionalBatchNorm2d(n_channels, nums_classes, hidden_dim=embedding_dim)
            for i in range(1, n_middle_blocks+1)
        ])

        self.last_blocks = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=same_padding(3, 1)),
            AdaptiveBatchNorm2d(n_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels, 3, kernel_size=1)
        )

    def forward(self, x, class_idxs):
        x = self.first_block(x)
        for conv, cbn in zip(self.middle_convs, self.middle_cbns):
            x = conv(x)
            x = cbn(x, class_idxs)
            x = nn.functional.leaky_relu(x, 0.2)
        x = self.last_blocks(x)
        return x


class UNet(nn.Module):
    #TODO
    """
      Standard Unet
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class NaiveCNN(nn.Module):
    """
    Naive CNN with a bunch of convolutions and a fully connected at the end
    Output in [-1,1]
    """
    def __init__(self, imsize=32, n_channels=1, dropout_rate=0.2):
        super().__init__()
        self.imsize = imsize

        self.conv1 = nn.Conv2d(n_channels, 4, kernel_size=5)
        current_shape = conv_out_shape((imsize, imsize), self.conv1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
        current_shape = conv_out_shape(current_shape, self.conv2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5)
        self.shape_before_dense = conv_out_shape(current_shape, self.conv3)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.linear = nn.Linear(16*self.shape_before_dense[0]*self.shape_before_dense[1], imsize*imsize) 
        
    def forward(self, x):
        batch_size, n_channels = x.size(0), x.size(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.dropout(x)
        #Reshape before fully connected
        x = x.view(batch_size, -1)
        x = torch.tanh(self.linear(x))
        x = x.view(batch_size, n_channels, self.imsize, self.imsize)
        return x

class LittleUnet(nn.Module):
    """A little U-net style CNN based on concatenations and transposed convolution
    Output in [-1,1]
    """
    def __init__(self, imsize=32, n_channels=1, initial_1by1=False):
        super().__init__()
        if initial_1by1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, 255, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(255, 255, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(255, 32, kernel_size=3, stride=2),
                AdaptiveBatchNorm2d(32),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, 32, kernel_size=3, stride=2),
                AdaptiveBatchNorm2d(32),
                nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            AdaptiveBatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3), 
            AdaptiveBatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_tran1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            AdaptiveBatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_tran2 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32, kernel_size=3, stride=2),
            AdaptiveBatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_tran3 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 1, kernel_size=4, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        x = self.conv_tran1(out3)
        x = self.conv_tran2(torch.cat((out2, x), dim=1))
        x = self.conv_tran3(torch.cat((out1, x), dim=1))
        return x


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg16_bn(True).features
        self.mean = torch.Tensor([123.68,  116.779,  103.939]).view(1,3,1,1)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Normalize the images since we have [-1,1] and vgg wants [0,1]
        x = (x*0.5)+0.5
        x = x*255 - self.mean
        x = self.model(x)
        return x

if __name__ == "__main__":
    im = torch.randn(8, 1, 32, 32)

    mlp = MLP()
    #Test mlp forward
    assert mlp(im).size() == im.size()

    naive_cnn = NaiveCNN()
    #Test naive cnn forward
    assert naive_cnn(im).size() == im.size()
    
    unet = LittleUnet(initial_1by1=True)
    #Test little unet forward
    assert unet(im).size() == im.size()
    
    im = torch.randn(8, 3, 60, 80)
    can32 = CAN()
    #Test can32 forward
    assert can32(im).size() == im.size()
    
    class_idxs = (torch.LongTensor([1]), torch.LongTensor([0]), torch.LongTensor([0]), torch.LongTensor([1]))
    nums_classes = (6, 3, 3, 4)
    cbn = MultipleConditionalBatchNorm2d(n_channels=3, nums_classes=nums_classes, hidden_dim=32)
    assert cbn(im, class_idxs).size() == im.size()
    
    c_can32 = ConditionalCAN(nums_classes)
    assert c_can32(im, class_idxs).size() == im.size()

    print("Tests run correctly!")
