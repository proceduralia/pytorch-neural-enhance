import torch
import torch.nn as nn

def conv_out_shape(dims, conv):
    """Computes the output shape for given convolution module
    Args:
        dims (tuples): a tuple of kind (w, h)
        conv (module): a pytorch convolutional module
    """
    kernel_size, stride, pad, dilation = conv.kernel_size, conv.stride, conv.padding, conv.dilation
    return tuple(int(((dims[i] + (2 * pad[i]) - (dilation[i]*(kernel_size[i]-1))-1)/stride[i])+1) for i in range(len(dims)))

def general_same_padding(i, k, d=1, s=1, dims=2):
    """Compute the padding to obtain the same output shape when using convolution
    Args: 
      - input_size, kernel_size, dilation, stride (tuple or ints)
      - dims (int): number of dimensions for the padding
    """
    #Convert i, k and d to tuples if they are int
    i = tuple([i for j in range(dims)]) if type(i) == int else i
    k = tuple([k for j in range(dims)]) if type(k) == int else k
    d = tuple([d for j in range(dims)]) if type(d) == int else d
    s = tuple([s for j in range(dims)]) if type(s) == int else s
    
    return tuple([int(0.5*(d[j]*(k[j]-1)-(1-i[j])*(s[j]-1))) for j in range(dims)])

def same_padding(k, d=1, dims=2):
    """Compute the padding to obtain the same output shape when using convolution,
       considering the case when the stride is unitary
    Args: 
      - input_size, kernel_size, dilation, stride (tuple or ints)
      - dims (int): number of dimensions for the padding
    """
    #Convert i, k and d to tuples if they are int
    k = tuple([k for j in range(dims)]) if type(k) == int else k
    d = tuple([d for j in range(dims)]) if type(d) == int else d
    
    return tuple([int(0.5*(d[j]*(k[j]-1))) for j in range(dims)])


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
       "Fast Image Processing with Fully-Convolutiona Nets"
    """
    def __init__(self, n_channels=32):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=3, padding=same_padding(3, 1)),
            AdaptiveBatchNorm2d(n_channels),
            nn.LeakyReLU(0.2),
        )
        
        #Layers from 2 to 8
        blocks = []
        for i in range(1, 8):
            d = 2**1
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
    
    im = torch.randn(8, 3, 300, 200)
    can32 = CAN()
    #Test can32 forward
    assert can32(im).size() == im.size()
