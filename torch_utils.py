import torch
from torch import nn

class JoinedDataLoader:
    """Loader for sampling from multiple loaders with probability proportional to their length. Stops when all loaders are exausthed.
    Useful in case you can't join samples of different datasets in a single batch.
    """
    def __init__(self, loaderA, loaderB):
        self.probA = len(loaderA)/(len(loaderA)+len(loaderB))
        self.loaderAiter, self.loaderBiter = iter(loaderA), iter(loaderB)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        loader_choice = torch.rand(1).item()
        if loader_choice < self.probA:
            try:
                n = next(self.loaderAiter)
            except StopIteration:
                try:
                    n = next(self.loaderBiter)
                except:
                    raise StopIteration
        else:
            try:
                n = next(self.loaderBiter)
            except StopIteration:
                try:
                    n = next(self.loaderAiter)
                except:
                    raise StopIteration
        return n
                
    def __len__(self):
        return len(self.loaderAiter) + len(self.loaderBiter) 

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

