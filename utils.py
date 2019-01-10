import numpy as np
import matplotlib.pyplot as plt

def display_transforms(im, ops, figsize=(12, 12)):
    """
    Apply all the transforms in ops to the image and display the result
    
    Args:
        - im (array): image to be processed
        - ops (dict): dictionary of operations to be applied
        - figsize (tuple): tuple with size of the figure
    """
    n_rows = (len(ops)+1)//2 + (0 if (len(ops)+1)%2 == 0 else 1)
    n_cols = n_rows
    plt.figure(figsize=figsize)
    
    #Display original image first
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(im);
    plt.axis('off')
    plt.title('Original')
    
    for i, op_name in enumerate(ops):
        plt.subplot(n_rows, n_cols, i+2)
        plt.imshow(ops[op_name](im))
        plt.axis('off')
        plt.title(op_name)