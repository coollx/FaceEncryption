import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import random

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from matplotlib_inline import backend_inline

def plot_history(train_history, model_name, filename = None):
    '''
    Plot the training history.
    
    Parameters
    ----------
    train_history: a pandas DataFrame
    model_name: a string
    filename: a string
    '''
    num_epochs = train_history.shape[0]
    #plot the training metrics
    backend_inline.set_matplotlib_formats('svg') #set the backend to svg to avoid blurry images
    
    #set the figure size
    plt.figure(figsize = (8, 6))
    
    plt.plot(train_history['epoch'], train_history['train_loss'], label='train_loss', linestyle = '-', color = 'blue')
    plt.plot(train_history['epoch'], train_history['train_acc'] / 100, label='train_acc', linestyle = '-', color = 'red')
    plt.plot(train_history['epoch'], train_history['test_loss'], label='test_loss', linestyle = '--', color = 'violet')
    plt.plot(train_history['epoch'], train_history['test_acc'] / 100, label='test_acc', linestyle = '--', color = 'green')

    plt.legend()
    plt.title(model_name + ' Training History')
    
    #set the x and y axis labels
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Accuracy')
    #set the x and y axis limits
    plt.xlim(0, num_epochs-1)
    y_min, y_max = 0, max(max(max(train_history['train_loss']), max(train_history['test_loss'])) + 0.1, 1.1)
    plt.ylim(y_min, y_max)
    # #set the x and y axis ticks
    plt.xticks(np.arange(0, num_epochs-1, 2))
    plt.yticks(np.arange(y_min, y_max, 0.5))
    #add gridlines
    plt.grid(True)
    
    if filename is not None:
        plt.savefig(filename, format='svg', dpi=1200)
        
    plt.show()
