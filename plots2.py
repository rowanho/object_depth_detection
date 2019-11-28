import matplotlib.pyplot as plt
import numpy as np

# Given  a grayscale image, plots the relevant histogram
# Saves as filename name_to_save

def plot_histogram(img, name_to_save, plot_name=''):
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    #plt.title(plot_name)
    plt.hist(img.ravel(), 256, [0,np.max(img)])
    plt.savefig(name_to_save)
    plt.close()
    
