# For Graphing

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def plot2d(x,y):
    green_patch = mpatches.Patch(color='green', label='NB')
    blue_patch = mpatches.Patch(color='blue', label='LR')
    plt.legend(handles=[green_patch,blue_patch],loc=4)

    plt.plot(x,y)
    plt.title("Classifier Comparisons")
    plt.ylabel("Accuracy Rate")
    plt.xlabel("Sample Size")
    plt.show()
    
plot2d([20000,50000,80000,100000,120000,150000,200000,300000,500000,700000],[0.411,0.42096,0.488,0.49816,0.4889,0.4346,0.43904,0.48921,0.531,0.5404])
plot2d([20000,50000,80000,100000,120000,150000,200000,300000,500000,700000],[0.46,0.49,0.548,0.55784,0.547,0.5078,0.5048,0.5524,0.5807,0.5811])
 