import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# parsing the log files manually is actually the quickest solution...
x_data = ["4","8","16","32","64","128","256","512","1024"]
y_data = [0.8797719371024364,
          0.9258625836798763,
          0.9602057318895961,
          0.9790404418009239,
          0.9872927408755435,
          0.9906736446591519,
          0.9935587761674718,
          0.9969135802469135,
          0.9753086419753088]

def plot_accuracy():
    plt.figure(figsize=(5,3))
    #sns.boxplot(x=data, y=name_list, order=name_list_sorted)
    ax = plt.plot(y_data, lw=2)
    #ax.set_xticklabels(x_data)
    plt.xlabel('batch size')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(9), x_data)
    plt.savefig("batch_size_acc.svg", format="svg")
    plt.show()

if __name__ == "__main__":
    plot_accuracy()








































