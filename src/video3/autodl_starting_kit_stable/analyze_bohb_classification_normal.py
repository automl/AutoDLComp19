import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# parsing the log files manually is actually the quickest solution...
x_data = ["8","16","32","64","128","256","512"]
y_data = [0.949405093356336,
          0.9554580903987269,
          0.9756209018393189,
          0.9924173049173051,
          0.9951697770217307,
          0.9974747474747474,
          0.9898989898989898]

def plot_accuracy():
    plt.figure(figsize=(6,3))
    #sns.boxplot(x=data, y=name_list, order=name_list_sorted)
    ax = plt.plot(y_data, lw=2)
    #ax.set_xticklabels(x_data)
    plt.xlabel('batch size')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(7), x_data)
    plt.show()

if __name__ == "__main__":
    plot_accuracy()








































