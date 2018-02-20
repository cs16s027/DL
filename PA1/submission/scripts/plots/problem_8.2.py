import os
import sys
from matplotlib import pyplot as plt

def loadLog(log_name):
    log = [line.strip().split(',') for line in open(log_name, 'r').readlines()]
    epochs, losses = [], []
    for item in log:
        epoch = int(item[0].split()[1])
        loss = float(item[2].split(': ')[-1])
        epochs.append(epoch)
        losses.append(loss)
    return epochs, losses

def plot(points, plot_name, plot_title):
    fig = plt.figure(0)
    ax = fig.gca()
    # Label the graph
    ax.set_title('{} loss for different batch sizes'.format(plot_title))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # Set limits
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 25])
    # Turn on grid, add x-axis and y-axis
    ax.grid()
    # Plot the train and validation loss
    ax.plot(points['1'][0], points['1'][1], linewidth = 1, color = 'red')
    ax.plot(points['20'][0], points['20'][1], linewidth = 1, color = 'blue')
    ax.plot(points['100'][0], points['100'][1], linewidth = 1, color = 'green')
    ax.plot(points['1000'][0], points['1000'][1], linewidth = 1, color = 'black')

    red = plt.Line2D((0,1), (0,0), color = 'red', marker='o', linestyle = '')
    blue = plt.Line2D((0,1), (0,0), color = 'blue', marker='o', linestyle = '')
    green = plt.Line2D((0,1), (0,0), color = 'green', marker='o', linestyle = '')
    black = plt.Line2D((0,1), (0,0), color = 'black', marker='o', linestyle = '')
    ax.legend([red, blue, green, black], ['1', '20', '100', '1000'])
    plt.savefig(plot_name)

_, stage, plot_name = sys.argv

points = {}
for item in os.listdir('logs/problem_8'):
    stage_ = item.split('.')[-2]
    batch_size = item.split('-')[-4]
    if stage.lower() == stage_:
        points[batch_size] = loadLog(os.path.join('logs/problem_8', item))
    
plot(points, plot_name, stage)

