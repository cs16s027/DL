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

def plot(train_points, valid_points, plot_name, plot_title):
    fig = plt.figure(0)
    ax = fig.gca()
    # Label the graph
    ax.set_title('Train vs Valid loss for {}'.format(plot_title))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # Set limits
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 25])
    # Turn on grid, add x-axis and y-axis
    ax.grid()
    # Specify colors
    colors = ['red', 'blue']
    # Plot the train and validation loss
    ax.plot(train_points[0], train_points[1], linewidth = 1, color = 'red')
    ax.plot(valid_points[0], valid_points[1], linewidth = 1, color = 'blue')

    red = plt.Line2D((0,1), (0,0), color = 'red', marker='o', linestyle = '')
    blue = plt.Line2D((0,1), (0,0), color = 'blue', marker='o', linestyle = '')
    ax.legend([red, blue], ['Training-loss', 'Validation-loss'])
    plt.savefig(plot_name)

_, train_log, valid_log, plot_name, plot_title = sys.argv
train_points = loadLog(train_log)
valid_points = loadLog(valid_log)
plot(train_points, valid_points, plot_name, plot_title)

