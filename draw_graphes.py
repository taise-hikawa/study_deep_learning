import numpy as np
import matplotlib.pylab as plt
from activation_functions import step_function
from activation_functions import sigmoid
from activation_functions import relu

def main():
    x = np.arange(-5.0, 5.0, 0.1)
    draw_step_function(x)
    draw_sigmoid(x)
    draw_relu(x)

def draw_step_function(x):
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def draw_sigmoid(x):
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def draw_relu(x):
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


if __name__ == "__main__":
    main()