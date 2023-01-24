# 13. Creating & Visualizing Neural Network for the given data. (Use python)
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork():
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )
        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target)
            self._update_parameters(derror_dbias, derror_dweights)
            if current_iteration % 100 == 0:
                cumulative_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]
                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)
                    cumulative_error = cumulative_error + error
                    cumulative_errors.append(cumulative_error)
        return cumulative_errors


learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)

input_vectors = np.array([
    [3, 1.5],
    [2, 1],
    [4, 1.5],
    [3, 4],
    [3.5, 0.5],
    [2, 0.5],
    [5.5, 1],
    [1, 1],
]
)
targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)
plt.plot(training_error)
plt.xlabel("Iteons")
plt.ylabel("Error for ratiall training instances")
plt.savefig("LAB 13 Network.png")






# Second Network 
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    for n, layer_size in enumerate(layer_sizes):
         layer_top = v_spacing*(layer_size - 1)/2. + (top +bottom)/2.
         for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top -m*v_spacing), v_spacing/4.,color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top +bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top +bottom)/2.
        
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n +1)*h_spacing + left],[layer_top_a - m*v_spacing,layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
            
    for n, (layer_size) in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top +bottom) / 2.
        for m in range(layer_size):
            center = (n * h_spacing + left, layer_top - m *v_spacing)
            radius = v_spacing / 4.
            if n > 0:
                wedge_left = Wedge(center, r=radius, theta1=90,theta2=270, color='w', fc='g', ec='k', zorder=4)
                wedge_right = Wedge(center, r=radius,theta1=270, theta2=90, color='w', fc='r', ec='k', zorder=4)
                ax.add_artist(wedge_left)
                ax.add_artist(wedge_right)
            else:
                circle = plt.Circle(center, radius, color='w',ec='k', zorder=4)
                ax.add_artist(circle)

fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, [4, 7, 2])
fig.savefig('LAB 13 Network 2.png')

