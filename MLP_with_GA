import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from geneticalgorithm import geneticalgorithm as ga

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the objective function to minimize
def objective_function(params):
    # Reshape the parameter vector into the weights and biases of the MLP
    weights1 = params[0:12].reshape((4,3))
    biases1 = params[12:15]
    weights2 = params[15:18].reshape((3,1))
    biases2 = params[18]
    
    # Define the MLP with the current weights and biases
    model = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', solver='adam', random_state=42)
    model.coefs_ = [weights1, weights2]
    model.intercepts_ = [biases1, biases2]
    
    # Train the MLP with the current weights and biases
    model.fit(X_train, y_train)
    
    # Return the error rate on the training set as the fitness value to minimize
    return model.score(X_train, y_train)

# Define the bounds for the weights and biases of the MLP
bounds = np.array([[0, 1]] * 19)

# Use the GA algorithm to find the optimal weights and biases of the MLP
algorithm_param = {'max_num_iteration': 1000, 'population_size': 100, 'mutation_probability': 0.1, 'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform', 'max_iteration_without_improv': 200}
model = ga(function=objective_function, dimension=19, variable_type='real', variable_boundaries=bounds, algorithm_parameters=algorithm_param)
model.run()

# Reshape the optimal parameter vector into the weights and biases of the MLP
weights1 = model.output_dict['variable'][0:12].reshape((4,3))
biases1 = model.output_dict['variable'][12:15]
weights2 = model.output_dict['variable'][15:18].reshape((3,1))
biases2 = model.output_dict['variable'][18]

# Define the MLP with the optimal weights and biases
model = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', solver='adam', random_state=42)
model.coefs_ = [weights1, weights2]
model.intercepts_ = [biases1, biases2]

# Evaluate the MLP on the testing set
score = model.score(X_test, y_test)
print("Test accuracy: {:.2f}%".format(score * 100))
