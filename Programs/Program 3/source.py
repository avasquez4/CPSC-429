#Andres Vasquez
#CPSC 429: Machine Learning
import matplotlib.pyplot as plt
import numpy as np
import math

#Open file and read data
def open_file(file_name, target_index):
    global feature_data
    feature_data = []

    global target_data
    target_data = []
    
    global data_len

    print('Opening file ...', file_name)

    file = open(file_name, 'r')
    while file:
        line  = file.readline().strip().split(',')

        if line[0] == '':
            break

        ##Feature Data
        temp = [1] + line[1:target_index] + line[target_index+1:]
        for i in range(0, len(temp)):
            try:
                temp[i] = float(temp[i])
            except (ValueError, TypeError):
                temp[i] = temp[i]
                continue
        feature_data.append(temp)

        #Target Data
        temp = line.pop(target_index)
        try:
            target_data.append(float(temp))
        except (ValueError, TypeError):
            target_data.append(0)
            continue

        data_len = len(target_data)

    file.close()
    del temp

def calculate_predictions(list_of_weights):
    weights = list_of_weights
    predictions = []

    for i in range(data_len):
        total = 0
        for j in range(len(weights)):
            total += weights[j]*feature_data[i][j]
        predictions.append(total)

    error = calculate_error(predictions)

    return predictions, error, calculate_error_delta(list_of_weights, error)

def calculate_error(list_of_predictions):
    error = []

    for i in range(data_len):
        temp_list = []
        curr_error = target_data[i] - list_of_predictions[i]
        temp_list.append(curr_error)
        temp_list.append(math.pow(curr_error, 2))
        error.append(temp_list)

    del temp_list

    return error

def calculate_error_delta(list_of_weights, list_of_errors):
    weights = list_of_weights
    error = list_of_errors

    error_delta = []

    for i in range(data_len):
        temp_list = []
        for j in range(len(weights)):
            total = error[i][0] * feature_data[i][j]
            temp_list.append(total)
        error_delta.append(temp_list)

    del temp_list

    return error_delta

def calculate_new_weights(list_of_weights, list_of_error_delta, learning_rate):
    weights = list_of_weights
    error_delta = list_of_error_delta

    for i in range(len(weights)):
        total = weights[i] + learning_rate * sum([arg[i] for arg in error_delta])
        weights[i] = total

    return weights

def print_data(list_of_predictions, list_of_errors, list_of_error_delta):
    predictions = list_of_predictions
    error = list_of_errors
    error_delta = list_of_error_delta

    print("{0:^5} {1:^10} {2:^10} {3:^10} {4:^10}".format("ID", "DATA" ,"PREDICTION", "ERROR", "SQUARED ERROR"), end = ' ')
    for i in range(len(error_delta[0])):
        print ('ERRDELTA w[{0}]'.format(i), end = ' ')
    print()

    for i in range(data_len):
        print( "{0:^5} {1:^10.2f} {2:^10.2f} {3:^10.2f} {4:^10.2f}".format(i + 1, target_data[i], predictions[i], error[i][0], error[i][1]), end = ' ')
        for j in range(len(error_delta[i])):
            print("{0:^15.2f}".format(error_delta[i][j]), end = ' ')
        print()

def compute_cost(list_of_errors):
    squared_errors = []
    for arg in list_of_errors:
        squared_errors.append(*arg[-1:])

    dot_product = 0
    for arg in squared_errors:
        dot_product += math.pow(arg, 2)

    J = (1.0 / (2 * data_len)) * dot_product

    return J

def gradient_descent(iterations, list_of_weights, learning_rate):
    weights = list_of_weights
    J_history = np.zeros(shape=(iterations, 1))

    print("Initial weights: {0}".format(weights))

    for iteration in range(iterations):
        predictions, error, error_delta = calculate_predictions(weights)

        if (iteration == 0):
            print_data(predictions, error, error_delta)
            print('\n')

        if (iteration == 1 or iteration == iterations - 1):
            print ("New weights after iteration {0}: {1}".format(iteration + 1, weights))
            print('\n')

        weights = calculate_new_weights(weights, error_delta, learning_rate)

        cost = compute_cost(error)

        J_history[iteration][0] = cost

    plt.plot(np.arange(iterations), J_history)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()


def main():
    open_file('prog3.txt', target_index = 5)
    gradient_descent(iterations = 100, list_of_weights = [-0.146, 0.185, -0.044, 0.119], learning_rate = 0.00000002)

    open_file('prog3_2.txt', target_index = 1)
    gradient_descent(iterations = 100, list_of_weights = [-59.50, -0.15, 0.60], learning_rate = 0.000002)

main()