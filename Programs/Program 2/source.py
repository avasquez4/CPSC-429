#Andres Vasquez
#CPSC 429: Machine Learning
from sklearn.preprocessing import MinMaxScaler
import math

list_of_countries, data, normalized_data, list_of_cpi = [], [], [], []

#Open file and read data
def open_file(file_name):
    file = open(file_name, 'r')
    while file:
        line  = file.readline().strip()
        if line == "":
            break
        ##Countries
        list_of_countries.append(line.split(',').pop(0))
        ##Data
        temp = line.split(',')[1:][:-1]
        for i in range(0, len(temp)):
            temp[i] = float(temp[i])
        data.append(temp)
        ##CPI
        temp = line.split(',')[-1:]
        for i in range(0, len(temp)):
            temp[i] = float(temp[i])
            list_of_cpi.append(temp[i])
    file.close()
    del temp

#Calculate Euclid Dists
def calculate_euclidean_dists(_data, _query_data):
    global list_of_euclidean_dists
    list_of_euclidean_dists = []
    for i in range(0, len(_data)):
        sum = 0
        for j in range(0, len(_data[i])):
            sum+=math.pow(_query_data[j]-_data[i][j], 2)
        list_of_euclidean_dists.append(round(math.sqrt(sum), 4))

#(Unweighted data) Calculate and output nearest neighbor(s) and predicted CPI
def calculate_nn(k):
    calcuation = round(sum(list_of_cpi[:k]) / k , 4)
    print("CPI for {0}-NN: {1}".format(k, calcuation),'\n')

#Calculate weights for each country
def calculate_weighted_data():
    global list_of_weights
    list_of_weights = []
    for arg in list_of_euclidean_dists:
        list_of_weights.append(round(1/math.pow(arg,2), 4))
    
    global list_of_weighted_cpi
    list_of_weighted_cpi = []
    for i in range(0, len(list_of_cpi)):
        list_of_weighted_cpi.append(round(list_of_cpi[i] * list_of_weights[i], 4))

#(Weighted data) Calculate and output nearest neighbor(s) and predicted CPI
def calculate_weighted_nn(k):
    weight_total, cpi_total = 0, 0
    for i in range(k):
        weight_total += list_of_weights[i]
        cpi_total += list_of_weighted_cpi[i]

    calcuation = round(cpi_total/weight_total, 4)
    print("CPI for {0}-NN:".format(k), calcuation, '\n')

#Normalized data using Scikit-Learn MinMaxScalar
def normalize_data(_data):
    data_normalized = MinMaxScaler().fit_transform(_data)
    return data_normalized

def print_data(is_weighted_data):
    if not (is_weighted_data):
        print ("{0:<20} {1:<10} {2:<10}".format('Country','Euclid','CPI'))
        for i in range(0, len(list_of_countries)):
            print ("{0:<20} {1:<10} {2:<10}".format(list_of_countries[i], list_of_euclidean_dists[i], list_of_cpi[i]))
    elif (is_weighted_data):
        print ("{0:<20} {1:<10} {2:<10} {3:<10} {4:<10}".format('Country','Euclid','CPI', 'Weight', 'W*CPI'))
        for i in range(0, len(list_of_countries)):
            print ("{0:<20} {1:<10} {2:<10} {3:<10} {4:<10}".format(list_of_countries[i], list_of_euclidean_dists[i], list_of_cpi[i], list_of_weights[i], list_of_weighted_cpi[i]))

open_file('CPI.txt')
queryCountry = 'Russia'

#Unnormalized Data:
query_data = [67.62, 31.68, 10.00, 3.87, 12.90]

##Unweighted Data:
calculate_euclidean_dists(data, query_data)
###Sort lists based on Euclid Dist
list_of_euclidean_dists, data, list_of_cpi, list_of_countries = (list(t) for t in zip(*sorted(zip(list_of_euclidean_dists, data, list_of_cpi, list_of_countries))))
print_data(is_weighted_data = False)
calculate_nn(3)
##Weighted Data:
calculate_weighted_data()
print_data(is_weighted_data = True)
calculate_weighted_nn(16)

#Normalized Data:
normalized_query = [0.6099, 0.3754, 0.0948, 0.5658, 0.9058]
normalized_data = normalize_data(data)

##Unweighted Data:
calculate_euclidean_dists(normalized_data, normalized_query)
###Sort lists based on Euclid Dist
list_of_euclidean_dists, normalized_data, list_of_cpi, list_of_countries = (list(t) for t in zip(*sorted(zip(list_of_euclidean_dists, normalized_data, list_of_cpi, list_of_countries))))
print_data(is_weighted_data = False)
calculate_nn(3)
##Weighted Data:
calculate_weighted_data()
print_data(is_weighted_data = True)
calculate_weighted_nn(16)