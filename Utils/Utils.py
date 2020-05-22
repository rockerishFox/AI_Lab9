import numpy as np

def normalize(train, test):
    train_n = [[] for _ in range(len(train))]
    test_n = [[] for _ in range(len(test))]

    print(len(train[0]))
    for findex in range(len(train[0])):
        f_train = [train[i][findex] for i in range(len(train))]
        f_test = [test[i][findex] for i in range(len(test))]

        medie = sum(f_train) / len(f_train)
        standard_deviation = (1 / len(f_train) * sum([(element - medie) ** 2 for element in f_train])) ** 0.5


        f_train_n = aux(f_train, medie, standard_deviation)
        f_test_n = aux(f_test, medie, standard_deviation)

        for index in range(len(f_train_n)):
            train_n[index].append(f_train_n[index])

        for index in range(len(f_test_n)):
            test_n[index].append(f_test_n[index])

    return train_n, test_n



def aux(data, medie, deviation):
    return [(element - medie) / deviation for element in data]

def get_accuracy(real, computed):
    suma = 0
    for i in range(len(real)):
        if real[i] == computed[i]:
            suma += 1
    return suma / len(real)
    
    
