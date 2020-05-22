
from Utils.Utils import normalize, get_accuracy
from LR.LR import LR
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def main():
    data = load_iris()
    #luam datele
    inputs, outputs = data['data'], data['target']

    indexes = [i for i in range(len(inputs))]

    train_sample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    test_sample = [i for i in indexes if not i in train_sample]

    training_inputs = [inputs[i] for i in train_sample]
    training_ouputs = [outputs[i] for i in train_sample]
    test_inputs = [inputs[i] for i in test_sample]
    test_outputs = [outputs[i] for i in test_sample]

    training_inputs_normalized, test_inputs_normalized = normalize(training_inputs, test_inputs)

    while True:
        print("1 - With tool")
        print("2 - Without tool")
        ans = input(">")
        if ans == '1':
            lr = LogisticRegression()
            lr.fit(training_inputs_normalized, training_ouputs)
            computed_outputs = lr.predict(test_inputs_normalized)

            # error = get_accuracy(test_outputs, computed_outputs)
            # print("Acc:  " + str(error))
            error = accuracy_score(test_outputs, computed_outputs)
            print("Accuracy using sklearn:  " + str(error))


        elif ans == '2':
            ##  0 = setoa, 1 = versicolor, 2 = virginica
            lr = LR(noOfClasses=3, labels=[0, 1, 2])

            lr.fit(training_inputs_normalized, training_ouputs, learning_rate=0.001, number_of_epochs=100)
            computed_outputs = lr.predict(test_inputs_normalized)

            # error = get_accuracy(test_outputs, computed_outputs)
            # print("Acc:  " + str(error))
            error = accuracy_score(test_outputs, computed_outputs)
            print("Accuracy using sklearn:  " + str(error))



main()
