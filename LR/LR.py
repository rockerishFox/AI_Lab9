from math import exp


def sig(x):
    return 1 / (1 + exp(-x))


class LR:
    def __init__(self, noOfClasses=0, labels=[]):
        self.__intercept = [0.0 for _ in range(noOfClasses)]
        self.__w = [[] for _ in range(noOfClasses)]
        self.__classesNr = noOfClasses # solutiile posibile
        self.__labels = labels


    @property
    def coefficients(self):
        return self.__w

    @property
    def intercept(self):
        return self.__intercept

    def fit(self, inputs, outputs, learning_rate=0.001, number_of_epochs=150):
        for ind in range(self.__classesNr):
            label_class = self.__labels[ind]

            out_class = [1 if outputs[i] == label_class else 0 for i in range(len(outputs))]

            self.__w[ind] = [0.0 for _ in range(len(inputs[0]) + 1)]

            # ne folosim de SGD pentru calcularea si identificarea coeficientilor
            for _ in range(number_of_epochs):
                new_coefficients = self.__w[ind]

                for i in range(len(inputs)):

                    error = sig(self.eval(inputs[i], self.__w[ind])) - out_class[i]

                    for j in range(len(inputs[0])):
                        # formula curs
                        new_coefficients[j + 1] = self.__w[ind][j + 1] - learning_rate * error * inputs[i][j]

                    new_coefficients[0] = self.__w[ind][0] - learning_rate * error

                self.__w[ind] = new_coefficients

            #update coef
            self.__intercept[ind] = self.__w[ind][0]
            self.__w[ind] = self.__w[ind][1:]
         
         
    def eval(self, xi, coef):
        #evaluation
        yi = coef[0]
        for j in range(len(xi)):
            yi += coef[j + 1] * xi[j]
        return yi


    #
    # def predictOneSample(self, sampleFeatures):
    #     threshold = 0.5
    #     coefficients = [self.__intercept] + [c for c in self.coef_]
    #     computedFloatValue = self.eval(sampleFeatures, coefficients)
    #     computed01Value = sigmoid(computedFloatValue)
    #     computedLabel = 0 if computed01Value < threshold else 1
    #     return computedLabel

    def predict(self, data):
        #predicting data
        out = []
        for i in range(len(data)):
            rez = []

            for clasa in range(self.__classesNr):
                eval = self.eval(data[i], [self.__intercept[clasa]] + self.__w[clasa])
                rez.append([sig(eval)])

            # output e format din pozitia labelurilor cu valorile cele mai mari
            out.append(self.__labels[rez.index(max(rez))])

        return out

