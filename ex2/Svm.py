from collections import Counter
import numpy as np
class Svm:
    def __init__(self, x_train, y_train, num_of_label):
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        # self.num_of_feature = num_of_feature
        self.num_of_label = num_of_label
        # self.weights = np.zeros(len(x_train) + 1)
        self.weights = np.zeros((num_of_label,self.x_train.shape[1]))
        self.LR = 0.01
        self.Lambda = 0.4
        self.epochs = 10
        self.counter = Counter(y_train)
        self.set_of_clssification = set(y_train)

    def predict(self,inputs):
        return np.argmax(np.dot(self.weights, inputs))

    def other(self,lable,prediction):
        # classification = [0,1,2]
        classification = self.set_of_clssification.copy()
        classification.remove(lable)
        classification.remove(prediction)
        return classification.pop()

    def train(self):
        for e in range(self.epochs):
            # shuffle
            # self.y_train = np.asarray(self.y_train)
            # ind = np.arange(self.x_train.shape[0])
            # np.random.shuffle(ind)
            # self.x_train = self.x_train[ind]
            # self.y_train = self.y_train[ind]
            mapIndexPosition = list(zip(self.x_train, self.y_train))
            np.random.shuffle(mapIndexPosition)
            self.x_train, self.y_train = zip(*mapIndexPosition)
            self.y_train = np.asarray(self.y_train)
            self.x_train = np.asarray(self.x_train)
            if e != 0:
                self.LR = self.LR / e
            for inputs, label in zip(self.x_train, self.y_train):
                prediction = self.predict(inputs)
                if label != prediction:
                    self.weights[label, :] = (1 - (self.Lambda * self.LR)) * self.weights[label, :] + self.LR * inputs
                    self.weights[prediction, :] = (1 - (self.Lambda * self.LR)) * self.weights[prediction, :] - self.LR * inputs
                    other = self.other(label, prediction)
                    self.weights[other, :] = (1 - (self.Lambda * self.LR)) * self.weights[other, :]
                else:
                    for i in range(len(self.counter)):
                        self.weights[i, :] = (1 - self.Lambda * self.LR) * self.weights[i, :]

    def accuracy(self,test_x,test_y):
        count = 0
        for x,y in zip(test_x,test_y):
            y_hat = self.predict(x)
            if y != y_hat:
                count = count + 1
        return 1-float(count/len(test_y))

    def restart_and_set_epoch_lambda_and_lr(self,new_epochs,new_lambda,new_LR):
        self.weights = np.zeros((self.num_of_label, self.x_train.shape[1]))
        self.epochs = new_epochs
        self.Lambda = new_lambda
        self.LR = new_LR