import numpy as np
class Perceptron:
    def __init__(self, x_train, y_train, num_of_label):
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        # self.num_of_feature = self.x_train.shape[1]
        self.num_of_label = num_of_label
        self.weights = np.zeros((num_of_label,self.x_train.shape[1]))
        self.eta = 0.01
        self.epochs = 20

    def predict(self,inputs):
        return np.argmax(np.dot(self.weights, inputs))

    def train(self):
        for e in range(self.epochs):
            # shuffle
            mapIndexPosition = list(zip(self.x_train, self.y_train))
            np.random.shuffle(mapIndexPosition)
            self.x_train, self.y_train = zip(*mapIndexPosition)
            self.y_train = np.asarray(self.y_train)
            self.x_train = np.asarray(self.x_train)
            # ind = np.arange(self.x_train.shape[0])
            # np.random.shuffle(ind)
            # self.x_train = self.x_train[ind]
            # self.y_train = self.y_train[ind]
            if e != 0:
                self.eta = self.eta / e
            # c = zip(self.x_train.tolist(),self.y_train)
            # random.shuffle(c)
            # self.x_train,self.y_train = zip(*c)
            for inputs, label in zip(self.x_train, self.y_train):
                prediction = self.predict(inputs)
                if label != prediction:
                    self.weights[label, :] += self.eta * (inputs)
                    self.weights[prediction, :] -= self.eta * (inputs)

    def accuracy(self,test_x,test_y):
        count = 0
        for x,y in zip(test_x,test_y):
            y_hat = self.predict(x)
            if y != int(y_hat):
                count = count + 1
        return 1 - float(count/len(test_y))

    def restart_and_set_epoch_and_lr(self,new_epochs,new_LR):
        self.weights = np.zeros((self.num_of_label, self.x_train.shape[1]))
        self.epochs = new_epochs
        self.LR = new_LR