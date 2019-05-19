import numpy as np
class Pa:
    def __init__(self, x_train, y_train, num_of_label):
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        # self.num_of_feature = num_of_feature
        self.num_of_label = num_of_label
        self.weights = np.zeros((num_of_label,self.x_train.shape[1]))
        # self.LR = 0.001
        self.epochs = 20

    def predict(self,inputs):
        return np.argmax(np.dot(self.weights, inputs))

    def loss_function(self, wy, wy_hat, x):
        return max(0, 1 - np.dot(wy,x) + np.dot(wy_hat,x))

    def train(self):
        for e in range(self.epochs):
            # self.y_train = np.asarray(self.y_train)
            # ind = np.arange(self.x_train.shape[0])
            # np.random.shuffle(ind)
            # self.x_train = self.x_train[ind]
            # self.y_train = self.y_train[ind]
            # np.random.seed(0)
            np.random.seed(3)
            mapIndexPosition = list(zip(self.x_train, self.y_train))
            np.random.shuffle(mapIndexPosition)
            self.x_train, self.y_train = zip(*mapIndexPosition)
            self.y_train = np.asarray(self.y_train)
            self.x_train = np.asarray(self.x_train)


            for inputs, label in zip(self.x_train, self.y_train):
                prediction = self.  predict(inputs)
                if label != prediction:
                    devider = (np.power(np.linalg.norm(inputs),2) * 2)
                    l_function = self.loss_function(self.weights[label,:], self.weights[prediction,:], inputs) / devider
                    self.weights[label,:] += l_function * inputs
                    self.weights[prediction,:] -= l_function * inputs

    def accuracy(self,test_x,test_y):
        count = 0
        for x,y in zip(test_x,test_y):
            y_hat = self.predict(x)
            if y != y_hat:
                count = count + 1
        return 1-float(count/len(test_y))

    def restart_and_set_epoch(self,new_epochs):
        self.weights = np.zeros((self.num_of_label, self.x_train.shape[1]))
        self.epochs = new_epochs

        # def training(self):
    #     """
    #             epohes = 100
    #     for e in range(epohes):
    #     :return:
    #     """
    #     for x, y in zip(self.train_x, self.train_y):
    #         y_hat = np.argmax(np.dot(self.weights, x))
    #         y_hat = float(y_hat)
    #         y_hat = int(y_hat)
    #         i_y = float(y[0])
    #         i_y = int(i_y)
    #         loss = max(0, 1 - np.dot(self.weights[i_y], x) + np.dot(self.weights[y_hat], x))
    #         tau = loss / 2 * (np.power(np.linalg.norm(x, ord=2), 2))
    #         if (i_y != int(y_hat)):
    #             self.weights[i_y, :] = self.weights[i_y, :] + tau * x
    #             self.weights[y_hat, :] = self.weights[y_hat, :] - tau * x