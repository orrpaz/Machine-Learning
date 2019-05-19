from Perceptron import Perceptron
from Svm import Svm
from Pa import Pa
from collections import Counter
import numpy as np
import sys
def read_from_file(file_name):
    return np.genfromtxt(file_name,dtype='str',delimiter=",")

def print_predict(perecptron_pre, svm_pre, pa_pre):
    for predict1, predict2, predict3 in zip(perecptron_pre,svm_pre,pa_pre):
        print("perceptron: " + str(predict1) + ", svm: "
              + str(predict2) + ", pa: " + str(predict3))

######### normalization ###########
def min_max_norm(train_x):
    '''
    min max mormalize
    :param x_train: set
    :return: normalize set
    '''
    train = train_x.copy()
    min_x = []
    max_x = []
    rows,num_of_feature = train_x.shape
    for i in range(num_of_feature):
        min_x.append(min(train_x[:, i]))
        max_x.append(max(train_x[:, i]))
        if min_x[i] == max_x[i]:
            train[:, i] = 1
        else:
            train[:, i] = ((train[:, i]) - min_x[i]) / (max_x[i] - min_x[i])
    return train,min_x,max_x

def min_max_norm_by_min_max(data, min_train, max_train):
    copy_data = data.copy()
    rows,num_of_feature = data.shape
    for i in range(num_of_feature):
        if min_train[i] == max_train[i]:
            copy_data[:, i] = 1
        else:
            copy_data[:, i] = ((copy_data[:, i]) - min_train[i]) / (max_train[i] - min_train[i])
    return copy_data

def z_score_norm(x_train):
    train = x_train.copy()
    num_of_feature = x_train.shape[1]
    mean= np.zeros((1,num_of_feature))
    std_dev = np.zeros((1,num_of_feature))
    for i in range(num_of_feature):
        mean[:,i] = np.mean(x_train[:, i])
        std_dev[:,i] = np.std(x_train[:, i])
        if (std_dev[:,i] is 0):
            train[:, i] = 1
        else:
            train[:, i] = (train[:, i] -  mean[:,i]) / std_dev[:,i]
    return train,mean,std_dev

def z_score_norm_by_mean_std(data,mean,std):
    copy_data = data.copy()
    num_of_feature = data.shape[1]
    for i in range(num_of_feature):
        if (std[:,i] is 0):
            copy_data[:, i] = 1
        else:
            copy_data[:, i] = (copy_data[:, i] -  mean[:,i]) / std[:,i]
    return copy_data

def one_hot_encode(data):
    '''
    encoded categorical to numeric value
    :param data: data
    :return: data encoded
    '''
    data = np.asarray(data)
    dict_of_gender = {}
    d = len(data)
    if data.ndim > 1:
        gender_array_from_train = data[:,0]
        for i in range(len(gender_array_from_train)):
            if gender_array_from_train[i] not in dict_of_gender:
                dict_of_gender[gender_array_from_train[i]] = []

        list_of_genders = list(dict_of_gender.keys())
        gender_to_num = dict((c, i) for i, c in enumerate(list_of_genders))
        arr = np.zeros((len(data), len(list_of_genders)))
        for i, c in enumerate(gender_array_from_train):
            a = [0 for _ in range(len(dict_of_gender))]
            myarray = np.asarray(a)
            myarray[gender_to_num[c]] = 1
            arr[i] = myarray
        return np.append(arr, data[:, 1:], axis=1)







def main(argv):
    train_x = read_from_file(sys.argv[1])
    train_x = one_hot_encode(train_x).astype(float)

    train_y = read_from_file(sys.argv[2])
    train_y = train_y.astype(float).astype(int)
    num_of_labels = len(Counter(train_y).keys())
    # np.random.seed(5)
    # mapIndexPosition = list(zip(train_x, train_y))
    # np.random.shuffle(mapIndexPosition)
    # train_x, train_y = zip(*mapIndexPosition)
    # train_y = np.asarray(train_y)
    # train_x = np.asarray(train_x)

    ############## prediction:################
    test_x = read_from_file(sys.argv[3])
    test_x = one_hot_encode(test_x).astype(float)
    #
    # test_y = read_from_file("test_y.txt").astype(float).astype(int).tolist()

    # ###### cross validation #######
    # trains_x = [all_train_x[:657],all_train_x[657:1314],all_train_x[1314:1971],all_train_x[1971:2628],all_train_x[2628:]]
    # trains_y = [all_train_y[:657], all_train_y[657:1314], all_train_y[1314:1971], all_train_y[1971:2628],all_train_y[2628:]]

    # for K in range(5):
    #     test_x = trains_x[K]
    #     test_y = trains_y[K]
    #     train_x = []
    #     train_y = []
    #     for i in range(5):
    #         if i is not K:
    #             for example, lable in zip(trains_x[i],trains_y[i]):
    #                 train_x.append(example)
    #                 train_y.append(lable)
    #
    #     train_x = np.asarray(train_x)
    #     train_y = np.asarray(train_y)
    # d = {"I": 0, "M": 1, "F": 2}
    # temp = train_x
    # temp = scipy.stats.zscore(temp)
    train_x_Z_score, mean, std_dev = z_score_norm(train_x)
    train_x_min_max, min_train, max_train = min_max_norm(train_x)

    test_x_z_score = z_score_norm_by_mean_std(test_x, mean, std_dev)
    test_x_min_max = min_max_norm_by_min_max(test_x, min_train, max_train)

    # perceptron_z_score = Perceptron(train_x_Z_score, train_y, num_of_feature, num_of_labels)
    # svm_z_score = Svm(train_x_Z_score, train_y, num_of_feature, num_of_labels)
    # pa_z_score = Pa(train_x_Z_score, train_y, num_of_feature, num_of_labels)
    #
    # perceptron_min_max = Perceptron(train_x_min_max, train_y, num_of_feature, num_of_labels)
    # svm_min_max = Svm(train_x_min_max, train_y, num_of_feature, num_of_labels)
    # pa_min_max = Pa(train_x_min_max, train_y, num_of_feature, num_of_labels)
    ############# training:#################

    perceptron = Perceptron(train_x_min_max, train_y,num_of_labels)
    svm = Svm(train_x_min_max, train_y,num_of_labels)
    pa = Pa(train_x_Z_score, train_y,num_of_labels)
    perceptron.train()
    svm.train()
    pa.train()

    predict_pereceptron = []
    predict_svm = []
    predict_pa = []
    for test_min_max,test_z_score in zip(test_x_min_max,test_x_z_score):
       predict_pereceptron.append(perceptron.predict(test_min_max))
       predict_svm.append(svm.predict(test_min_max))
       predict_pa.append(pa.predict(test_z_score))
    # for test in test_x_z_score:
    #     predict_pa.append(pa.predict(test))

    print_predict(predict_pereceptron, predict_svm, predict_pa)
    # print("Perceptron accuracy is: ", perceptron.accuracy(test_x_min_max, test_y))
    # print("SVM accuracy is: ", svm.accuracy(test_x_min_max, test_y))
    # print("PA accuracy is: ", pa.accuracy(test_x_z_score, test_y))


if __name__ == "__main__":
    main(sys.argv[1:])