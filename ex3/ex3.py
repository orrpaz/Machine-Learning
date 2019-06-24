import numpy as np


def relu(x):
    return np.maximum(0, x)


def dx_relu(x):
    return (x > 0).astype(np.float)


def softmax(x):
    exps = np.exp(x)
    return exps / exps.sum()


def loss_func(prediction, y):
    return -np.log(prediction[int(y)])


def foward_prop(x,y,params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1,x) + b1
    h1 = relu(z1)
    z2 = np.dot(W2, h1) + b2
    y_predict = softmax(z2)
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'y_predict': y_predict}
    for key in params:
        ret[key] = params[key]
    return ret


def back_prop(fprop_cache):
    x, y, z1, h1, z2, h2, W2 = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'y_predict', 'W2')]
    y_prob_vec = np.zeros((10, 1))
    y_prob_vec[int(y)] = 1
    dz2 = (h2 - y_prob_vec)  # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = np.copy(dz2)  # dL/dz2 * dz2/db2
    dz1 = np.dot(W2.T, dz2) * dx_relu(z1)  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = np.copy(dz1)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def train(epochs, eta, train_x, train_y, weight_bias_params):
    for e in range(epochs):
        loss_sum = 0
        train_x, train_y = shuffle(train_x,train_y)
        for x, y in zip(train_x, train_y):
            fprop_cache = foward_prop(x.reshape(x.size, 1), y, weight_bias_params)
            loss = loss_func(fprop_cache['y_predict'], y)
            loss_sum += loss
            bprop_cache = back_prop(fprop_cache)
            for key in weight_bias_params:
                weight_bias_params[key] -= bprop_cache[key] * eta
    return weight_bias_params


def shuffle(x, y):
    xy = list(zip(x, y))
    np.random.shuffle(xy)
    x, y = zip(*xy)
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def test_predict(param, test):
    with open("test_y", "w") as f:
        for x in test:
            y_hat = np.argmax(foward_prop(x.reshape(x.size, 1), 0, param)['y_predict'])
            f.write(str(y_hat))
            f.write("\n")


def check_accuracy(param, val_x, val_y):
    correct = 0
    loss_sum = 0
    for xi,yi in zip(val_x, val_y):
        y_predict = foward_prop(xi.reshape(xi.size, 1), yi, param)['y_predict']
        loss = loss_func(y_predict, yi)
        loss_sum += loss
        if np.argmax(y_predict) == yi:
            correct += 1
    return correct / val_y.size, loss_sum / val_y.size


def main():
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    train_x, train_y = shuffle(train_x, train_y)
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    train_y = train_y.astype(int)

    # hyper parameter
    hidden_layer_size = 100
    classes = 10
    eta = 0.01
    epochs = 20

    W1 = np.random.uniform(-0.05, 0.05, (hidden_layer_size, train_x.shape[1]))
    b1 = np.random.uniform(-0.05, 0.05, (hidden_layer_size, 1))
    W2 = np.random.uniform(-0.05, 0.05, (classes, hidden_layer_size))
    b2 = np.random.uniform(-0.05, 0.05, (classes, 1))
    weight_bias_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    param = train(epochs, eta, train_x, train_y, weight_bias_params)
    test_predict(param, test_x)


if __name__ == '__main__':
    main()
