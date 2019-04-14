import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import init_centroids
# data preperation (loading, normalizing, reshaping)
from scipy.misc import imread


def load_image(path):
    # path = 'dog.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    return X, img_size

def rgb_classification(data,centroid):
    assignments = []
    for entry in data:
        shortest = np.math.inf  # positive infinity
        shortest_index = 0
        for i in range(0, len(centroid)):
            distance = (np.linalg.norm(entry - centroid[i])) ** 2
            if distance < shortest:
                shortest = distance
                shortest_index = i
        assignments.append(shortest_index)
    return assignments
def update_centroid(dataset, centroid, classification):
    new_means = defaultdict(list)
    for assignment, point in zip(classification, dataset):
        new_means[assignment].append(point)
    for key, value in new_means.items():
        centroid[key] = np.asarray(value).mean(axis=0,dtype=np.float64)
    return centroid

def k_means(dataset,centroid,k):
    loss = []
    classification = []
    for x in range(11):
        print("iter " + str(x) + ":", end=' ')
        print(print_cent(centroid))
        classification = rgb_classification(dataset, centroid)
        z = np.asarray([np.asarray(centroid[assignment]) for assignment in classification])
        y = [(np.linalg.norm(c - entry)**2) for c, entry in zip(z, dataset)]
        loss.append(np.mean(y))
        centroid = update_centroid(dataset, centroid, classification)

    # return classification
    return np.asarray([np.asarray(centroid[assignment]) for assignment in classification]), loss



def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]

def main():
    dataset, img_size = load_image('dog.jpeg')
    # losss = []
    # f = open("output.txt", "w")

    for k in [2, 4, 8, 16]:
        centroid = init_centroids.init_centroids(dataset,k)
        print("k=" + str(k) + ":")
        classification, loss = k_means(dataset,centroid,k)
        # x = np.asarray([np.asarray(xi) for xi in loss])
        # plt.plot(x)
        # plt.xlabel('Number of Iteration')
        # plt.ylabel('Loss Avg')
        # plt.title('Loss Graph for k = ' + str(k))

        # plt.show()
        # z = np.reshape(classification, img_size)
        # plt.imshow(z)
        # plt.grid(False)
        # plt.title('Image for k = ' + str(k))
        # plt.show()

if __name__ == '__main__':
    main()