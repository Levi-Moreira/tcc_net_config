#!/usr/bin/env python

from __future__ import print_function
import cv2
import gc
import numpy

import numpy as np
from os import listdir
from sklearn.svm import LinearSVC
import pickle
from os.path import isfile, join

SZ = 227
CLASS_N = 10

DEFAULT_WIN_SIGMA = -1
DEFAULT_NLEVELS = 64


def load_images(fn):
    positive_path = fn + '/pos'
    negative_path = fn + '/neg'

    positive_files = [f for f in listdir(positive_path) if isfile(join(positive_path, f))]
    negative_files = [f for f in listdir(negative_path) if isfile(join(negative_path, f))]

    pos_images = numpy.empty(len(positive_files), dtype=object)
    neg_images = numpy.empty(len(negative_files), dtype=object)

    index = 0
    total_positive_images = len(positive_files)
    print("Positive images to load: {}".format(total_positive_images))
    for n in range(0, len(positive_files)):
        gc.collect()
        positive_image_path = join(positive_path, positive_files[n])

        image = cv2.imread(positive_image_path)
        height, width, channels = image.shape
        new_image = cv2.resize(image, (0, 0), fx=float(SZ) / width, fy=float(SZ) / height)

        pos_images[n] = new_image
        index = index + 1
        #print("Loading positive images {}%\r".format(int(index / float(total_positive_images) * 100)), end='\r')
	print(index)

    index = 0
    total_negative_images = len(negative_files)
    print("Negative images to load: {}".format(total_negative_images))
    for n in range(0, 8068):
        gc.collect()
        negative_image_path = join(negative_path, negative_files[n])

        image = cv2.imread(negative_image_path)
        height, width, channels = image.shape
        new_image = cv2.resize(image, (0, 0), fx=float(SZ) / width, fy=float(SZ) / height)

        neg_images[n] = new_image
        index = index + 1
        #print("Loading negative images {}%\r".format(int(index / float(total_negative_images) * 100)), end='\r')
	print(index)

    pos_labels = np.repeat(1, len(pos_images))
    neg_labels = np.repeat(0, len(neg_images))

    dataset = np.concatenate((pos_images, neg_images), axis=0)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)

    return dataset, labels


class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.50625):
        self.model = cv2.ml.SVM_create()
        # self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_LINEAR)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err) * 100))

    confusion = np.zeros((2, 2), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    # print('confusion matrix:')
    # print(confusion)

    return ((1 - err) * 100), confusion


def get_hog():
    winSize = (128, 128)
    blockSize = (32, 32)
    blockStride = (16, 16)
    cellSize = (16, 16)
    nbins = 9
    derivAperture = 1
    winSigma = DEFAULT_WIN_SIGMA
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = DEFAULT_NLEVELS
    signedGradient = True

    return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                             histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)



if __name__ == '__main__':
    print('Loading images  from dataset 1... ')
    # Load data.
    dataset, labels = load_images('dataset')

    # print('Loading images  from dataset 2... ')
    # dataset2, labels2 = load_images('/home/levi/TCC/digits-classification/dataset2')

    # dataset = np.concatenate((dataset1, dataset2), axis=0)
    # labels = np.concatenate((labels1, labels2), axis=0)

    #dataset = dataset1
    #labels = labels1

    # dataset, labels = load_images('/home/levi/TCC/digits-classification/dataset2')

    print(len(dataset))
    print(len(labels))

    print('Shuffle data ... ')
    # Shuffle data
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(dataset))
    dataset, labels = dataset[shuffle], labels[shuffle]

    print('Defining HoG parameters ...')
    # HoG feature descriptor
    hog = get_hog()

    print('Calculating HoG descriptor for every image ... ')
    hog_descriptors = []

    total = len(dataset)
    index = 0
    print('\n')
    for img in dataset:
        computed_hog = hog.compute(img)
        del img
        hog_descriptors.append(computed_hog)
        index = index + 1
        print("Hoggin {}%".format(int(index / float(total) * 100)), end='\r')

    hog_descriptors = np.squeeze(hog_descriptors)

    #del dataset1
    del dataset
    # del dataset

    numpy.savetxt("/mnt/TCC/descriptorsfinal2.csv", hog_descriptors, delimiter=",")

    cross_validation = []
    matrices = []
    for index in range(0, 50):
        rand = np.random.RandomState(10)
        shuffle = rand.permutation(len(hog_descriptors))
        hog_descriptors, labels = hog_descriptors[shuffle], labels[shuffle]

        print('Spliting data into training (90%) and test set (10%)... ')
        train_n = int(0.7 * len(hog_descriptors))

        hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])

        print('Training SVM model ...')
        model = SVM()
        model.train(hog_descriptors_train, labels_train)

        print('Saving SVM model ...')
        model.save('/mnt/TCC/digits_svm{}.dat'.format(index))

        # #
        print('Evaluating model 1... ')
        acc, confusion = evaluate_model(model, hog_descriptors_test, labels_test)
        cross_validation.append(acc)
        matrices.append(confusion)

    print("Mean accuracy for 10-cross validation: {}%".format(np.mean(cross_validation)))
    numpy.save("/mnt/TCC/matrices.dat", matrices)
    numpy.save("/mnt/TCC/cross_validation.dat", cross_validation)
