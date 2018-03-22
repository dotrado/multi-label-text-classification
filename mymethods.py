from math import sqrt

import numpy as np

from classifier.knn_classifier import KNNClassifier
from classifier.naive_bayes_classifier import NaiveBayesClassifier


def derivative_feature_vectors(_vocabulary):
    feature_set = []
    for feature, df in _vocabulary.items():
        feature_set.append((feature, df))
    feature_set = sorted(feature_set, key=lambda pair: pair[1], reverse=True)
    feature_vector_1_tmp = {}
    feature_vector_2_tmp = {}
    for feature in feature_set[0:127]:
        feature_vector_1_tmp[feature[0]] = len(feature_vector_1_tmp)
    for feature in feature_set[0:min(len(feature_set), 300)]:
        feature_vector_2_tmp[feature[0]] = len(feature_vector_2_tmp)
    return feature_vector_1_tmp, feature_vector_2_tmp


def generate_tf_idf_feature(bag_of_features, raw_documents, _vocabulary):
    m = len(bag_of_features)
    feature_matrix = np.zeros((len(raw_documents), len(bag_of_features)))
    i = 0
    for _document in raw_documents:
        feature_vector = np.zeros(m)
        for feature, col in bag_of_features.items():
            if feature in _document.tf_idf.keys():
                feature_vector[col] = _document.tf_idf[feature]
        _document.feature_vector = feature_vector
        feature_matrix[i] = feature_vector
        i += 1
    return feature_matrix


def calculate_accuracy(y_predict, y_test_origin):
    """ Calculate the accuracy of Y_predict. The cardinality of Y_predict and Y_test_origin is same.

    accuracy = avg∑(true labels in predicted class labels) / len(predicted class labels)

    :param y_predict: [[class labels]] a series of class label of articles
    :param y_test_origin: [[test labels]] original class labels
    :return: accuracy
    """
    _accuracy = 0.0
    for y1, y2 in zip(y_predict, y_test_origin):
        counter = 0
        for y in y1:
            if y in y2:
                counter += 1
        _accuracy += float(counter) / len(y1)
    _accuracy /= len(y_predict)
    return _accuracy


def knn_predict(feature_vector,
                df_of_topics,
                vocabulary_,
                train_documents,
                test_documents,
                feature_matrix,
                y_test_original):
    knn_classifier = KNNClassifier(df_of_topics=df_of_topics, vocabulary=vocabulary_)
    accuracy_list = []
    y_knn_feature_128, y_knn_feature_128_accuracy = [], 0.0
    for k in range(3, int(sqrt(len(train_documents)))):
        knn_classifier.k = k
        y_knn_feature_128 = knn_classifier.knn_predict(feature_vector,
                                                       train_documents,
                                                       test_documents,
                                                       feature_matrix)
        y_knn_feature_128_accuracy = calculate_accuracy(y_knn_feature_128,
                                                        y_test_original)
        accuracy_list.append((y_knn_feature_128_accuracy, k))
        print("k is: {}, accuracy is: {}".format(k, y_knn_feature_128_accuracy))

    accuracy_list = sorted(accuracy_list, key=lambda pair: pair[0], reverse=True)
    i = 0
    for accuracy in accuracy_list:
        print("k is: {}, accuracy is: {}".format(accuracy[1], accuracy[0]))
        i += 1
        if i > 10:
            break
    return y_knn_feature_128, y_knn_feature_128_accuracy


def naive_predict(feature_vector,
                  bag_of_topics,
                  vocabulary_,
                  train_documents,
                  test_documents,
                  y_test_original):
    naive_classifier = NaiveBayesClassifier(feature_vector=feature_vector,
                                            vocabulary=vocabulary_,
                                            bag_of_topics=bag_of_topics,
                                            n=len(train_documents))
    naive_classifier.fit(train_documents)
    y_predict = naive_classifier.predict(test_documents, k=0)
    y_accuracy = calculate_accuracy(y_predict, y_test_original)

    print("When the cardinality of feature vector is {}, "
          "the accuracy of Naive Bayes Classifier is {}."
          .format(len(feature_vector), y_accuracy))
