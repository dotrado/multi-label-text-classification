import csv
import os

import numpy as np

from classifier.knn_classifier import KNNClassifier
from classifier.naive_bayes_classifier import NaiveBayesClassifier
from data_structure.data_structure import StaticData
from metric.metric import calculate_tf_idf, add_value


def derivative_feature_vectors(_vocabulary):
    """Set up the fixed length feature vector.

    :param _vocabulary:
    :return: feature vector with artificial cardinality
    """

    feature_vector_1_tmp = {}
    feature_vector_2_tmp = {}
    for feature in _vocabulary[0:min(len(_vocabulary), 125)]:
        feature_vector_1_tmp[feature] = len(feature_vector_1_tmp)
    for feature in _vocabulary[0:min(len(_vocabulary), 270)]:
        feature_vector_2_tmp[feature] = len(feature_vector_2_tmp)
    return feature_vector_1_tmp, feature_vector_2_tmp


def generate_tf_idf_feature(bag_of_features, raw_documents):
    m = len(bag_of_features)
    feature_matrix = np.zeros((len(raw_documents), len(bag_of_features)))
    i = 0
    for _document in raw_documents:
        feature_vector = np.zeros(m)
        for feature, col in bag_of_features.items():
            if feature in _document.tfs['all']:
                tf = _document.tfs['all'][feature]
                df = StaticData.df_term[feature]
                tf_idf = calculate_tf_idf(tf=tf, df=df, doc_num=StaticData.n_train_documents)
                add_value(_document.tf_idf, feature, tf_idf)
                feature_vector[col] = tf_idf

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
                train_documents,
                test_documents,
                feature_matrix,
                y_test_original):
    knn_classifier = KNNClassifier(df_of_classes=StaticData.df_of_classes)
    accuracy_list = []
    y_knn_feature_128, y_knn_feature_128_accuracy = [], 0.0
    for k in range(5, 6):
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
        print("KNN Classifier: The number of neighbors : k is {}, accuracy is: {}".format(accuracy[1], accuracy[0]))
        i += 1
        if i > 10:
            break
    return y_knn_feature_128, y_knn_feature_128_accuracy


def naive_predict(feature_vector,
                  vocabulary_,
                  train_documents,
                  test_documents,
                  y_test_original):
    naive_classifier = NaiveBayesClassifier(feature_vector=feature_vector,
                                            vocabulary=vocabulary_,
                                            n=len(train_documents))
    naive_classifier.fit(train_documents)

    y_predict = naive_classifier.predict(test_documents, k=0)
    y_accuracy = calculate_accuracy(y_predict, y_test_original)

    print("When the cardinality of feature vector is {}, "
          "the accuracy of Naive Bayes Classifier is {}."
          .format(len(feature_vector), y_accuracy))


def generate_dataset(documents, vocab):
    # check whether the subdirectory exists or not if not create a subdirectory
    subdirectory = "output"
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    print("Start writing data to vocabulary.csv")
    with open('output/vocabulary.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Term", "Index"])
        writer.writerows(vocab.items())
    print("Finish writing data to vocabulary.csv")
    print("Start writing data to dataset.csv")
    with open('output/dataset.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["document_id - (feature, vector) - [class labels]"])
        writer.writerow('')
        document_id = 0
        for document in documents:
            print("Writing document {}".format(document_id))
            document.id = document_id
            writer.writerow(["document {}".format(document_id)])
            writer.writerow(["class labels:"])
            writer.writerow(document.class_list)
            writer.writerow(['feature vector:'])
            rows = []
            for feature, frequency in document.tfs['all'].items():
                output_str = "({},{})".format(feature, frequency)
                rows.append(output_str)
            writer.writerow(rows)
            writer.writerow('')
            document_id += 1
    print("Finish writing data to dataset.csv")


def write_to_file(iterator, filename):
    # check whether the subdirectory exists or not if not create a subdirectory
    subdirectory = "output"
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    print("\nStart writing data to {}...".format(filename))
    with open('output/' + filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows([iterator])
    print("Finish writing data to {}.".format(filename))
