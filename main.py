import os
import time
from sys import argv

import numpy as np

from classifier.knn_classifier import KNNClassifier
from data_preprocess.preprocess import DataProcessor, MyVectorizer


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


def generate_tfidf_feature(bag_of_features, raw_documents, _vocabulary):
    m = len(bag_of_features)
    for document in raw_documents:
        feature_vector = np.zeros(m)
        for feature, col in bag_of_features.items():
            if feature in document.tf_idf.keys():
                feature_vector[col] = document.tf_idf[feature]
        document.feature_vector['topics'] = feature_vector


if __name__ == "__main__":
    A1 = time.time()
    if len(argv) > 1:
        data_dir = argv[1]
    else:
        data_dir = 'data'

    if not os.path.exists(data_dir):
        raise OSError(
            'Please store original data files in data/ directory or '
            'type "python3 preprocess.py data_path" to input path of data')

    data_dir = os.path.abspath(data_dir)
    data_processor = DataProcessor()
    train_documents, test_documents, bag_of_topics, df_of_topics = data_processor.data_preprocess(data_dir)

    # binarize the class label to class vectors
    count_vectorizer = MyVectorizer(max_df=0.9, bag_of_topics=bag_of_topics)
    train_documents, vocabulary_ = count_vectorizer.fit_transform(train_documents)

    # construct more than 256 cardinality and less than 128 cardinality feature vectors
    feature_vector_1, feature_vector_2 = derivative_feature_vectors(vocabulary_)
    generate_tfidf_feature(feature_vector_1, train_documents, vocabulary_)
    knn_classifer = KNNClassifier(df_of_topics=df_of_topics, bag_of_features=feature_vector_1, vocabulary=vocabulary_)
    knn_classifer.knn_predict(feature_vector_1, train_documents, test_documents, df_of_topics)

    # generate_dataset(documents=train_documents, vocab=vocabulary_)
    # knn_classifier = KNN_Classifier(feature_vector_1, train_documents)
    # for test_document in test_documents:
    #     class_labels = knn_classifier.predict(test_document)
    #     accuracy = calculate_accuracy(class_labels, test_document.class_['topics'])
    # print("\nData preprocess termination message:")
    # print("The data preprocess is completed and successful.")
    # print("Two output files are in output/")
    # print("Process time: {} s.".format(time.time() - A1))
