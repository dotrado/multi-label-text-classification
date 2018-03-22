import queue

import numpy as np

from data_preprocess.preprocess import add_value
from metric.metric import calculate_tf_idf


class KNNClassifier:

    def __init__(self, bag_of_features=None, vocabulary=None, k=5, df_of_topics=None):
        self.bag_of_features = bag_of_features
        self.vocabulary = vocabulary
        self.k = k
        self.df_of_topics = df_of_topics
        self.p_topic_appear = {}
        self.k_neighbors = []

    def generate_feature_vector(self, test_document, n):
        """Generate tf-idf feature vector for a test document

        :param test_document:
        :param n:
        :return: npArray (1, m) feature vector
        """
        m = len(self.bag_of_features)
        feature_vector = np.zeros(m)
        for feature, col in self.bag_of_features.items():
            if feature in test_document.tfs['all'].keys():
                tf = test_document.tfs['all'][feature]
                df = self.vocabulary[feature]
                feature_vector[col] = calculate_tf_idf(tf=tf, df=df, doc_num=n)
        test_document.feature_vector = feature_vector
        return feature_vector

    def find_knn(self, train_documents, test_document, feature_matrix):
        k_queue = queue.PriorityQueue(self.k)
        k_neighbors = []
        n = len(train_documents)
        test_feature_vector = self.generate_feature_vector(test_document, n)
        z = np.linalg.norm(feature_matrix - test_feature_vector, axis=1, keepdims=True)
        for row in range(len(z)):
            if k_queue.qsize() == self.k:
                if k_queue.queue[0][0] < -z[row][0]:
                    k_queue.get()
                    k_queue.put((-z[row][0], row))
            else:
                k_queue.put((-z[row][0], row))
        for i in range(k_queue.qsize()):
            temp = k_queue.get()
            k_neighbors.append(train_documents[temp[1]])

        return k_neighbors

    def find_knn_prepare(self, train_documents, test_document, feature_matrix):
        k_neighbors = []
        n = len(train_documents)
        test_feature_vector = self.generate_feature_vector(test_document, n)
        z = np.linalg.norm(feature_matrix - test_feature_vector, axis=1, keepdims=True)
        for row, document in zip(range(len(z)), train_documents):
            k_neighbors.append((z[row][0], document))

        k_neighbors = sorted(k_neighbors, key=lambda pair: pair[0])
        return k_neighbors

    def prepare(self, train_documents, test_documents, feature_matrix):
        k_neighbors = []
        document_count = 0
        for test_document in test_documents:
            # print("Processing test document {}...".format(document_count))
            k_neighbors.append(self.find_knn_prepare(train_documents, test_document, feature_matrix))
            document_count += 1
        return k_neighbors

    def knn_predict(self, bags_of_features, train_documents, test_documents, feature_matrix):
        self.bag_of_features = bags_of_features
        if len(self.k_neighbors) == 0:
            self.k_neighbors = self.prepare(train_documents, test_documents, feature_matrix)
        y = []
        for k_neighbors in self.k_neighbors:
            # print("Processing test document {}...".format(document_count))
            # k_neighbors = self.find_knn(train_documents, test_document, feature_matrix)
            count_class_labels = {}
            for neighbor in k_neighbors[0:self.k]:
                document = neighbor[1]
                for topic in document.class_['topics']:
                    add_value(count_class_labels, topic, 1)
            votes = []
            for topic, df in count_class_labels.items():
                votes.append((topic, df))
            votes = sorted(votes, key=lambda pair: pair[1], reverse=True)
            predict_class = []
            for vote in votes:
                if vote[1] > self.k / 2:
                    predict_class.append(vote[0])
            if len(predict_class) == 0:
                for vote in votes:
                    if vote[0] == votes[0][0]:
                        predict_class.append(vote[0])
                    else:
                        break
            y.append(predict_class)
        return y
