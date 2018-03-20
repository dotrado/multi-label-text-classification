import numpy as np

from data_preprocess.preprocess import add_value
from metric.metric import calculate_tf_idf


class KNNClassifier:

    def __init__(self, bag_of_features=None, vocabulary=None, k=3, df_of_topics=None):
        self.bag_of_features = bag_of_features
        self.vocabulary = vocabulary
        self.k = k
        self.df_of_topics = df_of_topics
        self.p_topic_appear = {}

    def calculate_prior_prob(self, n):
        """Calculate the prior probability of class i appears in a document

        :param n:
        :return:
        """
        p_topic_appear = {}
        for topic, df in self.df_of_topics.items():
            p_topic_appear[topic] = (1.0 + float(df)) / (2.0 + n)
        self.p_topic_appear = p_topic_appear

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
        test_document.feature_vector['topics'] = feature_vector
        return feature_vector

    def find_knn(self, train_documents, test_document):
        k_neighbors = []
        n = len(train_documents)
        test_feature_vector = self.generate_feature_vector(test_document, n)
        for document in train_documents:
            dist = np.linalg.norm(test_feature_vector - document.feature_vector['topics'])
            k_neighbors.append((dist, document))
        return sorted(k_neighbors, key=lambda pair: pair[0])

    def knn_predict(self, train_documents, test_documents):
        Y = []
        for test_document in test_documents:
            k_neighbors = self.find_knn(train_documents, test_document)[0:self.k]
            count_class_labels = {}
            for neighbor in k_neighbors:
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
                predict_class.append(votes[0][0])
            Y.append(votes)
        return Y
