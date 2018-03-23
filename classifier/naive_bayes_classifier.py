import numpy as np

from data_structure.data_structure import StaticData
from metric.metric import calculate_tf_idf, add_value


class NaiveBayesClassifier:
    """ Naive Bayes Classifier

    """

    def __init__(self, feature_vector=None, vocabulary=None, n=0):
        self.class_possibility_list = []
        self.feature_vector = feature_vector
        self.p_class = {}
        self.p_class_attribute = {}
        self.vocabulary = vocabulary
        self.n = n
        self.bag_of_topics = StaticData.bag_of_classes

    def generate_feature_vector(self, test_document, n):
        """Generate tf-idf feature vector for a test document

        :param test_document:
        :param n:
        :return: npArray (1, m) feature vector
        """
        m = len(self.feature_vector)
        feature_vector = np.zeros(m)
        for feature, col in self.feature_vector.items():
            if feature in test_document.tfs['all'].keys():
                tf = test_document.tfs['all'][feature]
                df = StaticData.df_term[feature]
                feature_vector[col] = calculate_tf_idf(tf=tf, df=df, doc_num=n)
                # feature_vector[col] = 1

        norm = np.linalg.norm(feature_vector)
        if norm != 0:
            feature_vector = feature_vector / norm
        test_document.feature_vector = feature_vector
        return feature_vector

    def prepare(self, train_documents):
        p_class = {}
        p_class_attribute = {}
        document_count = 0
        for document in train_documents:
            print('Processing training document {}'.format(document_count))
            document_count += 1
            for topic in document.class_['topics']:
                add_value(p_class, topic, 1)
                tmp = np.zeros(len(self.feature_vector))
                i = 0
                for feature in self.feature_vector.keys():
                    if feature in document.tfs['all'].keys():
                        if topic not in p_class_attribute.keys():
                            p_class_attribute[topic] = {}

                        tf = document.tfs['all'][feature]
                        df = StaticData.df_term[feature]
                        tf_idf = calculate_tf_idf(tf=tf, df=df, doc_num=StaticData.n_train_documents)
                        tmp[i] = tf_idf * StaticData.chi_2_term_class[feature][topic] \
                                 * StaticData.entropy_term_class[feature][topic]
                    i += 1

                #  normalization here， each document will contribute 1 to p[class_], also contribute some
                #   less than 1 value to p[class_][feature]
                norm = np.linalg.norm(tmp)
                if norm != 0:
                    tmp = tmp / norm
                i = 0
                for feature in self.feature_vector.keys():
                    if feature in document.tfs['all'].keys():
                        add_value(p_class_attribute[topic], feature, tmp[i])
                    i += 1

        n = len(train_documents)
        for class_, count in p_class.items():
            n_topic = p_class[class_]
            for feature, count_ in p_class_attribute[class_].items():
                p_class_attribute[class_][feature] = float(count_) / n_topic
            p_class[class_] = float(count) / n
        return p_class, p_class_attribute

    def fit(self, train_documents):
        """ Build classifier.

        :param train_documents:
        :return:
        """
        self.n = len(train_documents)
        self.p_class, self.p_class_attribute = self.prepare(train_documents)

    def predict(self, test_documents, k):
        y_predict = []
        class_possibility_list = []
        if k > 1:
            for class_possibility in self.class_possibility_list:
                temp = []
                for i in range(k):
                    temp.append(class_possibility[i][1])
                y_predict.append(temp)
            return y_predict
        document_count = 0
        for document in test_documents:
            print("Naive Bayes classifier is predicting test document {}...".format(document_count))
            document_count += 1
            test_feature_vector = self.generate_feature_vector(document, self.n)
            class_possibility = []
            for topic in self.bag_of_topics:
                p = 0.0
                for feature, i in self.feature_vector.items():
                    if feature in self.p_class_attribute[topic]:
                        p += np.log(self.p_class_attribute[topic][feature] * test_feature_vector[i] + 1.0)
                p *= self.p_class[topic]
                class_possibility.append((p, topic))
            class_possibility = sorted(class_possibility, key=lambda pair: pair[0], reverse=True)
            y_predict.append([class_possibility[0][1]])
            class_possibility_list.append(class_possibility)
        self.class_possibility_list = class_possibility_list
        return y_predict
