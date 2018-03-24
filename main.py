import os
import time
from sys import argv

from data_preprocess.preprocess import DataProcessor, MyVectorizer
from data_structure.data_structure import StaticData
from mymethods import derivative_feature_vectors, naive_predict, generate_tf_idf_feature, knn_predict, write_to_file, \
    write_predict, write_termination_messages

if __name__ == "__main__":

    A1 = time.time()
    if len(argv) > 1:
        data_dir = argv[1]
    else:
        data_dir = 'data'

    if not os.path.exists(data_dir):
        raise OSError(
            'Please store original data files in data/ directory or type "python3 main.py data_path" to input path of data')

    """ Preprocessing """
    print("========== Parse data files ==========")
    data_dir = os.path.abspath(data_dir)
    data_processor = DataProcessor()
    train_documents, test_documents = data_processor.data_preprocess(data_dir)

    # test_documents = test_documents[0:10]
    # binarize the class label to class vectors
    print("\n========== Constructing bag of words ==========")
    count_vectorizer = MyVectorizer(max_df=0.9)
    train_documents, vocabulary_ = count_vectorizer.fit_transform(train_documents)
    count_vectorizer_test = MyVectorizer(max_df=0.9)
    count_vectorizer_test.count_vocab(test_documents)

    # get the Y test
    Y_test_original = []
    for document in test_documents:
        Y_test_original.append(document.class_['topics'])

    StaticData.preprocessing_time = time.time() - A1
    print("Preprocessing time: {} s.".format(StaticData.preprocessing_time))

    # construct more than 256 cardinality and less than 128 cardinality feature vectors
    print("\n========== Generate two derivative feature vectors of selected feature vector ==========")
    feature_vector_1, feature_vector_2 = derivative_feature_vectors(vocabulary_)
    print("Generate feature vector 1 which has a cardinality 125...")
    write_to_file(feature_vector_1, "feature_vector_1.csv")
    print("Generate feature vector 2 which has a cardinality 270...")
    write_to_file(feature_vector_2, "feature_vector_2.csv")

    """ knn predict """
    print("\n++++++++++ Start predicting ++++++++++")
    print("Select two classifiers: knn classifier and naive bayes classifier.")
    print("\n========== KNN Classifier ==========")
    print("Select k = 5 as the number of neighbors.")
    print("\nPredict using feature vector 1 ({} cardinality):".format(len(feature_vector_1)))
    print("")
    StaticData.A1 = time.time()
    feature_matrix_1 = generate_tf_idf_feature(feature_vector_1, train_documents)

    Y_knn_128_predict, Y_knn_128_accuracy = knn_predict(feature_vector=feature_vector_1,
                                                        train_documents=train_documents,
                                                        test_documents=test_documents,
                                                        feature_matrix=feature_matrix_1,
                                                        y_test_original=Y_test_original)
    write_predict(Y_test_original, Y_knn_128_predict, "KNN_predict_class_labels_125_feature_vector.txt")
    print("\nPredict using feature vector 2 ({} cardinality):".format(len(feature_vector_2)))
    StaticData.A1 = time.time()
    feature_matrix_2 = generate_tf_idf_feature(feature_vector_2, train_documents)
    Y_knn_256_predict, Y_knn_256_accuracy = knn_predict(feature_vector=feature_vector_2,
                                                        train_documents=train_documents,
                                                        test_documents=test_documents,
                                                        feature_matrix=feature_matrix_2,
                                                        y_test_original=Y_test_original)
    write_predict(Y_test_original, Y_knn_256_predict, "KNN_predict_class_labels_270_feature_vector.txt")

    """ Naive Bayes predict """
    print("\n========== Naive Bayes Classifier ==========")
    print("\nPredict using feature vector 1 ({} cardinality):".format(len(feature_vector_1)))
    Y_naive_128_predict = naive_predict(feature_vector_1,
                                        vocabulary_,
                                        train_documents,
                                        test_documents,
                                        Y_test_original)
    write_predict(Y_test_original, Y_naive_128_predict, "Naive_predict_class_labels_125_feature_vector.txt")
    print("\nPredict using feature vector 2 ({} cardinality):".format(len(feature_vector_2)))
    Y_naive_256_predict = naive_predict(feature_vector_2,
                                        vocabulary_,
                                        train_documents,
                                        test_documents,
                                        Y_test_original)
    write_predict(Y_test_original, Y_naive_256_predict, "Naive_predict_class_labels_270_feature_vector.txt")

    print("\n========== Termination message ==========")
    print("Mission completed.")
    print("We select knn classifier and naive classifier.")
    print("\nFor feature vector with {} cardinality:".format(len(feature_vector_1)))
    print("\nThe accuracy of knn classifier is {}.".format(StaticData.knn_accuracy[0]))
    print("The offline efficient cost of knn classifier is {} s.".format(StaticData.knn_build_time[0]))
    print("The online efficient cost of knn classifier is {} s.".format(StaticData.knn_predict_time[0]))

    print("\nThe accuracy of naive classifier is {}.".format(StaticData.naiver_accuracy[0]))
    print("The offline efficient cost of naive classifier is {} s.".format(StaticData.naive_build_time[0]))
    print("The online efficient cost of naive classifier is {} s.".format(StaticData.naive_predict_time[0]))

    print("\nFor feature vector with {} cardinality:".format(len(feature_vector_2)))
    print("\nThe accuracy of knn classifier is {}.".format(StaticData.knn_accuracy[1]))
    print("The offline efficient cost of knn classifier is {} s.".format(StaticData.knn_build_time[1]))
    print("The online efficient cost of knn classifier is {} s.".format(StaticData.knn_predict_time[1]))

    print("\nThe accuracy of naive classifier is {}.".format(StaticData.naiver_accuracy[1]))
    print("The offline efficient cost of naive classifier is {} s.".format(StaticData.naive_build_time[1]))
    print("The online efficient cost of naive classifier is {} s.".format(StaticData.naive_predict_time[1]))

    write_termination_messages("termination_messages.txt")
