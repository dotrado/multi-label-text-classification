import os
import time
from sys import argv

from data_preprocess.preprocess import DataProcessor, MyVectorizer
from mymethods import derivative_feature_vectors, naive_predict

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
    count_vectorizer_test = MyVectorizer(max_df=0.9)
    count_vectorizer_test.count_vocab(test_documents, False)

    # get the Y test
    Y_test_original = []
    for document in test_documents:
        Y_test_original.append(document.class_['topics'])

    # construct more than 256 cardinality and less than 128 cardinality feature vectors
    feature_vector_1, feature_vector_2 = derivative_feature_vectors(vocabulary_)

    """ knn predict """
    # feature_matrix_1 = generate_tf_idf_feature(feature_vector_1, train_documents, vocabulary_)
    # feature_matrix_2 = generate_tf_idf_feature(feature_vector_2, train_documents, vocabulary_)
    # Y_knn_128_predict, Y_knn_128_accuracy = knn_predict(feature_vector=feature_vector_1,
    #                                                     df_of_topics=df_of_topics,
    #                                                     vocabulary_=vocabulary_,
    #                                                     train_documents=train_documents,
    #                                                     test_documents=test_documents,
    #                                                     feature_matrix=feature_matrix_1,
    #                                                     y_test_original=Y_test_original)
    # Y_knn_256_predict, Y_knn_256_accuracy = knn_predict(feature_vector=feature_vector_2,
    #                                                     df_of_topics=df_of_topics,
    #                                                     vocabulary_=vocabulary_,
    #                                                     train_documents=train_documents,
    #                                                     test_documents=test_documents,
    #                                                     feature_matrix=feature_matrix_2,
    #                                                     y_test_original=Y_test_original)

    """ Naive Bayes predict """
    Y_naive_128_predict = naive_predict(feature_vector_1,
                                        bag_of_topics,
                                        vocabulary_,
                                        train_documents,
                                        test_documents,
                                        Y_test_original)
    Y_naive_256_predict = naive_predict(feature_vector_2,
                                        bag_of_topics,
                                        vocabulary_,
                                        train_documents,
                                        test_documents,
                                        Y_test_original)

    # generate_dataset(documents=train_documents, vocab=vocabulary_)
    # knn_classifier = KNN_Classifier(feature_vector_1, train_documents)
    # for test_document in test_documents:
    #     class_labels = knn_classifier.predict(test_document)
    #     accuracy = calculate_accuracy(class_labels, test_document.class_['topics'])
    # print("\nData preprocess termination message:")
    # print("The data preprocess is completed and successful.")
    # print("Two output files are in output/")
    # print("Process time: {} s.".format(time.time() - A1))
