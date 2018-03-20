import numpy as np


def calculate_tf_idf(tf, df, doc_num):
    """

    :param tf: term frequency
    :param df: document frequency where term appears
    :return: td-idf importance
    """
    idf = np.log(float(doc_num + 1) / (df + 1)) + 1.0
    tf = 1.0 + np.log(tf)
    return tf * idf
