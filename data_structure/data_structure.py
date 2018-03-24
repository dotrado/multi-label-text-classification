import numpy


class Document:
    """A document instance for further processing.

    The structure of class document:
        @dict['words'] is a dictionary.
        @dict['words']['title'] is a list which contains words of title.
        @dict['words']['body'] is a list which contains words of article body.
        @dict['topics'] is a list of TOPICS class labels.
        @dict['places'] is a list of PLACES class labels.
    """

    def __init__(self):
        self.title = ""
        self.text = ""
        self.class_list = []
        self.tfs = dict(title={}, body={}, all={})
        self.class_ = dict(topics=set(), places=set(), all=set())
        self.feature_vector = numpy.array([])
        self.class_vector = dict(topics=[], places=[])
        self.tf_idf = {}
        self.train = False


class StaticData:
    """ Static Count """
    # set of classes, class -> document frequency, initialized in data_preprocess
    bag_of_classes = set()
    df_of_classes = {}

    # number of train documents, initialized in data_preprocess
    n_train_documents = 0
    n_classes = 0.0

    # the amount of documents vs class
    class_has_documents = {}

    """ Metric"""
    # chi square metric of importance of a term to a class
    chi_2_term_class = {}
    # alpha
    tf_avg_term_class = {}
    # entropy
    entropy_term_class = {}
    # beta modified factor
    beta = {}

    # my ichi importance metric of a term to a class
    i_chi = {}
    i_chi_term_class = {}
    i_chi_list = []

    """ relation between term and class """
    # term frequency of a term to a class, initialized in calculate_static_data
    tf_term_class = {}
    df_term_class = {}

    """ relation between term and document """
    # how many documents have term i, initialized in preprocess.count_vocab
    df_term = {}

    """ sorted bag of words for building feature """
    vocabulary = []

    """ time counter """
    preprocessing_time = 0.0
    knn_build_time = []
    knn_predict_time = []
    naive_build_time = []
    naive_predict_time = []
    A1 = 0.0

    """ accuracy """
    knn_accuracy = []
    naiver_accuracy = []
