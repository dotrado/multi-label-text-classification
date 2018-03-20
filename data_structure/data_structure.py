class Document:
    """A document instance for further processing.

    The structure of class document:
        @dict['words'] is a dictonary.
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
        self.feature_vector = dict()
        self.class_vector = dict(topics=[], places=[])
        self.tf_idf = {}
        self.train = False


class Class_labels:

    def __init__(self):
        self.bag_of_topics = set()
