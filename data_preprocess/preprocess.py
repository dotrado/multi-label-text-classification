#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gjxhlan
"""
import csv
import numbers
import os
import re
import string

import numpy as np

from data_structure.data_structure import Document
from metric.metric import calculate_tf_idf


class MyVectorizer:
    """Convert a collection of text documents to a matrix of token counts.

    Parameters
    ----------
    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.
    """

    # stop words
    stop_words_ = frozenset(["a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually",
                             "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost",
                             "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an",
                             "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways",
                             "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around",
                             "as",
                             "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be",
                             "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand",
                             "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between",
                             "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't",
                             "cannot",
                             "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com",
                             "come",
                             "comes", "concerning", "consequently", "consider", "considering", "contain", "containing",
                             "contains", "corresponding", "could", "couldn't", "course", "currently", "d",
                             "definitely",
                             "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing",
                             "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight",
                             "either",
                             "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever",
                             "every",
                             "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except",
                             "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for",
                             "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get",
                             "gets",
                             "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings",
                             "h",
                             "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he",
                             "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby",
                             "herein",
                             "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully",
                             "how",
                             "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored",
                             "immediate",
                             "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner",
                             "insofar",
                             "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its",
                             "itself",
                             "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last",
                             "lately",
                             "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked",
                             "likely", "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may",
                             "maybe",
                             "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly",
                             "much",
                             "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary",
                             "need",
                             "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non",
                             "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "n't",
                             "o",
                             "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones",
                             "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves",
                             "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per",
                             "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides",
                             "q",
                             "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding",
                             "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same",
                             "saw",
                             "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed",
                             "seeming",
                             "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven",
                             "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some",
                             "somebody",
                             "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere",
                             "soon",
                             "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t",
                             "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that",
                             "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence",
                             "there",
                             "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon",
                             "these",
                             "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough",
                             "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to",
                             "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try",
                             "trying",
                             "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until",
                             "unto",
                             "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v",
                             "value",
                             "various", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we",
                             "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't", "what",
                             "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter",
                             "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
                             "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing",
                             "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y",
                             "yes",
                             "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
                             "yourselves", "z", "zero"])
    remove_table = str.maketrans('', '', string.digits + string.punctuation)

    # lemmatizer and stemmer

    def __init__(self, encoding='utf-8',
                 decode_error='strict', strip_accents='ascii',
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=0.9, min_df=3, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64, bag_of_topics=set()):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.strip_accents = strip_accents
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.preprocessor = preprocessor
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        if stop_words is None:
            self.stop_words = self.stop_words_
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.dfs = {}
        self.bag_of_topics = bag_of_topics

    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            self.fixed_vocabulary_ = True
        else:
            self.fixed_vocabulary_ = False

    def my_tokenizer(self, raw_text) -> {}:
        """ Extract tokenized words from text.

        :param
            text: string representing the text.
        :return
            A dictionary of tokenized words mapped to frequency.
        """
        # tokenize words, remove stop words
        tokenizer = re.compile(r'\b[a-z][a-z.-]*\b')
        tokens = tokenizer.findall(raw_text)
        valid_tokens = {}
        for token in tokens:
            if token not in self.stop_words and len(token) >= 4:
                valid_tokens[token] = valid_tokens.setdefault(token, 0) + 1

        """
        # lemmatization process
        lemmas = []
        for word, tag in tagged_tokens:
            wn_tag = self.get_word_net_pos(tag)
            if wn_tag is None:
                lemmas.append(self.lemmatizer.lemmatize(word))
            else:
                lemmas.append(self.lemmatizer.lemmatize(word, pos=wn_tag))

        # stemming process, remove short stems
        stems = []
        for token in lemmas:
            term = self.stemmer.stem(token)
            if len(term) >= 4:
                stems.append(term)

        return stems
        """

        return valid_tokens

    def _build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        if self.tokenizer is not None:
            return self.tokenizer

        return self.my_tokenizer

    def _build_preprocessor(self):
        """Return a function to preprocess the text before tokenization"""
        if self.preprocessor is not None:
            return self.preprocessor

        # remove digits, punctuations
        return lambda raw_text: re.sub('[^a-z,.-]', ' ', raw_text)

    def _build_analyzer(self):
        """Return a callable that handles pre-processing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self._build_preprocessor()
        tokenize = self._build_tokenizer()

        return lambda doc: tokenize(preprocess(doc))

    def count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        vocabulary = {}
        if fixed_vocab:
            vocabulary = self.vocabulary

        analyze = self._build_analyzer()

        for document in raw_documents:
            doc = document.text
            for feature, tfs in analyze(doc).items():
                if feature not in self.bag_of_topics:
                    try:
                        # calculate term frequency and dfs
                        add_value(document.tfs['all'], key=feature, value=tfs)
                        add_value(self.dfs, key=feature, value=1)
                    except KeyError:
                        # Ignore out-of-vocabulary items for fixed_vocab=True
                        continue

        return vocabulary

    def fit_transform(self, raw_documents):
        """Learn the vocabulary dictionary and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
            :param raw_documents:
        """
        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary = self.count_vocab(raw_documents,
                                      self.fixed_vocabulary_)

        vocabulary_ = {}
        if not self.fixed_vocabulary_:
            n_doc = len(raw_documents)
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            stop_words_, vocabulary_ = self._limit_features(raw_documents, vocabulary,
                                                            max_doc_count,
                                                            min_doc_count,
                                                            max_features)
            self.stop_words = self.stop_words.union(stop_words_)

        return raw_documents, vocabulary_

    def _limit_features(self, raw_documents, vocabulary, high=None, low=None,
                        limit=None):
        """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.

        This does not prune samples with zero features.
        """
        if high is None and low is None and limit is None:
            return raw_documents, set()

        # Calculate a mask based on document frequencies
        for term, dfs in self.dfs.items():
            if low <= dfs <= high:
                vocabulary[term] = dfs

        removed_terms = self.dfs.keys() - vocabulary.keys()

        # Calculate tf-idf
        valid_terms = {}
        n = len(raw_documents)
        for document in raw_documents:
            pairs = []
            for term, tf in document.tfs['all'].items():
                if term in vocabulary:
                    add_value(document.tf_idf, term, calculate_tf_idf(tf=tf, df=vocabulary[term], doc_num=n))
                    pairs.append((term, document.tf_idf[term]))
            pairs = sorted(pairs, key=lambda a: a[1], reverse=True)
            len_ = int(0.2 * len(pairs))
            for pair in pairs[0:min(10, len_)]:
                if pair[0] not in valid_terms.keys():
                    valid_terms[pair[0]] = vocabulary[pair[0]]

        if len(valid_terms) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return removed_terms, valid_terms


def add_value(dict_, key, value):
    if key not in dict_:
        dict_[key] = 0
    dict_[key] += value


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


class DataProcessor:
    """ Class DataProcessor for data pre-processing.

    data_preprocess(): pre-process the raw data into feature vectors
    """

    def __init__(self):
        # variables for removing stop words, digits, punctuation
        # two class labels are dictionary, key is class, value is list of documents
        self.bag_of_topics = set()
        self.bag_of_places = set()
        self.df_of_topics = {}

    @staticmethod
    def parse_article(article):
        """ Parse the article to generate a document object.

        Args:
            @article: represent an article.

        Returns:
            A document instance.
        """
        document = Document()

        # extract word list

        text_compile = re.compile('<text.*?</text>', re.DOTALL).search(article)
        if text_compile is None:
            return None
        else:
            text = text_compile.group(0)

        text = re.sub(pattern='</?[a-z]*?>', repl='', string=text)
        document.text = text

        # extract class label
        topic_labels = set()
        for topics in re.compile('<topics.*?</topics>').findall(article):
            for topic in re.compile('<d>[a-z]*?</d>').findall(topics):
                topic_labels.add(re.sub(pattern='</?d>', repl='', string=topic))
        if len(topic_labels) == 0:
            return None
        document.class_['topics'] = topic_labels
        document.class_list = list(topic_labels)

        # train or test
        document.train = re.search('lewissplit="train"', string=article) is not None

        return document

    def extract_documents(self, directory) -> []:
        """ Extract documents from raw data.

        Args:
            @data: raw data read from .sgn file.

        Returns:
            A list of document.
        """

        documents = []
        filecount = 0
        for file in os.listdir(directory):
            # open each .sgm

            if file.startswith('reut2'):
                filecount = filecount + 1
                print('Processing file {}...'.format(filecount))

                with open(directory + '/' + file, 'rb') as datafile:
                    data = datafile.read().decode('utf8', 'ignore')
                    soup = re.compile('<REUTERS.*?</REUTERS>', re.DOTALL)
                    document_count = 0
                    for article in soup.findall(data):
                        document_count += 1
                        print('Processing document {}...'.format(document_count))
                        document = self.parse_article(article.lower())
                        if document is not None:
                            documents.append(document)

                print('Finished processing file {}...'.format(filecount))

            if filecount == 0:
                raise OSError('No data file detected. Please make sure the filename start with "reut2".')
        return documents

    def data_preprocess(self, directory):
        """ Read data from the /data folder. Transform the data to a list of documents. Generate feature vectors for documents.

        In this case, I choose unique terms to construct feature vector.

        """
        # generate list of document objects for constructing feature vector
        documents = self.extract_documents(directory)
        _train_documents = []
        _test_documents = []
        for document in documents:
            if len(document.class_list) > 0:
                if document.train:
                    _train_documents.append(document)
                    self.bag_of_topics = self.bag_of_topics.union(document.class_['topics'])
                    for topic in document.class_['topics']:
                        add_value(self.df_of_topics, topic, 1)
                else:
                    _test_documents.append(document)

        return _train_documents, _test_documents, self.bag_of_topics, self.df_of_topics

# def pause():
#    programPause = input("Press the <ENTER> key to continue...")
