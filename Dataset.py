class DataSet():

    def __init__(self, data, r_seed=None, filter_fn=None):
        """
        A DataSet is a container for Comment instances.
        :param data: List of Comment instances
        :param r_seed: A integer value that can be used to ensure stable random generation of the data set.
        :param filter_fn: A function that can be used to filter which Comments are included in the training data
        on a per comment basis.
        """
        self.filter_fn = filter_fn

        # Remove all deleted comments before we do any further processing.
        self.data = list(filter(lambda x: x.deleted, data))

        # Shuffle the given data to ensure that data is sampled from random subreddits
        import random
        random.seed(r_seed)
        random.shuffle(self.data)


    def generate_train_test(self, percent):
        """
        Generate a training and validation set from the dataset.
        :param percent: Cutoff of how much of the dataset will be included in the training set. The remaining percent
        will be invluded in the testing set.
        :return: Tuple(training sparse matrix, training labels, validation sparse matrix, validation labels)
        """

        # Compute where the training and testing data will be split.
        divider = int(len(self.data) * percent)

        # Separate training and validation sets
        train = []
        for i in range(0, divider):
            if (self.filter_fn is None or self.filter_fn(self.data[i])):
                train.append(self.data[i])

        validation = []
        for i in range(divider, len(self.data)):
            validation.append(self.data[i])

        # Fit and transform the data into training and testing sparse matrices.
        training_sparse_matrix, training_labels, validation_sparse_matrix, validation_labels = self.fit_transform(train, validation)

        return {
            "training_sparse_matrix": training_sparse_matrix,
            "training_labels": training_labels,
            "validation_sparse_matrix": validation_sparse_matrix,
            "validation_labels": validation_labels
        }


    def generate_n_cross_validation_sets(self, n):
        """
        Generate n cross validation sets.
        :param n: Number of folds to generate cross validation sets from.
        :return: List[ Dict{data set size, training sparse matrix,
        training labels, validation sparse matrix, validation labels} * n]
        """
        if n <= 1:
            raise ValueError('must have n > 1')

        # Compute the size of each validation set.
        import math
        validation_set_size = int(math.floor(len(self.data) / n))

        sets = []

        for fold in range(0, n):

            # Divide the dataset into a training and validation based on the current fold.
            # Filter the training data if a filter_fn is given.
            training_set = filter(self.filter_fn, self.data[:fold * validation_set_size] + self.data[(fold + 1) * validation_set_size:])
            validation_set = self.data[fold * validation_set_size:(fold + 1) * validation_set_size]

            # Fit and transform the training and validation sets into sparse matrices.
            training_sparse_matrix, training_labels, validation_sparse_matrix, validation_labels = self.fit_transform(training_set, validation_set)

            sets.append({
                "size": training_sparse_matrix.shape[0] + validation_sparse_matrix.shape[0],
                "training_sparse_matrix": training_sparse_matrix,
                "training_labels": training_labels,
                "validation_sparse_matrix": validation_sparse_matrix,
                "validation_labels": validation_labels
            })

        return sets


    def generate_human(self):
        """
        Generate a set of data with comments and labels that can be used to evaluate human performance.
        :return: Tuple(text, labels)
        """
        text = []
        labels = []

        for comment in self.data:
            text.append(comment.original_body)
            labels.append(comment.subreddit)

        return (text, labels)


    def fit_transform(self, training, validation):
        """
        Accepts a training set and validation set which will be fitted to TfidfVectorizer and DictVectorizer and then
        transformed to two distinct sparse matrices.
        :param training: List[Comment]
        :param validation: List[Comment]
        :return: Tuple(training sparse matrix, training labels, validation sparse matrix, validation labels)
        """

        # From the training List of Comments collect the processed text body and hand generated features.
        X_train_text = []
        X_train_features = []
        Y_train = []

        for item in training:
            X_train_text.append(item.processed_body)
            X_train_features.append(item.features())
            Y_train.append(item.subreddit)

        # From the validation List of Comments collect the processed text body and hand generated features
        X_validation_text = []
        X_validation_features = []
        Y_validation = []

        for item in validation:
            X_validation_text.append(item.processed_body)
            X_validation_features.append(item.features())
            Y_validation.append(item.subreddit)

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction import DictVectorizer
        from nltk.corpus import stopwords
        from scipy.sparse import hstack

        # Fit TfidfVectorizer and DictVectorizer to the training data set.
        train__tfidf_transformer = TfidfVectorizer(stop_words=stopwords.words('english')).fit(X_train_text)
        train_dict_transformer = DictVectorizer().fit(X_train_features)

        # Transform the training and validation sets using the fitted transformers then merge the sparse matrices by row.
        combined_train = hstack([train__tfidf_transformer.transform(X_train_text), train_dict_transformer.transform(X_train_features)], format='csr')
        combined_test = hstack([train__tfidf_transformer.transform(X_validation_text), train_dict_transformer.transform(X_validation_features)], format='csr')

        return (combined_train, Y_train, combined_test, Y_validation)
