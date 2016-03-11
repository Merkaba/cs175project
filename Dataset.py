class DataSet():

    def __init__(self, data, r_seed=None, filter_fn=None):
        self.filter_fn = filter_fn

        # Remove all deleted comments before we do any further processing.
        self.data = list(filter(lambda x: x.original_body != "[deleted]", data))

        # Shuffle the given data to ensure that data is sampled from random subreddits
        import random
        random.seed(r_seed)
        random.shuffle(self.data)


    def generate_train_test(self, percent, filter_function=None):
        divider = int(len(self.data) * percent)

        train = []
        for i in range(0, divider):
            if (self.filter_fn is None or self.filter_fn(self.data[i])):
                train.append(self.data[i])

        validation = []
        for i in range(divider, len(self.data)):
            validation.append(self.data[i])

        training_sparse_matrix, training_labels, validation_sparse_matrix, validation_labels = self.fit_transform(train, validation)

        return {
            "training_sparse_matrix": training_sparse_matrix,
            "training_labels": training_labels,
            "validation_sparse_matrix": validation_sparse_matrix,
            "validation_labels": validation_labels
        }


    def generate_n_cross_validation_sets(self, n):
        if n <= 1:
            raise ValueError('must have n > 1')

        import math

        validation_set_size = int(math.floor(len(self.data) / n))

        sets = []

        for fold in range(0, n):
            training_set = filter(self.filter_fn, self.data[:fold * validation_set_size] + self.data[(fold + 1) * validation_set_size:])
            validation_set = self.data[fold * validation_set_size:(fold + 1) * validation_set_size]

            training_sparse_matrix, training_labels, validation_sparse_matrix, validation_labels = self.fit_transform(training_set, validation_set)

            sets.append({
                "size": training_sparse_matrix.shape[0] + validation_sparse_matrix.shape[0],
                "training_sparse_matrix": training_sparse_matrix,
                "training_labels": training_labels,
                "validation_sparse_matrix": validation_sparse_matrix,
                "validation_labels": validation_labels
            })

        return sets


    def fit_transform(self, training, validation):
        X_train_text = []
        X_train_features = []
        Y_train = []

        for item in training:
            X_train_text.append(item.processed_body)
            X_train_features.append(item.features())
            Y_train.append(item.subreddit)

        X_test_text = []
        X_test_features = []
        Y_test = []

        for item in validation:
            X_test_text.append(item.processed_body)
            X_test_features.append(item.features())
            Y_test.append(item.subreddit)

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction import DictVectorizer
        from nltk.corpus import stopwords
        from scipy.sparse import hstack

        train__tfidf_transformer = TfidfVectorizer(stop_words=stopwords.words('english')).fit(X_train_text)
        train_dict_transformer = DictVectorizer().fit(X_train_features)

        combined_train = hstack([train__tfidf_transformer.transform(X_train_text), train_dict_transformer.transform(X_train_features)], format='csr')
        combined_test = hstack([train__tfidf_transformer.transform(X_test_text), train_dict_transformer.transform(X_test_features)], format='csr')

        return (combined_train, Y_train, combined_test, Y_test)
