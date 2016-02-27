class Classifier():

    def predict(self, features):
        return self.pipeline.predict(features)


class MultinomialNaiveBayes(Classifier):


    def __init__(self, data):
        self.data = data

    def trainCountVectorizer(self):
        # Returns a Pipeline Object.
        #
        # comments: A list of (?)
        # subreddits:    A list of (?)

        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from nltk.corpus import stopwords
        from sklearn.naive_bayes import MultinomialNB

        features = []
        labels = []

        for item in self.data:
            features.append(item.processed_body)
            labels.append(item.subreddit)

        self.pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ]).fit(features, labels)


    def trainDictVectorizer(self):
        # Returns a Pipeline Object.
        #
        # comments: A list of (?)
        # subreddits:    A list of (?)

        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.naive_bayes import MultinomialNB

        features = []
        labels = []

        for item in self.data:
            features.append(item.features())
            labels.append(item.subreddit)

        self.pipeline = Pipeline([
            ('vect', DictVectorizer()),
            ('clf', MultinomialNB())
        ]).fit(features, labels)


class LogisticRegression(Classifier):

    def __init__(self, data):
        self.data = data


    def trainCountVectorizer(self):
        # Returns a Pipeline Object.
        #
        # comments: A list of (?)
        # subreddits:    A list of (?)

        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from nltk.corpus import stopwords
        from sklearn.linear_model import LogisticRegression

        features = []
        labels = []

        for item in self.data:
            features.append(item.processed_body)
            labels.append(item.subreddit)

        self.pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression())
        ]).fit(features, labels)


    def trainDictVectorizer(self):
        # Returns a Pipeline Object.
        #
        # comments: A list of (?)
        # subreddits:    A list of (?)

        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression

        features = []
        labels = []

        for item in self.data:
            features.append(item.features())
            labels.append(item.subreddit)

        self.pipeline = Pipeline([
            ('vect', DictVectorizer()),
            ('clf', LogisticRegression())
        ]).fit(features, labels)