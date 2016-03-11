class Classifier():


    def fit(self, X, Y):
        self.classifier.fit(X, Y)


    def predict(self, X):
        return self.classifier.predict(X)


    def test(self, X, Y):
        import numpy as np
        return np.mean(self.predict(X) == Y)


class MultinomialNaiveBayes(Classifier):


    def __init__(self):
        from sklearn.naive_bayes import MultinomialNB
        self.classifier = MultinomialNB()


class LogisticRegression(Classifier):


    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.classifier = LogisticRegression()


class SupportVectorMachine(Classifier):


    def __init__(self):
        from sklearn.svm import LinearSVC
        self.classifier = LinearSVC()