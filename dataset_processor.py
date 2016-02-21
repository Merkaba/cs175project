import json
from Comment import Comment
from Subreddit import Subreddit
from collections import Counter

# In this module, we load the dataset, which is in JSON, and parse it. The
# resulting data is tallied to create a list of Subreddit objects defined as
# subreddits. Each comment was posted in one of many Subreddits, so as we go
# through the comments, we add comments to a list in their respective Subreddit
# and generate a new Subreddit object if one does not already exist. In this way,
# our collection of Subreddits contains every subreddit that had an associated
# comment, and this collection contains a list of comments and various data
# attached analyzing it, such as BoW.
#
# subreddits: List of Subreddit Object, we contain our JSON data parsed here

def load_comments(filename, max_iteration=None):
    # Yields a Comment object
    #
    # filename:      Str, a filename as a path
    # max_iteration: Int, an optional argument which defines max ammount of comments to
    #                yield.
    #
    # Comments are loaded using json library which loads one comment at a time (?)

    current_iteration = 0

    with open(filename) as dataset:
        for line in dataset:
            if max_iteration is not None and current_iteration >= max_iteration:
                return
            else:
                current_iteration += 1
                yield Comment(json.loads(line))


def filter(input_filename, output_filename, subreddits):
    with open(input_filename) as input_file:
        with open(output_filename, 'w') as output_file:
            for line in input_file:
                if json.loads(line)['subreddit'] in subreddits:
                    output_file.write(line)


def split_features_and_label(filename, max_iteration=None, filter_function=None):
    # Returns features and labels for graphing purposes
    #
    # filename:            Str, a filename as a path
    # max_iteration:       Int, an optional argument which defines max ammount of
    #                      comments to yield.
    # filter_function      Function, only keep functions that pass filter function.
    features = []
    labels = []

    comments_included = 0

    for comment in load_comments(filename, max_iteration):
        if filter_function is None or filter_function(comment):
            comments_included += 1
            print(comments_included)
            features.append(comment.processed_body)
            labels.append(comment.subreddit)

    return features, labels

def train_multinomialNB(comments, subreddits):
    # Returns a Pipeline Object.
    #
    # comments: A list of (?)
    # subreddits:    A list of (?)

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB

    return Pipeline([
        ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ]).fit(comments, subreddits)


def train_LR(comments, subreddits):
    # Returns a Pipeline Object.
    #
    # comments: A list of (?)
    # subreddits:    A list of (?)

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    return Pipeline([
        ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ]).fit(comments, subreddits)


def seperate_train_test_set(features, labels, percent):
    train_features = features[0:int(len(features) * percent)]
    test_features = features[int(len(features) * percent):]

    train_labels = labels[0:int(len(labels) * percent)]
    test_labels = labels[int(len(labels) * percent):]

    return train_features, test_features, train_labels, test_labels

if __name__ == "__main__":

    sample_size = 100000

    excluded_subreddits = ["AskReddit"]

    def filter_function(comment):
        return comment.subreddit not in excluded_subreddits and comment.processed_body != "deleted" and comment.score >= 100

    # I'm excluding AskReddit for now as it somehow dominates all other labels when it is included.
    features, labels = split_features_and_label("/Users/nick/RC_2015-01_mc10", sample_size, filter_function)

    # Separate training and test sets
    train_features, test_features, train_labels, test_labels = seperate_train_test_set(features, labels, 0.75)

    # Should classify as nfl, nfl, videos
    sample_comments = [
        "Psh, and the 'experts' thought Norman was our top FA priority this offseason...",
        "Where do I buy his jersey?",
        "Awesome! 10/10 Would watch again. Damn it.."
    ]

    multNB_classifier = train_multinomialNB(features, labels)
    LR_classifier = train_LR(features, labels)

    import numpy as np

    print("Naive Bayes plain text accuracy: {}".format(np.mean(multNB_classifier.predict(test_features) == test_labels)))
    print("Logistic Regression plain text accurracy: {}".format(np.mean(LR_classifier.predict(test_features) == test_labels)))
