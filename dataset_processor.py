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


def split_comments_and_label(filename, max_iteration=None, filter_function=None):
    # Returns features and labels for graphing purposes
    #
    # filename:            Str, a filename as a path
    # max_iteration:       Int, an optional argument which defines max ammount of
    #                      comments to yield.
    # filter_function      Function, only keep functions that pass filter function.
    comments = []
    labels = []

    for comment in load_comments(filename, max_iteration):
        comments.append(comment)
        labels.append(comment.subreddit)

    return comments, labels

def train_multinomialNB(comments, subreddits):
    # Returns a Pipeline Object.
    #
    # comments: A list of (?)
    # subreddits:    A list of (?)

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from nltk.corpus import stopwords
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
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from nltk.corpus import stopwords
    from sklearn.linear_model import LogisticRegression

    return Pipeline([
        ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ]).fit(comments, subreddits)


def seperate_train_test_set(features, labels, percent, filter_function=None):

    train_features = []
    train_labels = []

    divider = int(len(features) * percent)

    for i in range(0, divider):
        if filter_function is None or filter_function(features[i]):
            train_features.append(features[i].processed_body)
            train_labels.append(labels[i])

    test_features = []
    test_labels = []

    for i in range(divider, len(features)):
        if filter_function is None or filter_function(features[i]):
            test_features.append(features[i].processed_body)
            test_labels.append(labels[i])

    return train_features, test_features, train_labels, test_labels

if __name__ == "__main__":

    sample_size = 10000

    def filter_function(comment):
        return comment.processed_body != "deleted"

    # Create two lists, one with the entire Comment object and the other with it's corresponding subreddit.
    comments, labels = split_comments_and_label("/Users/nick/RC_2015-01_mc10", sample_size)

    # Separate training and test sets
    train_features, test_features, train_labels, test_labels = seperate_train_test_set(comments, labels, 0.75)

    # Should classify as nfl, nfl, videos
    sample_comments = [
        "Psh, and the 'experts' thought Norman was our top FA priority this offseason...",
        "Where do I buy his jersey?",
        "Awesome! 10/10 Would watch again. Damn it.."
    ]

    multNB_classifier = train_multinomialNB(train_features, train_labels)
    LR_classifier = train_LR(train_features, train_labels)

    import numpy as np

    print("Naive Bayes plain text accuracy: {}".format(np.mean(multNB_classifier.predict(test_features) == test_labels)))
    print("Logistic Regression plain text accurracy: {}".format(np.mean(LR_classifier.predict(test_features) == test_labels)))
