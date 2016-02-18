import json
from Comment import Comment
from Subreddit import Subreddit
from collections import Counter

def load_comments(filename, max_iteration=None):

    current_iteration = 0

    with open(filename) as dataset:
        for line in dataset:
            if max_iteration is not None and current_iteration >= max_iteration:
                return
            else:
                current_iteration += 1
                yield Comment(json.loads(line))


def categorize_comments(filename, max_iteration=None):

    subreddits = {}

    for comment in load_comments(filename, max_iteration):
        if comment.subreddit not in subreddits:
            subreddits[comment.subreddit] = Subreddit(comment.subreddit)

        subreddits[comment.subreddit].add_comment(comment)

    return subreddits


def count_subreddits(filename):

    counts = Counter()

    for comment in load_comments(filename):
        counts[comment.subreddit] += 1

    return counts


def filter(input_filename, output_filename, subreddits):

    with open(input_filename) as input_file:
        with open(output_filename, 'w') as output_file:
            for line in input_file:
                if json.loads(line)['subreddit'] in subreddits:
                    output_file.write(line)


def split_text_and_label(filename, score_threshold=None, max_iteration=None, excluded_subreddits=[]):

    text = []
    labels = []

    for comment in load_comments(filename, max_iteration):
        if score_threshold is None or comment.score >= score_threshold and comment.subreddit not in excluded_subreddits:
            text.append(comment.body)
            labels.append(comment.subreddit)

    return text, labels


def train_multinomialNB(train_data, Y_train):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB

    return Pipeline([
        ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ]).fit(train_data, Y_train)


def train_LR(train_data, Y_train):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    return Pipeline([
        ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ]).fit(train_data, Y_train)


if __name__ == "__main__":

    sample_size = 1000000

    # I'm excluding AskReddit for now as it somehow dominates all other labels when it is included.
    text, labels = split_text_and_label("/Users/nick/RC_2015-01_mc10", 100, sample_size, ["AskReddit"])

    train_text = text[0:int(len(text) * 0.9)]
    test_text = text[int(len(text) * 0.9):]

    train_labels = labels[0:int(len(text) * 0.9)]
    test_labels = labels[int(len(text) * 0.9):]

    # Should classify as nfl, nfl, videos
    sample_comments = [
        "Psh, and the 'experts' thought Norman was our top FA priority this offseason...",
        "Where do I buy his jersey?",
        "Awesome! 10/10 Would watch again. Damn it.."
    ]

    multNB_classifier = train_multinomialNB(text, labels)
    LR_classifier = train_LR(text, labels)

    import numpy as np

    print np.mean(multNB_classifier.predict(test_text) == test_labels)
    print np.mean(LR_classifier.predict(test_text) == test_labels)
