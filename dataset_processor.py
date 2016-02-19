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


def split_text_and_label(filename, words_to_pos, score_threshold=None, max_iteration=None, excluded_subreddits=[]):
    # Returns text and labels for graphing purposes
    #
    # filename:            Str, a filename as a path
    # score_threshold:     Int, optional argument which filters out low-score comments
    # max_iteration:       Int, an optional argument which defines max ammount of
    #                      comments to yield.
    # excluded_subreddits: List of Str, removes certain subreddits from consideration
    text = []
    labels = []

    iteration = 0

    for comment in load_comments(filename, max_iteration):
        if score_threshold is None or comment.score >= score_threshold and \
                        comment.subreddit not in excluded_subreddits and comment.body != "deleted":
            text.append(comment.parts_of_speech(words_to_pos))
            labels.append(comment.subreddit)
            iteration += 1
            print(iteration)

    return text, labels


def train_multinomialNB(comments, subreddits):
    # Returns a Pipeline Object.
    #
    # comments: A list of (?)
    # subreddits:    A list of (?)

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction import CountVectorizer
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


if __name__ == "__main__":

    sample_size = 1000000

    # I'm excluding AskReddit for now as it somehow dominates all other labels when it is included.
    text, labels = split_text_and_label("/Users/nick/RC_2015-01_mc10", 5, 100, sample_size, ["AskReddit"])

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
