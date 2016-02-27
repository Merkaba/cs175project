import json
from Comment import Comment
from Classifiers import MultinomialNaiveBayes, LogisticRegression
from Dataset import DataSet
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


def evaluate_logistic_regression(train, test):
    LR_classifier = LogisticRegression(train)
    LR_classifier.trainCountVectorizer()
    return LR_classifier.test(test)


def evaluate_naive_bayes(train, test):
    multNB_classifier = MultinomialNaiveBayes(train)
    multNB_classifier.trainCountVectorizer()
    return multNB_classifier.test(test)


def general_test(filename, sample_size, random_seed, length_threshold):
    import numpy as np

    data_set = DataSet([comment for comment in load_comments(filename, sample_size)], random_seed, lambda comment: comment.length > length_threshold)
    sets = data_set.generate_n_cross_validation_sets(5)

    naive_bayes_results = [evaluate_naive_bayes(set[0], set[1]) for set in sets]
    logistic_regression_results = [evaluate_logistic_regression(set[0], set[1]) for set in sets]

    return {
        "length_threshold": length_threshold,
        "dataset_size": len(sets[0][0]) + len(sets[0][1]),
        "naive_bayes_average": np.mean(naive_bayes_results),
        "logistic_regression_average": np.mean(logistic_regression_results),
        "naive_bayes_results": naive_bayes_results,
        "logistic_regression_results": logistic_regression_results
    }

if __name__ == "__main__":

    filename = "/Users/nick/RC_2015-01_mc10"
    sample_size = 1000000
    random_seed = None

    results = [general_test(filename, sample_size, random_seed, length_threshold) for length_threshold in range(0, 100, 10)]

    max_naive_bayes = max(results, key=lambda x:x['naive_bayes_average'])
    max_logistic_regression = max(results, key=lambda x:x['logistic_regression_average'])

    print("Best average for Naive Bayes {} occurred at length threshold {}".format(max_naive_bayes['naive_bayes_average'], max_naive_bayes['length_threshold']))
    print("Best average for Logistic Regression {} occurred at length threshold {}".format(max_logistic_regression['logistic_regression_average'], max_logistic_regression['length_threshold']))




