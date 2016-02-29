import json
from Comment import Comment
from Classifiers import MultinomialNaiveBayes, LogisticRegression, SupportVectorMachine
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

def evaluate_support_vector_machine(train, test):
    svm_classifier = SupportVectorMachine(train)
    svm_classifier.trainCountVectorizer()
    return svm_classifier.test(test)


def general_test(filename, sample_size, n_cross_validation=5, random_seed=None, filter_fn=None):
    import numpy as np

    data_set = DataSet([comment for comment in load_comments(filename, sample_size)], random_seed, filter_fn)
    sets = data_set.generate_n_cross_validation_sets(n_cross_validation)

    naive_bayes_results = [evaluate_naive_bayes(set[0], set[1]) for set in sets]
    logistic_regression_results = [evaluate_logistic_regression(set[0], set[1]) for set in sets]
    support_vector_machine_results = [evaluate_support_vector_machine(set[0], set[1]) for set in sets]

    return {
        "dataset_size": len(sets[0][0]) + len(sets[0][1]),
        "naive_bayes_average": np.mean(naive_bayes_results),
        "logistic_regression_average": np.mean(logistic_regression_results),
        "support_vector_average": np.mean(support_vector_machine_results),
        "naive_bayes_results": naive_bayes_results,
        "logistic_regression_results": logistic_regression_results,
        "support_vector_machine_results": support_vector_machine_results
    }

def search_for_ideal_threshold():
    filename = "/Users/nick/RC_2015-01_mc10"
    sample_size = 200000
    random_seed = None
    n_cross_validation = 5

    results = [general_test(filename, sample_size, n_cross_validation, random_seed, lambda comment: comment.length > length_threshold) for length_threshold in range(0, 100, 10)]

    max_naive_bayes = max(results, key=lambda x:x['naive_bayes_average'])
    max_logistic_regression = max(results, key=lambda x:x['logistic_regression_average'])

    print("Best average for Naive Bayes {} occurred at length threshold {}".format(max_naive_bayes['naive_bayes_average'], max_naive_bayes['length_threshold']))
    print("Best average for Logistic Regression {} occurred at length threshold {}".format(max_logistic_regression['logistic_regression_average'], max_logistic_regression['length_threshold']))


if __name__ == "__main__":
    filename = "/Users/nick/RC_2015-01_mc10"
    sample_size = 500000

    results = general_test(filename, sample_size, filter_fn=lambda x: x.length > 90)

    print("data set size: {}".format(results['dataset_size']))
    print("naive bayes average: {}".format(results['naive_bayes_average']))
    print("logistic regression average: {}".format(results['logistic_regression_average']))
    print("support vector average: {}".format(results['support_vector_average']))


# Test run with sample_size = 1000000, random_seed = None, n_cross_validation = 5, for lengths 0:100 with step size 10
# Best average for Naive Bayes 0.548185415489 occurred at length threshold 60
# Best average for Logistic Regression 0.675748149737 occurred at length threshold 90



