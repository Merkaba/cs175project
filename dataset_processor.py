import json
from Comment import Comment
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


def evaluate_classifier(classifier, data):
    classifier.fit(data['training_sparse_matrix'], data['training_labels'])
    return classifier.test(data['validation_sparse_matrix'], data['validation_labels'])


def optimize_params(classifier, data_set, params):
    from sklearn.grid_search import GridSearchCV
    return GridSearchCV(classifier, params, error_score=0).fit(data_set['training_sparse_matrix'], data_set['training_labels'])


def general_test(filename, sample_size, n_cross_validation=5, random_seed=None, filter_fn=None):
    from Classifiers import MultinomialNaiveBayes, LogisticRegression, SupportVectorMachine
    import numpy as np

    data_set = DataSet([comment for comment in load_comments(filename, sample_size)], random_seed, filter_fn)
    print("Finished loading DataSet")

    sets = data_set.generate_n_cross_validation_sets(n_cross_validation)
    print("Finished generating cross validation sets")

    naive_bayes_results = [evaluate_classifier(MultinomialNaiveBayes(), set) for set in sets]
    print("Finished training MultinomialNaiveBayes")

    logistic_regression_results = [evaluate_classifier(LogisticRegression(), set) for set in sets]
    print("Finished training LogisticRegression")

    support_vector_machine_results = [evaluate_classifier(SupportVectorMachine(), set) for set in sets]
    print("Finished training SupportVectorMachine")

    return {
        "dataset_size": sets[0]['size'],
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
    max_support_vector_machine = max(results, key=lambda x:x['support_vector_average'])

    print("Best average for Naive Bayes {} occurred at length threshold {}".format(max_naive_bayes['naive_bayes_average'], max_naive_bayes['length_threshold']))
    print("Best average for Logistic Regression {} occurred at length threshold {}".format(max_logistic_regression['logistic_regression_average'], max_logistic_regression['length_threshold']))
    print("Best average for Support Vector Machine {} occurred at length threshold {}".format(max_support_vector_machine['support_vector_average'], max_support_vector_machine['length_threshold']))


if __name__ == "__main__":
    filename = "/Users/nick/RC_2015-01_mc10"

    results = general_test(filename, 10000, filter_fn=lambda x: x.length > 0)

    print("data set size: {}".format(results['dataset_size']))
    print("naive bayes average: {}".format(results['naive_bayes_average']))
    print("logistic regression average: {}".format(results['logistic_regression_average']))
    print("support vector average: {}".format(results['support_vector_average']))


    # data_set = DataSet([comment for comment in load_comments(filename, 1000)])
    #
    # from sklearn.linear_model import LogisticRegression
    # # C, penalty, class_weight[balanced]
    # from sklearn.naive_bayes import MultinomialNB
    # # {'alpha': 1.0, 'fit_prior': True, 'class_prior': None}
    # from sklearn.svm import LinearSVC
    # # C, penalty, loss
    #
    # from numpy import linspace, logspace
    #
    # lr_params = {'penalty': ['l2', 'l1'], 'C': logspace(10^-4, 10, 50), 'class_weight': ['balanced']}
    # nb_params = {'alpha': linspace(0.1, 1.0, 100), 'fit_prior': [True, False]}
    # svm_params = {'C': logspace(10^-4, 10, 50), 'penalty': ['l2', 'l1'], 'loss': ['hinge', 'squared_hinge']}
    #
    # lr_search = optimize_params(LogisticRegression(), data_set.generate_train_test(0.75), lr_params)
    # nb_search = optimize_params(MultinomialNB(), data_set.generate_train_test(0.75), nb_params)
    # svm_search = optimize_params(LinearSVC(), data_set.generate_train_test(0.75), svm_params)
    #
    # print("score: {}, params: {}".format(lr_search.best_score_, lr_search.best_params_))
    # print("score: {}, params: {}".format(nb_search.best_score_, nb_search.best_params_))
    # print("score: {}, params: {}".format(svm_search.best_score_, svm_search.best_params_))



