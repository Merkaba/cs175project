import json
from Comment import Comment
from Dataset import DataSet
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

"""
This module handles the loading of JSON objects from a given file.
It also contains hyperparameter optimization functions, ideal length threshold search,
a classifier evaluation function as well as a human evaluation function.

In the __name__ == "__main__" we have statements commented out that will run all of the previously listed functions.

"""

def load_comments(filename, max_iteration=None):
    """
    Yields a Comment object
    :param filename: Str, a filename as a path
    :param max_iteration: Int, an optional argument which defines max ammount of comments to yield.
    :yield: Comment instance.
    """

    current_iteration = 0

    with open(filename) as dataset:
        for line in dataset:
            if max_iteration is not None and current_iteration >= max_iteration:
                return
            else:
                current_iteration += 1
                yield Comment(json.loads(line))


def optimize_params(classifier, data_set, params):
    """
    Classifier to optimize params for on data_set
    :param classifier: sklearn classifier instance.
    :param data_set: Dict containing a training sparse matrix and labels.
    :param params: Dict containing the hyperparameters and the allowed range to search.
    :return: Optimal hyperparameters for the given classifier for the given range of parameters.
    """
    from sklearn.grid_search import GridSearchCV
    return GridSearchCV(classifier, params, error_score=0).fit(data_set['training_sparse_matrix'], data_set['training_labels'])


def search_for_ideal_threshold():
    """
    Searches for the ideal Comment length (in words) cutoff that optimizes accuracy.
    :return: Dict{ Best comment length cutoff found for the given dataset. }
    """
    filename = "/Users/nick/RC_2015-01_mc10"
    sample_size = 10
    random_seed = None
    n_cross_validation = 5
    classifiers = [MultinomialNB(), LogisticRegression(), LinearSVC()]

    best = {}

    for classifier in classifiers:
        best[type(classifier).__name__] = {"average": -1, "cutoff": -1}

    for cutoff in range(0, 100, 20):
        results = evaluate(filename, sample_size, classifiers, n_cross_validation, random_seed, lambda x: x.length > cutoff)

        for key, value in results[1].iteritems():
            if value['average'] > best[key]['average']:
                best[key]['average'] = value['average']
                best[key]['cutoff'] = cutoff

    return best


def evaluate(filename, sample_size, classifiers, n_cross_validation=5, random_seed=None, filter_fn=None, verbose=False):
    """
    Evaluates the given list of Classifier instances using n cross validation.
    :param filename: Str, a filename as a path
    :param sample_size: Number of Comment instances to use for each cross validation set.
    :param classifiers: List of Classifier instances which will each be fitted on the training data and scored on
    the testing data.
    :param n_cross_validation: Number of cross validation sets to generate during evaluation of each Classifier.
    :param random_seed: A integer value that can be used to ensure stable random generation of the data set.
    :param filter_fn: A function that can be used to filter which Comments are included in the training data on a per
    comment basis.
    :param verbose: If enabled output will be given detailing the progress on the evaluation.
    :return: Tuple (data set size, Dict{ classifier results: Dict{ n cross result accuracy, average accuracy } })
    """

    sets = DataSet([comment for comment in load_comments(filename, sample_size)], random_seed, filter_fn).generate_n_cross_validation_sets(n_cross_validation)

    if verbose:
        print("Finished generating cross validation sets")

    results = {}

    import numpy as np
    for classifier in classifiers:
        result = [classifier.fit(set['training_sparse_matrix'], set['training_labels']).score(set['validation_sparse_matrix'], set['validation_labels']) for set in sets],
        results[type(classifier).__name__] = {"result": result, "average":  np.mean(result)}
        if verbose:
            print("Finished testing {}".format(type(classifier).__name__))

    return (sets[0]['size'], results)


def human_test(filename, sample_size, random_seed=None, filter_fn=None):
    """
    Presents a textual interface that allows human accuracy to be evaluated on the given data. A human will be asked
    to classify a given comment into a specific subreddit.
    :param filename: Str, a filename as a path
    :param sample_size: Number of Comment instances to use for each cross validation set.
    :param random_seed: A integer value that can be used to ensure stable random generation of the data set.
    :param filter_fn: A function that can be used to filter which Comments are included in the training data on a per
    comment basis.
    :return: Accuracy of correctly predicted comment => subreddit labels.
    """

    data = DataSet([comment for comment in load_comments(filename, sample_size)], random_seed, filter_fn).generate_human(filter_fn)
    correct = 0
    potential_labels = list(set(data[1]))

    for i in range(0, len(data[0])):
        print(data[0][i])

        for j in range(0, len(potential_labels)):
            print("[{}]: {}".format(j, potential_labels[j]))
        choice = input("Enter the # corresponding to the correct subreddit: ")
        if potential_labels[choice] == data[1][i]:
            correct += 1

    return {"human_average": correct / float(len(data[0]))}


if __name__ == "__main__":
    filename = "/Users/nick/RC_2015-01_mc10"

    # Primary classifier evaluation. This is where we are running our experiments and generating our result data.

    # sample_size = 100
    # classifiers = [MultinomialNB(), LogisticRegression(), LinearSVC(), DummyClassifier(strategy="most_frequent")]
    # filter_fn = lambda x: x.length > 0
    # results = evaluate(filename=filename, sample_size=sample_size, classifiers=classifiers, filter_fn=filter_fn, verbose=True)
    #
    # print("data set size: {}".format(results[0]))
    # for key, value in results[1].iteritems():
    #     print("{} average: {}".format(key, value['average']))




    # # Length cutoff search. This is where we attempt to locate an ideal comment length (in words) cutoff.
    #
    # for key, value in search_for_ideal_threshold().iteritems():
    #     print("Best average for {}: {} occured at length cutoff {}".format(key, value["average"], value["cutoff"]))




    # # Hyperparameter optimization. This is where we attempt to locate the ideal hyperparameters in the
    # # defined ranges of values.

    # set = DataSet([comment for comment in load_comments(filename, 1000)]).generate_train_test(0.75)
    #
    # from numpy import linspace, logspace
    #
    # lr_params = {'C': logspace(10^-4, 10, 50), 'penalty': ['l2', 'l1'], 'class_weight': ['balanced']}
    # nb_params = {'alpha': linspace(0.1, 1.0, 100), 'fit_prior': [True, False]}
    # svm_params = {'C': logspace(10^-4, 10, 50), 'penalty': ['l2', 'l1'], 'loss': ['hinge', 'squared_hinge']}
    #
    # lr_search = optimize_params(LogisticRegression(), set, lr_params)
    # nb_search = optimize_params(MultinomialNB(), set, nb_params)
    # svm_search = optimize_params(LinearSVC(), set, svm_params)
    #
    # print("score: {}, params: {}".format(lr_search.best_score_, lr_search.best_params_))
    # print("score: {}, params: {}".format(nb_search.best_score_, nb_search.best_params_))
    # print("score: {}, params: {}".format(svm_search.best_score_, svm_search.best_params_))




    # # Human evaluation. This is where we evaluate human performance on our dataset.

    # results = human_test(filename, 20)
    # print("human average: {}".format(results['human_average']))