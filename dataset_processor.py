import json
from Comment import Comment
from Dataset import DataSet
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

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


def optimize_params(classifier, data_set, params):
    from sklearn.grid_search import GridSearchCV
    return GridSearchCV(classifier, params, error_score=0).fit(data_set['training_sparse_matrix'], data_set['training_labels'])


def test(filename, sample_size, classifiers, n_cross_validation=5, random_seed=None, filter_fn=None, verbose=False):

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


def search_for_ideal_threshold():
    filename = "/Users/nick/RC_2015-01_mc10"
    sample_size = 10
    random_seed = None
    n_cross_validation = 5
    classifiers = [MultinomialNB(), LogisticRegression(), LinearSVC()]

    best = {}

    for classifier in classifiers:
        best[type(classifier).__name__] = {"average": -1, "cutoff": -1}

    for cutoff in range(0, 100, 20):
        results = general_test(filename, sample_size, classifiers, n_cross_validation, random_seed, lambda x: x.length > cutoff)

        for key, value in results[1].iteritems():
            if value['average'] > best[key]['average']:
                best[key]['average'] = value['average']
                best[key]['cutoff'] = cutoff

    return best


def human_test(filename, sample_size, random_seed=None, filter_fn=None):
    import numpy as np
    data_set = DataSet([comment for comment in load_comments(filename, sample_size)], random_seed, filter_fn)

    data = data_set.generate_human(filter_fn)
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
    sample_size = 100
    classifiers = [MultinomialNB(), LogisticRegression(), LinearSVC(), DummyClassifier(strategy="most_frequent")]
    filter_fn = lambda x: x.length > 0

    results = test(filename=filename, sample_size=sample_size, classifiers=classifiers, filter_fn=filter_fn, verbose=True)

    print("data set size: {}".format(results[0]))
    for key, value in results[1].iteritems():
        print("{} average: {}".format(key, value['average']))

    # for key, value in search_for_ideal_threshold().iteritems():
    #     print("Best average for {}: {} occured at length cutoff {}".format(key, value["average"], value["cutoff"]))


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

    # results = human_test(filename, 20)
    # print("human average: {}".format(results['human_average']))