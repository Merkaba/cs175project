# cs175project
CS 175 Project Repository

Data set: nickmorri.com/downloads/RC_2015-01_mc10.zip

dataset_processor.py -

A function, load_comments, accepts a file path and continues to yield up to max_iteration Comment instances that have
been read and decoded from their JSON representation contained in the given file path.

A function, search_for_ideal_threshold, searches for the best length cutoff (in words) threshold that would yield the
best accuracy by limiting Comments included in the training set to only this >= the optimal length cutoff.

A function, evaluation, accepts a list of classifiers that should be evaluated on the given dataset.
For each classifier a 5 cross validation sets of training and validation data are generated. Each classifier is fitted
on the training sparse matrix and it’s associated labels. The sparse matrix that is used for fitting is created by the
DataSet class fit_transform. The average accuracies for each classifier across the 5 cross validation evaluations is
computed using the classifier score method which accepts the testing sparse matrix and it’s associated labels.
Currently we are evaluating MultinomialNB, LogisticRegression, LinearSVC and DummyClassifier (using the “most_frequent”
strategy which will always predict the label which is most frequent).

A function, human_test, accepts a list of unprocessed comment bodies with their associated subreddit labels are
retrieved from the DataSet. The user is presented with the unprocessed body for each of these comments and is asked to
choose which subreddit they think the comment came from. The percent of correct choices is returned.

A function, optimize_params, accepts set of training and testing sparse matrices are generated from the DataSet
(75% training data, 25% testing data). For each classifier we are evaluating we build a dictionary that contains the
relevant hyper parameters for that classifier. The key being the name of the hyper parameter and the value being the
range of values that will be evaluated. An exhaustive GridSearch is performed that will return the combination of
hyper parameters that performs best on the given training and testing data sets. These hyper parameters are then used
during evaluation of the classifiers in the main evaluation pipeline.


Comment.py -

Contains comment data and provides useful methods for feature extraction. During Comment instantiation process
comment text body by converting all characters to lowercase, removing punctuation, and stripping all leading and
trailing whitespace leaving only text that has relevant vocabulary.


Dataset.py -

A DataSet object is initialized with a list of all JSON objects which have been converted to Comment instances.
The DataSet object will automatically remove all Comment instances which have been indicated as deleted as there would
not be any useful vocabulary remaining as the comment body will have been removed. The given list of Comment instances
will then be randomly shuffled to ensure that distribution of Comments is normalized and will not favor any specific
subreddit. Provides useful methods for extracting subsets of data for training and validation sets. Can generate
sparse matrices for fitting sklearn Classifiers. Handles filtering of data based on given filter_fn.



