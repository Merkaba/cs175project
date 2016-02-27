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


def filter_json(input_filename, output_filename, subreddits):
    with open(input_filename) as input_file:
        with open(output_filename, 'w') as output_file:
            for line in input_file:
                if json.loads(line)['subreddit'] in subreddits:
                    output_file.write(line)



if __name__ == "__main__":

    def filter_fn(comment):
        return comment.length > 5

    data_set = DataSet([comment for comment in load_comments("/Users/nick/RC_2015-01_mc10", 200000)], 3, filter_fn)

    sets = data_set.generate_n_cross_validation_sets(5)

    import numpy as np

    for set in sets:

        multNB_classifier = MultinomialNaiveBayes(set[0])
        multNB_classifier.trainCountVectorizer()

        LR_classifier = LogisticRegression(set[0])
        LR_classifier.trainCountVectorizer()

        validation_features = []
        validation_labels = []

        for item in set[1]:
            # If using DictVectorizer ensure you are appending the item.features()
            # If using CountVectorizer ensure you are appending the item.processed_body
            validation_features.append(item.processed_body)
            validation_labels.append(item.subreddit)

        print("Naive Bayes plain text accuracy: {}".format(np.mean(multNB_classifier.predict(validation_features) == validation_labels)))
        print("Logistic Regression plain text accurracy: {}".format(np.mean(LR_classifier.predict(validation_features) == validation_labels)))

