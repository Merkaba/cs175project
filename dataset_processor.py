import json
from collections import Counter

class Comment():

    def __init__(self, object):
        self.body= object['body'].split()
        self.subreddit = object['subreddit']
        self.score = object['score']
        self.body_bag_of_words = None


    def bag_of_words(self):
        if self.body_bag_of_words is None:
            self.body_bag_of_words = Counter()
            for word in self.body:
                self.body_bag_of_words [word] += 1

        return self.body_bag_of_words

    def length(self):
        return len(self.body)


class Subreddit():

    def __init__(self, name):
        self.name = name
        self.comments = []
        self.bag_of_words = Counter()

    def add_comment(self, comment):
        self.comments.append(comment)
        self.bag_of_words.update(comment.bag_of_words())

    def comment_count(self):
        return len(self.comments)

    def sum_comment_length(self):
        count = 0
        for comment in self.comments:
            count += comment.length()
        return count


    def min_comment_length(self):
        return min([len(comment) for comment in self.comments])


    def max_comment_length(self):
        return max([len(comment) for comment in self.comments])


    def avg_comment_length(self):
        return self.sum_comment_length() / self.comment_count()


    def sum_comment_score(self):
        count = 0
        for comment in self.comments:
            count += comment.score
        return count


    def min_comment_score(self):
        return min([comment.score for comment in self.comments])


    def max_comment_score(self):
        return max([comment.score for comment in self.comments])


    def avg_comment_score(self):
        return self.sum_comment_score() / self.comment_count()


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

if __name__ == "__main__":

    subreddits = categorize_comments("/Users/nick/Desktop/RC_2015-01", 10000)

    for name, subreddit in subreddits.items():
        print(subreddit.avg_comment_score())
