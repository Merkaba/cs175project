from collections import Counter
from nltk import FreqDist

class Subreddit():
# This class defines a subreddit and contains data about it for easy usage
#
# name:         Str containing the subreddits name
# comments:     List of Str, contains all comments made in this subreddit
# bag_of_words: FreqDist, contains the frequency distribution of all comments
#               made

    def __init__(self, name):
        self.name = name
        self.comments = []
        self.bag_of_words = FreqDist()


    def add_comment(self, comment):
        # Adds a comment to the comment list and unions the comment's FreqDist
        # with the subreddit's FreqDist, stored in bag_of_words.
        self.comments.append(comment)
        self.bag_of_words.update(comment.bag_of_words())


    def comment_count(self):
        # Returns an Int, a count of how many comments are in the subreddit
        return len(self.comments)


    def sum_comment_length(self):
        # Returns an Int, the total of all comment lengths in a subreddit
        count = 0
        for comment in self.comments:
            count += comment.length()
        return count


    def min_comment_length(self):
        # Returns an Int, the smallest comment's length in characters
        return min([len(comment) for comment in self.comments])


    def max_comment_length(self):
        # Returns an Int, the largest comment's length in characters
        return max([len(comment) for comment in self.comments])


    def avg_comment_length(self):
        # Returns an Int, the average of all comment lengths in characters
        return self.sum_comment_length() / self.comment_count()


    def sum_comment_score(self):
        # Returns an Int, the total of all comment scores in this subreddit
        count = 0
        for comment in self.comments:
            count += comment.score
        return count


    def min_comment_score(self):
        # Returns an Int, the lowest score of all comments in this subreddit
        return min([comment.score for comment in self.comments])


    def max_comment_score(self):
        # Returns an Int, the highest score of all comments in this subreddit
        return max([comment.score for comment in self.comments])


    def avg_comment_score(self):
        # Returns an Int, the average score of all comments in this subreddit
        return self.sum_comment_score() / self.comment_count()