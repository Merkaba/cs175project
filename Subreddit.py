from collections import Counter

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