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