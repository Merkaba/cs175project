from nltk.corpus import stopwords
from nltk import FreqDist
import re

class Comment():


    def __init__(self, object):
        self.body = re.sub(r'\W+', ' ', object['body'].lower()).strip()
        self.subreddit = object['subreddit']
        self.score = object['score']
        self.freq_dist = None


    def bag_of_words(self):
        if self.freq_dist is None:
            self.freq_dist = FreqDist([word for word in self.body.split() if word not in set(stopwords.words("english"))])

        return self.freq_dist


    def length(self):
        return len(self.body)