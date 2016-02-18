from nltk.corpus import stopwords
from nltk import FreqDist
import re

class Comment():
'''This class defines a comment and keeps track of data for easy usage

body:      Str of comment body
subreddit: Str of subreddit name, used to verify
score:     Int of comment score, how many upvotes it received
freq_dist: FreqDist of str, contains frequency of all words in a comment
'''

    def __init__(self, object):
        self.body = re.sub(r'\W+', ' ', object['body'].lower()).strip()
        self.subreddit = object['subreddit']
        self.score = object['score']
        self.freq_dist = None


    def bag_of_words(self):
        '''
        Returns a FreqDist, a frequency distribution of words in this 
        comment, fracturing the comment body (self.body) with standard 
        deliminator in .split() which is whitespace
        '''
        if self.freq_dist is None:
            self.freq_dist = FreqDist([word for word in self.body.split() if word not in set(stopwords.words("english"))])

        return self.freq_dist


    def length(self):
        '''returns an int, the length of this comment's body'''
        return len(self.body)