from nltk.corpus import stopwords
from nltk import FreqDist
import re
import nltk.data, nltk.tag

tagger = nltk.tag.PerceptronTagger()

class Comment():

# This class defines a comment and keeps track of data for easy usage
#
# body:      Str of comment body
# subreddit: Str of subreddit name, used to verify
# score:     Int of comment score, how many upvotes it received

    def __init__(self, object):
        lower_case_body = object['body'].lower()
        punctuation_removed = re.sub("[^a-z0-9]", " ", lower_case_body)
        whitespace_removed = re.sub("\s+", " ", punctuation_removed).strip()
        self.body = whitespace_removed
        self.subreddit = object['subreddit']
        self.score = object['score']

    def parts_of_speech(self, words_to_pos=10):
        body_split = self.body.split()

        pos_list = [item[0] + " " + item[1] for item in tagger.tag(body_split[:words_to_pos])]
        pos_list.extend(body_split[words_to_pos:])

        return " ".join(pos_list)

    def get_features(self):
        return {
            "POS": self.parts_of_speech(),
            "score": self.score
        }

    def length(self):
        '''returns an int, the length of this comment's body'''
        return len(self.body)