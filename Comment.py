# This class defines a comment and keeps track of data for easy usage
#
# original_body:      Str of comment body
# processed_body:     Str of processed body
# subreddit:          Str of subreddit name, used to verify
# score:              Int of comment score, how many upvotes it received
# length:             Int of the number of words

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from collections import defaultdict

class Comment():
    import nltk.data, nltk.tag

    # Store parts of speech tagger as a class variable for speed.
    tagger = nltk.tag.PerceptronTagger()

    def __init__(self, object):
        import re
        self.original_body = object['body']
        self.processed_body = re.sub("\s+", " ", re.sub("[^a-z0-9]", " ", object['body'].lower())).strip()
        self.tokenized = self.processed_body.split()
        self.subreddit = object['subreddit']
        self.score = object['score']
        self.length = len(self.tokenized)
        self.contains_common_slang = 0

        self.common_slang()


    def parts_of_speech(self, max_words_to_pos=None):
        # Split the processed_body string into a list elements at every space.
        body_split = self.processed_body.split()

        result=defaultdict(int)
        # If the max_words_to_pos is None then we will process the entire list.
        if max_words_to_pos is None:
            for each in Comment.tagger.tag(body_split):
                result[each[1]] += 1
            return result
        # Otherwise we will process the first max_words_to_pos words. The remaining words will remain as regular text.
        else:
            pos_list = [item[0] + " " + item[1] for item in Comment.tagger.tag(body_split[:max_words_to_pos])]
            pos_list.extend(body_split[max_words_to_pos:])
            for each in Comment.tagger.tag(body_split[:max_words_to_pos]):
                result[each[1]] += 1
            return result

    def common_slang(self):
        common_slang = ['lol', 'rofl', 'lmao', 'xd', ':)', ':(', ':p']

        for each in self.tokenized:
            if each in common_slang:
                self.contains_common_slang = 1
            break

    def features(self, k_beginning = 2, k_ending = 2):
        # "pos": self.parts_of_speech()
        return {
            'score': self.score if self.score>0 else 0,
            'k_begin': str(self.tokenized[:k_beginning]),
            'k_end': str(self.processed_body[-k_ending:]),
            'length': self.length,
            'contains_url': 1 if '.com' in self.processed_body else 0,
            'contains_youtube': 1 if 'youtu' in self.processed_body else 0,
            'contains_common_slang': self.contains_common_slang
        }