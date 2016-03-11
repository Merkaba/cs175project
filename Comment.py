# This class defines a comment and keeps track of data for easy usage
#
# original_body:      Str of comment body
# processed_body:     Str of processed body
# subreddit:          Str of subreddit name, used to verify
# score:              Int of comment score, how many upvotes it received
# length:             Int of the number of words

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict

class Comment():
    import nltk.data, nltk.tag

    # Store parts of speech tagger as a class variable for speed.
    tagger = nltk.tag.PerceptronTagger()

    def __init__(self, object):
        import re
        self.original_body = object['body']
        self.processed_body = re.sub("\s+", " ", re.sub("[^a-z0-9]", " ", object['body'].lower())).strip()
        self.subreddit = object['subreddit']
        self.tokenized = self.processed_body.split()
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
        return {
            "spaghetti_spaghetti": "spaghetti spaghetti" in self.processed_body, #AdviceAnimals
            "yum_yum": "yum yum" in self.processed_body, #AdviceAnimals
            "united_states":"United States" in self.processed_body,  #News
            "r_worldnews": "r wordlnews" in self.processed_body, #News
            "r_videos": "r videos" in self.processed_body, #videos
            "watch_v":"watch v" in self.processed_body, #videos
            "r_funny":"r funny" in self.processed_body, #funny
            "funny_comments":"funny comments" in self.processed_body, #funny
            "super_bowl":"super_bowl" in self.processed_body, #nfl
            "this_season": "this season" in self.processed_body, #nfl
            "r_leagueoflegends" : "r leagueoflegends" in self.processed_body, #leagueoflegends
            "gem_gem" :"gem gem" in self.processed_body, #funny
            "high_school":"high school" in self.processed_body, #todayIlearned
            "star_wars":"star wars" in self.processed_body, #todayIlearned
            'score': self.score if self.score>0 else 0,
            'k_begin': str(self.tokenized[:k_beginning]),
            'k_end': str(self.processed_body[-k_ending:]),
            'length': self.length,
            "amp_amp":"amp_amp" in self.processed_body, #AskReddit
            'contains_youtube': 1 if 'youtu' in self.processed_body else 0,
            'contains_common_slang': self.contains_common_slang,
            'amp_amp': 'amp amp'in self.processed_body,
            "r_automoderator":"r automoderator" in self.processed_body, #AskReddit
            "r_askreddit":"r askreddit" in self.processed_body, #AskReddit
        }