# This class defines a comment and keeps track of data for easy usage
#
# body:      Str of comment body
# subreddit: Str of subreddit name, used to verify
# score:     Int of comment score, how many upvotes it received
class Comment():
    import nltk.data, nltk.tag

    # Store parts of speech tagger as a class variable for speed.
    tagger = nltk.tag.PerceptronTagger()

    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    def __init__(self, object):
        import re
        self.original_body = object['body']
        self.processed_body = re.sub("\s+", " ", re.sub("[^a-z0-9]", " ", object['body'].lower())).strip()
        self.lemmatized_body = " ".join([Comment.lemmatizer.lemmatize(word) for word in self.processed_body.split()])
        self.subreddit = object['subreddit']
        self.score = object['score']
        self.length = len(self.processed_body)

    def parts_of_speech(self, max_words_to_pos=None):
        # Split the processed_body string into a list elements at every space.
        body_split = self.processed_body.split()

        # If the max_words_to_pos is None then we will process the entire list.
        if max_words_to_pos is None:
            return " ".join([item[0] + " " + item[1] for item in Comment.tagger.tag(body_split)])
        # Otherwise we will process the first max_words_to_pos words. The remaining words will remain as regular text.
        else:
            pos_list = [item[0] + " " + item[1] for item in Comment.tagger.tag(body_split[:max_words_to_pos])]
            pos_list.extend(body_split[max_words_to_pos:])

            return " ".join(pos_list)

    def features(self):
        return {
            "body": self.processed_body,
            "parts_of_speech": self.parts_of_speech(),
            "score": self.score,
            "length": self.length
        }