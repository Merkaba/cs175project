import json
from Comment import Comment
from Subreddit import Subreddit
from collections import Counter

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


def count_subreddits(filename):

    counts = Counter()

    for comment in load_comments(filename):
        counts[comment.subreddit] += 1

    return counts


def filter(input_filename, output_filename, subreddits):

    with open(input_filename) as input_file:
        with open(output_filename, 'w') as output_file:
            for line in input_file:
                if json.loads(line)['subreddit'] in subreddits:
                    output_file.write(line)

if __name__ == "__main__":
    input_filename = "/Users/nick/RC_2015-01"
    output_filename = "/Users/nick/RC_2015-01_mc10"
    # most_common_subreddits_computed = [entry[0] for entry in count_subreddits(input_filename)]
    most_common_subreddits_cached = [u'AskReddit', u'nfl', u'funny', u'leagueoflegends', u'pics', u'worldnews', u'todayilearned', u'DestinyTheGame', u'AdviceAnimals', u'videos']
    filter(input_filename, output_filename, most_common_subreddits_cached)

