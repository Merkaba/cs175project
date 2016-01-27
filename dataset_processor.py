import json
import nltk
from collections import Counter

def load_comments(filename, count):
    comments = []

    with open(filename) as dataset:
        for i in range(0, count):
            comments.append(json.loads(dataset.readline()))

    return comments


def average_comment_length(comments):
    return sum([len(comment['body']) for comment in comments]) / len(comments)


def max_comment_length(comments):
    return max([len(comment['body']) for comment in comments])


def min_comment_length(comments):
    return min([len(comment['body']) for comment in comments])


def average_comment_score(comments):
    return sum([comment['score'] for comment in comments]) / len(comments)


def max_comment_score(comments):
    return max([comment['score'] for comment in comments])


def min_comment_score(comments):
    return min([comment['score'] for comment in comments])


def sub_reddit_count(comments):
    counts = Counter()

    for comment in comments:
        counts[comment['subreddit']] += 1

    return counts


if __name__ == "__main__":
    comments = load_comments("RC_2015-01", 1000)

    print("Comment length: Average={}, Minimum={}, Maximum={}"
          .format(average_comment_length(comments), min_comment_length(comments), max_comment_length(comments)))
    print("Comment score: Average={}, Minimum={}, Maximum={}"
          .format(average_comment_score(comments), min_comment_score(comments), max_comment_score(comments)))
    print("{} subreddits".format(len(sub_reddit_count(comments))))
    print(sub_reddit_count(comments).most_common(10))
