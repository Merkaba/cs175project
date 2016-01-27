import json
from Comment import Comment
from Subreddit import Subreddit

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


if __name__ == "__main__":

    subreddits = categorize_comments("/Users/nick/Desktop/RC_2015-01", 10000)

    for name, subreddit in subreddits.items():
        print(subreddit.avg_comment_score())
