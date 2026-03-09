#import shap
import math
import numpy as np
import random
import re
from shap.maskers import Text

class RandomTextMasker(Text):
    def clustering(self, s):
        """Compute the clustering of tokens for the given string."""
        self._update_s_cache(s)
        special_tokens = []
        sep_token = getattr_silent(self.tokenizer, "sep_token")
        if sep_token is None:
            special_tokens = []
        else:
            special_tokens = [sep_token]

        # convert the text segments to tokens that the partition tree function expects
        tokens = []
        space_end = re.compile(r"^.*\W$")
        letter_start = re.compile(r"^[A-Za-z]")
        for i, v in enumerate(self._segments_s):
            if i > 0 and space_end.match(self._segments_s[i - 1]) is None and letter_start.match(v) is not None and \
                    tokens[i - 1] != "":
                tokens.append("##" + v.strip())
            else:
                tokens.append(v.strip())

        pt = random_partition(tokens, special_tokens)

        # use the rescaled size of the clusters as their height since the merge scores are just a
        # heuristic and not scaled well
        pt[:, 2] = pt[:, 3]
        pt[:, 2] /= pt[:, 2].max()

        return pt

openers = {
    "(": ")"
}
closers = {
    ")": "("
}
enders = [".", ","]
connectors = ["but", "and", "or"]

class TokenGroup:
    """A token group (substring) representation used for token clustering."""

    def __init__(self, group, index=None):
        self.g = group
        self.index = index

    def __repr__(self):
        return self.g.__repr__()

    def __getitem__(self, index):
        return self.g[index]

    def __add__(self, o):
        return TokenGroup(self.g + o.g)

    def __len__(self):
        return len(self.g)

def merge_score(group1, group2, special_tokens):
    """Compute the score of merging two token groups.

    special_tokens: tokens (such as separator tokens) that should be grouped last
    """
    score = 0

    # 
    score -= random.random()

    # ensures special tokens are combined last, so 1st subtree is 1st sentence and 2nd subtree is 2nd sentence

    # Remove every score logic that isn't balance
    # if len(special_tokens) > 0:
    #     if group1[-1].s in special_tokens and group2[0].s in special_tokens:
    #         score -= math.inf # subtracting infinity to create lowest score and ensure combining these groups last

    # Remove every score logic that isn't balance
    # merge broken-up parts of words first
    # if group2[0].s.startswith("##"):
    #     score += 20

    # Remove every score logic that isn't balance
    # merge apostrophe endings next
    # if group2[0].s == "'" and (len(group2) == 1 or (len(group2) == 2 and group2[1].s in ["t", "s"])):
    #     score += 15
    # if group1[-1].s == "'" and group2[0].s in ["t", "s"]:
    #     score += 15

    # start_ctrl = group1[0].s.startswith("[") and group1[0].s.endswith("]")
    # end_ctrl = group2[-1].s.startswith("[") and group2[-1].s.endswith("]")

    # if (start_ctrl and not end_ctrl) or (end_ctrl and not start_ctrl):
    #     score -= 1000
    # if group2[0].s in openers and not group2[0].balanced:
    #     score -= 100
    # if group1[-1].s in closers and not group1[-1].balanced:
    #     score -= 100

    # # attach surrounding an openers and closers a bit later
    # if group1[0].s in openers and group2[-1] not in closers:
    #     score -= 2
    #
    # # reach across connectors later
    # if group1[-1].s in connectors or group2[0].s in connectors:
    #     score -= 2
    #
    # # reach across commas later
    # if group1[-1].s == ",":
    #     score -= 10
    # if group2[0].s == ",":
    #     if len(group2) > 1: # reach across
    #         score -= 10
    #     else:
    #         score -= 1

    # # reach across sentence endings later
    # if group1[-1].s in [".", "?", "!"]:
    #     score -= 20
    # if group2[0].s in [".", "?", "!"]:
    #     if len(group2) > 1: # reach across
    #         score -= 20
    #     else:
    #         score -= 1

    # Remove the balance logic that isn't random
    #score -= len(group1) + len(group2)
    #print(group1, group2, score)
    return score

def merge_closest_groups(groups, special_tokens):
    """Finds the two token groups with the best merge score and merges them."""
    scores = [merge_score(groups[i], groups[i+1], special_tokens) for i in range(len(groups)-1)]
    #print(scores)
    ind = np.argmax(scores)
    groups[ind] = groups[ind] + groups[ind+1]
    #print(groups[ind][0].s in openers, groups[ind][0])
    if groups[ind][0].s in openers and groups[ind+1][-1].s == openers[groups[ind][0].s]:
        groups[ind][0].balanced = True
        groups[ind+1][-1].balanced = True


    groups.pop(ind+1)

def random_partition(decoded_tokens, special_tokens):
    """Build a heriarchial clustering of tokens that align with sentence structure.

    Note that this is fast and heuristic right now.
    TODO: Build this using a real constituency parser.
    """
    token_groups = [TokenGroup([Token(t)], i) for i, t in enumerate(decoded_tokens)]
#     print(token_groups)
    M = len(decoded_tokens)
    new_index = M
    clustm = np.zeros((M-1, 4))
    for i in range(len(token_groups)-1):
        scores = [merge_score(token_groups[i], token_groups[i+1], special_tokens) for i in range(len(token_groups)-1)]
#         print(scores)
        ind = np.argmax(scores)

        lind = token_groups[ind].index
        rind = token_groups[ind+1].index
        clustm[new_index-M, 0] = token_groups[ind].index
        clustm[new_index-M, 1] = token_groups[ind+1].index
        clustm[new_index-M, 2] = -scores[ind]
        clustm[new_index-M, 3] = (clustm[lind-M, 3] if lind >= M else 1) + (clustm[rind-M, 3] if rind >= M else 1)

        token_groups[ind] = token_groups[ind] + token_groups[ind+1]
        token_groups[ind].index = new_index

        # track balancing of openers/closers
        if token_groups[ind][0].s in openers and token_groups[ind+1][-1].s == openers[token_groups[ind][0].s]:
            token_groups[ind][0].balanced = True
            token_groups[ind+1][-1].balanced = True

        token_groups.pop(ind+1)
        new_index += 1

    # negative means we should never split a group, so we add 10 to ensure these are very tight groups
    # (such as parts of the same word)
    clustm[:, 2] = clustm[:, 2] + 10

    return clustm

def getattr_silent(obj, attr):
    """This turns of verbose logging of missing attributes for huggingface transformers.

    This is motivated by huggingface transformers objects that print error warnings
    when we access unset properties.
    """
    reset_verbose = False
    if getattr(obj, 'verbose', False):
        reset_verbose = True
        obj.verbose = False

    val = getattr(obj, attr, None)

    if reset_verbose:
        obj.verbose = True

    # fix strange huggingface bug where `obj.verbose = False` causes val to change from None to "None"
    if val == "None":
        val = None

    return val

class Token:
    """A token representation used for token clustering."""

    def __init__(self, value):
        self.s = value
        if value in openers or value in closers:
            self.balanced = False
        else:
            self.balanced = True

    def __str__(self):
        return self.s

    def __repr__(self):
        if not self.balanced:
            return self.s + "!"
        return self.s

# Use it
# masker = CustomTextMasker(your_tokenizer)
# explainer = shap.Explainer(model, masker)