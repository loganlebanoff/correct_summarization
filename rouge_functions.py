from __future__ import division
import collections
import six

def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)

def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))

def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)

def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result

def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0

def rouge_n(peer, models, n, alpha, metric='f1'):
    """
    Compute the ROUGE-N score of a peer with respect to one or more models, for
    a given value of `n`.
    """
    if len(models) == 0:
        return 0.

    if type(models[0]) is not list:
        models = [models]

    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    if metric == 'f1':
        return _safe_f1(matches, recall_total, precision_total, alpha)
    elif metric == 'precision':
        return _safe_divide(matches, precision_total)
    elif metric == 'recall':
        return _safe_divide(matches, recall_total)
    else:
        raise Exception('must be one of {f1, recall, precision}')

def rouge_1(peer, models, alpha, metric='f1'):
    """
    Compute the ROUGE-1 (unigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 1, alpha, metric='f1')

def rouge_2(peer, models, alpha, metric='f1'):
    """
    Compute the ROUGE-2 (bigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 2, alpha, metric='f1')