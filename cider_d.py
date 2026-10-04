# cider_d.py
# CIDEr-D for SCST rewards: document frequencies are precomputed once over the
# training corpus (not per batch), and every sample is scored against ALL
# references of its image. Follows pycocoevalcap's CIDEr-D formulation
# (tf-idf n-grams n=1..4, clipped tf, Gaussian length penalty sigma=6, x10).
from collections import Counter, defaultdict

import numpy as np


def _ngrams(tokens, n=4):
    counts = Counter()
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            counts[tuple(tokens[i:i + k])] += 1
    return counts


class CiderD:
    def __init__(self, refs_corpus, n=4, sigma=6.0):
        """refs_corpus: list over images of list of reference token lists."""
        self.n, self.sigma = n, sigma
        self.df = defaultdict(float)
        for refs in refs_corpus:
            for ng in {ng for r in refs for ng in _ngrams(r, n)}:
                self.df[ng] += 1
        self.log_n = np.log(float(len(refs_corpus)))

    def _vec(self, tokens):
        vec = [dict() for _ in range(self.n)]
        norm = np.zeros(self.n)
        for ng, tf in _ngrams(tokens, self.n).items():
            k = len(ng) - 1
            v = tf * (self.log_n - np.log(max(1.0, self.df.get(ng, 0.0))))
            vec[k][ng] = v
            norm[k] += v * v
        return vec, np.sqrt(norm), len(tokens)

    def _sim(self, h, r):
        (vh, nh, lh), (vr, nr, lr) = h, r
        val = np.zeros(self.n)
        for k in range(self.n):
            for ng, v in vh[k].items():
                if ng in vr[k]:
                    val[k] += min(v, vr[k][ng]) * vr[k][ng]
            if nh[k] != 0 and nr[k] != 0:
                val[k] /= nh[k] * nr[k]
            val[k] *= np.e ** (-((lh - lr) ** 2) / (2 * self.sigma ** 2))
        return val

    def score(self, hyp, refs):
        """hyp: token list; refs: list of token lists. Returns CIDEr-D of one caption."""
        h = self._vec(hyp)
        total = sum(self._sim(h, self._vec(r)) for r in refs)
        return float(np.mean(total) / len(refs) * 10.0)

    def batch_scores(self, hyps, refs_list):
        return np.array([self.score(h, r) for h, r in zip(hyps, refs_list)], dtype=np.float32)
