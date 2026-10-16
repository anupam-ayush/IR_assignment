"""Spelling correction using Damerau-Levenshtein distance and frequency ranking."""

from typing import List, Tuple, Optional, Dict
from lexis.index.positional_index import PositionalIndex


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Damerau-Levenshtein distance (insertions, deletions, substitutions, transpositions)."""
    d = {}
    len1 = len(s1)
    len2 = len(s2)

    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )
            # Transposition of adjacent characters
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)

    return d[(len1 - 1, len2 - 1)]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute classical Levenshtein distance (insertions, deletions, substitutions)."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[m][n]


class SpellingCorrector:
    """Detects OOV terms and suggests deterministic corrections based on edit distance."""

    def __init__(self, index: PositionalIndex):
        self.index = index
        self.vocab = index.vocabulary()
        self.term_freqs = {
            t: (index.dictionary.get(t).total_tf if index.dictionary.get(t) else 0)
            for t in self.vocab
        }

    def is_in_vocabulary(self, term: str) -> bool:
        return self.index.dictionary.contains(term.lower())

    def suggest_corrections(
        self,
        term: str,
        max_distance: int = 2,
        top_k: int = 5,
    ) -> List[Tuple[str, int]]:
        """Return top corrections as (word, distance) tuples.
        
        Ranked deterministically by:
        1. Edit Distance (ascending)
        2. Total corpus TF (descending)
        3. Lexicographical order (ascending)
        """
        w = term.lower()
        if self.is_in_vocabulary(w):
            return [(w, 0)]

        candidates: List[Tuple[str, int, int]] = []  # (term, dist, tf)
        w_len = len(w)

        for cand in self.vocab:
            # Length filter heuristic: length difference cannot exceed max_distance
            if abs(len(cand) - w_len) > max_distance:
                continue

            dist = damerau_levenshtein_distance(w, cand)
            if dist <= max_distance:
                tf = self.term_freqs.get(cand, 0)
                candidates.append((cand, dist, tf))

        # Deterministic sorting
        candidates.sort(key=lambda x: (x[1], -x[2], x[0]))
        return [(c[0], c[1]) for c in candidates[:top_k]]

    def suggest_correction(self, term: str, max_distance: int = 2) -> Optional[str]:
        """Return single best suggested correction or None if no candidate within max_distance."""
        corrections = self.suggest_corrections(term, max_distance=max_distance, top_k=1)
        return corrections[0][0] if corrections else None
