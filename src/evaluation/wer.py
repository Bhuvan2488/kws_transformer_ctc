# src/evaluation/wer.py

def wer(reference, hypothesis):
    """
    Word Error Rate using Levenshtein distance.
    Args:
        reference: list[str]
        hypothesis: list[str]
    Returns:
        WER (float)
    """
    n = len(reference)
    m = len(hypothesis)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1], # substitution
                )

    return dp[n][m] / max(1, n)
