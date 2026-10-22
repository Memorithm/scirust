//! Pattern matching in sequences: KMP, Boyer-Moore, LCS, DTW.

/// KMP (Knuth-Morris-Pratt) pattern matching.
pub fn kmp(text: &[usize], pattern: &[usize]) -> Vec<usize> {
    if pattern.is_empty()
    {
        return (0..=text.len()).collect();
    }
    if text.len() < pattern.len()
    {
        return Vec::new();
    }

    let lps = compute_lps(pattern);
    let mut results = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;

    while i < text.len()
    {
        if text[i] == pattern[j]
        {
            i += 1;
            j += 1;
            if j == pattern.len()
            {
                results.push(i - j);
                j = lps[j - 1];
            }
        }
        else if j > 0
        {
            j = lps[j - 1];
        }
        else
        {
            i += 1;
        }
    }

    results
}

fn compute_lps(pattern: &[usize]) -> Vec<usize> {
    let m = pattern.len();
    let mut lps = vec![0usize; m];
    let mut len = 0usize;
    let mut i = 1usize;

    while i < m
    {
        if pattern[i] == pattern[len]
        {
            len += 1;
            lps[i] = len;
            i += 1;
        }
        else if len > 0
        {
            len = lps[len - 1];
        }
        else
        {
            lps[i] = 0;
            i += 1;
        }
    }

    lps
}

/// Boyer-Moore pattern matching using the bad-character heuristic.
pub fn boyer_moore(text: &[usize], pattern: &[usize]) -> Vec<usize> {
    if pattern.is_empty()
    {
        return (0..=text.len()).collect();
    }
    if text.len() < pattern.len()
    {
        return Vec::new();
    }

    let n = text.len();
    let m = pattern.len();
    let alphabet_size = 256;

    let mut bad_char = vec![-1i64; alphabet_size];
    for i in 0..m
    {
        bad_char[pattern[i] % alphabet_size] = i as i64;
    }

    let mut results = Vec::new();
    let mut s = 0i64;

    while s <= (n as i64 - m as i64)
    {
        let mut j = m as i64 - 1;

        while j >= 0 && pattern[j as usize] == text[(s + j) as usize]
        {
            j -= 1;
        }

        if j < 0
        {
            results.push(s as usize);
            s += 1;
        }
        else
        {
            let bad_idx = text[(s + j) as usize] % alphabet_size;
            let shift = (j - bad_char[bad_idx]).max(1);
            s += shift;
        }
    }

    results
}

/// Longest Common Subsequence length (space-optimized).
pub fn longest_common_subsequence_len(seq1: &[usize], seq2: &[usize]) -> usize {
    let n = seq1.len();
    let m = seq2.len();
    let mut prev = vec![0usize; m + 1];
    let mut curr = vec![0usize; m + 1];

    for i in 1..=n
    {
        for j in 1..=m
        {
            if seq1[i - 1] == seq2[j - 1]
            {
                curr[j] = prev[j - 1] + 1;
            }
            else
            {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

/// Longest Common Subsequence returning the actual subsequence.
pub fn longest_common_subsequence(seq1: &[usize], seq2: &[usize]) -> Vec<usize> {
    let n = seq1.len();
    let m = seq2.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for i in 1..=n
    {
        for j in 1..=m
        {
            if seq1[i - 1] == seq2[j - 1]
            {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }
            else
            {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    let mut result = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0
    {
        if seq1[i - 1] == seq2[j - 1]
        {
            result.push(seq1[i - 1]);
            i -= 1;
            j -= 1;
        }
        else if dp[i - 1][j] > dp[i][j - 1]
        {
            i -= 1;
        }
        else
        {
            j -= 1;
        }
    }
    result.reverse();
    result
}

/// Dynamic Time Warping distance between two 1-D sequences.
///
/// Returns `sqrt` of the minimal sum of squared differences over warping
/// paths from `(0, 0)` to `(n − 1, m − 1)` with steps `(1, 0)`, `(0, 1)` and
/// `(1, 1)`.
///
/// `step_size = Some(s)` restricts the paths to a Sakoe–Chiba band of
/// half-width `s` around the diagonal scaled to the length ratio: row `i`
/// (1-based) keeps the columns that the straight line from `(0, 0)` to
/// `(n, m)` crosses between rows `i − 1` and `i`, widened by `s` on each side,
/// i.e. `⌊(i − 1)·m/n⌋ + 1 − s ..= ⌈i·m/n⌉ + s`, clamped to `1..=m`. For
/// equal lengths this is `i − s ..= i + s`. Consecutive rows overlap and the
/// first and last rows reach columns `1` and `m`, so the band always contains
/// a path and the result is finite for finite, non-empty inputs, whichever
/// sequence is longer. `None` uses the full grid. If exactly one sequence
/// is empty no path exists and the result is `+∞`.
pub fn dynamic_time_warping(seq1: &[f64], seq2: &[f64], step_size: Option<usize>) -> f64 {
    let n = seq1.len();
    let m = seq2.len();
    let inf = f64::INFINITY;

    let mut dp = vec![vec![inf; m + 1]; n + 1];
    dp[0][0] = 0.0;

    for i in 1..=n
    {
        // Columns crossed by the scaled diagonal within row i, widened by s.
        // A band centred on a single rounded column per row misses column 1
        // in the first row (and leaves gaps between rows) once m/n exceeds
        // about s + 1, which left dp[n][m] at INFINITY when seq2 is the
        // longer sequence.
        let (j_start, j_end) = match step_size
        {
            Some(s) =>
            {
                let start = ((i - 1) * m / n + 1).saturating_sub(s).max(1);
                let end = (i * m).div_ceil(n).saturating_add(s).min(m);
                (start, end)
            },
            None => (1, m),
        };

        for j in j_start..=j_end
        {
            let cost = (seq1[i - 1] - seq2[j - 1]).powi(2);
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }

    dp[n][m].sqrt()
}

/// DTW that also returns the optimal warping path.
///
/// The distance is the same as [`dynamic_time_warping`] with `step_size =
/// None`. The path lists 0-based index pairs from `(0, 0)` to
/// `(n − 1, m − 1)`. If both sequences are empty the result is `(0.0, [])`;
/// if exactly one is empty no warping path exists and the result is
/// `(+∞, [])`, matching [`dynamic_time_warping`]. If the minimal cost is
/// not finite (squared differences overflow to `+∞`, or an input is NaN),
/// the distance is `+∞` or NaN as in [`dynamic_time_warping`] and the path is
/// still a valid monotone path from `(0, 0)` to `(n − 1, m − 1)`, but it is
/// not meaningful as an optimal alignment.
pub fn dynamic_time_warping_with_path(seq1: &[f64], seq2: &[f64]) -> (f64, Vec<(usize, usize)>) {
    let n = seq1.len();
    let m = seq2.len();
    let inf = f64::INFINITY;

    // With one empty sequence the traceback below would start on row or
    // column 0 and compute `0 - 1` on a `usize`.
    if n == 0 || m == 0
    {
        let dist = if n == m { 0.0 } else { inf };
        return (dist, Vec::new());
    }

    let mut dp = vec![vec![inf; m + 1]; n + 1];
    let mut traceback = vec![vec![0u8; m + 1]; n + 1];
    dp[0][0] = 0.0;

    for i in 1..=n
    {
        for j in 1..=m
        {
            let cost = (seq1[i - 1] - seq2[j - 1]).powi(2);
            let candidates = [dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]];
            let mut best = candidates[0];
            let mut best_idx = 0u8;
            for (k, &c) in candidates.iter().enumerate()
            {
                if c < best
                {
                    best = c;
                    best_idx = k as u8;
                }
            }
            dp[i][j] = cost + best;
            traceback[i][j] = best_idx;
        }
    }

    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0
    {
        path.push((i - 1, j - 1));
        // Row 1 and column 1 have a single valid predecessor. When every
        // candidate cost is +inf (squared differences that overflow, e.g.
        // 1e200 vs -1e200) or NaN, the stored choice defaults to "up", which
        // on row 1 would step to row 0 with j > 0 and underflow `i - 1` on
        // the next iteration.
        let step = if i == 1 && j > 1
        {
            1
        }
        else if j == 1 && i > 1
        {
            0
        }
        else
        {
            traceback[i][j]
        };
        match step
        {
            0 => i -= 1,
            1 => j -= 1,
            _ =>
            {
                i -= 1;
                j -= 1;
            },
        }
    }
    path.reverse();
    (dp[n][m].sqrt(), path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmp_basic() {
        let text = vec![0, 1, 2, 0, 1, 2, 3];
        let pattern = vec![0, 1, 2];
        assert_eq!(kmp(&text, &pattern), vec![0, 3]);
    }

    #[test]
    fn kmp_no_match() {
        let text = vec![0, 1, 2, 3];
        assert!(kmp(&text, &[5, 6]).is_empty());
    }

    #[test]
    fn kmp_empty_pattern() {
        let text = vec![1, 2, 3];
        assert_eq!(kmp(&text, &[]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn kmp_pattern_equals_text() {
        let text = vec![1, 2, 3];
        assert_eq!(kmp(&text, &text.clone()), vec![0]);
    }

    #[test]
    fn bm_basic() {
        let text = vec![0, 1, 2, 0, 1, 2, 3];
        let pattern = vec![0, 1, 2];
        assert_eq!(boyer_moore(&text, &pattern), vec![0, 3]);
    }

    #[test]
    fn bm_no_match() {
        let text = vec![0, 1, 2, 3];
        assert!(boyer_moore(&text, &[5, 6]).is_empty());
    }

    #[test]
    fn bm_repeated_pattern() {
        let text = vec![1, 1, 1, 1, 1];
        let pattern = vec![1, 1];
        assert_eq!(boyer_moore(&text, &pattern), vec![0, 1, 2, 3]);
    }

    #[test]
    fn lcs_identical() {
        let seq = vec![1, 2, 3, 4];
        assert_eq!(longest_common_subsequence_len(&seq, &seq), 4);
    }

    #[test]
    fn lcs_empty() {
        assert_eq!(longest_common_subsequence_len(&[], &[]), 0);
        assert_eq!(longest_common_subsequence_len(&[1, 2], &[]), 0);
    }

    #[test]
    fn lcs_classic() {
        let s1 = vec![1, 2, 3, 4, 5];
        let s2 = vec![2, 4, 5, 1, 3];
        assert_eq!(longest_common_subsequence_len(&s1, &s2), 3);
    }

    #[test]
    fn lcs_subsequence_values() {
        let s1 = vec![1, 0, 0, 1, 0, 1, 0, 1];
        let s2 = vec![0, 1, 0, 1, 0, 0, 1, 0];
        let lcs = longest_common_subsequence(&s1, &s2);
        assert_eq!(lcs.len(), longest_common_subsequence_len(&s1, &s2));
    }

    /// Returns true if `sub` appears in order (not necessarily contiguous) within `seq`.
    fn is_subsequence(sub: &[usize], seq: &[usize]) -> bool {
        let mut it = seq.iter();
        sub.iter().all(|x| it.any(|y| y == x))
    }

    #[test]
    fn lcs_returns_actual_subsequence_values() {
        let s1 = vec![1, 2, 3, 4, 5];
        let s2 = vec![2, 4, 5, 1, 3];
        let lcs = longest_common_subsequence(&s1, &s2);
        assert_eq!(lcs.len(), 3);
        assert!(
            is_subsequence(&lcs, &s1),
            "{:?} not a subsequence of {:?}",
            lcs,
            s1
        );
        assert!(
            is_subsequence(&lcs, &s2),
            "{:?} not a subsequence of {:?}",
            lcs,
            s2
        );
    }

    #[test]
    fn dtw_identical() {
        let seq = vec![1.0, 2.0, 3.0, 4.0];
        let dist = dynamic_time_warping(&seq, &seq, None);
        assert!((dist - 0.0).abs() < 1e-10);
    }

    #[test]
    fn dtw_shifted() {
        let s1 = vec![0.0, 1.0, 2.0, 3.0];
        let s2 = vec![1.0, 2.0, 3.0, 4.0];
        let dist = dynamic_time_warping(&s1, &s2, None);
        assert!(dist > 0.0);
        assert!(dist < 5.0);
    }

    #[test]
    fn dtw_with_path_consistency() {
        let s1 = vec![1.0, 2.0, 3.0];
        let s2 = vec![1.0, 3.0, 5.0];
        let (dist, path) = dynamic_time_warping_with_path(&s1, &s2);
        assert!(dist > 0.0);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(2, 2)));
        assert!(path.len() >= 3);
    }

    #[test]
    fn dtw_with_path_exact_distance() {
        let s1 = vec![1.0, 2.0, 3.0];
        let s2 = vec![1.0, 3.0, 5.0];
        let (dist, path) = dynamic_time_warping_with_path(&s1, &s2);
        let expected = 5.0f64.sqrt();
        assert!(
            (dist - expected).abs() < 1e-10,
            "dist={} expected={}",
            dist,
            expected
        );
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(2, 2)));
    }

    #[test]
    fn dtw_banded_unequal_lengths_reaches_corner() {
        // Optimal unconstrained DTW is 0 (every element of seq1 matches a 0 or the
        // trailing 1 in seq2). A length-adaptive band must still reach dp[n][m].
        let s1 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let s2 = vec![0.0, 1.0];
        let banded = dynamic_time_warping(&s1, &s2, Some(1));
        assert!(
            banded.is_finite(),
            "banded DTW returned non-finite: {}",
            banded
        );
        assert!((banded - 0.0).abs() < 1e-10, "banded={}", banded);
    }

    #[test]
    fn dtw_banded_reaches_corner_when_second_sequence_is_longer() {
        // Mirror of `dtw_banded_unequal_lengths_reaches_corner` with the longer
        // sequence second. The band used to start row 1 at column 2 for
        // s <= 1, so every cell stayed at INFINITY. Reference values: the same
        // banded recursion written out in Python (row 1 keeps columns
        // 1..=3 + s, row 2 keeps 4 - s..=6).
        let s1 = vec![0.0, 1.0];
        let s2 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let expected = [2.0_f64.sqrt(), 1.0, 0.0];
        for (s, &want) in expected.iter().enumerate()
        {
            let banded = dynamic_time_warping(&s1, &s2, Some(s));
            assert!((banded - want).abs() < 1e-12, "s={s}: banded={banded}");
        }
    }

    #[test]
    fn dtw_band_always_contains_a_path_and_never_beats_full() {
        // For every shape up to 12x12 and every half-width, the banded
        // distance is finite and no smaller than the unconstrained one (the
        // band only removes paths), and a band at least as wide as the grid
        // gives the unconstrained distance.
        for n in 1..=12usize
        {
            for m in 1..=12usize
            {
                let a: Vec<f64> = (0..n).map(|k| (0.9 * k as f64).sin()).collect();
                let b: Vec<f64> = (0..m).map(|k| (0.5 * k as f64).cos()).collect();
                let full = dynamic_time_warping(&a, &b, None);
                for s in 0..4
                {
                    let banded = dynamic_time_warping(&a, &b, Some(s));
                    assert!(banded.is_finite(), "n={n} m={m} s={s}: {banded}");
                    assert!(banded >= full - 1e-12, "n={n} m={m} s={s}");
                }
                let wide = dynamic_time_warping(&a, &b, Some(n.max(m)));
                assert!((wide - full).abs() <= 1e-12 * full.max(1.0), "n={n} m={m}");
            }
        }
    }

    #[test]
    fn dtw_banded_long_second_sequence_matches_reference() {
        // 10 vs 100 samples of the same sine at matching phase steps.
        // Reference values from the same recursion in Python (full grid and
        // the documented band). Before the fix s = 0, 1 and 5 gave INFINITY.
        let a: Vec<f64> = (0..10).map(|k| (0.7 * k as f64).sin()).collect();
        let b: Vec<f64> = (0..100).map(|k| (0.07 * k as f64).sin()).collect();
        let full = dynamic_time_warping(&a, &b, None);
        assert!((full - 1.720_978_946_203_408_4).abs() < 1e-9, "full={full}");
        let cases = [
            (0, 2.726_278_105_070_878),
            (1, 2.392_055_296_736_038_4),
            (5, 1.729_101_440_046_996_4),
            (10, 1.720_978_946_203_408_4),
        ];
        for (s, want) in cases
        {
            let banded = dynamic_time_warping(&a, &b, Some(s));
            assert!((banded - want).abs() < 1e-9, "s={s}: banded={banded}");
        }
    }

    #[test]
    fn dtw_banded_matches_full() {
        let s1: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let s2: Vec<f64> = (0..10).map(|x| (x as f64) + 0.5).collect();
        let full = dynamic_time_warping(&s1, &s2, None);
        let banded = dynamic_time_warping(&s1, &s2, Some(10));
        assert!((full - banded).abs() < 1e-10);
    }

    #[test]
    fn dtw_with_path_one_empty_input_returns_infinity_without_panicking() {
        // The traceback used to compute `0 - 1` on a usize when exactly one
        // input was empty (overflow panic in debug builds). No warping path
        // exists in that case, as in `dynamic_time_warping`.
        let s = vec![1.0, 2.0, 3.0];
        let empty: Vec<f64> = Vec::new();
        for (a, b) in [(&s, &empty), (&empty, &s)]
        {
            let (dist, path) = dynamic_time_warping_with_path(a, b);
            assert!(dist.is_infinite() && dist > 0.0, "dist={dist}");
            assert!(path.is_empty(), "path={path:?}");
            assert_eq!(dist, dynamic_time_warping(a, b, None));
        }
        let (dist, path) = dynamic_time_warping_with_path(&empty, &empty);
        assert_eq!(dist, 0.0);
        assert!(path.is_empty());
    }

    #[test]
    fn dtw_with_path_agrees_with_distance_and_path_cost() {
        // For non-empty inputs the path variant gives the unconstrained
        // distance, and the returned path is a monotone warping path from
        // (0, 0) to (n - 1, m - 1) whose squared cost equals that distance.
        for n in 1..=7usize
        {
            for m in 1..=7usize
            {
                let a: Vec<f64> = (0..n).map(|k| (0.8 * k as f64).sin()).collect();
                let b: Vec<f64> = (0..m).map(|k| (0.6 * k as f64).cos()).collect();
                let (dist, path) = dynamic_time_warping_with_path(&a, &b);
                let full = dynamic_time_warping(&a, &b, None);
                assert!((dist - full).abs() <= 1e-12 * full.max(1.0), "n={n} m={m}");
                assert_eq!(path.first(), Some(&(0, 0)));
                assert_eq!(path.last(), Some(&(n - 1, m - 1)));
                for w in path.windows(2)
                {
                    let (di, dj) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
                    assert!(matches!((di, dj), (1, 0) | (0, 1) | (1, 1)), "n={n} m={m}");
                }
                let cost: f64 = path.iter().map(|&(i, j)| (a[i] - b[j]).powi(2)).sum();
                assert!(
                    (cost.sqrt() - dist).abs() <= 1e-12 * dist.max(1.0),
                    "n={n} m={m}"
                );
            }
        }
    }
    #[test]
    fn dtw_with_path_non_finite_costs_do_not_panic() {
        // (1e200 - (-1e200))^2 overflows to +inf, so every cell on row 1 is
        // +inf and the traceback used to step from row 1 to row 0 with j > 0.
        let cases: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1e200], vec![-1e200, 0.0]),
            (vec![-1e200, 0.0], vec![1e200]),
            (vec![1e200, 1.0], vec![-1e200, -1e200, 3.0]),
            (vec![f64::NAN], vec![1.0, 2.0, 3.0]),
            (vec![1.0, 2.0, 3.0], vec![f64::NAN]),
        ];
        for (a, b) in cases
        {
            let (dist, path) = dynamic_time_warping_with_path(&a, &b);
            let plain = dynamic_time_warping(&a, &b, None);
            assert!(!dist.is_finite(), "dist = {dist}");
            assert_eq!(dist.is_nan(), plain.is_nan());
            assert_eq!(path.first(), Some(&(0, 0)));
            assert_eq!(path.last(), Some(&(a.len() - 1, b.len() - 1)));
            for w in path.windows(2)
            {
                let (di, dj) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
                assert!(matches!((di, dj), (1, 0) | (0, 1) | (1, 1)), "step {w:?}");
            }
        }
    }
}
