"""
Lab 3 - Encoding, Distance Metrics, and Descriptive Statistics
----------------------------------------------------------------
A2  label_encode / one_hot_encode
A3  encodedataset            -> apply encoding to the marketing campaign data
A4  minkowsi                 -> Minkowski distance from scratch
A5  distance                 -> Minkowski distance for p = 1..maxp
A6  compare                  -> validate against scipy.spatial.distance.minkowski
A7  dotproduct / norm        -> from scratch, validated against numpy
A8  mean / variance / std    -> from-scratch descriptive stats, per feature
A9  comparestats             -> validate A8 stats against numpy.mean / numpy.std
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\CHANDRA SEKHAR\OneDrive\Documents\MY_SUBJECTS\ML\LAB\Lab Session Data (1).xlsx"
SHEET_NAME = "marketing_campaign"


# ----------------------------------------------------------------------
# A2 - Encoding utilities
# ----------------------------------------------------------------------
def label_encode(input_data: pd.DataFrame, column_name: str):
    """Integer-encode a single column. Returns (new_df, fitted_encoder)."""
    encoder = LabelEncoder()
    out = input_data.copy()
    out[column_name] = encoder.fit_transform(out[column_name].astype(str))
    return out, encoder


def one_hot_encode(input_data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """One-hot encode a single column, dropping the original column."""
    out = input_data.copy()
    dummies = pd.get_dummies(out[column_name], prefix=column_name, drop_first=False)
    return pd.concat([out.drop(columns=column_name), dummies], axis=1)


# ----------------------------------------------------------------------
# A3 - Apply encoding to the real dataset
# ----------------------------------------------------------------------
def encodedataset(data: pd.DataFrame):
    new = data.drop(columns=["Dt_Customer"])
    new, edumap = label_encode(new, "Education")
    new = one_hot_encode(new, "Marital_Status")
    maritalcols = [c for c in new.columns if c.startswith("Marital_Status_")]
    return new, edumap, maritalcols


# ----------------------------------------------------------------------
# A4 - Minkowski distance, built from scratch
# ----------------------------------------------------------------------
def minkowsi(a, b, p):
    """sum(|a_i - b_i|^p) ** (1/p), computed with a single vectorized pass."""
    diffs = (abs(x - y) ** p for x, y in zip(a, b))
    return sum(diffs) ** (1 / p)


# ----------------------------------------------------------------------
# A5 - Sweep p = 1..maxp
# ----------------------------------------------------------------------
def distance(a, b, maxp):
    plist = list(range(1, maxp + 1))
    dlist = [minkowsi(a, b, p) for p in plist]
    return plist, dlist


# ----------------------------------------------------------------------
# A6 - Cross-check against scipy
# ----------------------------------------------------------------------
def compare(a, b, maxp):
    rows = []
    for p in range(1, maxp + 1):
        mine = minkowsi(a, b, p)
        pkg = scipy.spatial.distance.minkowski(a, b, p)
        rows.append([p, mine, pkg, abs(mine - pkg)])
    return rows


# ----------------------------------------------------------------------
# A7 - Dot product / norm, built from scratch
# ----------------------------------------------------------------------
def dotproduct(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return dotproduct(a, a) ** 0.5


# ----------------------------------------------------------------------
# A8 - Descriptive statistics, built from scratch (no numpy/pandas builtins)
# ----------------------------------------------------------------------
def mean(data):
    return sum(data) / len(data)


def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def std(data):
    return variance(data) ** 0.5


def matrixstats(matrix):
    """
    matrix: list of rows (samples), each row a list of feature values.
    Transpose via zip(*matrix) to iterate feature-by-feature instead of
    manually indexing columns.
    """
    columns = list(zip(*matrix))
    means = [mean(col) for col in columns]
    variances = [variance(col) for col in columns]
    stds = [std(col) for col in columns]
    return means, variances, stds


# ----------------------------------------------------------------------
# A9 - Validate A8 against numpy's built-in mean/std
# ----------------------------------------------------------------------
def comparestats(matrix, means, stds):
    """
    Uses numpy for the trusted reference computation (axis=0 -> per-feature,
    collapsing down each column across all samples), then diffs against the
    from-scratch A8 results.
    """
    arr = np.asarray(matrix, dtype=float)
    np_means = arr.mean(axis=0)
    np_stds = arr.std(axis=0)          # population std (ddof=0), matches A8's /N

    mean_diff = np.abs(np.array(means) - np_means)
    std_diff = np.abs(np.array(stds) - np_stds)

    return np_means, np_stds, mean_diff, std_diff


def build_stats_table(columns, means, variances, stds, np_means, np_stds, mean_diff, std_diff):
    """Assemble everything into one tidy DataFrame for easy printing/inspection."""
    return pd.DataFrame({
        "feature": columns,
        "mean_mine": means,
        "mean_numpy": np_means,
        "mean_diff": mean_diff,
        "variance_mine": variances,
        "std_mine": stds,
        "std_numpy": np_stds,
        "std_diff": std_diff,
    })


# ----------------------------------------------------------------------
def main():
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 10)

    # --- Demo on a tiny toy DataFrame -----------------------------------
    sample_df = pd.DataFrame({
        "Product": ["Apple", "Banana", "Apple", "Orange", "Banana"],
        "Color": ["Red", "Yellow", "Red", "Orange", "Yellow"],
        "Price": [100, 50, 100, 75, 50],
    })

    print("Original Data")
    print(sample_df)

    label_df, encoder = label_encode(sample_df, "Product")
    print("\nLabel Encoding")
    print(label_df)

    one_hot_df = one_hot_encode(sample_df, "Color")
    print("\nOne Hot Encoding")
    print(one_hot_df)

    # --- Real dataset -----------------------------------------------------
    data = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    encoded, edumap, maritalcols = encodedataset(data)

    print("\nEncoded Dataset")
    print(encoded.head())

    # --- A4: sanity check on tiny hand-verifiable vectors ------------------
    v1, v2 = [1, 2, 3], [4, 6, 8]
    print("\nA4")
    for p in (1, 2, 3):
        print(f"p={p} : {minkowsi(v1, v2, p)}")

    # All downstream numeric work needs a clean float matrix
    # (one_hot columns come back as bool -> must cast before arithmetic)
    vectors = encoded.fillna(0).astype(float)
    veca = list(vectors.iloc[0])
    vecb = list(vectors.iloc[1])

    # --- A5 ------------------------------------------------------------
    print("\nA5")
    plist, dlist = distance(veca, vecb, 10)
    for p, d in zip(plist, dlist):
        print(f"p = {p} distance = {d}")

    # --- A6 ------------------------------------------------------------
    print("\nA6")
    a6_df = pd.DataFrame(compare(veca, vecb, 10),
                          columns=["p", "mine", "scipy", "abs_diff"])
    print(a6_df.to_string(index=False))

    # --- A7 ------------------------------------------------------------
    print("\nA7")
    print("Dot Product :", dotproduct(veca, vecb))
    print("Numpy Dot   :", np.dot(veca, vecb))
    print("Norm A      :", norm(veca))
    print("Norm B      :", norm(vecb))

    # --- A8 ------------------------------------------------------------
    matrix = vectors.values.tolist()
    means, variances, stds = matrixstats(matrix)

    print("\nA8")
    a8_df = pd.DataFrame({
        "feature": vectors.columns,
        "mean": means,
        "variance": variances,
        "std": stds,
    })
    print(a8_df.to_string(index=False))

    # --- A9 ------------------------------------------------------------
    np_means, np_stds, mean_diff, std_diff = comparestats(matrix, means, stds)
    a9_df = build_stats_table(vectors.columns, means, variances, stds,
                               np_means, np_stds, mean_diff, std_diff)

    print("\nA9")
    print(a9_df[["feature", "mean_mine", "mean_numpy", "mean_diff",
                 "std_mine", "std_numpy", "std_diff"]].to_string(index=False))

    max_mean_diff = a9_df["mean_diff"].max()
    max_std_diff = a9_df["std_diff"].max()
    print(f"\nMax |mean diff| across all features : {max_mean_diff:.2e}")
    print(f"Max |std diff|  across all features : {max_std_diff:.2e}")
    print("(Both should be ~0 -> confirms the from-scratch A8 functions "
          "agree with numpy's built-in mean()/std().)")


if __name__ == "__main__":
    main()