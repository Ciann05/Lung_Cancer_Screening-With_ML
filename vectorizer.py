import numpy as np

MISSING_TOKEN = "__MISSING__"


def is_missing(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text in ("", "NA", "NaN", "nan", ".", "None")


def to_float(value):
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.num_bins = num_bins
        self.feature_transforms = {}
        self.feature_order = []
        self.is_fit = False

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to [z-score, missing_indicator].
                 Missing values are mean-imputed before normalization.
        """
        observed = [v for v in values if v is not None]
        if len(observed) == 0:
            mean, std = 0.0, 1.0
        else:
            arr = np.asarray(observed, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if std == 0:
                std = 1.0

        if verbose:
            print(f"Numerical feature mean={mean:.4f}, std={std:.4f}, n_obs={len(observed)}")

        def vectorizer(x):
            parsed = to_float(x)
            missing = 1.0 if parsed is None else 0.0
            value = mean if parsed is None else parsed
            return np.array([(value - mean) / std, missing], dtype=float)

        return vectorizer

    def get_ordinal_vectorizer(self, values, verbose=False):
        """
        Integer-encode ordinal features with mean imputation + standardization
        and a missingness indicator.
        """
        observed = [v for v in values if v is not None]
        if len(observed) == 0:
            mean, std = 0.0, 1.0
        else:
            arr = np.asarray(observed, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if std == 0:
                std = 1.0

        if verbose:
            print(f"Ordinal feature mean={mean:.4f}, std={std:.4f}, n_obs={len(observed)}")

        def vectorizer(x):
            parsed = to_float(x)
            missing = 1.0 if parsed is None else 0.0
            value = mean if parsed is None else parsed
            return np.array([(value - mean) / std, missing], dtype=float)

        return vectorizer

    def get_histogram_vectorizer(self, values, verbose=False):
        """
        Bin a numerical feature into quantile buckets and one-hot encode the bucket.
        The last slot is a missingness indicator.
        """
        observed = [v for v in values if v is not None]
        if len(observed) == 0:
            edges = np.array([], dtype=float)
        else:
            quantiles = np.linspace(0, 1, self.num_bins + 1)[1:-1]
            edges = np.unique(np.quantile(np.asarray(observed, dtype=float), quantiles))

        if verbose:
            print(f"Histogram feature bin edges={edges.tolist()}")

        n_bins = len(edges) + 1

        def vectorizer(x):
            parsed = to_float(x)
            one_hot = np.zeros(n_bins + 1, dtype=float)
            if parsed is None:
                one_hot[-1] = 1.0
            else:
                one_hot[int(np.searchsorted(edges, parsed, side="right"))] = 1.0
            return one_hot

        return vectorizer

    def get_categorical_vectorizer(self, values):
        """
        :return: function to map categorical x to one-hot feature vector.
                 Missing values become their own category.
        """
        categories = sorted({MISSING_TOKEN if is_missing(v) else str(v).strip() for v in values})
        if MISSING_TOKEN not in categories:
            categories.append(MISSING_TOKEN)
        index = {cat: i for i, cat in enumerate(categories)}
        n = len(categories)

        def vectorizer(x):
            key = MISSING_TOKEN if is_missing(x) else str(x).strip()
            if key not in index:
                key = MISSING_TOKEN
            one_hot = np.zeros(n, dtype=float)
            one_hot[index[key]] = 1.0
            return one_hot

        return vectorizer

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.
        """
        self.feature_transforms = {}
        self.feature_order = []

        for feature_name in self.feature_config.get("numerical", []):
            values = [to_float(row[feature_name]) for row in X]
            self.feature_transforms[feature_name] = self.get_numerical_vectorizer(values)
            self.feature_order.append(feature_name)

        for feature_name in self.feature_config.get("ordinal", []):
            values = [to_float(row[feature_name]) for row in X]
            self.feature_transforms[feature_name] = self.get_ordinal_vectorizer(values)
            self.feature_order.append(feature_name)

        for feature_name in self.feature_config.get("histogram", []):
            values = [to_float(row[feature_name]) for row in X]
            self.feature_transforms[feature_name] = self.get_histogram_vectorizer(values)
            self.feature_order.append(feature_name)

        for feature_name in self.feature_config.get("categorical", []):
            values = [row[feature_name] for row in X]
            self.feature_transforms[feature_name] = self.get_categorical_vectorizer(values)
            self.feature_order.append(feature_name)

        self.is_fit = True

    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """
        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set")

        transformed_data = []
        for row in X:
            pieces = [self.feature_transforms[name](row[name]) for name in self.feature_order]
            transformed_data.append(np.concatenate(pieces))

        return np.asarray(transformed_data, dtype=float)
