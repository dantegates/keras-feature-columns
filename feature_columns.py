import collections
import functools
import itertools

import keras
import numpy as np


def _cached_property(fn):
    fn._cached_property = True
    return property(functools.lru_cache(1)(fn))


class FeatureSet:
    def __init__(self, *features):
        self.features = features

    def fit(self, X):
        self.features = [f.fit(X[f.name]) for f in self.features]
        return self

    def transform(self, X):
        return [f.transform(X[f.name]) for f in self.features]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @_cached_property
    def inputs(self):
        return [f.input for f in self.features]

    @_cached_property
    def output(self):
        concat = keras.layers.Concatenate(axis=-1)
        return concat([f.output for f in self.features])
    
    @classmethod
    def combine(cls, *feature_sets):
        features = itertools.chain.from_iterable(fs.features for fs in feature_sets)
        return cls(*list(features))

    def __getitem__(self, name):
        # inefficient - but self.features is probably small anyway. lazy first pass
        feature = None
        for f in self.features:
            if f.name == name:
                feature = f
        return feature


class Feature:
    _class_instances = collections.defaultdict(int)

    def __init__(self, name, input_dim=1):
        self.name = name
        self.input = keras.layers.Input((input_dim,), name=self._uid)

    @_cached_property
    def output(self):
        return self.input
    
    def fit(self, X):
        return self

    def transform(self, X):
        return getattr(X, 'values', X)  # pd.DataFrame friendly

    @_cached_property
    def _uid(self):
        class_name = self.__class__.__name__
        self._class_instances[class_name] += 1
        return f'{self.name}_{self._class_instances[class_name]}'


class NumericFeature(Feature):
    def __init__(self, *args, normalizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = normalizer

    def transform(self, X):
        if self.normalizer is not None:
            return self.normalizer(X)
        return super().transform(X)


class Categories:
    def __init__(self, categories):
        self.indices = {v: i for i, v in enumerate(categories)}
        self.map = np.vectorize(lambda x: self.indices.get(x, len(self.indices)))

    def __len__(self):
        return len(self.indices)
    

class CategoricalFeature(Feature):
    def __init__(self, *args, X=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.categories = None
        if X is not None:
            self.fit(X)

    def fit(self, X):
        X = getattr(X, self.name, X)  # pandas.DataFrame friendly
        if self.categories is None:
            self.categories = Categories(set(X))
        return self

    def transform(self, X):
        return self.categories.map(X)


class OneHotFeature(CategoricalFeature, Feature):
    def __init__(self, *args, input_dim=1, X=None, **kwargs):
        input_dim = len(set(X)) + 1 if X is not None else input_dim
        super().__init__(*args, X=X, input_dim=input_dim, **kwargs)

    def fit(self, X):
        super().fit(X)
        return self

    def transform(self, X):
        one_hots = np.zeros((len(X), len(self.categories)+1))
        one_hots[np.arange(len(X)), self.categories.map(X)] = 1
        return one_hots

    
class EmbeddedFeature(CategoricalFeature, Feature):
    def __init__(self, *args, embedding_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.embedding = keras.layers.Embedding(
            input_dim=len(self.categories)+1,  # +1 for OOV
            output_dim=self.embedding_dim,
            input_length=1)

    @_cached_property
    def output(self):
        return keras.layers.Flatten()(self.embedding(self.input))
