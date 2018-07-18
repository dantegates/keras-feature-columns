import keras
import numpy as np


class FeatureColumn:
    def __init__(self, name):
        self.name = name
        self.input = keras.layers.Input((1,), name=self.name)

    @property
    def output(self):
        return self.input

    def transform(self, X):
        return X.values


class NumericColumn(FeatureColumn):
    def __init__(self, *args, normalizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = normalizer

    def fit(self, X):
        if self.normalizer is not None:
            self.normalizer.fit(X)
        return self

    def transform(self, X):
        if self.normalizer is not None:
            return self.normalizer.transform(X).values
        return X.values
    
    
class EmbeddingColumn(FeatureColumn):
    def __init__(self, *args, vocab_size, output_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.output_dim = output_dim

    def fit(self, X):
        self.vocab_map = {v: i for i, v in enumerate(set(X))}
        @np.vectorize
        def apply_mapping(X):
            return self.vocab_map.get(x, self.vocab_size)
        self._apply_mapping = apply_mapping
        return self

    def transform(self, X):
        return self._apply_mapping(X)

    @property
    def output(self):
        embedding = keras.layers.Embedding(
            input_dim=self.vocab_size+1,  # +1 for OOV
            output_dim=self.output_dim,
            input_length=1)(self.input)
        return keras.layers.Flatten()(embedding)


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

    @property
    def inputs(self):
        return [f.input for f in self.features]

    @property
    def output(self):
        concat = keras.layers.Concatenate(axis=-1)
        return concat([f.output for f in self.features])


class Scaler:
    def __init__(self):
        self.mean = self.std = None

    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std
