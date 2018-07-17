import keras


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

    def transform(self, X):
        if self.normalizer is not None:
            return self.normalizer(X).values
        return X.values


class EmbeddingColumn(FeatureColumn):
    def __init__(self, *args, vocabulary, output_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocabulary = vocabulary
        self.vocab_map = {v: i for i, v in enumerate(vocabulary)}
        self.output_dim = output_dim

    @property
    def output(self):
        embedding = keras.layers.Embedding(
            input_dim=len(self.vocabulary)+1,  # +1 for OOV
            output_dim=self.output_dim,
            input_length=1)(self.input)
        return keras.layers.Flatten()(embedding)

    def transform(self, X):
        mapping = lambda x: self.vocab_map.get(x, len(self.vocabulary))
        return X.apply(mapping).values


class Features:
    def __init__(self, *features):
        self.features = features

    @property
    def inputs(self):
        return [f.input for f in self.features]

    @property
    def output(self):
        concat = keras.layers.Concatenate(axis=-1)
        return concat([f.output for f in self.features])

    def transform(self, X):
        return [f.transform(X[f.name]) for f in self.features]


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    def normalize(column):
        mean = X[column].mean()
        std = X[column].std()
        def normalizer(X, mean=mean, std=std):
            return (X - std) / mean
        return normalizer

    X = pd.DataFrame({
        'feature1': np.random.randint(10, size=100),
        'feature2': np.random.randint(100, size=100),
        'feature3': np.random.rand(size=100)
    })
    y = np.random.rand(size=100)


    features = Features(
        EmbeddingColumn('feature1', output_dim=10),
        EmbeddingColumn('feature2', output_dim=10),
        NumericColumn('feature3'),
    )

    x = keras.layers.Dense(50, activation='relu')(features.output)
    x = keras.layers.Dense(50, activation='relu')(x)
    x = keras.layers.Dense(1, activation='sigmoid')
    model = keras.models.Model(inputs=features.inputs, outputs=x)
    model.compile(loss='mean_squarred_error', optimizer='adam')
    model.fit(features.transform(X), y)
