# DeepRank
An implementation of deep ordinal regression in Keras for ranking problems.

## Getting started

### Prerequisites
The latest versions of:
```
tensorflow
keras
numpy
```

## Example

### Generating data
Let's generate some quick and dirty ordinal data using sklearn.

First, we'll generate some basic regression data.

```
x, y = make_regression(
    n_samples=10000,
    n_features=20,
    n_informative=15)
```

We need to "squish" this data into an ordinal form. Let's do 4 ordered categories (0, 1, 2, 3).
```
qt = QuantileTransformer()
y = qt.fit_transform(y.reshape(-1, 1))[:,0]
y = np.floor(y * 4) # Encode to 4 uniformly distributed ranks
y = to_categorical(y)
x_train, x_val, y_train, y_val = train_test_split(x, y)
```
```
y.mean(axis=0)
array([ 0.2501,  0.2501,  0.2501,  0.2497])
```
Looks good.

### Fitting a deep ordinal model
```
from deeprank import OrdinalOutput
```
The activation directly before the OrdinalOutput layer should be 1D. This is our "ordered logit".

For simplicity of notation, if we have `K` ordered, 0-indexed classes, we define `K+1` thresholds `t(k)` such that:
```
P(y<k|Xi) = sigmoid(t(k) - logit)
```
Naturally:
```
t(0) = -inf
t(K) = inf
```

The `OrdinalOutput` layer has been designed for use with `categorical_crossentropy` loss. The layer will learn the appropriate category thresholds for the ordered logit simultaneously with the weights from any prior layers. During initialization, the thresholds are randomly generated based on a chosen initializer, but then *ordered*. This is crucial for the loss function to behave properly. 

Let's spin up a super simple model for our example:

```
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=20))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.add(OrdinalOutput(output_dim=4))
```

Load your favorite callbacks and fit just as you would any other Keras model. That's it!
```
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[model_checkpoint, early_stopping])
```
After training, we can observe the thresholds learnt by the `OrdinalOutput` layer.
```
model.load_weights('best_weights.h5')
model.layers[-1].get_weights()[0]
array([[-11.39258575,  -0.43373951,  10.62213516]], dtype=float32)
```
The thresholds have slowly diverged from zero. This checks out nicely.