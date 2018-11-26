import numpy as np

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers


class OrdinalOutput(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.t0 = K.constant(-np.inf, shape=(1,1))
        self.tK = K.constant(np.inf, shape=(1,1))
        super(OrdinalOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(input_shape[1], self.output_dim - 1),
            initializer=self.sorted_initializer('glorot_uniform'),
            trainable=True)
        self.thresholds = K.concatenate(
            [self.t0, self.thresholds, self.tK],
            axis=-1)
        super(OrdinalOutput, self).build(input_shape)

    def call(self, x):
        output = (
            K.sigmoid(self.thresholds[:, 1:] - x) - 
            K.sigmoid(self.thresholds[:, :-1] - x))
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def sorted_initializer(self, initializer):
        # Returns a function that returns a sorted
        # initialization based on an initializer string
        def sorter(shape, dtype=None):
            # Returns a sorted initialization
            init_func = initializers.get(initializer)
            init = K.eval(init_func(shape, None))
            init = np.sort(init)
            init = K.cast_to_floatx(init)
            return init
        return sorter