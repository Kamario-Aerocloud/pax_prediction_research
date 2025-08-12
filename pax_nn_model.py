import tensorflow as tf


def pax_model():
    def residual_block(x, units, dropout_rate=0.3, l2_reg=1e-4):
        shortcut = x
        # First dense
        x = tf.keras.layers.Dense(units, activation='swish',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        # Second dense (no activation before skip)
        x = tf.keras.layers.Dense(units, activation=None,
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        # Skip connection
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation('swish')(x)
        return x

    inputs = tf.keras.Input(shape=(14,))

    # Normalize inputs to help convergence
    x = tf.keras.layers.BatchNormalization()(inputs)

    # First projection layer
    x = tf.keras.layers.Dense(48, activation='swish',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    # Residual stack
    for _ in range(4):
        x = residual_block(x, 48, dropout_rate=0.35)

    # Bottleneck before output
    x = tf.keras.layers.Dense(24, activation='swish',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
