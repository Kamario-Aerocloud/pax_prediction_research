import tensorflow as tf


def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling1D()(tf.expand_dims(input_tensor, axis=1))
    se = tf.keras.layers.Dense(channels // reduction, activation='relu')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((channels,))(se)
    return tf.keras.layers.Multiply()([input_tensor, se])


def residual_block(x, units, dropout_rate=0.3, l2_reg=1e-4):
    shortcut = x
    x = tf.keras.layers.Dense(units, activation='swish',
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units, activation=None,
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)

    # Apply SE block here
    x = se_block(x)

    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('swish')(x)
    return x


def pax_model():
    inputs = tf.keras.Input(shape=(14,))

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(48, activation='swish',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    for _ in range(4):
        x = residual_block(x, 48, dropout_rate=0.35)

    x = tf.keras.layers.Dense(24, activation='swish',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
