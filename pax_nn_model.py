import tensorflow as tf
from typing import Optional


def se_block(input_tensor: tf.Tensor, channels: Optional[int] = None, reduction: int = 16) -> tf.Tensor:
    """
    Squeeze-and-Excitation block implemented with Keras layers only.
    - input_tensor: shape (batch, channels)
    - channels: number of channels (int). If None, try to infer statically.
    """
    # Try to infer channels from static shape if not provided
    if channels is None:
        inferred = tf.keras.backend.int_shape(input_tensor)[-1]
        if inferred is None:
            raise ValueError(
                "se_block requires a concrete `channels` argument when `input_tensor` has unknown last dimension."
            )
        channels = int(inferred)

    # Expand to (batch, 1, channels) using Keras Reshape layer (no raw tf.expand_dims)
    x_reshaped = tf.keras.layers.Reshape((1, channels))(input_tensor)  # (batch, 1, channels)

    # Global average pooling over the temporal dimension (length=1) -> (batch, channels)
    se = tf.keras.layers.GlobalAveragePooling1D()(x_reshaped)

    # Bottleneck dense layers for excitation
    se = tf.keras.layers.Dense(max(1, channels // reduction), activation='relu')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid')(se)

    # Restore shape (batch, channels)
    se = tf.keras.layers.Reshape((channels,))(se)

    # Scale input by the excitation vector
    scaled = tf.keras.layers.Multiply()([input_tensor, se])
    return scaled


def residual_block(x: tf.Tensor, units: int, dropout_rate: float = 0.3, l2_reg: float = 1e-4) -> tf.Tensor:
    """
    Residual block using only Keras layers and ops.
    - x: input tensor, shape (batch, units) expected
    - units: number of units for dense layers (kept the same as original)
    """
    shortcut = x

    x = tf.keras.layers.Dense(
        units,
        activation='swish',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(
        units,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)

    # Use SE block with explicit channels = units
    x = se_block(x, channels=units, reduction=16)

    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('swish')(x)
    return x


def pax_model() -> tf.keras.Model:

    inputs = tf.keras.Input(shape=(15,), dtype='float32')

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(
        48,
        activation='swish',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    for _ in range(4):
        x = residual_block(x, 48, dropout_rate=0.35)

    x = tf.keras.layers.Dense(
        24,
        activation='swish',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
