""" Module to train a Keras model with TensorFlow Transform preprocessing. """

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers  # pylint: disable=import-error
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERIC_FEATURES = [
    'age', 'daily_steps', 'heart_rate', 'physical_activity_level',
    'quality_of_sleep', 'sleep_duration', 'stress_level'
]

CATEGORICAL_FEATURES = [
    'bmi_category', 'blood_pressure', 'gender', 'occupation'
]

LABEL_KEY = "sleep_disorder"


def transformed_name(key: str) -> str:
    """
    Transforms the feature name by appending '_xf' to the original key.

    Args:
        key (str): The original feature name.

    Returns:
        str: The transformed feature name.
    """
    return f"{key}_xf"


def gzip_reader_fn(filenames: str) -> tf.data.TFRecordDataset:
    """
    Reads a GZIP file into a TensorFlow TFRecord dataset.

    Args:
        filenames (str): The file path(s) to read from.

    Returns:
        tf.data.TFRecordDataset: The dataset containing the records.
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern: str,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int = 128) -> tf.data.Dataset:
    """
    Creates an input function that loads and preprocesses the dataset.

    Args:
        file_pattern (str): The file path(s) to load.
        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.
        batch_size (int): The batch size for data loading.

    Returns:
        tf.data.Dataset: The dataset ready for training or evaluation.
    """
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=None,
        shuffle=True,
        shuffle_buffer_size=1000,
        label_key=transformed_name(LABEL_KEY)
    )

    return dataset


def _get_serve_tf_examples_fn(model: tf.keras.Model,
                              tf_transform_output: tft.TFTransformOutput):
    """
    Returns a serving function for the TensorFlow model.

    Args:
        model (tf.keras.Model): The trained TensorFlow model.
        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.

    Returns:
        function: The serving function to handle incoming requests.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples: tf.Tensor):
        """
        Process and transform the input examples for serving.

        Args:
            serialized_tf_examples (tf.Tensor): Serialized TF examples to process.

        Returns:
            tf.Tensor: Model predictions.
        """
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """
    Main function to train the model with the provided arguments.

    Args:
        fn_args (FnArgs): The arguments passed to the trainer function.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Get dataset and compute steps per epoch
    batch_size = 128
    train_dataset = input_fn(
        fn_args.train_files, tf_transform_output, batch_size)
    eval_dataset = input_fn(
        fn_args.eval_files, tf_transform_output, batch_size)

    # Load best hyperparameters
    hp = fn_args.hyperparameters['values']

    # Build model with best hyperparameters
    model = build_model(hp, tf_transform_output)

    steps_per_epoch = 26
    validation_steps = 7

    # Train model
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )]
    )

    # Save model with serving signature
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)


def build_model(hp: dict, tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
    """
    Build and compile a Keras model based on hyperparameters.

    Args:
        hp (dict): The dictionary of hyperparameters.
        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    # Create feature processing layers
    numeric_layers = _build_numeric_features(
        NUMERIC_FEATURES)
    categorical_layers = _build_categorical_features(
        CATEGORICAL_FEATURES, hp, tf_transform_output)

    # Concatenate all features
    concat_features = layers.concatenate(numeric_layers + categorical_layers)

    # Add hidden layers dynamically from tuner
    hidden_layers = _build_hidden_layers(concat_features, hp)

    # Output layer
    outputs = layers.Dense(7, activation='softmax')(hidden_layers)

    model = tf.keras.Model(inputs=numeric_layers +
                           categorical_layers, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def _build_numeric_features(numeric_feature_names):
    """
    Builds and processes numeric features for the model.

    Args:
        numeric_feature_names (list): The list of numeric feature names.

    Returns:
        list: A list of layers processing the numeric features.
    """
    numeric_layers = []
    for feature_name in numeric_feature_names:
        input_layer = layers.Input(
            shape=(1,), name=transformed_name(feature_name))
        norm_layer = layers.BatchNormalization()(input_layer)
        numeric_layers.append(norm_layer)
    return numeric_layers


def _build_categorical_features(categorical_feature_names, hp, tf_transform_output):
    """
    Builds and processes categorical features with embeddings for the model.

    Args:
        categorical_feature_names (list): The list of categorical feature names.
        hp (dict): Hyperparameters for the model.
        tf_transform_output: The transform output containing preprocessing details.

    Returns:
        list: A list of layers processing the categorical features.
    """
    categorical_layers = []
    for feature_name in categorical_feature_names:
        vocab_size = tf_transform_output.vocabulary_size_by_name(feature_name)
        input_layer = layers.Input(
            shape=(1,), name=transformed_name(feature_name), dtype=tf.int64)

        safe_input = tf.where(
            tf.logical_or(input_layer < 0, input_layer >= vocab_size),
            tf.zeros_like(input_layer),
            input_layer
        )

        embedding = layers.Embedding(
            vocab_size,
            hp.get('embedding_dim'),
            mask_zero=True
        )(safe_input)

        embedding_flat = layers.Flatten()(embedding)
        categorical_layers.append(embedding_flat)

    return categorical_layers


def _build_hidden_layers(concat_features, hp):
    """
    Adds hidden layers to the model based on the hyperparameters.

    Args:
        concat_features (tensor): The concatenated feature layer.
        hp (dict): Hyperparameters for the model.

    Returns:
        tensor: The layer after applying all hidden layers and dropout.
    """
    hidden_layer = concat_features
    for i in range(hp.get('num_hidden_layers')):
        units = hp.get(f'units_{i}')
        dropout_rate = hp.get(f'dropout_{i}')
        hidden_layer = layers.Dense(units, activation='relu')(hidden_layer)
        hidden_layer = layers.BatchNormalization()(hidden_layer)
        hidden_layer = layers.Dropout(dropout_rate)(hidden_layer)

    return hidden_layer
