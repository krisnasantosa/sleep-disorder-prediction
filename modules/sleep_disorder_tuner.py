""" Module for hyperparameter tuning with Keras Tuner. """

from typing import NamedTuple, Dict, Text, Any
import kerastuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers  # pylint: disable=import-error
from kerastuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs

# Constants for feature categories and label key
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
    Generate transformed feature name by appending '_xf' to the original key.

    Args:
        key (str): The original feature name.

    Returns:
        str: The transformed feature name.
    """
    return f"{key}_xf"


# Named tuple for Tuner function result
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])


def _gzip_reader_fn(filenames: str) -> tf.data.TFRecordDataset:
    """
    Create a GZIP reader for dataset files.

    Args:
        filenames (str): Path to the dataset file.

    Returns:
        tf.data.TFRecordDataset: A GZIP reader for the given dataset file.
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern: str,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 128) -> tf.data.Dataset:
    """
    Input function to load and preprocess the dataset using the transformed features.

    Args:
        file_pattern (str): File pattern for the dataset files.
        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.
        batch_size (int): The batch size for dataset loading.

    Returns:
        tf.data.Dataset: The dataset with transformed features and labels.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=1,
        shuffle=True,
        label_key=transformed_name(LABEL_KEY)
    )

    return dataset


def create_numeric_input_layer(feature_name: str) -> layers.Layer:
    """
    Helper function to create the input layer for numeric features.

    Args:
        feature_name (str): The name of the numeric feature.

    Returns:
        layers.Layer: The input layer for the numeric feature.
    """
    input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name))
    norm_layer = layers.BatchNormalization()(input_layer)
    return norm_layer


def create_categorical_input_layer(feature_name: str,
                                   vocab_size: int,
                                   embedding_dim: int) -> layers.Layer:
    """
    Helper function to create the input layer for categorical features with embedding.

    Args:
        feature_name (str): The name of the categorical feature.
        vocab_size (int): The vocabulary size for the feature.
        embedding_dim (int): The embedding dimension.

    Returns:
        layers.Layer: The input layer for the categorical feature.
    """
    input_layer = layers.Input(
        shape=(1,), name=transformed_name(feature_name), dtype=tf.int64)
    safe_input = tf.where(
        tf.logical_or(input_layer < 0, input_layer >= vocab_size),
        tf.zeros_like(input_layer),
        input_layer
    )
    embedding = layers.Embedding(
        vocab_size, embedding_dim, mask_zero=True, name=f'embedding_{feature_name}')(safe_input)
    return layers.Flatten()(embedding)


def model_builder(hp, tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
    """
    Build a Keras model for sleep disorder prediction with hyperparameter tuning support.

    Args:
        hp: The hyperparameter object used for tuning.
        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    inputs = []
    encoded_features = []

    # Process numeric features
    for feature_name in NUMERIC_FEATURES:
        encoded_features.append(create_numeric_input_layer(feature_name))

    # Process categorical features
    embedding_dim = hp.Int('embedding_dim', 8, 32, step=8)
    for feature_name in CATEGORICAL_FEATURES:
        vocab_size = tf_transform_output.vocabulary_size_by_name(feature_name)
        encoded_features.append(create_categorical_input_layer(
            feature_name, vocab_size, embedding_dim))

    # Concatenate encoded features
    concat_features = layers.concatenate(encoded_features)

    # Add hidden layers with hyperparameter tuning
    num_hidden_layers = hp.Int('num_hidden_layers', 2, 4)
    for i in range(num_hidden_layers):
        units = hp.Int(f'units_{i}', 32, 256, step=32)
        dropout_rate = hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)
        concat_features = layers.Dense(
            units, activation='relu')(concat_features)
        concat_features = layers.BatchNormalization()(concat_features)
        concat_features = layers.Dropout(dropout_rate)(concat_features)

    # Output layer
    outputs = layers.Dense(3, activation='softmax')(concat_features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model with learning rate tuning
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """
    Create a hyperparameter tuning function for the sleep disorder prediction model.

    Args:
        fn_args (FnArgs): Arguments passed to the tuning function.

    Returns:
        TunerFnResult: Named tuple containing the tuner and fitting arguments.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Hyperparameter tuner setup
    tuner = kt.Hyperband(
        hypermodel=lambda hp: model_builder(hp, tf_transform_output),
        objective='val_accuracy',
        max_epochs=30,
        factor=3,
        directory=fn_args.working_dir,
        project_name='sleep_disorder_tuning',
        overwrite=True
    )

    # Prepare train and eval datasets
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'callbacks': [early_stopping]
        }
    )
