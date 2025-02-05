"""Defines the preprocessing functions for transforming features using TensorFlow Transform."""

import tensorflow_transform as tft

NUMERIC_FEATURES = [
    'age',
    'daily_steps',
    'heart_rate',
    'physical_activity_level',
    'quality_of_sleep',
    'sleep_duration',
    'stress_level'
]

CATEGORICAL_FEATURES = [
    'bmi_category',
    'blood_pressure',
    'gender',
    'occupation'
]

LABEL_KEY = 'sleep_disorder'


def transformed_name(feature_name):
    """Append '_xf' to feature name to denote transformed feature."""
    return f'{feature_name}_xf'


def scale_numeric_features(inputs):
    """Scale numeric features to z-score."""
    return {
        transformed_name(feature): tft.scale_to_z_score(inputs[feature])
        for feature in NUMERIC_FEATURES
    }


def encode_categorical_features(inputs):
    """Convert categorical features to indices using vocabulary."""
    return {
        transformed_name(feature): tft.compute_and_apply_vocabulary(
            inputs[feature], vocab_filename=feature)
        for feature in CATEGORICAL_FEATURES
    }


def encode_label(inputs):
    """Convert label to int64 using vocabulary."""
    return {
        transformed_name(LABEL_KEY): tft.compute_and_apply_vocabulary(
            inputs[LABEL_KEY], vocab_filename=LABEL_KEY)
    }


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features."""
    outputs = {}

    # Process numeric, categorical features and label
    outputs.update(scale_numeric_features(inputs))
    outputs.update(encode_categorical_features(inputs))
    outputs.update(encode_label(inputs))

    return outputs
