#!/usr/bin/env python
# coding: utf-8

# # Prediksi Gangguan Tidur (Sleep Disorder)
# 
# Permasalahan yang ingin diselesaikan dalam proyek ini adalah memprediksi kualitas tidur seseorang berdasarkan faktor gaya hidup. Tidur yang berkualitas dan gaya hidup sehat memiliki peran penting dalam menjaga kesejahteraan fisik dan mental. Dengan memahami hubungan antara berbagai faktor ini, kita dapat memberikan rekomendasi yang lebih baik untuk meningkatkan kualitas tidur dan kesehatan secara keseluruhan.

# Dataset yang digunakan dalam proyek ini adalah "[Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data)" yang diambil dari platform Kaggle. Dataset ini berisi 400 record dengan 13 atribut yang mencakup berbagai aspek terkait kebiasaan tidur dan gaya hidup.
# 
# Atribut dalam Dataset:
# 
# - **Demografi:**
#   - Gender (Jenis Kelamin: Male/Female)
#   - Age (Usia dalam tahun)
#   - Occupation (Pekerjaan atau profesi)
# 
# - **Faktor Tidur:**
#   - Sleep Duration (Lama tidur dalam jam per hari)
#   - Quality of Sleep (Kualitas tidur, skala 1-10)
#   
# - **Faktor Gaya Hidup:**
#   - Physical Activity Level (Tingkat aktivitas fisik, menit/hari)
#   - Stress Level (Tingkat stres, skala 1-10)
#   - BMI Category (Kategori BMI: Underweight, Normal, Overweight)
#   
# - **Indikator Kesehatan:**
#   - Blood Pressure (Tekanan darah, sistolik/diastolik)
#   - Heart Rate (Detak jantung, bpm)
#   - Daily Steps (Jumlah langkah harian)
#   
# - **Gangguan Tidur:**
#   - Sleep Disorder (Gangguan tidur: None, Insomnia, Sleep Apnea)
# 
# Detail tentang Kolom Sleep Disorder:
# - **None:** Individu tidak memiliki gangguan tidur.
# - **Insomnia:** Kesulitan untuk tidur atau tetap tidur, yang berdampak pada kualitas tidur.
# - **Sleep Apnea:** Gangguan tidur akibat berhentinya pernapasan selama tidur, yang dapat menyebabkan gangguan kesehatan.
# 

# Langkah pertama yang akan dilakukan adalah dengan import library yang dibutuhkan dan membaca dataset yang akan digunakan.

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner
from tfx.proto import example_gen_pb2, trainer_pb2, tuner_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.components import Evaluator, Pusher
from tfx.proto import pusher_pb2
import os


# ## Set Variables
# 
# Selaanjutnya, kita akan menentukan variabel untuk membuat end-to-end pipeline menggunakan TFX dengan mendefinisikan beberapa konfigurasi seperti nama pipeline, lokasi dataset, lokasi metadata, dan lain-lain.

# In[41]:


PIPELINE_NAME = "krisna_santosa-pipeline"
PIPELINE_ROOT = os.path.join('output', PIPELINE_NAME)
METADATA_PATH = os.path.join('output/metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = 'output/serving_model'
MODULE_DIR = "modules"
DATA_ROOT = "data"


# Lakukan inisialisasi instance `InteractiveContext` untuk mengatur dan menjalankan pipeline TFX secara interaktif yang menerima parameter berupa nama pipeline dan lokasi metadata.

# In[3]:


# Initialize InteractiveContext
interactive_context = InteractiveContext(
    pipeline_root=PIPELINE_ROOT
)


# ## Data Ingestion
# 
# Langkah pertama dalam pipeline adalah melakukan data ingestion. Dalam kasus ini, dataset yang digunakan adalah dataset obesitas yang telah dijelaskan sebelumnya. Dataset ini akan dibaca menggunakan komponen `CsvExampleGen` yang akan menghasilkan output berupa dataset yang telah di-preprocess. Kode di bawah ini akan membagi dataset menjadi dua bagian, yaitu 80% untuk training dan 20% untuk testing.

# In[4]:


output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
        example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
    ])
)

example_gen = CsvExampleGen(
    input_base=DATA_ROOT,
    output_config=output
)


# Untuk melihat komponen `ExampleGen` secara interaktif, kita dapat menjalankan komponen tersebut menggunakan object InteractiveContext() yang telah kita definisikan sebelumnya.

# In[5]:


interactive_context.run(example_gen)


# ## Data Validation
# 
# Setelah data di-preprocess, langkah selanjutnya adalah melakukan data validation, ada tiga komponen yang digunakan dalam data validation, yaitu `StatisticsGen`, `SchemaGen`, dan `ExampleValidator`. Komponen `StatisticsGen` akan menghasilkan statistik deskriptif dari dataset, komponen `SchemaGen` akan menghasilkan skema dari dataset, dan komponen `ExampleValidator` akan memvalidasi data berdasarkan skema yang telah dihasilkan oleh komponen `SchemaGen`.

# ### Summary Statistics
# 
# Komponen ini akan berisi statistik deskriptif dari dataset, seperti jumlah data, rata-rata, standar deviasi, dan lain-lain. Kode di bawah ini akan menampilkan statistik deskriptif dari dataset. Komponen ini menerima input berupa dataset yang telah di-preprocess oleh komponen `ExampleGen`.

# In[6]:


statistics_gen = StatisticsGen(
    examples=example_gen.outputs["examples"]
)

interactive_context.run(statistics_gen)


# In[7]:


interactive_context.show(statistics_gen.outputs["statistics"])


# ### Data Schema
# 
# Komponen ini akan menghasilkan skema dari dataset, seperti tipe data, domain, dan lain-lain. Kode di bawah ini akan menampilkan skema dari dataset. Komponen ini menerima input berupa dataset yang telah di-preprocess oleh komponen `ExampleGen`.

# In[8]:


schema_gen = SchemaGen(
    statistics=statistics_gen.outputs["statistics"]
)

interactive_context.run(schema_gen)


# In[9]:


interactive_context.show(schema_gen.outputs["schema"])


# ### Anomalies Detection (Validator)
# 
# Pada komponen ini, kita akan melakukan validasi data berdasarkan skema yang telah dihasilkan oleh komponen `SchemaGen`. Komponen ini akan mendeteksi anomali pada dataset, seperti data yang hilang, data yang tidak sesuai dengan skema, dan lain-lain.

# In[10]:


example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)

interactive_context.run(example_validator)


# In[11]:


interactive_context.show(example_validator.outputs['anomalies'])


# Berdasarkan hasil tersebut tidak terdapat anomali yang ditemukan dalam dataset. Aritnya data siap masuk ke tahap selanjutnya, yaitu preprocessing.

# ## Data Preprocessing
# 
# Setelah tahap data validation, langkah selanjutnya adalah melakukan data preprocessing. Dalam kasus ini, kita akan melakukan data preprocessing dengan menggunakan komponen `Transform`. Komponen ini akan melakukan preprocessing data, seperti normalisasi, one-hot encoding, dan lain-lain. Untuk melakukan preprocessing data, kita perlu mendefinisikan file module yang berisi fungsi preprocessing data. 

# In[12]:


TRANSFORM_MODULE_FILE = os.path.join(MODULE_DIR, "sleep_disorder_transform.py")


# In[45]:


get_ipython().run_cell_magic('writefile', '{TRANSFORM_MODULE_FILE}', '"""Defines the preprocessing functions for transforming features using TensorFlow Transform."""\n\nimport tensorflow_transform as tft\n\nNUMERIC_FEATURES = [\n    \'age\',\n    \'daily_steps\',\n    \'heart_rate\',\n    \'physical_activity_level\',\n    \'quality_of_sleep\',\n    \'sleep_duration\',\n    \'stress_level\'\n]\n\nCATEGORICAL_FEATURES = [\n    \'bmi_category\',\n    \'blood_pressure\',\n    \'gender\',\n    \'occupation\'\n]\n\nLABEL_KEY = \'sleep_disorder\'\n\n\ndef transformed_name(feature_name):\n    """Append \'_xf\' to feature name to denote transformed feature."""\n    return f\'{feature_name}_xf\'\n\n\ndef scale_numeric_features(inputs):\n    """Scale numeric features to z-score."""\n    return {\n        transformed_name(feature): tft.scale_to_z_score(inputs[feature])\n        for feature in NUMERIC_FEATURES\n    }\n\n\ndef encode_categorical_features(inputs):\n    """Convert categorical features to indices using vocabulary."""\n    return {\n        transformed_name(feature): tft.compute_and_apply_vocabulary(\n            inputs[feature], vocab_filename=feature)\n        for feature in CATEGORICAL_FEATURES\n    }\n\n\ndef encode_label(inputs):\n    """Convert label to int64 using vocabulary."""\n    return {\n        transformed_name(LABEL_KEY): tft.compute_and_apply_vocabulary(\n            inputs[LABEL_KEY], vocab_filename=LABEL_KEY)\n    }\n\n\ndef preprocessing_fn(inputs):\n    """Preprocess input features into transformed features."""\n    outputs = {}\n\n    # Process numeric, categorical features and label\n    outputs.update(scale_numeric_features(inputs))\n    outputs.update(encode_categorical_features(inputs))\n    outputs.update(encode_label(inputs))\n\n    return outputs\n')


# Setelah file module preprocessing data telah dibuat, kita dapat mendefinisikan komponen `Transform` dengan mendefinisikan fungsi preprocessing data yang telah dibuat sebelumnya. Komponen ini menerima input berupa dataset yang telah di-preprocess oleh komponen `ExampleGen` dan output berupa dataset yang telah di-preprocess.

# In[14]:


transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(TRANSFORM_MODULE_FILE)
)

interactive_context.run(transform)


# Sampai tahap ini, kita telah melakukan data ingestion, data validation, dan data preprocessing. Langkah selanjutnya adalah melakukan training model menggunakan komponen `Trainer`.

# ## Model Development
# 
# Pada tahap ini, kita akan melakukan training model menggunakan komponen `Trainer`. Komponen ini akan melakukan training model menggunakan dataset yang telah di-preprocess oleh komponen `Transform`. Tetapi sebelum itu kita akan melakukan tuning hyperparameter menggunakan komponen `Tuner` terlebih dahulu. 

# In[15]:


TUNER_MODULE_FILE = os.path.join(MODULE_DIR, "sleep_disorder_tuner.py")


# ### Tuner
# 
# Komponen ini akan melakukan tuning hyperparameter pada model yang akan digunakan. Kita perlu mendefinisikan file module yang berisi fungsi untuk membuat model, fungsi untuk meng-compile model, dan fungsi untuk melakukan tuning hyperparameter.

# In[46]:


get_ipython().run_cell_magic('writefile', '{TUNER_MODULE_FILE}', '""" Module for hyperparameter tuning with Keras Tuner. """\n\nfrom typing import NamedTuple, Dict, Text, Any\nimport kerastuner as kt\nimport tensorflow as tf\nimport tensorflow_transform as tft\nfrom tensorflow.keras import layers  # pylint: disable=import-error\nfrom kerastuner.engine import base_tuner\nfrom tfx.components.trainer.fn_args_utils import FnArgs\n\n# Constants for feature categories and label key\nNUMERIC_FEATURES = [\n    \'age\', \'daily_steps\', \'heart_rate\', \'physical_activity_level\',\n    \'quality_of_sleep\', \'sleep_duration\', \'stress_level\'\n]\n\nCATEGORICAL_FEATURES = [\n    \'bmi_category\', \'blood_pressure\', \'gender\', \'occupation\'\n]\n\nLABEL_KEY = "sleep_disorder"\n\n\ndef transformed_name(key: str) -> str:\n    """\n    Generate transformed feature name by appending \'_xf\' to the original key.\n\n    Args:\n        key (str): The original feature name.\n\n    Returns:\n        str: The transformed feature name.\n    """\n    return f"{key}_xf"\n\n\n# Named tuple for Tuner function result\nTunerFnResult = NamedTuple(\'TunerFnResult\', [(\'tuner\', base_tuner.BaseTuner),\n                                             (\'fit_kwargs\', Dict[Text, Any])])\n\n\ndef _gzip_reader_fn(filenames: str) -> tf.data.TFRecordDataset:\n    """\n    Create a GZIP reader for dataset files.\n\n    Args:\n        filenames (str): Path to the dataset file.\n\n    Returns:\n        tf.data.TFRecordDataset: A GZIP reader for the given dataset file.\n    """\n    return tf.data.TFRecordDataset(filenames, compression_type=\'GZIP\')\n\n\ndef _input_fn(file_pattern: str,\n              tf_transform_output: tft.TFTransformOutput,\n              batch_size: int = 128) -> tf.data.Dataset:\n    """\n    Input function to load and preprocess the dataset using the transformed features.\n\n    Args:\n        file_pattern (str): File pattern for the dataset files.\n        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.\n        batch_size (int): The batch size for dataset loading.\n\n    Returns:\n        tf.data.Dataset: The dataset with transformed features and labels.\n    """\n    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()\n\n    dataset = tf.data.experimental.make_batched_features_dataset(\n        file_pattern=file_pattern,\n        batch_size=batch_size,\n        features=transformed_feature_spec,\n        reader=_gzip_reader_fn,\n        num_epochs=1,\n        shuffle=True,\n        label_key=transformed_name(LABEL_KEY)\n    )\n\n    return dataset\n\n\ndef create_numeric_input_layer(feature_name: str) -> layers.Layer:\n    """\n    Helper function to create the input layer for numeric features.\n\n    Args:\n        feature_name (str): The name of the numeric feature.\n\n    Returns:\n        layers.Layer: The input layer for the numeric feature.\n    """\n    input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name))\n    norm_layer = layers.BatchNormalization()(input_layer)\n    return norm_layer\n\n\ndef create_categorical_input_layer(feature_name: str,\n                                   vocab_size: int,\n                                   embedding_dim: int) -> layers.Layer:\n    """\n    Helper function to create the input layer for categorical features with embedding.\n\n    Args:\n        feature_name (str): The name of the categorical feature.\n        vocab_size (int): The vocabulary size for the feature.\n        embedding_dim (int): The embedding dimension.\n\n    Returns:\n        layers.Layer: The input layer for the categorical feature.\n    """\n    input_layer = layers.Input(\n        shape=(1,), name=transformed_name(feature_name), dtype=tf.int64)\n    safe_input = tf.where(\n        tf.logical_or(input_layer < 0, input_layer >= vocab_size),\n        tf.zeros_like(input_layer),\n        input_layer\n    )\n    embedding = layers.Embedding(\n        vocab_size, embedding_dim, mask_zero=True, name=f\'embedding_{feature_name}\')(safe_input)\n    return layers.Flatten()(embedding)\n\n\ndef model_builder(hp, tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:\n    """\n    Build a Keras model for sleep disorder prediction with hyperparameter tuning support.\n\n    Args:\n        hp: The hyperparameter object used for tuning.\n        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.\n\n    Returns:\n        tf.keras.Model: The compiled Keras model.\n    """\n    inputs = []\n    encoded_features = []\n\n    # Process numeric features\n    for feature_name in NUMERIC_FEATURES:\n        encoded_features.append(create_numeric_input_layer(feature_name))\n\n    # Process categorical features\n    embedding_dim = hp.Int(\'embedding_dim\', 8, 32, step=8)\n    for feature_name in CATEGORICAL_FEATURES:\n        vocab_size = tf_transform_output.vocabulary_size_by_name(feature_name)\n        encoded_features.append(create_categorical_input_layer(\n            feature_name, vocab_size, embedding_dim))\n\n    # Concatenate encoded features\n    concat_features = layers.concatenate(encoded_features)\n\n    # Add hidden layers with hyperparameter tuning\n    num_hidden_layers = hp.Int(\'num_hidden_layers\', 2, 4)\n    for i in range(num_hidden_layers):\n        units = hp.Int(f\'units_{i}\', 32, 256, step=32)\n        dropout_rate = hp.Float(f\'dropout_{i}\', 0.1, 0.5, step=0.1)\n        concat_features = layers.Dense(\n            units, activation=\'relu\')(concat_features)\n        concat_features = layers.BatchNormalization()(concat_features)\n        concat_features = layers.Dropout(dropout_rate)(concat_features)\n\n    # Output layer\n    outputs = layers.Dense(3, activation=\'softmax\')(concat_features)\n    model = tf.keras.Model(inputs=inputs, outputs=outputs)\n\n    # Compile model with learning rate tuning\n    learning_rate = hp.Float(\'learning_rate\', 1e-4, 1e-2, sampling=\'log\')\n    model.compile(\n        optimizer=tf.keras.optimizers.Adam(learning_rate),\n        loss=\'sparse_categorical_crossentropy\',\n        metrics=[\'accuracy\']\n    )\n\n    return model\n\n\ndef tuner_fn(fn_args: FnArgs) -> TunerFnResult:\n    """\n    Create a hyperparameter tuning function for the sleep disorder prediction model.\n\n    Args:\n        fn_args (FnArgs): Arguments passed to the tuning function.\n\n    Returns:\n        TunerFnResult: Named tuple containing the tuner and fitting arguments.\n    """\n    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n\n    # Hyperparameter tuner setup\n    tuner = kt.Hyperband(\n        hypermodel=lambda hp: model_builder(hp, tf_transform_output),\n        objective=\'val_accuracy\',\n        max_epochs=30,\n        factor=3,\n        directory=fn_args.working_dir,\n        project_name=\'sleep_disorder_tuning\',\n        overwrite=True\n    )\n\n    # Prepare train and eval datasets\n    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)\n    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)\n\n    # Early stopping callback\n    early_stopping = tf.keras.callbacks.EarlyStopping(\n        monitor=\'val_accuracy\',\n        patience=3,\n        restore_best_weights=True\n    )\n\n    return TunerFnResult(\n        tuner=tuner,\n        fit_kwargs={\n            \'x\': train_dataset,\n            \'validation_data\': eval_dataset,\n            \'callbacks\': [early_stopping]\n        }\n    )\n')


# Setelah file module tuning hyperparameter telah dibuat, kita dapat mendefinisikan komponen `Tuner` dengan mendefinisikan fungsi tuning hyperparameter yang telah dibuat sebelumnya. Komponen ini menerima input berupa dataset yang telah di-preprocess oleh komponen `Transform`.

# In[32]:


tuner = Tuner(
    module_file=os.path.abspath(TUNER_MODULE_FILE),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'])
)

interactive_context.run(tuner)


# ### Model Training
# 
# Setelah tuning hyperparameter selesai, kita dapat melakukan training model menggunakan komponen `Trainer`. Komponen ini akan melakukan training model menggunakan dataset yang telah di-preprocess oleh komponen `Transform` dan hyperparameter yang telah di-tuning oleh komponen `Tuner`. Kita akan definisikan file module yang berisi fungsi untuk membuat model, fungsi untuk meng-compile model, dan fungsi untuk melakukan training model.

# In[43]:


TRAINER_MODULE_FILE = os.path.join(MODULE_DIR,"sleep_disorder_trainer.py")


# In[47]:


get_ipython().run_cell_magic('writefile', '{TRAINER_MODULE_FILE}', '""" Module to train a Keras model with TensorFlow Transform preprocessing. """\n\nimport tensorflow as tf\nimport tensorflow_transform as tft\nfrom tensorflow.keras import layers  # pylint: disable=import-error\nfrom tfx.components.trainer.fn_args_utils import FnArgs\n\nNUMERIC_FEATURES = [\n    \'age\', \'daily_steps\', \'heart_rate\', \'physical_activity_level\',\n    \'quality_of_sleep\', \'sleep_duration\', \'stress_level\'\n]\n\nCATEGORICAL_FEATURES = [\n    \'bmi_category\', \'blood_pressure\', \'gender\', \'occupation\'\n]\n\nLABEL_KEY = "sleep_disorder"\n\n\ndef transformed_name(key: str) -> str:\n    """\n    Transforms the feature name by appending \'_xf\' to the original key.\n\n    Args:\n        key (str): The original feature name.\n\n    Returns:\n        str: The transformed feature name.\n    """\n    return f"{key}_xf"\n\n\ndef gzip_reader_fn(filenames: str) -> tf.data.TFRecordDataset:\n    """\n    Reads a GZIP file into a TensorFlow TFRecord dataset.\n\n    Args:\n        filenames (str): The file path(s) to read from.\n\n    Returns:\n        tf.data.TFRecordDataset: The dataset containing the records.\n    """\n    return tf.data.TFRecordDataset(filenames, compression_type=\'GZIP\')\n\n\ndef input_fn(file_pattern: str,\n             tf_transform_output: tft.TFTransformOutput,\n             batch_size: int = 128) -> tf.data.Dataset:\n    """\n    Creates an input function that loads and preprocesses the dataset.\n\n    Args:\n        file_pattern (str): The file path(s) to load.\n        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.\n        batch_size (int): The batch size for data loading.\n\n    Returns:\n        tf.data.Dataset: The dataset ready for training or evaluation.\n    """\n    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()\n\n    dataset = tf.data.experimental.make_batched_features_dataset(\n        file_pattern=file_pattern,\n        batch_size=batch_size,\n        features=transform_feature_spec,\n        reader=gzip_reader_fn,\n        num_epochs=None,\n        shuffle=True,\n        shuffle_buffer_size=1000,\n        label_key=transformed_name(LABEL_KEY)\n    )\n\n    return dataset\n\n\ndef _get_serve_tf_examples_fn(model: tf.keras.Model,\n                              tf_transform_output: tft.TFTransformOutput):\n    """\n    Returns a serving function for the TensorFlow model.\n\n    Args:\n        model (tf.keras.Model): The trained TensorFlow model.\n        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.\n\n    Returns:\n        function: The serving function to handle incoming requests.\n    """\n    model.tft_layer = tf_transform_output.transform_features_layer()\n\n    @tf.function\n    def serve_tf_examples_fn(serialized_tf_examples: tf.Tensor):\n        """\n        Process and transform the input examples for serving.\n\n        Args:\n            serialized_tf_examples (tf.Tensor): Serialized TF examples to process.\n\n        Returns:\n            tf.Tensor: Model predictions.\n        """\n        feature_spec = tf_transform_output.raw_feature_spec()\n        feature_spec.pop(LABEL_KEY)\n\n        parsed_features = tf.io.parse_example(\n            serialized_tf_examples, feature_spec)\n        transformed_features = model.tft_layer(parsed_features)\n\n        return model(transformed_features)\n\n    return serve_tf_examples_fn\n\n\ndef run_fn(fn_args: FnArgs):\n    """\n    Main function to train the model with the provided arguments.\n\n    Args:\n        fn_args (FnArgs): The arguments passed to the trainer function.\n    """\n\n    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n\n    # Get dataset and compute steps per epoch\n    batch_size = 128\n    train_dataset = input_fn(\n        fn_args.train_files, tf_transform_output, batch_size)\n    eval_dataset = input_fn(\n        fn_args.eval_files, tf_transform_output, batch_size)\n\n    # Load best hyperparameters\n    hp = fn_args.hyperparameters[\'values\']\n\n    # Build model with best hyperparameters\n    model = build_model(hp, tf_transform_output)\n\n    steps_per_epoch = 26\n    validation_steps = 7\n\n    # Train model\n    model.fit(\n        train_dataset,\n        validation_data=eval_dataset,\n        epochs=30,\n        steps_per_epoch=steps_per_epoch,\n        validation_steps=validation_steps,\n        callbacks=[tf.keras.callbacks.EarlyStopping(\n            monitor=\'val_accuracy\',\n            patience=5,\n            restore_best_weights=True\n        )]\n    )\n\n    # Save model with serving signature\n    signatures = {\n        \'serving_default\':\n        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(\n            tf.TensorSpec(\n                shape=[None],\n                dtype=tf.string,\n                name=\'examples\'))\n    }\n    model.save(fn_args.serving_model_dir,\n               save_format=\'tf\', signatures=signatures)\n\n\ndef build_model(hp: dict, tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:\n    """\n    Build and compile a Keras model based on hyperparameters.\n\n    Args:\n        hp (dict): The dictionary of hyperparameters.\n        tf_transform_output (tft.TFTransformOutput): TensorFlow Transform output.\n\n    Returns:\n        tf.keras.Model: The compiled Keras model.\n    """\n    # Create feature processing layers\n    numeric_layers = _build_numeric_features(\n        NUMERIC_FEATURES)\n    categorical_layers = _build_categorical_features(\n        CATEGORICAL_FEATURES, hp, tf_transform_output)\n\n    # Concatenate all features\n    concat_features = layers.concatenate(numeric_layers + categorical_layers)\n\n    # Add hidden layers dynamically from tuner\n    hidden_layers = _build_hidden_layers(concat_features, hp)\n\n    # Output layer\n    outputs = layers.Dense(7, activation=\'softmax\')(hidden_layers)\n\n    model = tf.keras.Model(inputs=numeric_layers +\n                           categorical_layers, outputs=outputs)\n\n    model.compile(\n        optimizer=tf.keras.optimizers.Adam(hp.get(\'learning_rate\')),\n        loss=\'sparse_categorical_crossentropy\',\n        metrics=[\'accuracy\']\n    )\n\n    return model\n\n\ndef _build_numeric_features(numeric_feature_names):\n    """\n    Builds and processes numeric features for the model.\n\n    Args:\n        numeric_feature_names (list): The list of numeric feature names.\n\n    Returns:\n        list: A list of layers processing the numeric features.\n    """\n    numeric_layers = []\n    for feature_name in numeric_feature_names:\n        input_layer = layers.Input(\n            shape=(1,), name=transformed_name(feature_name))\n        norm_layer = layers.BatchNormalization()(input_layer)\n        numeric_layers.append(norm_layer)\n    return numeric_layers\n\n\ndef _build_categorical_features(categorical_feature_names, hp, tf_transform_output):\n    """\n    Builds and processes categorical features with embeddings for the model.\n\n    Args:\n        categorical_feature_names (list): The list of categorical feature names.\n        hp (dict): Hyperparameters for the model.\n        tf_transform_output: The transform output containing preprocessing details.\n\n    Returns:\n        list: A list of layers processing the categorical features.\n    """\n    categorical_layers = []\n    for feature_name in categorical_feature_names:\n        vocab_size = tf_transform_output.vocabulary_size_by_name(feature_name)\n        input_layer = layers.Input(\n            shape=(1,), name=transformed_name(feature_name), dtype=tf.int64)\n\n        safe_input = tf.where(\n            tf.logical_or(input_layer < 0, input_layer >= vocab_size),\n            tf.zeros_like(input_layer),\n            input_layer\n        )\n\n        embedding = layers.Embedding(\n            vocab_size,\n            hp.get(\'embedding_dim\'),\n            mask_zero=True\n        )(safe_input)\n\n        embedding_flat = layers.Flatten()(embedding)\n        categorical_layers.append(embedding_flat)\n\n    return categorical_layers\n\n\ndef _build_hidden_layers(concat_features, hp):\n    """\n    Adds hidden layers to the model based on the hyperparameters.\n\n    Args:\n        concat_features (tensor): The concatenated feature layer.\n        hp (dict): Hyperparameters for the model.\n\n    Returns:\n        tensor: The layer after applying all hidden layers and dropout.\n    """\n    hidden_layer = concat_features\n    for i in range(hp.get(\'num_hidden_layers\')):\n        units = hp.get(f\'units_{i}\')\n        dropout_rate = hp.get(f\'dropout_{i}\')\n        hidden_layer = layers.Dense(units, activation=\'relu\')(hidden_layer)\n        hidden_layer = layers.BatchNormalization()(hidden_layer)\n        hidden_layer = layers.Dropout(dropout_rate)(hidden_layer)\n\n    return hidden_layer\n')


# In[35]:


trainer = Trainer(
    module_file=os.path.abspath(TRAINER_MODULE_FILE),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'])
)

interactive_context.run(trainer)


# ## Model Analysis and Validation
# 
# Setelah training model selesai, langkah selanjutnya adalah melakukan analisis model dan validasi model. Dalam kasus ini, kita akan menggunakan komponen `Resolver` dan `Evaluator`. Resolver berperan untuk menentukan baseline model yang akan digunakan untuk membandingkan model yang telah di-training. Sedangkan Evaluator berperan untuk mengevaluasi model yang telah di-training.

# ### Resolver Component
# 
# Pada komponen ini, kita akan menentukan baseline model yang akan digunakan untuk membandingkan model yang telah di-training.

# In[36]:


model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')

interactive_context.run(model_resolver)


# ### Evaluator Component
# 
# Pada komponen ini, kita akan mengevaluasi model yang telah di-training. Komponen ini akan menghasilkan beberapa metric evaluasi model, seperti accuracy, precision, recall, dan lain-lain. Kode di bawah ini akan menampilkan metric evaluasi model dengan threshold 0.85.

# In[37]:


LABEL_KEY = "sleep_disorder"

def transformed_name(key):
    return key + '_xf'

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(
            label_key=transformed_name(LABEL_KEY)
        )],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='SparseCategoricalAccuracy'),
            tfma.MetricConfig(
                class_name='SparseCategoricalAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.85}
                    ),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': 0.0001}
                    )
                )
            )
        ])
    ]
)


# Setelah membuat konfigurasi untuk komponen `Evaluator`, kita dapat mengevaluasi model yang telah di-training dengan menjalankan komponen `Evaluator` pada kode di bawah ini.

# In[38]:


evaluator = Evaluator(
    examples=transform.outputs['transformed_examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config
)

interactive_context.run(evaluator)


# Untuk dapat melihat hasil evaluasi model dengan visualisasi, kita menggunakan  `tfma.view.render_slicing_metrics` yang akan menampilkan metric evaluasi model dengan visualisasi.

# In[39]:


# Visualize evaluation results
eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)


# ## Pusher
# 
# Setelah model dievaluasi, langkah terakhir adalah melakukan push model ke production. Pada kasus ini, kita akan menggunakan komponen `Pusher` untuk melakukan push model ke production. Komponen ini akan melakukan menyimpan model yang telah di-training ke storage yang telah ditentukan.

# In[42]:


pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=SERVING_MODEL_DIR)
    )
)

interactive_context.run(pusher)

