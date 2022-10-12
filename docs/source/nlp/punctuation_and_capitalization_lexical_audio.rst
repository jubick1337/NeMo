.. _punctuation_and_capitalization_lexical_audio:

Punctuation and Capitalization Lexical Audio Model
====================================

<<<<<<< HEAD
Automatic Speech Recognition (ASR) systems typically generate text with no punctuation and capitalization of the words.
There are two issues with non-punctuated ASR output:

- it could be difficult to read and understand
- models for some downstream tasks, such as named entity recognition, machine translation, or text-to-speech, are
  usually trained on punctuated datasets and using raw ASR output as the input to these models could deteriorate their
  performance

But sometimes punctuation and capitalization cannot be restored based only on text. In this case we can use audio to improve model's accuracy.

You can find more details on each section in `Punctuation And Capitalization's page <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html>`_.
=======
Sometimes punctuation and capitalization cannot be restored based only on text. In this case we can use audio to improve model's accuracy.

Like in these examples:

.. code::
  
  Oh yeah? or Oh yeah.

  We need to go? or We need to go.

  Yeah, they make you work. Yeah, over there you walk a lot? or Yeah, they make you work. Yeah, over there you walk a lot.

You can find more details on text only punctuation and capitalization in `Punctuation And Capitalization's page <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html>`_. In this document, we focus on model changes needed to use acoustic features.
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

Quick Start Guide
-----------------

.. code-block:: python

    from nemo.collections.nlp.models import PunctuationCapitalizationLexicalAudioModel

    # to get the list of pre-trained models
    PunctuationCapitalizationLexicalAudioModel.list_available_models()

<<<<<<< HEAD
    # Download and load the pre-trained BERT-based model
    model = PunctuationCapitalizationLexicalAudioModel.from_pretrained("PLACEHOLDER")
=======
    # Download and load the pre-trained model
    model = PunctuationCapitalizationLexicalAudioModel.from_pretrained("<PATH to .nemo file>")
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

    # try the model on a few examples
    model.add_punctuation_capitalization(['how are you', 'great how about you'], audio_queries=['/path/to/1.wav', '/path/to/2.wav'], target_sr=16000)

Model Description
-----------------
<<<<<<< HEAD
In the addition to Punctuation and Capitalization model we add audio encoder (e.g. Conformer's encoder) and attention based fusion of lexical and audio features.
This model architecture is based on `Multimodal Semi-supervised Learning Framework for Punctuation Prediction in Conversational Speech <https://arxiv.org/pdf/2008.00702.pdf>`_.

.. note::

    We recommend you try this model in a Jupyter notebook (run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.): `NeMo/tutorials/nlp/Punctuation_and_Capitalization_Lexical_Audio.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Punctuation_and_Capitalization_Lexical_Audio.ipynb>`__.

    Connect to an instance with a GPU (**Runtime** -> **Change runtime type** -> select **GPU** for the hardware accelerator).

=======
In addition to `Punctuation And Capitalization model <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html>`_ we add audio encoder (e.g. Conformer's encoder) and attention based fusion of lexical and audio features.
This model architecture is based on `Multimodal Semi-supervised Learning Framework for Punctuation Prediction in Conversational Speech <https://arxiv.org/pdf/2008.00702.pdf>`__ :cite:`nlp-punct-sunkara20_interspeech`.

.. note::

>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
    An example script on how to train and evaluate the model can be found at: `NeMo/examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py>`__.

    The default configuration file for the model can be found at: `NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml>`__.

    The script for inference can be found at: `NeMo/examples/nlp/token_classification/punctuate_capitalize_infer.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuate_capitalize_infer.py>`__.

.. _raw_data_format_punct:

Raw Data Format
---------------
In addition to `Punctuation And Capitalization Raw Data Format <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#raw-data-format>`_ this model also requires audio data.
You have to provide ``audio_train.txt`` and ``audio_dev.txt`` (and optionally ``audio_test.txt``) which contain one valid path to audio per row.

Example of the ``audio_train.txt``/``audio_dev.txt`` file:

.. code::

    /path/to/1.wav
    /path/to/2.wav
    ....
In this case ``source_data_dir`` structure should look similar to the following:

.. code::

   .
   |--sourced_data_dir
     |-- dev.txt
     |-- train.txt
     |-- audio_train.txt
     |-- audio_dev.txt

.. _nemo-data-format-label:

Tarred dataset
--------------

It is recommended to use tarred dataset for training with large amount of data (>500 hours) due to large amount of RAM consumed by loading whole audio data into memory and CPU usage.

For creating of tarred dataset with audio you will need data in NeMo format:

.. code::

    python examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py \
        --text <PATH/TO/LOWERCASED/TEXT/WITHOUT/PUNCTUATION> \
        --labels <PATH/TO/LABELS/IN/NEMO/FORMAT> \
        --output_dir <PATH/TO/DIRECTORY/WITH/OUTPUT/TARRED/DATASET> \
        --num_batches_per_tarfile 100 \
        --use_audio \
        --audio_file <PATH/TO/AUDIO/PATHS/FILE> \
        --sample_rate 16000 

<<<<<<< HEAD
=======
.. note::
  You can change sample rate to any positive integer. It will be used in constructor of :class:`~nemo.collections.asr.parts.preprocessing.AudioSegment`. It is recomended to set ``sample_rate`` to the same value as data which was used during training of ASR model.


>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
Training Punctuation and Capitalization Model
---------------------------------------------

The audio encoder is initialized with pretrained ASR model. You can use any of ``list_available_models()`` of ``EncDecCTCModel`` or your own checkpoints, either one should be provided in ``model.audio_encoder.pretrained_model``.
You can freeze audio encoder during training and add additional ``ConformerLayer`` on top of encoder to reduce compute with ``model.audio_encoder.freeze``. You can also add `Adapters <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/core/adapters/components.html>`_ to reduce compute with ``model.audio_encoder.adapter``. Parameters of fusion module are stored in ``model.audio_encoder.fusion``.
An example of a model configuration file for training the model can be found at:
`NeMo/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml>`__.

<<<<<<< HEAD
.. _run-config-label:

Run config
^^^^^^^^^^

An example of a config file is
`here <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml>`_.

.. list-table:: Run config. The main config passed to a script `punctuation_capitalization_lexical_audio_train_evaluate.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py>`_
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **pretrained_model**
     - string
     - ``null``
     - Can be an NVIDIA's NGC cloud model or a path to a ``.nemo`` checkpoint. You can get list of possible cloud options
       by calling a method :py:meth:`~nemo.collections.nlp.models.PunctuationCapitalizationLexicalAudioModel.list_available_models`.
   * - **name**
     - string
     - ``'Punctuation_and_Capitalization_Lexical_Audio'``
     - A name of the model. Used for naming output directories and ``.nemo`` checkpoints.
   * - **do_training**
     - bool
     - ``true``
     - Whether to perform training of the model.
   * - **do_testing**
     - bool
     - ``false``
     - Whether ot perform testing of the model after training.
   * - **model**
     - :ref:`model config<model-config-label>`
     - :ref:`model config<model-config-label>`
     - A configuration for the :class:`~nemo.collections.nlp.models.PunctuationCapitalizationLexicalAudioModel`.
   * - **trainer**
     - trainer config
     -
     - Parameters of
       `pytorch_lightning.Trainer <https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api>`_.
   * - **exp_manager**
     - exp manager config
     -
     - A configuration with various NeMo training options such as output directories, resuming from checkpoint,
       tensorboard and W&B logging, and so on. For possible options see :ref:`exp-manager-label` description and class
       :class:`~nemo.utils.exp_manager.exp_manager`.

.. _model-config-label:
=======
Configs
^^^^^^^^^^^^
.. note::
  This page contains only parameters specific to lexical and audio model. Others parameters can be found in `Punctuation And Capitalization's page <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html>`_.
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

Model config
^^^^^^^^^^^^

<<<<<<< HEAD
.. list-table:: Location of model config in parent config
   :widths: 5 5
   :header-rows: 1

   * - **Parent config**
     - **Key in parent config**
   * - :ref:`Run config<run-config-label>`
     - ``model``

=======
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
A configuration of
:class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_lexical_audio_model.PunctuationCapitalizationLexicalAudioModel`
model.

.. list-table:: Model config
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
<<<<<<< HEAD
   * - **class_labels**
     - :ref:`class labels config<class-labels-config-label>`
     - :ref:`class labels config<class-labels-config-label>`
     - Cannot be omitted in `.yaml` config. The ``class_labels`` parameter containing a dictionary with names of label
       id files used in ``.nemo`` checkpoints. These file names can also be used for passing label vocabularies to the
       model. If you wish to use ``class_labels`` for passing vocabularies, please provide path to vocabulary files in
       ``model.common_dataset_parameters.label_vocab_dir`` parameter.
   * - **common_dataset_parameters**
     - :ref:`common dataset parameters config<common-dataset-parameters-config-label>`
     - :ref:`common dataset parameters config<common-dataset-parameters-config-label>`
     - Label ids and loss mask information.
   * - **train_ds**
     - :ref:`data config<data-config-label>` with string in  ``ds_item``
     - ``null``
     - A configuration for creating training dataset and data loader. Cannot be omitted in `.yaml` config if training
       is performed.
   * - **validation_ds**
     - :ref:`data config<data-config-label>` with string OR list of strings in ``ds_item``
     - ``null``
     - A configuration for creating validation datasets and data loaders.
   * - **test_ds**
     - :ref:`data config<data-config-label>` with string OR list of strings in ``ds_item``
     - ``null``
     - A configuration for creating test datasets and data loaders. Cannot be omitted in `.yaml` config if testing is
       performed.
   * - **punct_head**
     - :ref:`head config<head-config-label>`
     - :ref:`head config<head-config-label>`
     - A configuration for creating punctuation MLP head that is applied to a language model outputs.
   * - **capit_head**
     - :ref:`head config<head-config-label>`
     - :ref:`head config<head-config-label>`
     - A configuration for creating capitalization MLP head that is applied to a language model outputs.
   * - **tokenizer**
     - :ref:`tokenizer config<tokenizer-config-label>`
     - :ref:`tokenizer config<tokenizer-config-label>`
     - A configuration for creating source text tokenizer.
   * - **language_model**
     - :ref:`language model config<language-model-config-label>`
     - :ref:`language model config<language-model-config-label>`
     - A configuration of a BERT-like language model which serves as a model body.
   * - **optim**
     - optimization config
     - ``null``
     - A configuration of optimizer, learning rate scheduler, and L2 regularization. Cannot be omitted in `.yaml`
       config if training is performed. For more information see :ref:`Optimization <optimization-label>` and
       `primer <https://github.com/NVIDIA/NeMo/blob/main/tutorials/00_NeMo_Primer.ipynb>`_ tutorial.
=======
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
   * - **audio_encoder** 
     - :ref:`audio encoder config<audio-encoder-config-label>`
     - :ref:`audio encoder config<audio-encoder-config-label>`
     - A configuration for audio encoder.

<<<<<<< HEAD
.. _class-labels-config-label:

Class labels config
^^^^^^^^^^^^^^^^^^^

.. list-table:: Location of class labels config in parent configs
   :widths: 5 5
   :header-rows: 1

   * - **Parent config**
     - **Key in parent config**
   * - :ref:`Run config<run-config-label>`
     - ``model.class_labels``
   * - :ref:`Model config<model-config-label>`
     - ``class_labels``

.. list-table:: Class labels config
   :widths: 5 5 5 35
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **punct_labels_file**
     - string
     - ???
     - A name of a punctuation labels file. This parameter cannot be omitted in `.yaml` config. This name
       is used as a name of label ids file in ``.nemo`` checkpoint. It also can be used for passing label vocabulary to
       the model. If ``punct_labels_file`` is used as a vocabulary file, then you should provide parameter
       ``label_vocab_dir`` in :ref:`common dataset parameters<common-dataset-parameters-config-label>`
       (``model.common_dataset_parameters.label_vocab_dir`` in :ref:`run config<run-config-label>`). Each line of
       ``punct_labels_file`` file contains 1 label. The values are sorted, ``<line number>==<label id>``, starting
       from 0. A label with ``0`` id must contain neutral label which has to be
       equal to a ``pad_label`` parameter in :ref:`common dataset parameters<common-dataset-parameters-config-label>`.

   * - **capit_labels_file**
     - string
     - ???
     - Same as ``punct_labels_file`` for capitalization labels.

.. _common-dataset-parameters-config-label:

Common dataset parameters config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Location of common dataset parameters config in parent config
   :widths: 5 5
   :header-rows: 1

   * - **Parent config**
     - **Key in parent config**
   * - :ref:`Run config<run-config-label>`
     - ``model.common_dataset_config``
   * - :ref:`Model config<model-config-label>`
     - ``common_dataset_config``

A common dataset parameters config which includes label and loss mask information.
If you omit parameters ``punct_label_ids``, ``capit_label_ids``, ``label_vocab_dir``, then labels will be inferred
from a training dataset or loaded from a checkpoint.

Parameters ``ignore_extra_tokens`` and ``ignore_start_end`` are responsible for forming loss mask. A loss mask
defines on which tokens loss is computed.

.. list-table:: Common dataset parameters config
   :widths: 5 5 5 35
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **pad_label**
     - string
     - ???
     - This parameter cannot be omitted in `.yaml` config. The ``pad_label`` parameter contains label used for
       punctuation and capitalization label padding. It also serves as a neutral label for both punctuation and
       capitalization. If any of ``punct_label_ids``, ``capit_label_ids`` parameters is provided, then ``pad_label``
       must have ``0`` id in them. In addition, if ``label_vocab_dir`` is provided, then ``pad_label`` must be on the
       first lines in files ``class_labels.punct_labels_file`` and ``class_labels.capit_labels_file``.
   * - **ignore_extra_tokens**
     - bool
     - ``false``
     - Whether to compute loss on not first tokens in words. If this parameter is ``true``, then loss mask is ``false``
       for all tokens in a word except the first.
   * - **ignore_start_end**
     - bool
     - ``true``
     - If ``false``, then loss is computed on [CLS] and [SEP] tokens.
   * - **punct_label_ids**
     - ``Dict[str, int]``
     - ``null``
     - A dictionary with punctuation label ids. ``pad_label`` must have ``0`` id in this dictionary. You can omit this
       parameter and pass label ids through ``class_labels.punct_labels_file`` or let the model to infer label ids from
       dataset or load them from checkpoint.
   * - **capit_label_ids**
     - ``Dict[str, int]``
     - ``null``
     - Same as ``punct_label_ids`` for capitalization labels.
   * - **label_vocab_dir**
     - string
     - ``null``
     - A path to directory which contains class labels files. See :class:`ClassLabelsConfig`. If this parameter is
       provided, then labels will be loaded from files which are located in ``label_vocab_dir`` and have names
       specified in ``model.class_labels`` configuration section. A label specified in ``pad_label`` has to be on the
       first lines of ``model.class_labels`` files.

.. _data-config-label:
=======
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

Data config
^^^^^^^^^^^

.. list-table:: Location of data configs in parent configs
   :widths: 5 5
   :header-rows: 1

   * - **Parent config**
     - **Keys in parent config**
   * - :ref:`Run config<run-config-label>`
     - ``model.train_ds``, ``model.validation_ds``, ``model.test_ds``
   * - :ref:`Model config<model-config-label>`
     - ``train_ds``, ``validation_ds``, ``test_ds``

<<<<<<< HEAD
For convenience, items of data config are described in 4 tables:
:ref:`common parameters for both regular and tarred datasets<common-data-parameters-label>`,
:ref:`parameters which are applicable only to regular dataset<regular-dataset-parameters-label>`,
:ref:`parameters which are applicable only to tarred dataset<tarred-dataset-parameters-label>`,
:ref:`parameters for PyTorch data loader<pytorch-dataloader-parameters-label>`.

.. _common-data-parameters-label:

.. list-table:: Parameters for both regular (:class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset`) and tarred (:class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.BertPunctuationCapitalizationTarredDataset`) datasets
   :widths: 5 5 5 35
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **use_tarred_dataset**
     - bool
     - ???
     - This parameter cannot be omitted in `.yaml` config. The ``use_tarred_dataset`` parameter specifies whether to
       use tarred dataset or regular dataset. If ``true``, then you should provide ``ds_item``, ``tar_metadata_file``
       parameters. Otherwise, you should provide parameters ``ds_item``, ``text_file``, ``labels_file``,
       ``tokens_in_batch`` parameters.
   * - **ds_item**
     - **string** OR **list of strings** (only if used in ``model.validation_ds`` or ``model.test_ds``)
     - ???
     - This parameter cannot be omitted in `.yaml` config. The ``ds_item`` parameter contains a path to a directory
       with ``tar_metadata_file`` file (if ``use_tarred_dataset=true``) or ``text_file`` and ``labels_file``
       (if ``use_tarred_dataset=false``). For ``validation_ds`` or ``test_ds`` you may specify a list of paths in
       ``ds_item``. If ``ds_item`` is a list, then evaluation will be performed on several datasets. To override
       ``ds_item`` config parameter with a list use following syntax:
       ``python punctuation_capitalization_train_evaluate.py model.validation_ds.ds_item=[path1,path2]`` (no spaces after ``=``
       sign).
   * - **label_info_save_dir**
     - string
     - ``null``
     - A path to a directory where files created during dataset processing are stored. These files include label id
       files and label stats files. By default, it is a directory containing ``text_file`` or ``tar_metadata_file``.
       You may need this parameter if dataset directory is read-only and thus does not allow saving anything near
       dataset files.

=======
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
.. _regular-dataset-parameters-label:

.. list-table:: Parameters for regular (:class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset`) dataset
   :widths: 5 5 5 30
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
<<<<<<< HEAD
   * - **text_file**
     - string
     - ``null``
     - This parameter cannot be omitted in `.yaml` config if ``use_tarred_dataset=false``. The ``text_file``
       parameter is a name of a source text file which is located in ``ds_item`` directory.
   * - **labels_file**
     - string
     - ``null``
     - This parameter cannot be omitted in `.yaml` config if ``use_tarred_dataset=false``. The ``labels_file``
       parameter is a name of a file with punctuation and capitalization labels in
       :ref:`NeMo format <nemo-data-format-label>`. It has is located in ``ds_item`` directory.
   * - **tokens_in_batch**
     - int
     - ``null``
     - This parameter cannot be omitted in `.yaml` config if ``use_tarred_dataset=false``. The ``tokens_in_batch``
       parameter contains a number of tokens in a batch including paddings and special tokens ([CLS], [SEP], [UNK]).
       This config does not have ``batch_size`` parameter.
   * - **max_seq_length**
     - int
     - ``512``
     - Max number of tokens in a source sequence. ``max_seq_length`` includes [CLS] and [SEP] tokens. Sequences
       which are too long will be clipped by removal of tokens from the end of a sequence.
   * - **num_samples**
     - int
     - ``-1``
     - A number of samples loaded from ``text_file`` and ``labels_file`` which are used in the dataset. If this
       parameter equals ``-1``, then all samples are used.
   * - **use_cache**
     - bool
     - ``true``
     - Whether to use pickled features which are already present in ``cache_dir``.
       For large not tarred datasets, pickled features may considerably reduce time required for training to start.
       Tokenization of source sequences is not fast because sequences are split into words before tokenization.
       For even larger datasets (~4M), tarred datasets are recommended. If pickled features are missing, then
       new pickled features file will be created regardless of the value of ``use_cache`` parameter because
       pickled features are required for distributed training.
   * - **cache_dir**
     - string
     - ``null``
     - A path to a directory containing cache or directory where newly created cache is saved. By default, it is
       a directory containing ``text_file``. You may need this parameter if cache for a dataset is going to be created
       and the dataset directory is read-only. ``cache_dir`` and ``label_info_save_dir`` are separate parameters for
       the case when a cache is ready and this cache is stored in a read-only directory. In such a case you will
       separate ``label_info_save_dir``.
   * - **get_label_frequences**
     - bool
     - ``false``
     - Whether to show and save label frequencies. Frequencies are showed if ``verbose`` parameter is ``true``. If
       ``get_label_frequencies=true``, then frequencies are saved into ``label_info_save_dir``.
   * - **verbose**
     - bool
     - ``true``
     - If ``true``, then progress messages and examples of acquired features are printed.
   * - **n_jobs**
     - int
     - ``0``
     - Number of workers used for features creation (tokenization, label encoding, and clipping). If ``0``, then
       multiprocessing is not used; if ``null``, then ``n_jobs`` will be equal to the number of CPU cores. WARNING:
       there can be weird deadlocking errors with some tokenizers (e.g. SentencePiece) if ``n_jobs`` is greater than
       zero.
=======
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
   * - **use_audio**
     - bool
     - ``false``
     - If set to ``true`` dataset will return audio as well as text.
   * - **audio_file**
     - string
     - ``null``
     - A path to file with audio paths.
   * - **sample_rate**
     - int
     - ``null``
     - Target sample rate of audios. Can be used for up sampling or down sampling of audio.
   * - **use_bucketing**
     - bool
     - ``true``
     - If set to True will sort samples based on their audio length and assamble batches more efficently (less padding in batch). If set to False dataset will return ``batch_size`` batches instead of ``number_of_tokens`` tokens. 
   * - **preload_audios**
     - bool
     - ``true``
     - If set to True batches will include waveforms, if set to False will store audio_filepaths instead and load audios during ``collate_fn`` call.
    
<<<<<<< HEAD
.. _tarred-dataset-parameters-label:

.. list-table:: Parameters for tarred (:class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.BertPunctuationCapitalizationTarredDataset`) dataset
   :widths: 5 5 5 30
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **tar_metadata_file**
     - string
     - ``null``
     - This parameter cannot be omitted in `.yaml` config if ``use_tarred_dataset=true``. The ``tar_metadata_file``
       is a path to metadata file of tarred dataset. A tarred metadata file and
       other parts of tarred dataset are usually created by the script
       `examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py
       <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py>`_
   * - **tar_shuffle_n**
     - int
     - ``1``
     - The size of shuffle buffer of `webdataset <https://github.com/webdataset/webdataset>`_. The number of batches
       which are permuted.
   * - **shard_strategy**
     - string
     - ``scatter``
     - Tarred dataset shard distribution strategy chosen as a str value during ddp. Accepted values are ``scatter`` and ``replicate``.
       ``scatter``: Each node gets a unique set of shards, which are permanently pre-allocated and never changed at runtime, when the total
       number of shards is not divisible with ``world_size``, some shards (at max ``world_size-1``) will not be used.
       ``replicate``: Each node gets the entire set of shards available in the tarred dataset, which are permanently pre-allocated and never
       changed at runtime. The benefit of replication is that it allows each node to sample data points from the entire dataset independently
       of other nodes, and reduces dependence on value of ``tar_shuffle_n``.

       .. warning::
           Replicated strategy allows every node to sample the entire set of available tarfiles, and therefore more than one node may sample
           the same tarfile, and even sample the same data points! As such, there is no assured guarantee that all samples in the dataset will be
           sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific occasions (when the number of shards is not
           divisible with ``world_size``), will not sample the entire dataset. For these reasons it is not advisable to use tarred datasets as
           validation or test datasets.

.. _pytorch-dataloader-parameters-label:

.. list-table:: Parameters for PyTorch `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.DataLoader>`_
   :widths: 5 5 5 30
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **shuffle**
     - bool
     - ``true``
     - Shuffle batches every epoch. For usual training datasets, the parameter activates batch repacking every
       epoch. For tarred dataset it would be only batches permutation.
   * - **drop_last**
     - bool
     - ``false``
     - In cases when data parallelism is used, ``drop_last`` defines the way data pipeline behaves when some replicas
       are out of data and some are not. If ``drop_last`` is ``True``, then epoch ends in the moment when any replica
       runs out of data. If ``drop_last`` is ``False``, then the replica will replace missing batch with a batch from a
       pool of batches that the replica has already processed. If data parallelism is not used, then parameter
       ``drop_last`` does not do anything. For more information see
       `torch.utils.data.distributed.DistributedSampler
       <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler>`_
   * - **pin_memory**
     - bool
     - ``true``
     - See this parameter documentation in
       `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.DataLoader>`_
   * - **num_workers**
     - int
     - ``8``
     - See this parameter documentation in
       `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.DataLoader>`_
   * - **persistent_memory**
     - bool
     - ``true``
     - See this parameter documentation in
       `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.DataLoader>`_

.. _head-config-label:

Head config
^^^^^^^^^^^

.. list-table:: Location of head configs in parent configs
   :widths: 5 5
   :header-rows: 1

   * - **Parent config**
     - **Keys in parent config**
   * - :ref:`Run config<run-config-label>`
     - ``model.punct_head``, ``model.capit_head``
   * - :ref:`Model config<model-config-label>`
     - ``punct_head``, ``capit_head``

This config defines a multilayer perceptron which is applied to
outputs of a language model. Number of units in the hidden layer is equal to the dimension of the language model.

.. list-table:: Head config
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **num_fc_layers**
     - int
     - ``1``
     - A number of hidden layers in the multilayer perceptron.
   * - **fc_dropout**
     - float
     - ``0.1``
     - A dropout used in the MLP.
   * - **activation**
     - string
     - ``'relu'``
     - An activation used in hidden layers.
   * - **use_transformer_init**
     - bool
     - ``true``
     - Whether to initialize the weights of the classifier head with the approach that was used for language model
       initialization.

.. _language-model-config-label:

Language model config
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Location of language model config in parent configs
   :widths: 5 5
   :header-rows: 1

   * - **Parent config**
     - **Key in parent config**
   * - :ref:`Run config<run-config-label>`
     - ``model.language_model``
   * - :ref:`Model config<model-config-label>`
     - ``language_model``

A configuration of a language model which serves as a model body. BERT-like HuggingFace models are supported. Provide a
valid ``pretrained_model_name`` and, optionally, you may reinitialize model via ``config_file`` or ``config``.

Alternatively you can initialize the language model using ``lm_checkpoint``.

.. list-table:: Language model config
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **pretrained_model_name**
     - string
     - ???
     - This parameter cannot be omitted in `.yaml` config. The ``pretrained_model_name`` parameter contains a name of
       HuggingFace pretrained model. For example, ``'bert-base-uncased'``.
   * - **config_file**
     - string
     - ``null``
     - A path to a file with HuggingFace model config which is used to reinitialize the language model.
   * - **config**
     - dict
     - ``null``
     - A HuggingFace config which is used to reinitialize the language model.
   * - **lm_checkpoint**
     - string
     - ``null``
     - A path to a ``torch`` checkpoint of the language model.

.. _tokenizer-config-label:

Tokenizer config
^^^^^^^^^^^^^^^^

.. list-table:: Location of tokenizer config in parent configs
   :widths: 5 5
   :header-rows: 1

   * - **Parent config**
     - **Key in parent config**
   * - :ref:`Run config<run-config-label>`
     - ``model.tokenizer``
   * - :ref:`Model config<model-config-label>`
     - ``tokenizer``

A configuration of a source text tokenizer.

.. list-table:: Language model config
   :widths: 5 5 10 25
   :header-rows: 1

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **tokenizer_name**
     - string
     - ???
     - This parameter cannot be omitted in `.yaml` config. The ``tokenizer_name`` parameter containing a name of the
       tokenizer used for tokenization of source sequences. Possible
       options are ``'sentencepiece'``, ``'word'``, ``'char'``, HuggingFace tokenizers (e.g. ``'bert-base-uncased'``).
       For more options see function ``nemo.collections.nlp.modules.common.get_tokenizer``. The tokenizer must have
       properties ``cls_id``, ``pad_id``, ``sep_id``, ``unk_id``.
   * - **vocab_file**
     - string
     - ``null``
     - A path to vocabulary file which is used in ``'word'``, ``'char'``, and HuggingFace tokenizers.
   * - **special_tokens**
     - ``Dict[str, str]``
     - ``null``
     - A dictionary with special tokens passed to constructors of ``'char'``, ``'word'``, ``'sentencepiece'``, and
       various HuggingFace tokenizers.
   * - **tokenizer_model**
     - string
     - ``null``
     - A path to a tokenizer model required for ``'sentencepiece'`` tokenizer.
=======
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

.. _audio-encoder-config-label:

Audio Encoder config
^^^^^^^^^^^^^^^^

.. list-table:: Audio Encoder Config
<<<<<<< HEAD
    :widths: 5 5 10 25
    :header-rows: 1
=======
   :widths: 5 5 10 25
   :header-rows: 1
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **pretrained_model**
     - string
     - ``stt_en_conformer_ctc_medium``
     - Pretrained model name or path to ``.nemo``` file to take audio encoder from.
   * - **freeze**
     - :ref:`freeze config<freeze-config-label>`
     - :ref:`freeze config<freeze-config-label>`
     - Configuration for freezing audio encoder's weights.
   * - **adapter**
     - :ref:`adapter config<adapter-config-label>`
     - :ref:`adapter config<adapter-config-label>`
     - Configuration for adapter.
   * - **fusion**
     - :ref:`fusion config<fusion-config-label>`
     - :ref:`fusion config<fusion-config-label>`
     - Configuration for fusion.

<<<<<<< HEAD
.. _freeze-config-label:

.. list-table:: Freeze Config
    :widths: 5 5 10 25
    :header-rows: 1
=======

.. _freeze-config-label:

.. list-table:: Freeze Config
   :widths: 5 5 10 25
   :header-rows: 1
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **is_enabled**
     - bool
     - ``false``
     - If set to ``true`` encoder's weights will not be updated during training and aditional ``ConformerLayer`` layers will be added.
   * - **d_model**
     - int
     - ``256``
     - Input dimension of ``MultiheadAttentionMechanism`` and ``PositionwiseFeedForward`` of additional ``ConformerLayer`` layers.
   * - **d_ff**
     - int
     - ``1024``
     - Hidden dimension of ``PositionwiseFeedForward`` of additional ``ConformerLayer`` layers.
   * - **num_layers**
     - int
     - ``4``
     - Number of additional ``ConformerLayer`` layers.

<<<<<<< HEAD
.. _adapter-config-label:

.. list-table:: Adapter Config
    :widths: 5 5 10 25
    :header-rows: 1
=======

.. _adapter-config-label:

.. list-table:: Adapter Config
   :widths: 5 5 10 25
   :header-rows: 1
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **enable**
     - bool
     - ``false``
     - If set to ``true`` will enable adapters for audio encoder.
   * - **config**
     - ``LinearAdapterConfig``
     - ``null``
     - For more details see `nemo.collections.common.parts.LinearAdapterConfig <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/parts/adapter_modules.py#L141>`_ class.

<<<<<<< HEAD
.. _fusion-config-label:

.. list-table:: Fusion Config
    :widths: 5 5 10 25
    :header-rows: 1
=======

.. _fusion-config-label:

.. list-table:: Fusion Config
   :widths: 5 5 10 25
   :header-rows: 1
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

   * - **Parameter**
     - **Data type**
     - **Default value**
     - **Description**
   * - **num_layers**
     - int
     - ``4``
     - Number of layers to use in fusion.
   * - **num_attention_heads**
     - int
     - ``4``
     - Number of attention heads to use in fusion.
   * - **inner_size**
     - int
     - ``2048``
     - Fusion inner size.


<<<<<<< HEAD
=======

>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
Model training
^^^^^^^^^^^^^^

For more information, refer to the :ref:`nlp_model` section.

To train the model from scratch, run:

.. code::

      python examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py \
             model.train_ds.ds_item=<PATH/TO/TRAIN/DATA_DIR> \
             model.train_ds.text_file=<NAME_OF_TRAIN_INPUT_TEXT_FILE> \
             model.train_ds.labels_file=<NAME_OF_TRAIN_LABELS_FILE> \
             model.validation_ds.ds_item=<PATH/TO/DEV/DATA_DIR> \
             model.validation_ds.text_file=<NAME_OF_DEV_TEXT_FILE> \
             model.validation_ds.labels_file=<NAME_OF_DEV_LABELS_FILE> \
             trainer.devices=[0,1] \
             trainer.accelerator='gpu' \
             optim.name=adam \
             optim.lr=0.0001 \
             model.train_ds.audio_file=<NAME_OF_TRAIN_AUDIO_FILE> \
             model.validation_ds.audio_file=<NAME_OF_DEV_AUDIO_FILE>

The above command will start model training on GPUs 0 and 1 with Adam optimizer and learning rate of 0.0001; and the
trained model is stored in the ``nemo_experiments/Punctuation_and_Capitalization`` folder.

To train from the pre-trained model, run:

.. code::

      python examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py \
             model.train_ds.ds_item=<PATH/TO/TRAIN/DATA_DIR> \
             model.train_ds.text_file=<NAME_OF_TRAIN_INPUT_TEXT_FILE> \
             model.train_ds.labels_file=<NAME_OF_TRAIN_LABELS_FILE> \
             model.validation_ds.ds_item=<PATH/TO/DEV/DATA/DIR> \
             model.validation_ds.text_file=<NAME_OF_DEV_TEXT_FILE> \
             model.validation_ds.labels_file=<NAME_OF_DEV_LABELS_FILE> \
             model.train_ds.audio_file=<NAME_OF_TRAIN_AUDIO_FILE> \
             model.validation_ds.audio_file=<NAME_OF_DEV_AUDIO_FILE> \
             pretrained_model=<PATH/TO/SAVE/.nemo>


.. note::

    All parameters defined in the configuration file can be changed with command arguments. For example, the sample
    config file mentioned above has :code:`train_ds.tokens_in_batch` set to ``2048``. However, if you see that
    the GPU utilization can be optimized further by using a larger batch size, you may override to the desired value
    by adding the field :code:`train_ds.tokens_in_batch=4096` over the command-line. You can repeat this with
    any of the parameters defined in the sample configuration file.

Inference
---------

Inference is performed by a script `examples/nlp/token_classification/punctuate_capitalize_infer.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuate_capitalize_infer.py>`_

.. code::

    python punctuate_capitalize_infer.py \
        --input_manifest <PATH/TO/INPUT/MANIFEST> \
        --output_manifest <PATH/TO/OUTPUT/MANIFEST> \
<<<<<<< HEAD
        --pretrained_name PLACEHOLDER \
=======
        --pretrained_name <PATH to .nemo file> \
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
        --max_seq_length 64 \
        --margin 16 \
        --step 8 \
        --use_audio

<<<<<<< HEAD
:code:`<PATH/TO/INPUT/MANIFEST>` is a path to NeMo :ref:`ASR manifest<LibriSpeech_dataset>` with text in which you need to
restore punctuation and capitalization. If manifest contains :code:`'pred_text'` key, then :code:`'pred_text'` elements
will be processed. Otherwise, punctuation and capitalization will be restored in :code:`'text'` elements.

:code:`<PATH/TO/OUTPUT/MANIFEST>` is a path to NeMo ASR manifest into which result will be saved. The text with restored
punctuation and capitalization is saved into :code:`'pred_text'` elements if :code:`'pred_text'` key is present in the
input manifest. Otherwise result will be saved into :code:`'text'` elements.

Alternatively you can pass data for restoring punctuation and capitalization as plain text. See help for parameters :code:`--input_text`
and :code:`--output_text` of the script. In this case you also need to provide `--audio_file` parameter.
`punctuate_capitalize_infer.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuate_capitalize_infer.py>`_.

The script `punctuate_capitalize_infer.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuate_capitalize_infer.py>`_
can restore punctuation and capitalization in a text of arbitrary length. Long sequences are split into segments
:code:`--max_seq_length - 2` tokens each (this number does not include :code:`[CLS]` and :code:`[SEP]` tokens). Each
segment starts and ends with :code:`[CLS]` and :code:`[SEP]` tokens correspondingly. Every segment is offset to the
previous one by :code:`--step` tokens. For example, if every character is a token, :code:`--max_seq_length=5`,
:code:`--step=2`, then text :code:`"hello"` will be split into segments
:code:`[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`.

If segments overlap, then predicted probabilities for a token present in several segments are multiplied before
before selecting the best candidate.

Splitting leads to pour performance of a model near edges of segments. Use parameter :code:`--margin` to discard :code:`--margin`
probabilities predicted for :code:`--margin` tokens near segment edges. For example, if
every character is a token, :code:`--max_seq_length=5`, :code:`--step=1`, :code:`--margin=1`, then text :code:`"hello"` will be split into
segments :code:`[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'e', 'l', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`.
Before calculating final predictions, probabilities for tokens marked by asterisk are removed: :code:`[['[CLS]', 'h', 'e', 'l'*, '[SEP]'*], ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]']]`

Long audios are split into corresponding sequences of :code:`4000 * max_seq_length`.
=======
Long audios are split just like in text only case, audio sequences treated the same as text seqences except :code:`max_seq_length` for audio equals :code:`max_seq_length*4000`.
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62

Model Evaluation
----------------

Model evaluation is performed by the same script
`examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py
<https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.py>`_
as training.

Use :ref`config<run-config-lab>` parameter ``do_training=false`` to disable training and parameter ``do_testing=true``
to enable testing. If both parameters ``do_training`` and ``do_testing`` are ``true``, then model is trained and then
tested.

To start evaluation of the pre-trained model, run:

.. code::

<<<<<<< HEAD
    python punctuation_capitalization_train_evaluate.py \
           +model.do_training=false \
           +model.to_testing=true \
           model.test_ds.ds_item=<PATH/TO/TEST/DATA/DIR>  \
           pretrained_model=PLACEHOLDER \
=======
    python punctuation_capitalization_lexical_audio_train_evaluate.py \
           +model.do_training=false \
           +model.to_testing=true \
           model.test_ds.ds_item=<PATH/TO/TEST/DATA/DIR>  \
           pretrained_model=<PATH to .nemo file> \
>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
           model.test_ds.text_file=<NAME_OF_TEST_INPUT_TEXT_FILE> \
           model.test_ds.labels_file=<NAME_OF_TEST_LABELS_FILE> \
           model.test_ds.audio_file=<NAME_OF_TEST_AUDIO_FILE>


Required Arguments
^^^^^^^^^^^^^^^^^^

- :code:`pretrained_model`: pretrained Punctuation and Capitalization Lexical Audio model from ``list_available_models()`` or path to a ``.nemo``
<<<<<<< HEAD
  file. For example: ``PLACEHOLDER`` or ``your_model.nemo``.
- :code:`model.test_ds.ds_item`: path to the directory that contains :code:`model.test_ds.text_file`, :code:`model.test_ds.labels_file` and :code:`model.test_ds.audio_file`

During evaluation of the :code:`test_ds`, the script generates two classification reports: one for capitalization task and another
one for punctuation task. This classification reports include the following metrics:

- :code:`Precision`
- :code:`Recall`
- :code:`F1`

More details about these metrics can be found `here <https://en.wikipedia.org/wiki/Precision_and_recall>`__.

=======
  file. For example: ``your_model.nemo``.
- :code:`model.test_ds.ds_item`: path to the directory that contains :code:`model.test_ds.text_file`, :code:`model.test_ds.labels_file` and :code:`model.test_ds.audio_file`

>>>>>>> dd9a30f55b7ec308cc3b55751e8ae5aa40f36a62
References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-PUNCT
    :keyprefix: nlp-punct-

