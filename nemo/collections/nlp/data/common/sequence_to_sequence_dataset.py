# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import (
    get_indexed_dataset_,
    get_samples_mapping,
)
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import TextMemMapDataset
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['SequenceToSequenceDataset', 'TextMemmapSequenceToSequenceDataset']


class SequenceToSequenceDataset(Dataset):
    """Sequence to Sequence Dataset in memory."""

    def __init__(
        self,
        src_file_name: str,
        tgt_file_name: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
    ):
        super().__init__()
        self.src_file_name = src_file_name
        self.tgt_file_name = tgt_file_name
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_seq_length = max_src_seq_length
        self.max_tgt_seq_length = max_tgt_seq_length
        assert self.max_src_seq_length > 0
        assert self.max_tgt_seq_length > 0
        self._check_files_exist()
        self._get_examples()

    def _check_files_exist(self):
        if not os.path.exists(self.src_file_name):
            raise FileNotFoundError(f"Source file {self.src_file_name} not found")
        if not os.path.exists(self.tgt_file_name):
            raise FileNotFoundError(f"Source file {self.src_file_name} not found")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text_enc = example['src']
        text_dec = example['tgt'][:-1]
        labels = example['tgt'][1:]
        return {'text_enc': text_enc, 'text_dec': text_dec, 'labels': labels}

    def _get_examples(self):
        self.examples = []
        with open(self.src_file_name, encoding='utf8') as f_src, open(self.tgt_file_name, encoding='utf8') as f_tgt:
            for i, (src, tgt) in enumerate(zip(f_src, f_tgt)):
                if i % 10000 == 0 and i != 0:
                    logging.info(f"Read {i} lines from {self.src_file_name} & {self.tgt_file_name}")
                src = (
                    [self.src_tokenizer.bos_id]
                    + self.src_tokenizer.text_to_ids(src.strip())
                    + [self.src_tokenizer.eos_id]
                )
                tgt = (
                    [self.tgt_tokenizer.bos_id]
                    + self.tgt_tokenizer.text_to_ids(tgt.strip())
                    + [self.tgt_tokenizer.eos_id]
                )
                # Truncate to max sequence length.
                if len(src) > self.max_src_seq_length:
                    src = src[-self.max_src_seq_length + 1 :]
                if len(tgt) > self.max_tgt_seq_length:
                    tgt = tgt[-self.max_tgt_seq_length + 1 :]
                self.examples.append({'src': src, 'tgt': tgt})

        logging.info(f'Dataset Length : {len(self.examples)}')

    def collate_fn(self, batch):
        enc_query = [item['text_enc'] for item in batch]
        dec_input = [item['text_dec'] for item in batch]
        labels = [item['labels'] for item in batch]

        if isinstance(enc_query[0], np.ndarray):
            enc_query = [x.tolist() for x in enc_query]

        if isinstance(dec_input[0], np.ndarray):
            dec_input = [x.tolist() for x in dec_input]

        if isinstance(labels[0], np.ndarray):
            labels = [x.tolist() for x in labels]

        max_dec_input_length = max([len(item) for item in dec_input]) if dec_input else 0
        max_enc_query_length = max([len(item) for item in enc_query]) if enc_query else 0
        max_label_length = max([len(item) for item in labels]) if labels else 0

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_query = [item + [self.src_tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_query]
        dec_input = [item + [self.tgt_tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in dec_input]
        labels = [item + [self.tgt_tokenizer.pad_id] * (max_label_length - len(item)) for item in labels]

        enc_query = torch.LongTensor(enc_query)
        dec_input = torch.LongTensor(dec_input)
        labels = torch.LongTensor(labels)
        loss_mask = torch.LongTensor(loss_mask)

        enc_mask = (enc_query != self.src_tokenizer.pad_id).long()
        dec_mask = (dec_input != self.tgt_tokenizer.pad_id).long()

        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
        }


class IndexedSequenceToSequenceDataset(SequenceToSequenceDataset):
    """Abstract class for TextMemmapSequenceToSequenceDataset and BinarizedMemmapSequenceToSequenceDataset.
    This class is not meant to be used standalone and just as an abstract class for the two subclasses.
    """

    def __init__(
        self,
        src_file_name: str,
        tgt_file_name: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        seed: int = 1234,
        max_num_samples=None,
    ):
        """
        src_file_name: Path to a single source file on disk. This is either the path to a raw text file or the prefix to the processed src_file_name.bin/idx files.
        src_file_name: Path to a single target file on disk. This is either the path to a raw text file or the prefix to the processed tgt_file_name.bin/idx files.
        src_tokenizer: Tokenizer for the source dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        tgt_tokenizer: Tokenizer for the target dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_src_seq_length: Maximum length of the source sequences. Lines above this length will be truncated.
        max_tgt_seq_length: Maximum length of the target sequences. Lines above this length will be truncated.
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        """
        super().__init__(
            src_file_name=src_file_name,
            tgt_file_name=tgt_file_name,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length,
            max_tgt_seq_length=max_tgt_seq_length,
        )
        self.seed = seed
        self.max_num_samples = max_num_samples
        logging.info(f'Desired number of samples : {self.max_num_samples}')
        logging.info(f'Source Dataset Length : {len(self.src_indexed_dataset)}')
        logging.info(f'Target Dataset Length : {len(self.tgt_indexed_dataset)}')

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.src_indexed_dataset)
        else:
            return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):
                idx = idx.item()

        assert idx < len(self.src_indexed_dataset)
        src = self.src_indexed_dataset[idx]
        if len(src) > self.max_src_seq_length - 2:
            src = src[: self.max_src_seq_length - 2]
        text_enc = np.concatenate([[self.src_tokenizer.bos_id], src, [self.src_tokenizer.eos_id]])

        tgt = self.tgt_indexed_dataset[idx]
        if len(tgt) > self.max_tgt_seq_length - 2:
            tgt = tgt[: self.max_tgt_seq_length - 2]

        text_dec = np.concatenate([[self.tgt_tokenizer.bos_id], tgt])
        labels = np.concatenate([tgt, [self.tgt_tokenizer.eos_id]])

        return {'text_enc': text_enc, 'text_dec': text_dec, 'labels': labels}

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            # This means max src and max tgt sequence length need to be the same
            if self.max_src_seq_length != self.max_tgt_seq_length:
                raise ValueError(
                    f"max_src_seq_length ({self.max_src_seq_length}) != max_tgt_seq_length ({self.max_tgt_seq_length}). This is needed for max_samples based training for now."
                )

            self.samples_mapping = get_samples_mapping(
                indexed_dataset=self.src_indexed_dataset,
                data_prefix=self.src_file_name,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_src_seq_length - 2,
                short_seq_prob=0,
                seed=self.seed,
                name=self.src_file_name.split('/')[-1],
                binary_head=False,
            )
        else:
            self.samples_mapping = None


class TextMemmapSequenceToSequenceDataset(IndexedSequenceToSequenceDataset):
    """Memory-mapped text sequence to sequence dataset. Operates on raw text files and tokenizes the text on-the-fly."""

    def __init__(
        self,
        src_file_name: str,
        tgt_file_name: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        seed: int = 1234,
        max_num_samples: int = None,
    ):
        """
        src_file_name: Path to a single source file on disk. The file should contain one sentence per line and be raw text.
        tgt_file_name: Path to a single target file on disk. The file should contain one sentence per line aligned with src_file_name and be raw text.
        src_tokenizer: Tokenizer for the source dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        tgt_tokenizer: Tokenizer for the target dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_src_seq_length: Maximum length of the source sequences. Lines above this length will be truncated.
        max_tgt_seq_length: Maximum length of the target sequences. Lines above this length will be truncated.
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        """
        self.seed = seed
        self.max_num_samples = max_num_samples
        super().__init__(
            src_file_name=src_file_name,
            tgt_file_name=tgt_file_name,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length,
            max_tgt_seq_length=max_tgt_seq_length,
            seed=seed,
            max_num_samples=max_num_samples,
        )

    def _get_examples(self):
        self.src_indexed_dataset = TextMemMapDataset(dataset_paths=[self.src_file_name], tokenizer=self.src_tokenizer)
        self.tgt_indexed_dataset = TextMemMapDataset(dataset_paths=[self.tgt_file_name], tokenizer=self.tgt_tokenizer)

        assert len(self.src_indexed_dataset) == len(
            self.tgt_indexed_dataset
        ), "src and tgt has different number of lines"
        self._build_samples_mapping()


class BinarizedMemmapSequenceToSequenceDataset(IndexedSequenceToSequenceDataset):
    """Memory-mapped text sequence to sequence dataset. Operates pre-tokenized binarized data files."""

    def __init__(
        self,
        src_dataset_prefix: str,
        tgt_dataset_prefix: str,
        src_tokenizer: TokenizerSpec,
        tgt_tokenizer: TokenizerSpec,
        max_src_seq_length: int,
        max_tgt_seq_length: int,
        seed: int = 1234,
        max_num_samples: int = None,
    ):
        """
        src_dataset_prefix: Path to the *prefix* of a single source bin/idx file on disk. This necessitates the existance src_file_prefix.bin and src_file_prefix.idx.
        tgt_dataset_prefix: Path to the *prefix* of a single target aligned with source bin/idx file on disk. This necessitates the existance tgt_file_prefix.bin and tgt_file_prefix.idx.
        src_tokenizer: Tokenizer for the source dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        tgt_tokenizer: Tokenizer for the target dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_src_seq_length: Maximum length of the source sequences. Lines above this length will be truncated.
        max_tgt_seq_length: Maximum length of the target sequences. Lines above this length will be truncated.
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        """
        self.src_dataset_prefix = src_dataset_prefix
        self.tgt_dataset_prefix = tgt_dataset_prefix
        self.seed = seed
        self.max_num_samples = max_num_samples
        super().__init__(
            src_file_name=src_dataset_prefix,
            tgt_file_name=tgt_dataset_prefix,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_seq_length=max_src_seq_length,
            max_tgt_seq_length=max_tgt_seq_length,
            seed=seed,
            max_num_samples=max_num_samples,
        )

    def _check_files_exist(self):
        if not os.path.exists(self.src_dataset_prefix + ".bin") or not os.path.exists(
            self.src_dataset_prefix + ".idx"
        ):
            raise FileNotFoundError(f"{self.src_dataset_prefix}.bin or {self.src_dataset_prefix}.idx not found")
        if not os.path.exists(self.tgt_dataset_prefix + ".bin") or not os.path.exists(
            self.tgt_dataset_prefix + ".idx"
        ):
            raise FileNotFoundError(f"{self.tgt_dataset_prefix}.bin or {self.tgt_dataset_prefix}.idx not found")

    def _get_examples(self):
        self.src_indexed_dataset = self._get_indexed_dataset(
            self.src_dataset_prefix, data_impl='mmap', skip_warmup=True
        )
        self.tgt_indexed_dataset = self._get_indexed_dataset(
            self.tgt_dataset_prefix, data_impl='mmap', skip_warmup=True
        )
        assert len(self.src_indexed_dataset) == len(self.tgt_indexed_dataset)
        self._build_samples_mapping()

    def _get_indexed_dataset(self, data_prefix, data_impl, skip_warmup):
        indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)
        return indexed_dataset
