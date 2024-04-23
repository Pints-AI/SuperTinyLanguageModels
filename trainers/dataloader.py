"""
A collection of dataloaders
"""

import os

import numpy as np
import torch
from tqdm import tqdm
from trainers.utils import load_data


class StandardDataloader:
    """
    A basic dataloader that preprocesses a dataset via
    tokenization, and then randomly loads batches as
    necessary.
    Args:
        cfg: the data config
        preprocess: whether to preprocess the data
    """

    def __init__(self, cfg, data_dir):
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.dataset_path = os.path.join(
            self.data_dir, 
            self.cfg["trainer"]["dataset"],
            f'{self.cfg["model_shell"]["tokenizer"]}-{self.cfg["model_shell"]["vocab_size"]}',
            self.cfg["trainer"]["dataloader"]["name"]
        )

        self.context_window = self.cfg["model_shell"]["context_window"]
        self.batch_size = self.cfg["trainer"]["training"]["batch_size"]
        self.device = self.cfg["general"]["device"]

    def get_batch(self, split="train"):
        """
        Get a train/val batch
        """
        data = np.memmap(
            os.path.join(self.dataset_path, f"{split}.bin"), dtype=np.uint16, mode="r"
        )

        ix = torch.randint(len(data) - self.context_window, (self.batch_size,))
        X = torch.stack(
            [
                torch.from_numpy((data[i : i + self.context_window]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.context_window]).astype(np.int64)
                )
                for i in ix
            ]
        )

        X, y = X.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
            self.device, non_blocking=True
        )

        return X, y

    def check_processed(self):
        """
        Check if the data has been preprocessed
        """
        return os.path.exists(self.dataset_path)

    def prepare_data(self, tokenizer):
        """
        Tokenize and store the data
        """
        # create folder
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        else:
            return # already processed

        # load the dataset
        split_dataset = load_data(
            dataset_name=self.cfg["trainer"]["dataset"],
        )

        print(split_dataset.keys())

        def process(example):
            ids = tokenizer.encode(example["text"])
            ids.append(tokenizer.eot_token)
            return {"ids": ids, "len": len(ids)}

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=8,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(self.dataset_path, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()




class Seq2SeqDataloader:
    """
    A sequence to sequence dataloader that preprocess a dataset
    via tokenization, and then randomly loads batches of 
    X,y pairs where both X and y are sequences of tokens of 
    a specific length
    Args:
        cfg: the data config
        preprocess: whether to preprocess the data
    """

    def __init__(self, cfg, data_dir):
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.dataset_path = os.path.join(
            self.data_dir, 
            f'{self.cfg["model_shell"]["tokenizer"]}-{self.cfg["model_shell"]["vocab_size"]}',
        )

        self.context_window = self.cfg["model_shell"]["context_window"]
        self.batch_size = self.cfg["trainer"]["training"]["batch_size"]
        self.device = self.cfg["general"]["device"]

    def get_batch(self, split="train"):
        """
        Get a train/val batch
        """
        data = np.memmap(
            os.path.join(self.dataset_path, f"{split}.bin"), dtype=np.uint16, mode="r"
        )

        ix = torch.randint(len(data) - 2*self.context_window, (self.batch_size,))
        X = torch.stack(
            [
                torch.from_numpy((data[i : i + self.context_window]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + self.context_window : i + 2* self.context_window]).astype(np.int64)
                )
                for i in ix
            ]
        )

        X, y = X.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
            self.device, non_blocking=True
        )

        return X, y

    def check_processed(self):
        """
        Check if the data has been preprocessed
        """
        return os.path.exists(self.dataset_path)

    def prepare_data(self, tokenizer):
        """
        Tokenize and store the data
        """
        # create folder
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        else:
            return # already processed

        # load the dataset
        split_dataset = load_data(
            dataset_name=self.cfg["trainer"]["dataset"],
        )

        print(split_dataset.keys())

        def process(example):
            ids = tokenizer.encode_ordinary(example["text"])
            ids.append(tokenizer.eot_token)
            return {"ids": ids, "len": len(ids)}

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=8,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(self.dataset_path, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()






class BytePoolingDataloader:
    """
    A basic dataloader that preprocesses a dataset via
    tokenization, and then randomly loads batches as
    necessary.
    Args:
        cfg: the data config
        preprocess: whether to preprocess the data
    """

    def __init__(self, cfg, data_dir):
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.dataset_path = os.path.join(
            self.data_dir, 
            self.cfg["trainer"]["dataset"],
            "BytePooling",
            f'{self.cfg["model_shell"]["tokenizer"]}-{self.cfg["model_shell"]["vocab_size"]}',
            f'{self.cfg["model_shell"]["pooling_tokenizer"]}-{self.cfg["model_shell"]["pooling_vocab_size"]}',
            self.cfg["trainer"]["dataloader"]["name"]
        )

        self.context_window = self.cfg["model_shell"]["context_window"]
        self.batch_size = self.cfg["trainer"]["training"]["batch_size"]
        self.device = self.cfg["general"]["device"]

    def get_batch(self, split="train"):
        """
        Get a train/val batch
        """
        data = np.memmap(
            os.path.join(self.dataset_path, f"{split}.bin"), dtype=np.uint16, mode="r"
        )

        ix = torch.randint(len(data) - self.context_window, (self.batch_size,))
        X = torch.stack(
            [
                torch.from_numpy((data[i : i + self.context_window]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.context_window]).astype(np.int64)
                )
                for i in ix
            ]
        )

        X, y = X.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
            self.device, non_blocking=True
        )

        return X, y

    def check_processed(self):
        """
        Check if the data has been preprocessed
        """
        return os.path.exists(self.dataset_path)

    def prepare_data(self, byte_tokenizer, pooling_tokenizer):
        """
        Tokenize and store the data
        """
        # create folder
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        else:
            return # already processed

        # load the dataset
        split_dataset = load_data(
            dataset_name=self.cfg["trainer"]["dataset"],
        )

        print(split_dataset.keys())

        def process(example):
            """
            First use the pooling_tokenizer to get the sub-word units,
            decode them, re-encode them using the byte level tokenizer
            and store those as sub-world lists of byte tokens.
            """
            pooling_ids = pooling_tokenizer.encode(example["text"])
            example_tokens = []
            for i in range(len(pooling_ids)):
                # decode individual ids, re-encode them using the byte tokenizer
                sub_word_text = pooling_tokenizer.decode([pooling_ids[i]])
                byte_token_ids = byte_tokenizer.encode(sub_word_text)
                example_tokens.append(byte_token_ids)
            #ids = byte_tokenizer.encode(example["text"])
            #ids.append(byte_tokenizer.eot_token)
            #bounds = 
            """
            At this point, the structure of example_tokens is a list of lists of byte tokens,
            which are in the same size as gpt-2 tokenizer sub-word units.
            """
            return {"ids": example_tokens, "len": len(example_tokens)}

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=8,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(self.dataset_path, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
