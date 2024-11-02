"""
Data processing module for preparing datasets for training.
"""

import os
from tqdm import tqdm
from typing import Any, List
from pathlib import Path

import numpy as np
from datasets import Dataset

from models.experimental.byte_level_copy.custom_embedder.dataset_preparation.data_utils import load_data, get_preprocessed_data_path
from models.experimental.byte_level_copy.custom_embedder.chunker import CustomByteLevelEmbedder

# Base class for data processors
class BasePreProcessor:
    """
    Base class for data processors.
    Provides an interface to process data and write tokenized data to disk.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def process(self, example):
        """
        Tokenizes the input example.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        """
        Writes the tokenized data to disk.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class EmbedderPreProcessor(BasePreProcessor):
    """ TODO """
    def __init__(self, embedder, canonical_tokenizer):
        super().__init__(embedder)
        self.canonical_tokenizer = canonical_tokenizer

    def process(self, sample):
        ids = self.embedder.tokenize_input(sample["text"])
        canonical_tokens_ids = self.canonical_tokenizer.encode(sample["text"])
        canonical_tokens = [
            self.canonical_tokenizer.decode_single_token_bytes(token_id)
            for token_id in canonical_tokens_ids
        ]
        end_chars_pos = self._find_end_characters(canonical_tokens)
        if len(end_chars_pos) != len(ids): # should assert but just inform and ignore for now
            print("WARNING: mismatch between canonical tokenization length and embedder tokenization length.")
            return {"ids": ids, "len": len(ids), "labels": end_chars_pos, 'valid': False}
        return {"ids": ids, "len": len(ids), "labels": end_chars_pos, 'valid': True}

    def write_tokenized_data(self, tokenized: dict, tokenized_data_folder: Path):
        """ TODO """
        for split, dset in tokenized.items():
            filtered_dset = dset.filter(lambda row: row["valid"] == True)
            arr_len = np.sum(filtered_dset["len"], dtype=np.uint64)
            ids_file_name = tokenized_data_folder / f"{split}_ids.bin"
            ids_dtype = np.uint8  # Assumes token IDs are less than 2**8
            arr_ids = np.memmap(
                ids_file_name,
                dtype=ids_dtype,
                mode="w+",
                shape=(arr_len,),
            )
            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {ids_file_name}"):
                # Batch together samples for faster write
                batch = filtered_dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into memory-mapped array
                arr_ids[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr_ids.flush()

            labels_file_name = tokenized_data_folder / f"{split}_labels.bin"
            labels_dtype = np.uint8
            arr_labels = np.memmap(
                labels_file_name,
                dtype=labels_dtype,
                mode="w+",
                shape=(arr_len,),
            )
            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {labels_file_name}"):
                # Batch together samples for faster write
                batch = filtered_dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into memory-mapped array
                arr_labels[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr_labels.flush()

    def _find_end_characters(self, tokens: List[str]) -> List[int]:
        """
        Identify which characters are the end chars of the token they belong to
        """
        binary_values = []
        for canonical_token in tokens:
            binary_token = [0] * (len(canonical_token) - 1) + [1]
            binary_values.extend(binary_token)
        return binary_values


# Dictionary mapping processor names to classes
PROCESSORS = {
    "embedder_preprocessor": EmbedderPreProcessor,
}

def prepare_data(
        cfg: dict,
        embedder: CustomByteLevelEmbedder,
        canonical_tokenizer: Any):
    """ TODO """
    # Extract configuration parameters
    preprocessor_name = cfg["trainer"]["preprocessor_name"]
    dataset_names = cfg["trainer"]["dataset"]

    # Ensure dataset_names is a list
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # Define the tokenized data folder path
    tokenized_data_folder = Path(get_preprocessed_data_path(cfg))

    if tokenized_data_folder.exists():
        print(f"Tokenized data already exists at {tokenized_data_folder}.")
    # create dir if it does not exist
    tokenized_data_folder.parent.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    split_dataset = load_data(dataset_names=dataset_names)

    # Initialize the processor
    processor_class = PROCESSORS.get(preprocessor_name)
    if processor_class is None:
        raise ValueError(f"Processor '{preprocessor_name}' not recognized.")
    processor = processor_class(embedder=embedder, canonical_tokenizer=canonical_tokenizer)

    try:
        # Determine the number of processors to use
        max_procs = min(os.cpu_count(), cfg["general"].get("max_num_cores", 12))
        print(f"Using {max_procs} processes for tokenization.")

        # Tokenize the dataset
        tokenized = split_dataset.map(
            processor.process,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            num_proc=max_procs,
        )

        # Write tokenized data to disk
        processor.write_tokenized_data(
            tokenized=tokenized,
            tokenized_data_folder=tokenized_data_folder,
        )

    except Exception as exc:
        print(f"Error during data processing: {exc}")
        # Clean up partial files
        for file in tokenized_data_folder.iterdir():
            file.unlink()  # Remove each file
        raise RuntimeError("Failed to process and write data.") from exc
