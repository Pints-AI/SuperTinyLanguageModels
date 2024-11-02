"""
Data processing module for preparing datasets for training.
"""

import os
from tqdm import tqdm
from typing import Any, List
import random
from pathlib import Path

import numpy as np
from datasets import Dataset

from models.experimental.byte_level_copy.custom_embedder.dataset_preparation.data_utils import load_data, get_preprocessed_data_path
from models.build_models import build_embedding_model
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
    def __init__(self, embedder):
        super().__init__(embedder)

    def process(self, sample):
        bytes_tokens = self.embedder.tokenize_input(sample["text"])
        return {"ids": ids, "len": len(ids)}

    def write_tokenized_data(self, tokenized, tokenized_data_folder):
        """ TODO """
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(tokenized_data_folder, f"{split}.bin")
            dtype = np.uint16  # Assumes token IDs are less than 2**16
            arr = np.memmap(
                filename,
                dtype=dtype,
                mode="w+",
                shape=(arr_len,),
            )
            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into memory-mapped array
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush() 


# Dictionary mapping processor names to classes
PROCESSORS = {
    "embedder_preprocessor": EmbedderPreProcessor,
}

def prepare_data_for_model(
        embedder: CustomByteLevelEmbedder,
        canonical_tokenizer: Any,
        cfg: dict, 
        dataset: Dataset,
        preprocess_data_folder: Path):
    # Extract configuration parameters
    preprocessor_name = cfg["trainer"]["preprocessor_name"]

    # randomise data; create new
    shuffled_dataset = dataset.shuffle()

    # create dir if it does not exist
    preprocess_data_folder.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the processor
    processor_class = PROCESSORS.get(preprocessor_name)
    if processor_class is None:
        raise ValueError(f"Processor '{preprocessor_name}' not recognized.")
    processor = processor_class(embedder=embedder)

    try:
        # Determine the number of processors to use
        max_procs = min(os.cpu_count(), cfg["general"].get("max_num_cores", 12))
        print(f"Using {max_procs} processes for tokenization.")

        # Tokenize the dataset
        tokenized = shuffled_dataset.map(
            processor.process,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            num_proc=max_procs,
        )

        # Write tokenized data to disk
        processor.write_tokenized_data(
            tokenized=tokenized,
            tokenized_data_folder=preprocess_data_folder,
        )

    except Exception as exc:
        print(f"Error during data processing: {exc}")
        # Clean up partial files
        for file in preprocess_data_folder.iterdir():
            file.unlink()  # Remove each file
        raise RuntimeError("Failed to process and write data.") from exc
