import os
from typing import List
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from models.components.tokenizers import utils
from models.components.tokenizers.base_class import Tokenizer as BaseTokenizer
from trainers.utils import load_data

class BPETokenizer(BaseTokenizer):
    """Tokenizer for Byte Pair Encoding using Hugging Face tokenizers library."""

    def __init__(self, vocab_size: int, dataset_name: str):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size (int): The desired vocabulary size.
            dataset_name (str): The name of the dataset to use for training.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dataset_name = dataset_name
        self.special_tokens = {
            "<|pad|>": vocab_size - 3,
            "<|endoftext|>": vocab_size - 2,
            "<|unk|>": vocab_size - 1,
        }
        self.pad_token = self.special_tokens["<|pad|>"]
        self.eot_token = self.special_tokens["<|endoftext|>"]
        self.unk_token = self.special_tokens["<|unk|>"]

        assert self.vocab_size >= 256 + len(self.special_tokens), \
            f"Vocab size too small! Must be > {256 + len(self.special_tokens)})"

        if not utils.check_if_tokenizer_exists(
            tokenizer_type="bpe", vocab_size=vocab_size, dataset_name=dataset_name
        ):
            self._train_tokenizer()
            self._save()
        else:
            self._load()

    def encode(self, text: str) -> List[int]:
        """
        Encode the text into BPE tokens.
        
        Args:
            text (str): The input text to encode.
        
        Returns:
            List[int]: The list of token ids.
        """
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts into BPE tokens.
        
        Args:
            texts (List[str]): The list of input texts to encode.
        
        Returns:
            List[List[int]]: The list of token id lists.
        """
        return [self.encode(text) for text in texts]

    def decode(self, tokens: List[int]) -> str:
        """
        Decode the BPE tokens back into text.
        
        Args:
            tokens (List[int]): The list of token ids to decode.
        
        Returns:
            str: The decoded text.
        """
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        Decode a batch of BPE token lists back into text.
        
        Args:
            token_lists (List[List[int]]): The list of token id lists to decode.
        
        Returns:
            List[str]: The list of decoded texts.
        """
        if torch.is_tensor(token_lists):
            token_lists = token_lists.tolist()
        return [self.decode(tokens) for tokens in token_lists]

    def _train_tokenizer(self, verbose: bool = True):
        """
        Train the BPE tokenizer on the given dataset.
        
        Args:
            verbose (bool): Whether to show progress during training.
        """
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            min_frequency=2
        )

        dataset = load_data(dataset_name=self.dataset_name)
        files = [dataset["train"]["text"]]  # Adjust based on your dataset structure

        tokenizer.train(files, trainer)

        self.tokenizer = tokenizer
        self.vocab = {id: token for token, id in tokenizer.get_vocab().items()}
        self.merges = tokenizer.model.merges

    def _save(self):
        """
        Save the tokenizer as a .json file.
        """
        tokenizer_folder, tokenizer_path = utils.get_tokenizer_path(
            tokenizer_type="bpe",
            vocab_size=self.vocab_size,
            dataset_name=self.dataset_name,
        )
        if not os.path.exists(tokenizer_folder):
            os.makedirs(tokenizer_folder)

        self.tokenizer.save(tokenizer_path)

    def _load(self):
        """
        Load the tokenizer from a .json file.
        """
        _, tokenizer_path = utils.get_tokenizer_path(
            tokenizer_type="bpe",
            vocab_size=self.vocab_size,
            dataset_name=self.dataset_name,
        )

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab = {id: token for token, id in self.tokenizer.get_vocab().items()}
        self.merges = self.tokenizer.model.merges