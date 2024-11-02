"""
Experimental file for the following helper functions:
1. compute_loss_for_end_token - computes the loss between predicted and target OHE labels (that denote end bytes)
2. canonical tokenizer to get target tokens
3. 

workflow:
0. convert input string into raw bytes (using BPE via build_tokenizer) --> ?
1. given raw bytes, use embedder to get the embeddings for each byte
2. use delimiter model to split into chunks of embeddings (output)
3. end bytes classifier

1. select canonical tokenizer (tiktokenizer o200k_base)
2. tokenize input string into tokens
3. use our embedder to convert each token
4. do OHE to represent whether each original byte is an 'end token'

compute loss between the two OHE

"""
import torch
import torch.nn.functional as F
from typing import List

import tiktoken

from models.embedding_models import EmbedderInterface
from models.components.tokenizers import build_tokenizer, TokenizerClass
from models.components.transformer_blocks import GenericTransformerBlock


INPUT = [
    # "i like machine learning",
    # "anton is running",
    "finetuning LLMs is as simple as ordering a pint of beer!"
]

# This config is from byte_autoencoder_3.yaml. Copy-pasted for now.
CONFIG = {
    "model": {
        "core_model_type": "pass_through",
        "hidden_dim": 384,
        "byte_hidden": 96,
        "max_chunk_length": 12,
        "max_num_chunks": 1024,
        "num_delimiter_layers": 3,
        "num_byte_decoder_layers": 5,
        "target_chunk_len": 8.0,
        "chunk_len_loss_weight": 0.5,
        "chunk_len_penalty": 0.1,
        "context_window": 8192,
        "embedding_model_type": "byte_level",
        "tokenizer_type": "bpe",
        "tokenizer_dataset_name": "simple_en_wiki",
        "tokenizer_simplify_data": True,
        "vocab_size": 259,
        "lm_head_type": "byte_level",
        "lm_head_normalization": "rms_norm",
        "lm_head_bias": False,
        "lm_head_dropout": 0.0,
        "model_shell_type": "byte_autoencoder_shell",
        "embedding_weight_tying": True,
        "ffn_weight_tying": False,
        "cproj_weight_tying": False,
        "positional_encoding_type": "rope"
    },
    "trainer": {
        "preprocessor_name": "embedder_preprocessor",
        "trainer_type": "base_trainer",
        "dataset": "simple_en_wiki",
        "batch_size": 12,
        "gradient_accumulation_steps": 4,
        "max_iters": 10000,
        "eval_interval": 50000000,
        "log_interval": 1,
        "checkpoint_interval": 1000,
        "eval_iters": 1000,
        "run_eval": False,
        "eval": {
            "mcq_benchmarks": None,
            "mcq_num_samples": 1000,
            "eval_byte_metrics": False,
            "text_modeling_eval": False,
            "text_generation_eval": False
        },
        "optimizer": {
            "optimizer_name": "adamW",
            "lr": 5.0e-4,
            "min_lr": 5.0e-5,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0
        },
        "lr_scheduler": {
            "name": "cosine",
            "warmup_iters": 100
        },
        "dataloader": {
            "name": "autoencoder"
        },
        "datasampling": {
            "name": "standard"
        },
        "loss_fn": {
            "name": "pass_through"
        }
    },
    "general": {
        "logging": {
            "wandb_log": True,
            "wandb_project": "SuperTinyLanguageModels",
            "wandb_run_name": "Null",
            "group_name": "experimental_byte_level"
        },
        "paths": {
            "output_dir": "outputs",
            "data_dir": "data",
            "checkpoint_dir": "checkpoints",
            "eval_dir": "evals"
        },
        "seed": 489,
        "device": "cpu"
    }
}

def compute_loss_for_end_token(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    assert predictions.shape == targets.shape, (
        "Something went wrong when computing loss! "
        "<predictions> and <targets> should have the same shape."
    )
    
    mse_loss = F.mse_loss(predictions, targets, reduction='none')
    return mse_loss.mean(dim=tuple(range(1, mse_loss.ndim)), keepdim=True)


def initialize_byte_tokenizer():
    return build_tokenizer(
        tokenizer_type="bpe",
        vocab_size=259,
        simplify=False,
        dataset_name="simple_en_wiki",
    )

def conversion_to_bytes(
    byte_tokenizer: TokenizerClass,
    string: str
) ->  List[bytes]:
    return byte_tokenizer.encode(string)

def get_canonical_tokenizer(tokenizer: str='o200k_base'):
    return tiktoken.get_encoding(tokenizer)


# Referenced from ByteLevelEmbedder
class CustomByteLevelEmbedder(EmbedderInterface):
    """
    Input is a sequence of byte-level token ids
    """

    def __init__(self, model_cfg: dict, device: str="cpu"):
        super().__init__()
        self.model_cfg = model_cfg

        self.max_chunk_length = self.model_cfg["max_chunk_length"]
        self.max_num_chunks = self.model_cfg["max_num_chunks"]
        self.byte_hidden = self.model_cfg["byte_hidden"]
        self.hidden_dim = self.model_cfg["hidden_dim"]
        self.num_delimiter_layers = self.model_cfg["num_delimiter_layers"]


        self.byte_tokenizer = build_tokenizer(
            tokenizer_type="bpe",
            vocab_size=model_cfg["vocab_size"],
            simplify=False,
            dataset_name="simple_en_wiki",
        )

        self.byte_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=model_cfg["byte_hidden"],
            device=device
        ) #259*32  

        self.delimiter_model = EndByteClassifier(
            byte_hidden=self.byte_hidden,
            num_delimiter_layers=self.num_delimiter_layers,
        )

        # Store pad_token_id and eot_token as class attributes
        self.pad_token_id = self.byte_tokenizer.pad_token
        self.eot_token = self.byte_tokenizer.eot_token

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len)
        x_embedded = self.byte_embedder(x)  # (batch_size, seq_len, byte_hidden)

        # Pass through delimiter model
        probs_end_tokens = self.delimiter_model(
            x=x_embedded,
        )
        return probs_end_tokens

    def tokenize_input(self,
                       input_string:str,
                       truncate:bool=False,
                       add_eot:bool=False) -> List[int]:
        token_ids = self.byte_tokenizer.encode(input_string)
        if add_eot:
            token_ids.append(self.eot_token)

        return token_ids

    def decode(self, tokens: List[int]):
        """ Decode a tensor of tokens into a string. """
        return self.byte_tokenizer.decode(tokens)


# Referenced from TokenizerEncoder
class EndByteClassifier(torch.nn.Module):
    """
    Take seq of byte embeddings, return transformed sequence and attention mask.
    """

    def __init__(self, num_delimiter_layers: int, byte_hidden: int):
        super().__init__()

        self.num_delimiter_layers = num_delimiter_layers
        self.byte_hidden = byte_hidden

        self.transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=self.byte_hidden,
                    context_window=2048, #5 * 2048,
                    ffn_cfg={
                        "ffn_type": "generic",
                        "ffn_dim": 4 * self.byte_hidden,
                        "activation": "gelu",
                        "normalization": "rms_norm",
                        "bias": False,
                        "dropout": 0.0,
                    },
                    attn_cfg={
                        "attn_type": "causal",
                        "num_kv_heads": 8,
                        "num_q_heads": 8,
                        "normalization": "rms_norm",
                        "bias": False,
                        "dropout": False,
                        "pos_enc_cfg": {
                            "positional_encoding_type": "rope"
                        }
                    },
                )
                for _ in range(self.num_delimiter_layers)
            ]
        )

        self.end_of_seq_head = torch.nn.Linear(
            self.byte_hidden,
            1,  # Output logits for delimiter prediction
            bias=True,
        ) # [Hello ][World!] (12, 32) -> (12, 1) 

    def forward(self, x):
        # Pass through transformer blocks
        x_transformed = x
        for block in self.transformer:
            x_transformed = block(x_transformed)

        # Predict delimiters
        logits = self.end_of_seq_head(x_transformed).squeeze(-1)  # Shape: (batch, seq_len)

        # Apply sigmoid activation
        probs = torch.sigmoid(logits)

        return probs
    
def find_end_characters(tokens: List[str]) -> List[int]:
    """
    Identify which characters are the end chars of the token they belong to
    """
    binary_values = []
    for canonical_token in tokens:
        binary_token = [0] * (len(canonical_token) - 1) + [1]
        binary_values.extend(binary_token)
    return binary_values


def start(cfg):
    custom_embedder = CustomByteLevelEmbedder(cfg['model'], cfg['general']['device'])
    # user_input = input("Insert string: ")
    user_input = INPUT
    batch_input_bytes_tokens: List[List[int]] = []
    for input_string in user_input:
        input_bytes_tokens = custom_embedder.tokenize_input(input_string)
        batch_input_bytes_tokens.append(input_bytes_tokens)
        print(len(input_bytes_tokens))

    end_bytes_probs = custom_embedder(torch.tensor(batch_input_bytes_tokens))
    print("Using Custom Embedder:")
    print("="*80)
    print(f"Input bytes tokens: {input_bytes_tokens}")
    print(f"Output: {end_bytes_probs}")
    print("="*80)

    canonical_tokenizer = get_canonical_tokenizer()
    batch_canonical_tokens: List[bytes] = []
    batch_end_chars_pos: List[int] = []
    for input_string in user_input:
        canonical_tokens_ids = canonical_tokenizer.encode(input_string)
        canonical_tokens = [
            canonical_tokenizer.decode_single_token_bytes(token_id)
            for token_id in canonical_tokens_ids
        ]
        end_chars_pos = find_end_characters(canonical_tokens)

        batch_canonical_tokens.append(canonical_tokens)
        batch_end_chars_pos.append(end_chars_pos)

    print("Using Canonical Tokenizer: ")
    print("="*80)
    print(f"Canonical tokens:")
    for canonical_tokens, end_chars_pos in zip(batch_canonical_tokens, batch_end_chars_pos):
        print(canonical_tokens)
        print(end_chars_pos)
        print(len(end_chars_pos))
    print("="*80)
    print(f"Loss value: {compute_loss_for_end_token(torch.tensor(batch_end_chars_pos), end_bytes_probs)}")


if __name__ == "__main__":
    start(CONFIG) # from byte_autoencoder_3 yaml file
    