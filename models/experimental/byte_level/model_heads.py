"""
A collection of different model heads.
"""

import torch

from models.components.positional_encoding import LearnedPosEncoding
from models.experimental.byte_level.layers import (
    ByteLevelTransformerBlock,
    GenericTransformerBlockByte
)
from models.components.transformer_blocks import GenericTransformerBlock

from models.components.normalization import build_normalization
import time




class ByteLevelDecoder(torch.nn.Module):
    """
    Use multiple learned heads to decode into by hidden size,
    pre-append to the byte embeddings of the answers and
    autoregressively decode the next token, applying the
    LM (byte level) head only to the actual tokens, not
    the latent ecoded ones.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.hidden_dim = model_cfg["hidden_dim"]
        self.byte_hidden = model_cfg["byte_hidden_dim"]
        self.byte_vocab_size = model_cfg["vocab_size"]
        self.max_chunk_length = model_cfg["max_chunk_length"] + 1 
        self.num_byte_decoder_layers = model_cfg["num_byte_decoder_layers"]

        self.projection = torch.nn.Linear(
            in_features=self.hidden_dim, # 384
            out_features=self.max_chunk_length * self.byte_hidden,
            bias=False,
        )

        # build transformer block
        self.transformer = torch.nn.ModuleList(
            [
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_hidden_dim"],
                    output_dim=model_cfg["byte_hidden_dim"],
                    ffn_dim=model_cfg["byte_hidden_dim"]*4,
                    context_window=model_cfg["max_chunk_length"],
                    attn_cfg=model_cfg["byte_head_attn_dict"]
                ) for _ in range(self.num_byte_decoder_layers)
            ]
        )

        self.lm_head = torch.nn.Linear(
            in_features=self.byte_hidden, # 128
            out_features=self.byte_vocab_size, # 259 (256 bytes + 3 special)
            bias=False,
        )

    def forward(self, x):
        """
        Bidirectionally decode all tokens at once
        """
        # project the latent embeddings
        x = self.projection(x)
        x = x.view(x.size(0), x.size(1), self.max_chunk_length, self.byte_hidden)


        # Reshape for transformer
        B, S, _, _ = x.size()
        x = x.view(B * S, self.max_chunk_length, self.byte_hidden).contiguous()

        # pass through transformer
        for block in self.transformer:
            x = block(x)
        # pass final self.max_chunk_length byte tokens through lm head
        x = self.lm_head(x)

        # reshape and return
        x = x.view(B, S, self.max_chunk_length, self.byte_vocab_size)
        return x

    def inference(self, x):
        """
        inference
        """
        return self.forward(x)[0][:, -1, :, :]