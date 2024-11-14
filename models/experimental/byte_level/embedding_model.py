"""
A collection of embedding models. A collection model includes
the tokenizer(s), token embeddings and positional encodings
(if necessary).
"""

import torch

from models.components.positional_encoding import LearnedPosEncoding
from models.components.tokenizers import build_tokenizer
from models.embedding_models import EmbedderInterface
from models.experimental.byte_level.layers import ByteLevelTransformerBlock

from models.components.transformer_blocks import GenericTransformerBlock
from copy import deepcopy



# TODO
# - Different eot tokens for chunks and text 
# 



# 1. delimitation (byte, idx -> reshaped byte idx and target ids)
# 2. Byte embedder (reshaped byte idx -> global token embeddings)
class DelimiterModel(torch.nn.Module):
    """ TODO """
    def __init__(self, model_cfg):
        super().__init__()

        self.byte_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=model_cfg["byte_hidden_dim"],
        )

        self.transformer = torch.nn.ModuleList(
            [
                GenericTransformerBlock(
                    hidden_dim=model_cfg["byte_hidden_dim"],
                    context_window=model_cfg["context_window"],
                    ffn_cfg=model_cfg["delimiter_model"]["ffn"],
                    attn_cfg=model_cfg["delimiter_model"]["attn"],
                    depth=i
                )
                for i in range(model_cfg["delimiter_model"]["num_layers"])
            ]
        )

        self.end_of_seq_head = torch.nn.Linear(
            model_cfg["byte_hidden_dim"],
            1, # prob of sequence ending
        )
        self.device = "cuda" #model_cfg['general']['device']
        # self._load_model_weights() # loaded at train.py: 68 to avoid pytorch reinitialization


    def _load_model_weights(self):
        """ TODO """
        path = "../../../checkpoints/embedder_model_weights_wiki_en.pth"

        # load weight dict
        weight_dict = torch.load(path)
        weights = {k.replace("module.", ""): v for k, v in weight_dict.items()}

        transformer_weight_dict = {}
        embedder_weight_dict = {}
        head_dict = {}
        for weight_key, weight in weights.items():
            if "transformer" in weight_key:
                transformer_weight_dict[weight_key.replace("delimiter_model.", "").replace("transformer.", "")] = weight 
            elif "byte_embedder" in weight_key:
                embedder_weight_dict[weight_key.replace("byte_embedder.", "")] = weight
            elif "end_of_seq_head" in weight_key:
                head_dict[weight_key.replace("delimiter_model.", "").replace("end_of_seq_head.", "")] = weight

        # load both
        self.transformer.load_state_dict(transformer_weight_dict, strict=False)
        self.transformer.to(self.device)
        self.byte_embedder.load_state_dict(embedder_weight_dict)
        self.byte_embedder.to(self.device)
        self.end_of_seq_head.load_state_dict(head_dict)
        self.end_of_seq_head.to(self.device)

        # print("original", self.end_of_seq_head.weight)

        # Freeze all weights
        for param in self.transformer.parameters():
            param.requires_grad = False 
        for param in self.byte_embedder.parameters():
            param.requires_grad = False 
        for param in self.end_of_seq_head.parameters():
            param.requires_grad = False 



    def forward(self, x):
        # pass throug embedder 
        x = self.byte_embedder(x)
        # Pass through transformer blocks 
        for block in self.transformer:
            x = block(x)

        # Predict delimitations
        logits = self.end_of_seq_head(x).squeeze(-1) # Shape: (batch, seq_len)

        # Apply sigmoid 
        probs = torch.sigmoid(logits)
        return probs
        


class ByteBidirectionEncoding(torch.nn.Module):
    """ TODO """
    def __init__(self, model_cfg):
        super().__init__()

        # build the transformer blocks
        self.transformer = torch.nn.ModuleList(
            [
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_hidden_dim"],
                    output_dim=model_cfg["byte_hidden_dim"],
                    ffn_dim=model_cfg["byte_hidden_dim"]*4,
                    context_window=model_cfg["max_chunk_length"],
                    attn_cfg=model_cfg["chunk_encoding_attn_dict"]
                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_hidden_dim"],
                    output_dim=model_cfg["byte_hidden_dim"],
                    ffn_dim=model_cfg["byte_hidden_dim"]*4,
                    context_window=model_cfg["max_chunk_length"],
                    attn_cfg=model_cfg["chunk_encoding_attn_dict"]
                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_hidden_dim"],
                    output_dim=model_cfg["byte_hidden_dim"]*4,
                    ffn_dim=model_cfg["byte_hidden_dim"]*8,
                    context_window=model_cfg["max_chunk_length"],
                    attn_cfg=model_cfg["chunk_encoding_attn_dict"]
                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_hidden_dim"]*4,
                    output_dim=model_cfg["byte_hidden_dim"]*4,
                    ffn_dim=model_cfg["byte_hidden_dim"]*16,
                    context_window=model_cfg["max_chunk_length"],
                    attn_cfg=model_cfg["chunk_encoding_attn_dict"]
                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_hidden_dim"]*4,
                    output_dim=model_cfg["byte_hidden_dim"]*4,
                    ffn_dim=model_cfg["byte_hidden_dim"]*16,
                    context_window=model_cfg["max_chunk_length"],
                    attn_cfg=model_cfg["chunk_encoding_attn_dict"]
                ),
                ByteLevelTransformerBlock(
                    input_dim=model_cfg["byte_hidden_dim"]*4,
                    output_dim=model_cfg["hidden_dim"],
                    ffn_dim=model_cfg["byte_hidden_dim"]*16,
                    context_window=model_cfg["max_chunk_length"],
                    attn_cfg=model_cfg["chunk_encoding_attn_dict"]
                ),
            ]
        )


    def forward(self, x):
        # B, Num_chunck, Chunck_len, 128 
        B, C_num, C_len, h_b = x.size()
        # flatten first two dims 
        x = x.view(B*C_num, C_len, h_b)

        # pass through blocks
        for block in self.transformer:
            x = block(x)

        x = x.mean(-2) # TODO: should ignore the pad tokens
        # reshape it back to 3
        x = x.view(B, C_num, -1)  # batch, chunk_num, 512

        return x 


class ByteLevelEmbedder(EmbedderInterface):
    """ TODO """
    def __init__(self, model_cfg):
        super().__init__()
        self.max_chunk_length = model_cfg["max_chunk_length"]

        # Initialize the byte tokenizer
        self.byte_tokenizer = build_tokenizer(
            tokenizer_type="byte",
            vocab_size=model_cfg["vocab_size"],
            simplify=False,
            dataset_names=None,
            num_reserved_tokens=0,
        )
        

        # Initialize the byte embedding 
        self.byte_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=model_cfg["byte_hidden_dim"],
        )

        self.pad_token = self.byte_tokenizer.pad_token
        self.eot_token = self.byte_tokenizer.eot_token
        # eot_token_embedding = self.byte_embedder(torch.tensor([self.eot_token], device="cuda"))[0]
        # pad_token_embedding = self.byte_embedder(torch.tensor([self.pad_token], device="cuda"))[0]

        # self.register_buffer("eot_token_embedding", eot_token_embedding)
        # self.register_buffer("pad_token_embedding", pad_token_embedding)
        # Initialize the delimiter model
        self.delimiter_model = DelimiterModel(model_cfg=model_cfg)
        # print("Delimeter:", self.delimiter_model.end_of_seq_head.weight)

        # Initialize the bidirectional embedding model
        self.chunk_encoding_model = ByteBidirectionEncoding(
            model_cfg=model_cfg
        )


    def _reshape_sequence(self, x_embedded, x_ids, delimitation_probs):
        """ TODO """
        # turn probs into mask
        end_of_chunk = delimitation_probs > 0.5

        batch_size, seq_len = end_of_chunk.size()
        device = x_embedded.device

        chunk_indices = []
        avg_chunk_len = []
        max_observed_chunk_len = 0
        for batch in range(batch_size):
            # Get predicted end positions
            ends = torch.nonzero(end_of_chunk[batch], as_tuple=False).squeeze(-1)

            # If no ends detected, set to the last token
            if ends.numel() == 0:
                ends = torch.tensor([seq_len-1], device=device)
            else:
                ends = ends + 1 # Adjust to point after the chunk end

            # Initialize starts
            starts = torch.cat([torch.tensor([0], device=device), ends[:-1]])

            chunk_lengths = ends-starts
            max_observed_chunk_len = max(max_observed_chunk_len, chunk_lengths.max())
            chunk_indices.append((starts, ends))
            avg_chunk_len.append(chunk_lengths.float().mean())


        # Find the max num chunks and max chunk length from the chunked data
        max_num_chunks = torch.sum(end_of_chunk, dim=-1).max()
        max_chunk_length = self.max_chunk_length + 1#min(max_observed_chunk_len, self.max_chunk_length) + 1
        # Initialize output tensors by repeating pad_token_vector
        # output_tensor = self.pad_token_embedding.repeat(
        #     batch_size, 
        #     max_num_chunks, 
        #     max_chunk_length, 
        #     1
        # )

        output_tensor = self.byte_embedder(torch.tensor([self.pad_token], device="cuda"))[0].repeat(
            batch_size, 
            max_num_chunks, 
            max_chunk_length, 
            1
        )


        # Initialize output_token_ids with pad_token_id
        output_token_ids = torch.full(
            (batch_size, max_num_chunks, max_chunk_length),
            self.byte_tokenizer.pad_token,
            device=device,
            dtype=torch.long,
        )

        # Populate output_tensor and output_token_ids with actual chunk data 
        for batch in range(batch_size):
            starts, ends = chunk_indices[batch]
            num_chunks = min(len(ends), max_num_chunks)
            for i in range(num_chunks):
                chunk = x_embedded[batch, starts[i]:ends[i], :]
                chunk_ids = x_ids[batch, starts[i]:ends[i]]
                chunk_len = chunk.size(0)

                if chunk_len >= max_chunk_length:
                    chunk = chunk[:max_chunk_length-1]
                    chunk_ids = chunk_ids[:max_chunk_length-1]
                    chunk_len = chunk.size(0) # max_chunk_length


                output_tensor[batch, i, :chunk_len, :] = chunk 
                # add end token
                output_tensor[batch, i, chunk_len, :] = self.byte_embedder(torch.tensor([self.eot_token], device="cuda"))[0] #self.eot_token_embedding

                # add output ids
                output_token_ids[batch, i, :chunk_len] = chunk_ids 
                # add end token
                output_token_ids[batch, i, chunk_len] = self.eot_token

        return output_tensor, output_token_ids, sum(avg_chunk_len)/len(avg_chunk_len)



    def forward(self, x, delimitations):
        # delimit the sequence
        # delimitation_probs = self.delimiter_model(x)
        delimitation_probs = torch.tensor(delimitations, device=x.device)

        # embed x
        x_embedded = self.byte_embedder(x)

        # reshape based on delimitations
        x_reshaped, x_ids, avg_chunk_len = self._reshape_sequence(
            x_embedded=x_embedded,
            x_ids=x,
            delimitation_probs=delimitation_probs
        )

        # push to the same device as x
        x_reshaped = x_reshaped.to(x.device)
        x_ids = x_ids.to(x.device)

        # pass through the actual byte embedding model
        x_global = self.chunk_encoding_model(x_reshaped)

        return x_global, x_ids, avg_chunk_len

    def tokenize_input(self, input_string, truncate=False, add_eot=True):
        token_ids = self.byte_tokenizer.encode(input_string)
        # if add_eot:
        #     token_ids.append(self.eot_token)
        return token_ids

    def decode(self, tokens):
        """ Decode a tensor of tokens into a string. """
        return self.byte_tokenizer.decode_batch(tokens)
        
