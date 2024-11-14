"""
The standard Model Shell. It combines the embedding model,
core model and LM head.
"""

import torch

from models import core_models, embedding_models, model_heads
from models.model_shell import ModelShell 
import numpy as np
import time 

def cross_entropy_loss_fn(logits, y, ignore_index=-1):
    """
    Cross Entropy Loss Function that ignores specified index.

    Args:
        logits (Tensor): The output logits from the model (B, S, V).
        y (Tensor): The target token IDs (B, S).
        ignore_index (int): The index to ignore in the loss computation.

    Returns:
        Tensor: The computed cross entropy loss.
    """
    # print(logits.?size())
    # torch.Size([2, 641, 14, 259])
    # torch.Size([2, 641, 14])

    # torch.Size([2, 1781, 17, 259])
    # torch.Size([2, 1781, 17])

    # ValueError: Expected input batch_size (3562) to match target batch_size (8192).
    # input(y.size())
    logits = logits.reshape(-1, logits.size(-1))  # (B*S, V)
    y = y.reshape(-1)  # (B*S,)

    # if np.random.uniform() < 0.01:
    #     import matplotlib.pyplot as plt 
    #     plt.bar(range(len(logits[0])), logits[0].float().detach().cpu().numpy())
    #     plt.show()
    # print(logits[:16])
    # print(y[:16])
    # input()
    return torch.nn.functional.cross_entropy(logits, y, ignore_index=ignore_index)

class ByteModelShell(torch.nn.Module):
    """
    Unify the embedding model, core model, and LM head 
    into a single object; initializes the weights
    and prints basic model statistics.
    """

    def __init__(
        self,
        model_cfg,
        embedding_model,
        core_model,
        model_head,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.core_model = core_model
        self.model_head = model_head

        self.device = torch.device("cuda")  # Initialize to CPU or any default device

        # Initialize weights if necessary
        self._initialize_weights()

        # Print basic model statistics
        self._print_model_stats()


        # get loss hyperparameters
        # self.target_chunk_len = model_cfg["target_chunk_len"]
        # self.chunk_len_loss_weight = model_cfg["chunk_len_loss_weight"]
        # self.max_chunk_length = model_cfg["max_chunk_length"]
        # self.chunk_len_penalty = model_cfg["chunk_len_penalty"]



    def _initialize_weights(self):
        """
        Initialize weights of the model components.
        """
        # Example initialization (modify as needed)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def _print_model_stats(self):
        """
        Print basic statistics about the model.
        """
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Embedding Model Parameter Count: {count_parameters(self.embedding_model):,}")
        print(f"Core Model Parameter Count: {count_parameters(self.core_model):,}")
        print(f"Model Head Parameter Count: {count_parameters(self.model_head):,}")


        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")

    # Override the `to` method to set the device attribute
    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)

    def forward(self, token_ids, delimitations, attn_mask = None):
        """
        The default forward pass is used for training and
        accepts the token_ids as input.

        Args:
            token_ids (Tensor): Input token IDs (B, S).

        Returns:
            Tuple[Tensor, Tensor]: The core model output and the loss.
        """
        # Pass the token_ids through the embedding model
        # to get embeddings and target_ids (B, S, H) and (B, S)
        embeddings, target_ids, avg_chunk_len = self.embedding_model(token_ids, delimitations) 
        # print(embeddings.size())
        # print(f"Embeddings: {embeddings.size()}") # [2, 635, 384]
        # print(f"target_ids: {target_ids.size()}") # [2, 635, 14]
        # print(f"Avg. chunk len: {avg_chunk_len}")
        # exit()

        # autoregressive (ad'hoc)
        embeddings = embeddings[:, :-1]
        target_ids = target_ids[:, 1:]
        # print(f"Embeddings: {embeddings.size()}") # [2, 635, 384]
        # print(f"target_ids: {target_ids.size()}") # [2, 635, 14]
        # input()
        # Pass the embeddings through the core model
        core_output = self.core_model(embeddings, attn_mask)

        # Pass the core model output through the model head to get logits (B, S, V)
        logits = self.model_head(core_output)

        # input(logits.size()) # [2, 3456, 6, 259]
        # input(target_ids.size()) # [2, 3456, 6]
        # input(avg_chunk_len)
        # exit()
        
        # print(logits.size())
        # print(target_ids.size())
        # print(torch.argmax(logits[:, -1], -1))
        # print(target_ids[:, -1])


        # print(torch.argmax(logits[:, -1], -1))
        # print(target_ids[:, -1])


        # print(torch.argmax(logits[:, -2], -1))
        # print(target_ids[:, -2])


        # print(torch.argmax(logits[:, -3], -1))
        # print(target_ids[:, -3])


        # # just get the first byte
        # logits = torch.argmax(logits[:, :, 0], dim=-1)
        # targets = target_ids[:, :, 0]

        # print(logits)
        # input(targets)

        # input(self.embedding_model.pad_token)

        # Compute the loss, ignoring pad tokens
        loss = cross_entropy_loss_fn(
            logits=logits,
            y=target_ids.long(),
            ignore_index=self.embedding_model.pad_token
        )

        # Aux loss 1: Target Chunk length
        # chunk_loss = chunk_len_loss #* (avg_chunk_len - self.target_chunk_len) ** 2

        # Aux loss 2: Max Chunk length
        # over_length = torch.clamp(
        #     avg_chunk_len-self.max_chunk_length,
        #     min=0
        # )
        # length_loss = torch.sum(over_length)


        total_loss = loss #+ \
        # input(total_loss)
        #    self.chunk_len_loss_weight * chunk_loss #+ \
        #self.chunk_len_penalty * length_loss

        #print(f"Total Loss: {total_loss}, Chunk Loss: {chunk_loss}, BCE Loss: {loss}")

        additional_info_dict = {
            "average_chunk_length": avg_chunk_len.item(),
            # "chunk_len_loss": self.chunk_len_loss_weight*chunk_loss,
            # "chunk_len_penalty_loss": self.chunk_len_penalty*length_loss,
            "BCE-loss": loss,
            "total-loss": total_loss,
        }
        print(avg_chunk_len.item())
        return logits, total_loss #, additional_info_dict

    @torch.no_grad()
    def inference(self, model_input):
        """
        Takes a string or list of token ids as input,
        and returns the decoded model output. The actual
        decoding should happen in the decoding generator.
        Args:
            model_input: str or torch.tensor(B, S)
        Returns:
            logits: torch.tensor(B, S, V),
        """
        # check if input is string
        if isinstance(model_input, str):
            # use inference function of the embedding model
            model_input = self.embedding_model.tokenize_input(model_input, truncate=True, add_eot=False)

        if isinstance(model_input, torch.Tensor):
            x = model_input.to(device=self.device, dtype=torch.long).unsqueeze(0)
        else:
            x = torch.tensor(model_input, device=self.device, dtype=torch.long).unsqueeze(0)
        
        embeddings, target_ids, avg_chunk_len = self.embedding_model(model_input)

        core_output = self.core_model(embeddings, None)

        logits = self.model_head.inference(core_output)
        return logits, model_input
