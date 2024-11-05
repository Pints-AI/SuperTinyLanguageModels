"""
Simple LoRA FFN weight sharing but with softmax-weighted experts.
"""
import torch 
import math 
from models.core_models import GenericTransformer

from models.components.attention import build_attention
from models.components.feedforward import build_ffn
from models.components.normalization import build_normalization

from models.components.activations import build_activation

from typing import Optional


class MoELoRA(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            lora_rank: int,
            n_experts: int,
            lora_alpha: float = 1.0,
            global_gating: bool = False
        ):
        """
        LoRA MoE implementation

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            lora_rank (int): The rank of the LoRA matrices.
            n_experts (int): Number of experts.
            lora_alpha (float): Scaling factor for the LoRA update.
            global_gating (bool): If True, use the same expert for all sequence items.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.lora_rank = lora_rank 
        self.n_experts = n_experts
        self.lora_alpha = lora_alpha
        self.global_gating = global_gating
        self.scaling = self.lora_alpha / self.lora_rank

        # Initialize main weight matrix
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))

        # Initialize gate linear
        self.gate_linear = torch.nn.Linear(in_features, n_experts, bias=False)

        # Initialize LoRA matrices
        self.lora_experts_U = torch.nn.Parameter(
            torch.empty((n_experts, lora_rank, in_features))
        )
        self.lora_experts_V = torch.nn.Parameter(
            torch.zeros((n_experts, out_features, lora_rank))
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.gate_linear.reset_parameters()

        for i in range(self.n_experts):
            torch.nn.init.kaiming_uniform_(self.lora_experts_U[i], a=math.sqrt(5))
            # V is already initialized to zeros
            torch.nn.init.kaiming_uniform_(self.lora_experts_V[i], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Forward pass where the gating is applied to each element in the 
        sequence independently
        """
        if self.global_gating:
            gate = torch.nn.functional.softmax(self.gate_linear(x[:,-1]), dim=-1) # torch.Size([2, 8]) torch.Size([8, 32, 416]) torch.Size([8, 1072, 32])
            
            lora_weights = self.lora_experts_V @ self.lora_experts_U
            updated_weight = self.weight + self.scaling * lora_weights
            #print(x.size(), updated_weight.size()) # torch.Size([2, 512, 416]) torch.Size([8, 1072, 416])
            #input()
            output = torch.einsum('bsh,efh->besf', x, updated_weight) # torch.Size([2, 8, 1072])
            output = torch.sum(gate.unsqueeze(2).unsqueeze(3) * output, dim=1)
            return output 
            input(output.size())

            lora_weights_U = torch.einsum('bi,ijk->bjk', gate, self.lora_experts_U)  # resulting shape: [2, 32, 416]
            lora_weights_V = torch.einsum('bi,ijk->bjk', gate, self.lora_experts_V)  # resulting shape: [2, 1072, 32]
            # combine
            lora_weights = torch.einsum('bij,bjk->bik', lora_weights_V, lora_weights_U)  # resulting shape: [2, 1072, 416]
            # apply
            updated_weight = self.weight + self.scaling * lora_weights
            #print(x.size(), updated_weight.size()) # torch.Size([2, 512, 416]) torch.Size([2, 1072, 416])
            #input()
            output = torch.einsum("bsh,bfh->bsf", x, updated_weight)
            return output
        else:
            gate = torch.nn.functional.softmax(self.gate_linear(x), dim=-1) #torch.Size([24, 512, 8])
            B, S, E = gate.size()

            # flatten gate along the B & S dim
            gate = gate.view(-1, E) # torch.Size([12288, 8]) 
            x = x.view(-1, self.in_features) # torch.Size([12288, 416])

            lora_weights = self.lora_experts_V @ self.lora_experts_U # torch.Size([8, 1072, 416])
            # Add LoRA update to the main weight
            updated_weight = self.weight + self.scaling * lora_weights # torch.Size([8, 1072, 416])
            # apply weights
            #print(updated_weight.size(), x.size()) # torch.Size([8, 1072, 416]) torch.Size([1024, 416])
            output = torch.einsum('ijk,bk->bij', updated_weight, x) # torch.Size([1024, 8, 1072])
            # apply the gates across the second dim and pool
            #print(gate.size(), output.size()) # torch.Size([1024, 8]) torch.Size([1024, 8, 1072])
            output = torch.sum(gate.unsqueeze(2) * output, dim=1)
            #output = torch.einsum('ij,ikj->ik', gate, output) # torch.Size([12288, 1072])
            return output.view(B, S, self.out_features)

            input(output.size())
            output = torch.einsum('ij,ikj->ik', x, updated_weight) # torch.Size([12288, 1072])
            input(updated_weight.size())
            lora_weights_U = gate.unsqueeze(2).unsqueeze(3) * self.lora_experts_U # torch.Size([12288, 8, 32, 416])
            # now average over experts
            lora_weights_U = lora_weights_U.mean(dim=1) # torch.Size([12288, 1072, 416])

            lora_weights_V = gate.unsqueeze(2).unsqueeze(3) * self.lora_experts_V # torch.Size([12288, 8, 1072, 32])
            # now average over experts
            lora_weights_V = lora_weights_V.mean(dim=1) # torch.Size([12288, 1072, 32])

            lora_weights = lora_weights_V @ lora_weights_U # torch.Size([12288, 1072, 416])
            # Add LoRA update to the main weight
            updated_weight = self.weight + self.scaling * lora_weights


            # apply weights
            output = torch.einsum('ij,ikj->ik', x, updated_weight)
            return output.view(B, S, self.out_features)
    

class SharedMoEFFN(torch.nn.Module):
    """ """
    def __init__(self, hidden_dim, ffn_dim, lora_rank, n_experts, lora_alpha):
        super().__init__()
        self.linear_1 = MoELoRA(
            in_features=hidden_dim,
            out_features=ffn_dim,
            lora_rank=lora_rank,
            n_experts=n_experts,
            lora_alpha=lora_alpha
        )

        self.linear_2 = MoELoRA(
            in_features=ffn_dim,
            out_features=hidden_dim,
            lora_rank=lora_rank,
            n_experts=n_experts,
            lora_alpha=lora_alpha
        )
        self.linear_3 = MoELoRA(
            in_features=hidden_dim,
            out_features=ffn_dim,
            lora_rank=lora_rank,
            n_experts=n_experts,
            lora_alpha=lora_alpha
        )


    def forward(self, x):
        """
        A simple forward pass through the FFN
        """
        return self.linear_2(torch.nn.functional.silu(self.linear_1(x)) * self.linear_3(x))
    

class SharedTransformerBlock(torch.nn.Module):
    """
    LoRA shared transformer block
    """
    def __init__(self, hidden_dim, context_window, ffn_cfg, attn_cfg, depth: Optional[int]=None):
        super().__init__()

        # build the attn norm
        self.attn_norm = build_normalization(
            normalization_name=attn_cfg.get("normalization", "none"),
            dim=hidden_dim,
            bias=attn_cfg["params"]["bias"],
        )

        # build the attention
        self.attn = build_attention(
            attn_name=attn_cfg["name"],
            attn_params=attn_cfg["params"],
            hidden_dim=hidden_dim,
            context_window=context_window,
            depth=depth,
        )

        # build the ffn norm
        self.ffn_norm = build_normalization(
            normalization_name=ffn_cfg.get("normalization", "none"), # Default: none
            dim=hidden_dim,
            bias=ffn_cfg["params"]["bias"],
        )

        # build the ffn block
        self.ffn = SharedMoEFFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_cfg["ffn_dim"],
            lora_rank=ffn_cfg["lora_rank"],
            n_experts=ffn_cfg["n_experts"],
            lora_alpha=ffn_cfg["lora_alpha"],
        )

    def forward(self, x, attention_mask=None):
        """
        A simple, residual forward
        pass through the GPT block.
        Args:
            x: the input tensor (b, s, h)
            attention_mask: the attention mask
        Returns:
            x: the output tensor (b, s, h)
        """
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SharedMoE(torch.nn.Module):
    """
    core model class with shared MoE weights
    """
    def __init__(self, model_cfg):
        super().__init__()

        # build the transformer
        self.transformer = torch.nn.ModuleDict(
            {
                "drop": torch.nn.Dropout(),
                "h": torch.nn.ModuleList(
                    [
                        SharedTransformerBlock(
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"],
                            use_rope=model_cfg["positional_encoding_type"] == "rope",
                            ffn_cfg=model_cfg["core_model"]["ffn"],
                            attn_cfg=model_cfg["core_model"]["attn"],
                        )
                        for _ in range(model_cfg["core_model"]["num_layers"])
                    ]
                ),
            }
        )

        # share the weights between all ffn blocks
        ffn_0 = self.transformer.h[0].ffn
        """for i in range(1, len(self.transformer.h)):
            self.transformer.h[i].ffn.linear_1.weight = ffn_0.linear_1.weight
            self.transformer.h[i].ffn.linear_2.gate_linear.weight = ffn_0.linear_2.gate_linear.weight
            for j in range(ffn_0.linear_1.n_experts):
                self.transformer.h[i].ffn.linear_1.lora_experts_U[j] = ffn_0.linear_1.lora_experts_U[j]
                self.transformer.h[i].ffn.linear_1.lora_experts_V[j] = ffn_0.linear_1.lora_experts_V[j]

            self.transformer.h[i].ffn.linear_2.weight = ffn_0.linear_2.weight
            self.transformer.h[i].ffn.linear_2.gate_linear.weight = ffn_0.linear_2.gate_linear.weight
            for j in range(ffn_0.linear_2.n_experts):
                self.transformer.h[i].ffn.linear_2.lora_experts_U[j] = ffn_0.linear_2.lora_experts_U[j]
                self.transformer.h[i].ffn.linear_2.lora_experts_V[j] = ffn_0.linear_2.lora_experts_V[j]

            self.transformer.h[i].ffn.linear_3.weight = ffn_0.linear_3.weight
            self.transformer.h[i].ffn.linear_3.gate_linear.weight = ffn_0.linear_3.gate_linear.weight
            for j in range(ffn_0.linear_3.n_experts):
                self.transformer.h[i].ffn.linear_3.lora_experts_U[j] = ffn_0.linear_3.lora_experts_U[j]
                self.transformer.h[i].ffn.linear_3.lora_experts_V[j] = ffn_0.linear_3.lora_experts_V[j]"""
        for i in range(1, len(self.transformer.h)):
            self.transformer.h[i].ffn.linear_1.weight = ffn_0.linear_1.weight
            self.transformer.h[i].ffn.linear_1.gate_linear.weight = ffn_0.linear_1.gate_linear.weight
            self.transformer.h[i].ffn.linear_1.lora_experts_U = ffn_0.linear_1.lora_experts_U
            self.transformer.h[i].ffn.linear_1.lora_experts_V = ffn_0.linear_1.lora_experts_V

            self.transformer.h[i].ffn.linear_2.weight = ffn_0.linear_2.weight
            self.transformer.h[i].ffn.linear_2.gate_linear.weight = ffn_0.linear_2.gate_linear.weight
            self.transformer.h[i].ffn.linear_2.lora_experts_U = ffn_0.linear_2.lora_experts_U
            self.transformer.h[i].ffn.linear_2.lora_experts_V = ffn_0.linear_2.lora_experts_V

            self.transformer.h[i].ffn.linear_3.weight = ffn_0.linear_3.weight
            self.transformer.h[i].ffn.linear_3.gate_linear.weight = ffn_0.linear_3.gate_linear.weight
            self.transformer.h[i].ffn.linear_3.lora_experts_U = ffn_0.linear_3.lora_experts_U
            self.transformer.h[i].ffn.linear_3.lora_experts_V = ffn_0.linear_3.lora_experts_V

    def forward(self, x):
        """
        Pass an input through the model
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, H)
        """

        # apply dropout
        x = self.transformer.drop(x)

        # pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        return x
