import torch
import torch.optim as optim

from models.experimental.byte_level_copy.custom_embedder.chunker import (
    CustomByteLevelEmbedder, 
    compute_loss_for_end_token, 
    find_end_characters, 
    get_canonical_tokenizer
)


INPUT = [
    "i like machine learning",
    "anton is running",
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
        "trainer_type": "base_trainer",
        "dataset": "fineweb_edu_10B",
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

model_cfg: dict = CONFIG["model"]
device: str = CONFIG["general"]["device"]
learning_rate: float = CONFIG["trainer"]["optimizer"]["lr"]
num_epochs = 10  # Example number of epochs

custom_embedder = CustomByteLevelEmbedder(model_cfg, device=device).to(device)
optimizer = optim.AdamW(custom_embedder.parameters(), lr=learning_rate)

def train_step(input_string: str, target_end_positions: torch.Tensor):
    # tokenize and get byte embeddings
    input_tokens = custom_embedder.tokenize_input(input_string)
    input_tensor = torch.tensor([input_tokens], device=device)  # Add batch dimension
    #forward pass
    end_bytes_probs = custom_embedder(input_tensor)

    # compute loss and backpropagate
    loss = compute_loss_for_end_token(end_bytes_probs, target_end_positions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# train
for epoch in range(num_epochs):
    custom_embedder.train()
    total_loss = 0.0
    for input_string in INPUT:  # sample data
        canonical_tokenizer = get_canonical_tokenizer()
        canonical_tokens_ids = canonical_tokenizer.encode(input_string)
        canonical_tokens = [
            canonical_tokenizer.decode_single_token_bytes(token_id)
            for token_id in canonical_tokens_ids
        ]
        end_chars_pos = find_end_characters(canonical_tokens)
        target_end_positions = torch.tensor([end_chars_pos], device=device, dtype=torch.float)

        # Train step
        loss = train_step(input_string, target_end_positions)
        total_loss += loss

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(INPUT):.4f}")
