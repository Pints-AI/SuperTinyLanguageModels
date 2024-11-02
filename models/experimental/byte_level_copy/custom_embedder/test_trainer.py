import torch
import torch.optim as optim

from models.experimental.byte_level_copy.custom_embedder.chunker import (
    CustomByteLevelEmbedder, 
    compute_loss_for_end_token, 
    find_end_characters, 
    get_canonical_tokenizer,
    CONFIG
)


INPUT = [
    "i like machine learning",
    "anton is running",
    "finetuning LLMs is as simple as ordering a pint of beer!"
]

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
