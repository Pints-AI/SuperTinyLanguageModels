import torch
import torch.optim as optim
from pathlib import Path

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from models.experimental.byte_level_copy.custom_embedder.dataset_preparation.prepare import prepare_data
from models.experimental.byte_level_copy.custom_embedder.chunker import (
    get_canonical_tokenizer, 
    compute_MSEloss_for_end_token,
    compute_BCEloss_for_end_token,
    CustomByteLevelEmbedder, 
    CONFIG
)
from models.experimental.byte_level_copy.custom_embedder.dataset_preparation.dataset_loader import EndByteClassifierDataset

def train_step(
        optimizer: torch.optim.Optimizer,
        predictions: torch.Tensor,
        labels: torch.Tensor) -> float:
    # compute loss and backpropagate
    loss = compute_BCEloss_for_end_token(predictions, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

if __name__ == "__main__":
    model_cfg: dict = CONFIG["model"]
    device: str = CONFIG["general"]["device"]
    learning_rate: float = CONFIG["trainer"]["optimizer"]["lr"]
    num_epochs = 1  # Example number of epochs

    if device == "cpu":
        print("WARNING.. Using CPU..")

    embedder = CustomByteLevelEmbedder(model_cfg, device=device).to(device)

    if device == 'cuda':
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs..")
            embedder = DataParallel(embedder)

    optimizer = optim.AdamW(embedder.parameters(), lr=learning_rate)

    # 0. prepare dataset if not done
    # tokenizer = get_canonical_tokenizer()
    # prepare_data(CONFIG, embedder, tokenizer)

    x_train = Path("/home/pints/brewery/SuperTinyLanguageModels/data/simple_en_wiki/embedder_preprocessor/bpe-259-0/train_ids.bin")
    y_train = Path("/home/pints/brewery/SuperTinyLanguageModels/data/simple_en_wiki/embedder_preprocessor/bpe-259-0/train_labels.bin")
    x_val = Path("/home/pints/brewery/SuperTinyLanguageModels/data/simple_en_wiki/embedder_preprocessor/bpe-259-0/val_ids.bin")
    y_val = Path("/home/pints/brewery/SuperTinyLanguageModels/data/simple_en_wiki/embedder_preprocessor/bpe-259-0/val_labels.bin")

    train_dataset = EndByteClassifierDataset(x_train, y_train, CONFIG)
    test_dataset = EndByteClassifierDataset(x_val, y_val, CONFIG)

    # MUST NOT SHUFFLE! due to the way we concatenate during tokenization
    train_data_loader = DataLoader(train_dataset, batch_size=18, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=18, shuffle=False)

    for epoch in range(num_epochs):
        embedder.train()
        total_loss = 0.0
        num_train_samples = 0
        for x, y in train_data_loader:
            pred = embedder(x)
            loss = train_step(optimizer, pred, y) # Train step
            batch_size = x.size(0)
            total_loss += loss * batch_size
            num_train_samples += batch_size
            if num_train_samples % (1000*batch_size) == 0:
                print(f"Number samples processed: {num_train_samples} / {len(train_data_loader) * batch_size}")
        avg_loss = total_loss / num_train_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Test loop (evaluation)
    print("Evaluating..")
    embedder.eval()
    test_loss = 0.0
    num_test_samples = 0
    with torch.no_grad():
        for x, y in test_data_loader:
            pred = embedder(x)
            loss = compute_BCEloss_for_end_token(pred, y)
            batch_size = x.size(0)
            test_loss += loss * batch_size
            num_test_samples += batch_size
    
    avg_test_loss = test_loss / num_test_samples
    print(f"Final Test Loss: {avg_test_loss:.4f}")


    # save the embedder model
    model_save_path = Path("./embedder_model_weights.pth")
    torch.save(embedder.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")
