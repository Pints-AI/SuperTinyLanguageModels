from models.experimental.byte_level_copy.custom_embedder.dataset_preparation.data_utils import load_data
from models.experimental.byte_level_copy.custom_embedder.dataset_preparation.prepare import prepare_data
from models.experimental.byte_level_copy.custom_embedder.chunker import get_canonical_tokenizer, CustomByteLevelEmbedder, CONFIG


if __name__ == "__main__":
    # import torch
    # print("CUDA available:", torch.cuda.is_available()) 

    embedder = CustomByteLevelEmbedder(CONFIG['model'], CONFIG['general']['device'])
    tokenizer = get_canonical_tokenizer()
    prepare_data(CONFIG, embedder, tokenizer)
