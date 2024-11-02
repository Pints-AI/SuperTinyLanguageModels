from models.experimental.byte_level_copy.custom_embedder.dataset_preparation.data_utils import load_data
from models.experimental.byte_level_copy.custom_embedder.dataset_preparation.prepare import prepare_data
from models.experimental.byte_level_copy.custom_embedder.chunker import get_canonical_tokenizer, CustomByteLevelEmbedder, CONFIG


if __name__ == "__main__":
    # dataset = load_data(['simple_en_wiki'])
    # print(type(dataset))
    # print(dataset)
    # print(type(dataset['train']))
    # print(len(dataset['train']['id']))
    embedder = CustomByteLevelEmbedder(CONFIG['model'], CONFIG['general']['device'])
    tokenizer = get_canonical_tokenizer()
    prepare_data(CONFIG, embedder, tokenizer)