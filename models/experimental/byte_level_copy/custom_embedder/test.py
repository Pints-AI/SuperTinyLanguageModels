from models.experimental.byte_level_copy.custom_embedder.dataset_preparation.data_utils import load_data

if __name__ == "__main__":
    dataset = load_data(['en_wiki'])
    print(type(dataset))
    print(dataset)