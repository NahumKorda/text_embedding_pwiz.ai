from datasets import load_dataset
from sentence_transformers import InputExample
from torch.utils.data import DataLoader, Dataset


class DataSet(Dataset):
    def __init__(self, dataset: list[InputExample]):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_local_data(file_path: str) -> list[dict[str, str | list[str]]]:
    """
    Implement your own code here.
    Dataset should have the following schema:
        [
            {
                "query": "",
                "pos": [""],
                "neg": [""]
            }
        ]
    Negative samples should be more numerous than positive for best results.
    """
    pass


def get_examples_for_testing(example_dataset: str) -> list[dict[str, str | list[str]]]:
    dataset = load_dataset(example_dataset)
    return dataset['train']['set']


def get_data_loader(dataset: list[dict[str, str | list[str]]], batch_size: int):
    train_data = list()
    for i in range(len(dataset)):
        example = dataset[i]
        train_data.append(InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]))
    train_dataset = DataSet(train_data)
    return DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
