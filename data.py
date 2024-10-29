import pickle
from datasets import Dataset, load_dataset


def get_huggingface_examples():
    dataset = load_dataset("sentence-transformers/all-nli", "triplet")
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    test_dataset = dataset["test"]
    return train_dataset, eval_dataset, test_dataset


def load_local_data(file_path: str) -> list[dict[str, str | list[str]]]:
    with open(file_path, "rb") as input_file:
        return pickle.load(input_file)


def get_local_data(file_path: str) -> Dataset:
    raw_data = load_local_data(file_path)
    anchors, pos_examples, neg_examples = list(), list(), list()
    for i in range(len(raw_data)):
        example = raw_data[i]
        anchor = example["query"]
        for j in range(len(example["pos"])):
            if j >= len(example["neg"]):
                continue
            pos_example = example["pos"][j]
            neg_example = example["neg"][j]
            anchors.append(anchor)
            pos_examples.append(pos_example)
            neg_examples.append(neg_example)
    data = {
        "anchor": anchors,
        "positive": pos_examples,
        "negative": neg_examples
    }

    return Dataset.from_dict(data)
