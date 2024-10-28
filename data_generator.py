import os.path
import pickle
from random import shuffle
from Levenshtein import distance
from tqdm import tqdm


class DataGenerator:

    def __init__(self, positive_file_path: str, negative_file_path: str):
        self.__positives = self.__get_positives(positive_file_path)
        self.__negatives = self.__get_negatives(negative_file_path)

    @staticmethod
    def __get_positives(positive_file_path: str) -> list[list[str]]:
        with open(positive_file_path, "rb") as input_file:
            return pickle.load(input_file)

    @staticmethod
    def __get_negatives(negative_file_path: str) -> list[str]:
        with open(negative_file_path, "rb") as input_file:
            return pickle.load(input_file)

    def generate_data(self, data_directory: str):

        # Split positives into future datasets
        # Split must be carried out on the raw data
        # to ensure that val and test data are not visible during training
        train_positives, val_positives, test_positives = self.__split_datasets()

        # Generate datasets
        train_dataset = self.__generate_dataset(train_positives)
        val_dataset = self.__generate_dataset(val_positives)
        test_dataset = self.__generate_dataset(test_positives)

        # Save datasets
        file_path = os.path.join(data_directory, "train.pkl")
        self.__save(train_dataset, file_path)
        file_path = os.path.join(data_directory, "val.pkl")
        self.__save(val_dataset, file_path)
        file_path = os.path.join(data_directory, "test.pkl")
        self.__save(test_dataset, file_path)

        return f"Train dataset size: {len(train_dataset)}\nVal dataset size: {len(val_dataset)}\nTest dataset seize: {len(test_dataset)}"

    def __split_datasets(self) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:

        total = len(self.__positives)
        val_cutoff = int(0.1 * total)
        test_cutoff = val_cutoff + int(0.1 * total)

        shuffle(self.__positives)

        val_positives = self.__positives[:val_cutoff]
        test_positives = self.__positives[val_cutoff: test_cutoff]
        train_positives = self.__positives[test_cutoff:]

        return train_positives, val_positives, test_positives

    def __generate_dataset(self, positives: list[list[str]]):

        retval = list()

        for positive_alternatives in tqdm(positives, total=len(positives)):
            retval.extend(self.__generate_data_points(positive_alternatives))

        return retval

    def __generate_data_points(self, positive_alternatives: list[str]) -> list[dict]:

        retval = list()

        for positive in positive_alternatives:
            negative_samples = self.__select_negatives(positive, 3 * len(positive_alternatives))
            positive_samples = list()
            for positive_sample_candidate in positive_alternatives:
                if positive_sample_candidate != positive:
                    positive_samples.append(positive_sample_candidate)
            retval.append({
                "query": positive,
                "pos": positive_samples,
                "neg": negative_samples
            })

        return retval

    def __select_negatives(self, positive: str, total_negatives: int) -> list[str]:

        lower_bound = len(positive) - int(0.1 * len(positive))
        upper_bound = len(positive) + int(0.1 * len(positive))
        score_threshold = int(0.8 * len(positive))

        shuffle(self.__negatives)

        retval = list()
        for negative in self.__negatives:
            if len(retval) < total_negatives:
                if lower_bound <= len(negative) <= upper_bound:
                    score = distance(s1=positive, s2=negative)
                    if score > score_threshold:
                         retval.append(negative)
            else:
                break

        return retval

    @staticmethod
    def __save(dataset: list[dict[str, str | list[str]]], file_path: str):
        with open(file_path, "wb") as output_file:
            pickle.dump(dataset, output_file)


if __name__ == '__main__':

    positive_path = "/Users/nahumkorda/code/resources/pwiz/positive_train_data.pkl"
    negative_path = "/Users/nahumkorda/code/resources/dsec_embedding_model/wiki_categs.pkl"
    output_directory = "/Users/nahumkorda/code/resources/pwiz/data"

    data_generator = DataGenerator(positive_path, negative_path)

    stats = data_generator.generate_data(output_directory)
    print(stats)
