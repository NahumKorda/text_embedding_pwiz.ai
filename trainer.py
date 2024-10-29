import os.path
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from config import Config


class Trainer:

    def __init__(self, config: Config, train_dataset: Dataset, eval_dataset: Dataset):
        self.__config = config
        self.__model = self.__get_model(config)
        self.__trainer = self.__get_trainer(config, train_dataset, eval_dataset)

    @staticmethod
    def __get_model(config: Config):
        return SentenceTransformer(config.model)

    def __get_trainer(self, config: Config, train_dataset: Dataset, eval_dataset: Dataset):

        loss = TripletLoss(self.__model)

        dev_evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="pwiz.ai",
            show_progress_bar=False
        )
        dev_evaluator(self.__model)

        return SentenceTransformerTrainer(
            model=self.__model,
            args=config.get_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
        )

    def train(self):
        self.__trainer.train()

    def evaluate(self, test_dataset: Dataset):
        test_evaluator = TripletEvaluator(
            anchors=test_dataset["anchor"],
            positives=test_dataset["positive"],
            negatives=test_dataset["negative"],
            name="pwiz.ai",
            show_progress_bar=False
        )
        metrics = test_evaluator(self.__model)
        print(metrics)

    def save(self):
        output_dir = os.path.join(self.__config.output_dir, "final")
        self.__model.save_pretrained(output_dir)
