from sentence_transformers import SentenceTransformer, models, losses
from torch.utils.data import DataLoader
from config import Config


class Trainer:

    def __init__(self, config: Config):
        self.__config = config
        base_model = models.Transformer(config.model)
        pooling_model = models.Pooling(base_model.get_word_embedding_dimension())
        self.__model = SentenceTransformer(modules=[base_model, pooling_model])
        # Experiment with different loss algorithms
        self.__loss = losses.TripletLoss(model=self.__model)

    def train(self, data_loader: DataLoader):
        self.__model.fit(
            train_objectives=[(data_loader, self.__loss)],
            epochs=self.__config.epochs
        )

    def save(self):
        self.__model.save(self.__config.model_directory)
