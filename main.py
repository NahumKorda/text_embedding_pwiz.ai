from config import Config
from data import get_examples_for_testing, get_data_loader
from trainer import Trainer

"""
Trains a text-embedding model from scratch.
"""


if __name__ == '__main__':

  config = Config("config.yaml")

  trainer = Trainer(config)

  dataset = get_examples_for_testing(config.example_dataset)
  data_loader = get_data_loader(dataset, config.batch_size)

  trainer.train(data_loader)
