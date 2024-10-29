from config import Config
from data import get_local_data
from trainer import Trainer

"""
Trains a text-embedding model from scratch.
"""

def train_model():

  train_dataset = get_local_data("/Users/nahumkorda/code/resources/pwiz/data/train.pkl")
  eval_dataset = get_local_data("/Users/nahumkorda/code/resources/pwiz/data/val.pkl")
  test_dataset = get_local_data("/Users/nahumkorda/code/resources/pwiz/data/test.pkl")

  print(f"Train dataset size: {len(train_dataset)}")
  print(f"Eval dataset size: {len(eval_dataset)}")
  print(f"Test dataset size: {len(test_dataset)}")

  config = Config("config.yaml")

  trainer = Trainer(config, train_dataset, eval_dataset)
  trainer.train()

  trainer.evaluate(test_dataset)


if __name__ == '__main__':

  train_model()
