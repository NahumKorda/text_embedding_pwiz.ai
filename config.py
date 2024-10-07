from dataclasses import dataclass
import yaml


@dataclass
class Config:
    model: str
    example_dataset: str
    batch_size: int
    epochs: int
    model_directory: str

    def __init__(self, file_path: str):
        with open(file_path, "r") as input_file:
            config = yaml.safe_load(input_file)
        self.model = config["model"]
        self.example_dataset = config["example_dataset"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.model_directory = config["model_directory"]
