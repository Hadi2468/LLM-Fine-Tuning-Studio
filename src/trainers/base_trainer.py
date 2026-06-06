# Base trainer class defining the interface for all trainers in the LLM Fine-Tuning Studio

from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def prepare_dataset(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save_model(self):
        pass