from abc import ABC, abstractmethod

from omegaconf import DictConfig

from idspy.src.idspy.core.storage.dict import DictStorage


class Experiment(ABC):
    """
    Abstract base class for machine learning experiments.
    """

    @abstractmethod
    def preprocessing(self, cfg: DictConfig, storage: DictStorage) -> None:
        """
        Perform data preprocessing steps.
        """
        pass

    @abstractmethod
    def training(self, cfg: DictConfig, storage: DictStorage) -> None:
        """
        Train the model.
        """
        pass

    @abstractmethod
    def testing(self, cfg: DictConfig, storage: DictStorage) -> None:
        """
        Test the model and evaluate performance.
        """
        pass
