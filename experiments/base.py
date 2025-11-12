from datetime import datetime

from omegaconf import DictConfig

from idspy.src.idspy.core.storage.dict import DictStorage


class Experiment:
    """
    Abstract base class for machine learning experiments.
    """

    def __init__(self, cfg: DictConfig, storage: DictStorage) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"{cfg.path.logs}/{cfg.data.name}/{cfg.type}{'_bg' if not cfg.experiment.exclude_background else ''}/{cfg.seed}/{cfg.stage}_{ts}"
        self.cfg = cfg
        self.storage = storage
