import logging
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig

from idspy.src.idspy.common.logging import setup_logging
from idspy.src.idspy.common.utils import set_seeds
from idspy.src.idspy.common.config import load_config

from idspy.src.idspy.core.storage.dict import DictStorage
from idspy.src.idspy.nn.torch.helper import get_device

from experiments import ExperimentFactory

setup_logging()
logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    set_seeds(cfg.seed)

    # Setup device
    if cfg.device == "auto":
        device = get_device()
    else:
        device = torch.device(cfg.device)

    # Setup storage
    storage = DictStorage(
        {
            "device": device,
            "seed": cfg.seed,
            "stop_pipeline": False,
        }
    )

    exp = ExperimentFactory.create(
        class_name=cfg.type, config={"cfg": cfg, "storage": storage}
    )

    stage_method = getattr(exp, cfg.stage.lower(), None)

    if stage_method is None:
        raise AttributeError(f"Stage '{cfg.stage}' not found in experiment")

    if not callable(stage_method):
        raise TypeError(f"Stage '{cfg.stage}' is not callable")

    stage_method()


if __name__ == "__main__":
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    main(cfg)
