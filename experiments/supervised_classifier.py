import datetime


from omegaconf import DictConfig


from .base import Experiment
from . import ExperimentFactory

from idspy.src.idspy.nn.torch.prediction.classification import ArgMax

from idspy.src.idspy.core.storage.dict import DictStorage
from idspy.src.idspy.core.pipeline.base import PipelineEvent
from idspy.src.idspy.core.pipeline.observable import (
    ObservablePipeline,
    ObservableFittablePipeline,
    ObservableRepeatablePipeline,
)
from idspy.src.idspy.core.events.bus import EventBus
from idspy.src.idspy.core.events.event import only_source

from idspy.src.idspy.builtins.handler.logging import Logger, DataFrameProfiler

from idspy.src.idspy.builtins.step.data.io import LoadData, SaveData
from idspy.src.idspy.builtins.step.data.adjust import (
    DropNulls,
    RareClassFilter,
    ColsToNumpy,
    Filter,
)
from idspy.src.idspy.builtins.step.data.map import FrequencyMap, LabelMap
from idspy.src.idspy.builtins.step.data.scale import StandardScale
from idspy.src.idspy.builtins.step.data.split import (
    StratifiedSplit,
    ExtractSplitPartitions,
)

from idspy.src.idspy.builtins.step.nn.torch.builder.model import BuildModel
from idspy.src.idspy.builtins.step.nn.torch.builder.dataset import BuildDataset
from idspy.src.idspy.builtins.step.nn.torch.builder.dataloader import BuildDataLoader
from idspy.src.idspy.builtins.step.nn.torch.builder.optimizer import BuildOptimizer
from idspy.src.idspy.builtins.step.nn.torch.builder.loss import BuildLoss
from idspy.src.idspy.builtins.step.nn.torch.builder.scheduler import BuildScheduler

from idspy.src.idspy.builtins.step.nn.torch.engine.tensor import CatTensors
from idspy.src.idspy.builtins.step.nn.torch.engine.epoch import (
    TrainOneEpoch,
    ValidateOneEpoch,
)
from idspy.src.idspy.builtins.step.nn.torch.engine.forward import MakePredictions

from idspy.src.idspy.builtins.step.nn.torch.engine.early_stopping import EarlyStopping
from idspy.src.idspy.builtins.step.nn.torch.model.io import (
    LoadModelWeights,
    SaveModelWeights,
)
from idspy.src.idspy.builtins.step.metric.classification import ClassificationMetrics
from idspy.src.idspy.builtins.step.metric.clustering import ClusteringMetrics

from idspy.src.idspy.builtins.step.log.tensorboard import MetricsLogger, WeightsLogger
from idspy.src.idspy.builtins.step.log.projection import VectorsProjectionPlot


@ExperimentFactory.register()
class SupervisedClassifier(Experiment):

    def __init__(self, cfg: DictConfig) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"{cfg.path.logs}/supervised_classifier/{ts}"

    def preprocessing(self, cfg: DictConfig, storage: DictStorage) -> None:
        bus = EventBus()
        bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)
        bus.subscribe(
            callback=DataFrameProfiler(),
            event_type=PipelineEvent.PIPELINE_END,
            predicate=only_source("preprocessing_pipeline"),
        )

        fit_aware_pipeline = ObservableFittablePipeline(
            steps=[
                StandardScale(),
                FrequencyMap(max_levels=cfg.data.max_cat_levels),
                LabelMap(),
            ],
            name="fit_aware_pipeline",
            bus=bus,
            storage=storage,
        )

        full_pipeline = ObservablePipeline(
            steps=[
                LoadData(
                    file_path=cfg.path.data_raw,
                    file_name=cfg.data.file_name,
                    fmt="csv",
                    numerical_cols=cfg.data.numerical_columns,
                    categorical_cols=cfg.data.categorical_columns,
                    label_col=cfg.data.label_column,
                ),
                DropNulls(),
                RareClassFilter(
                    target_col=cfg.data.label_column,
                    min_count=3000,
                ),
                Filter(
                    query=f"{cfg.data.label_column} != '{cfg.data.benign_tag}'",
                ),
                StratifiedSplit(
                    class_col=cfg.data.label_column,
                    train_size=cfg.data.train_size,
                    val_size=cfg.data.val_size,
                    test_size=cfg.data.test_size,
                ),
                fit_aware_pipeline,
                SaveData(
                    file_path=cfg.path.data_processed,
                    file_name=cfg.data.file_name,
                    fmt=cfg.data.format,
                ),
            ],
            storage=storage,
            bus=bus,
            name="preprocessing_pipeline",
        )

        full_pipeline.run()

    def training(self, cfg: DictConfig, storage: DictStorage) -> None:
        bus = EventBus()
        bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)

        setup_pipeline = ObservablePipeline(
            steps=[
                LoadData(
                    file_path=cfg.path.data_processed,
                    file_name=cfg.data.file_name,
                    fmt=cfg.data.format,
                ),
                ExtractSplitPartitions(),
                BuildModel(model_args=cfg.model),
                BuildLoss(loss_args=cfg.loss),
                BuildOptimizer(optimizer_args=cfg.optimizer),
                BuildDataset(
                    df_key="train.data",
                    dataset_key="train.dataset",
                    label_col=cfg.data.label_column,
                ),
                BuildDataLoader(
                    dataloader_args=cfg.loops.train.dataloader,
                    dataset_key="train.dataset",
                    dataloader_key="train.dataloader",
                ),
                BuildScheduler(
                    scheduler_args=cfg.scheduler, dataloader_key="train.dataloader"
                ),
                BuildDataset(
                    df_key="val.data",
                    dataset_key="val.dataset",
                    label_col=cfg.data.label_column,
                ),
                BuildDataLoader(
                    dataloader_args=cfg.loops.val.dataloader,
                    dataset_key="val.dataset",
                    dataloader_key="val.dataloader",
                ),
            ],
            storage=storage,
            bus=bus,
        )

        training_pipeline = ObservableRepeatablePipeline(
            steps=[
                TrainOneEpoch(metrics_key="train.metrics"),
                MetricsLogger(log_dir=self.log_dir, metrics_key="train.metrics"),
                WeightsLogger(log_dir=self.log_dir, model_key="model"),
                ValidateOneEpoch(
                    dataloader_key="val.dataloader",
                    metrics_key="val.metrics",
                    loss_fn_key="loss_fn",
                ),
                EarlyStopping(
                    min_delta=0.001,
                    metrics_key="val.metrics",
                    stop_key="stop_pipeline",
                ),
            ],
            storage=storage,
            bus=bus,
            count=cfg.loops.train.epochs,
            clear_storage=False,
            predicate=lambda storage: storage.as_dict().get("stop_pipeline", False),
        )

        full_pipeline = ObservablePipeline(
            steps=[
                setup_pipeline,
                training_pipeline,
                SaveModelWeights(
                    file_path=cfg.path.model,
                    file_name=cfg.model._target_ + "_final",
                    fmt="pt",
                ),
            ],
            storage=storage,
            bus=bus,
        )

        full_pipeline.run()

    def testing(self, cfg: DictConfig, storage: DictStorage) -> None:
        bus = EventBus()
        bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)

        setup_pipeline = ObservablePipeline(
            steps=[
                LoadData(
                    file_path=cfg.path.data_processed,
                    file_name=cfg.data.file_name,
                    fmt=cfg.data.format,
                ),
                ExtractSplitPartitions(),
                ColsToNumpy(
                    df_key="test.data",
                    output_key="test.labels",
                    cols=[cfg.data.label_column],
                ),
                BuildModel(model_args=cfg.model),
                LoadModelWeights(
                    file_path=cfg.path.model,
                    file_name=cfg.model._target_ + "_final",
                    fmt="pt",
                ),
                BuildDataset(
                    df_key="test.data",
                    dataset_key="test.dataset",
                    label_col=cfg.data.label_column,
                ),
                BuildDataLoader(
                    dataloader_args=cfg.loops.test.dataloader,
                    dataset_key="test.dataset",
                    dataloader_key="test.dataloader",
                ),
            ],
            storage=storage,
            bus=bus,
        )

        testing_pipeline = ObservablePipeline(
            steps=[
                ValidateOneEpoch(
                    dataloader_key="test.dataloader",
                    metrics_key="test.metrics",
                    outputs_key="test.outputs",
                    save_outputs=True,
                ),
                CatTensors(
                    tensors_key="test.outputs",
                    section="logits",
                    output_key="test.logits_tensor",
                ),
                CatTensors(
                    tensors_key="test.outputs",
                    section="latents",
                    output_key="test.latents_tensor",
                ),
                MetricsLogger(log_dir=self.log_dir, metrics_key="test.metrics"),
                MakePredictions(
                    pred_fn=ArgMax(),
                    logits_key="test.logits_tensor",
                    outputs_key="test.preds",
                ),
                ClassificationMetrics(
                    labels_key="test.labels",
                    predictions_key="test.preds",
                    metrics_key="test.classification_metrics",
                ),
                ClusteringMetrics(
                    vectors_key="test.latents_tensor",
                    labels_key="test.labels",
                    metrics_key="test.clustering_metrics",
                ),
                VectorsProjectionPlot(
                    vectors_key="test.latents_tensor",
                    labels_key="test.labels",
                    n_components=2,
                    output_key="test.projection_plot",
                ),
                MetricsLogger(
                    log_dir=self.log_dir,
                    metrics_key="test.classification_metrics",
                ),
                MetricsLogger(
                    log_dir=self.log_dir,
                    metrics_key="test.clustering_metrics",
                ),
                MetricsLogger(
                    log_dir=self.log_dir,
                    metrics_key="test.projection_plot",
                ),
            ],
            storage=storage,
            bus=bus,
        )

        full_pipeline = ObservablePipeline(
            steps=[
                setup_pipeline,
                testing_pipeline,
            ],
            storage=storage,
            bus=bus,
        )

        full_pipeline.run()
