from datetime import datetime


from omegaconf import DictConfig


from ..base import Experiment
from .. import ExperimentFactory

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

from idspy.src.idspy.builtins.handler.logging import (
    Logger,
    DataFrameProfiler as DataFrameProfilerHandler,
)

from idspy.src.idspy.builtins.step.data.io import LoadData, SaveData
from idspy.src.idspy.builtins.step.data.sample import (
    ComputeIndicesByLabel,
    SelectSamplesByIndices,
    Downsample,
)
from idspy.src.idspy.builtins.step.data.adjust import (
    DropNulls,
    RareClassFilter,
    DFToNumpy,
    Filter,
    Clip,
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
from idspy.src.idspy.builtins.step.metric.classification import (
    SupervisedClassificationMetrics,
)
from idspy.src.idspy.builtins.step.metric.projection import VectorsProjectionPlot
from idspy.src.idspy.builtins.step.metric.clustering import ClusteringMetrics

from idspy.src.idspy.builtins.step.log.tensorboard import TBLogger, TBWeightsLogger
from idspy.src.idspy.builtins.step.log.profiler import DataFrameProfiler


@ExperimentFactory.register()
class SupervisedClassifier(Experiment):

    def __init__(self, cfg: DictConfig) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"{cfg.path.logs}/{cfg.data.name}/supervised_classifier{'_bg' if not cfg.experiment.exclude_background else ''}/{cfg.seed}/{cfg.stage}_{ts}"

    def preprocessing(self, cfg: DictConfig, storage: DictStorage) -> None:
        bus = EventBus()
        bus.subscribe(callback=Logger(), event_type=PipelineEvent.STEP_START)
        bus.subscribe(
            callback=DataFrameProfilerHandler(),
            event_type=PipelineEvent.PIPELINE_END,
            predicate=only_source("preprocessing_pipeline"),
        )

        fit_aware_pipeline = ObservableFittablePipeline(
            steps=[
                Clip(),
                StandardScale(),
                FrequencyMap(max_levels=cfg.data.max_cat_levels),
                LabelMap(target_col=f"multi_{cfg.data.label_column}"),
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
                    query=(
                        f"{cfg.data.label_column} != '{cfg.data.benign_tag}'"
                        if cfg.experiment.exclude_background
                        else None
                    ),
                ),
                StratifiedSplit(
                    class_col=cfg.data.label_column,
                    train_size=cfg.data.train_size,
                    val_size=cfg.data.val_size,
                    test_size=cfg.data.test_size,
                ),
                fit_aware_pipeline,
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="data.labels_mapping",
                ),
                SaveData(
                    file_path=cfg.path.data_processed,
                    file_name=(
                        cfg.data.file_name + "_bg"
                        if not cfg.experiment.exclude_background
                        else cfg.data.file_name
                    ),
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
                    file_name=(
                        cfg.data.file_name + "_bg"
                        if not cfg.experiment.exclude_background
                        else cfg.data.file_name
                    ),
                    fmt=cfg.data.format,
                ),
                ExtractSplitPartitions(),
                Downsample(
                    frac=cfg.experiment.labeled_fraction,
                    class_col=f"multi_{cfg.data.label_column}",
                    random_state=cfg.seed,
                    df_key="train.data",
                ),
                DataFrameProfiler(
                    df_key="train.data",
                    output_key="train.data_profile",
                ),
                BuildModel(model_name=cfg.model.name, model_args=cfg.model.args),
                BuildLoss(loss_name=cfg.loss.name, loss_args=cfg.loss.args),
                BuildOptimizer(
                    optimizer_name=cfg.optimizer.name, optimizer_args=cfg.optimizer.args
                ),
                BuildDataset(
                    df_key="train.data",
                    dataset_key="train.dataset",
                    label_col=f"multi_{cfg.data.label_column}",
                ),
                BuildDataLoader(
                    dataloader_args=cfg.loops.train.dataloader,
                    dataset_key="train.dataset",
                    dataloader_key="train.dataloader",
                ),
                BuildScheduler(
                    scheduler_name=cfg.scheduler.name,
                    scheduler_args=cfg.scheduler.args,
                    dataloader_key="train.dataloader",
                ),
                BuildDataset(
                    df_key="val.data",
                    dataset_key="val.dataset",
                    label_col=f"multi_{cfg.data.label_column}",
                ),
                BuildDataLoader(
                    dataloader_args=cfg.loops.val.dataloader,
                    dataset_key="val.dataset",
                    dataloader_key="val.dataloader",
                ),
                TBLogger(log_dir=self.log_dir, subject_key="train.data_profile"),
            ],
            storage=storage,
            bus=bus,
        )

        training_pipeline = ObservableRepeatablePipeline(
            steps=[
                TrainOneEpoch(metrics_key="train.metrics"),
                TBLogger(log_dir=self.log_dir, subject_key="train.metrics"),
                TBWeightsLogger(log_dir=self.log_dir, model_key="model"),
                ValidateOneEpoch(
                    dataloader_key="val.dataloader",
                    metrics_key="val.metrics",
                    loss_fn_key="loss_fn",
                ),
                EarlyStopping(
                    min_delta=cfg.loops.train.early_stopping.delta,
                    patience=cfg.loops.train.early_stopping.patience,
                    mode=cfg.loops.train.early_stopping.mode,
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
                    file_name=(
                        cfg.model.name + "_bg"
                        if not cfg.experiment.exclude_background
                        else cfg.model.name
                    ),
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
                    file_name=(
                        cfg.data.file_name + "_bg"
                        if not cfg.experiment.exclude_background
                        else cfg.data.file_name
                    ),
                    fmt=cfg.data.format,
                ),
                ExtractSplitPartitions(),
                DataFrameProfiler(
                    df_key="test.data",
                    output_key="test.data_profile",
                ),
                DFToNumpy(
                    df_key="test.data",
                    output_key="test.labels",
                    cols=f"multi_{cfg.data.label_column}",
                ),
                BuildModel(model_name=cfg.model.name, model_args=cfg.model.args),
                LoadModelWeights(
                    file_path=cfg.path.model,
                    file_name=(
                        cfg.model.name + "_bg"
                        if not cfg.experiment.exclude_background
                        else cfg.model.name
                    ),
                    fmt="pt",
                ),
                BuildDataset(
                    df_key="test.data",
                    dataset_key="test.dataset",
                    label_col=f"multi_{cfg.data.label_column}",
                ),
                BuildDataLoader(
                    dataloader_args=cfg.loops.test.dataloader,
                    dataset_key="test.dataset",
                    dataloader_key="test.dataloader",
                ),
                TBLogger(log_dir=self.log_dir, subject_key="test.data_profile"),
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
                TBLogger(log_dir=self.log_dir, subject_key="test.metrics"),
                MakePredictions(
                    pred_fn=ArgMax(),
                    logits_key="test.outputs.logits",
                    prediction_key="test.preds",
                    confidences_key="test.confidences",
                ),
                SupervisedClassificationMetrics(
                    labels_key="test.labels",
                    predictions_key="test.preds",
                    confidences_key="test.confidences",
                    metrics_key="test.classification_metrics",
                ),
                ComputeIndicesByLabel(
                    sample_size=10000,
                    stratify=True,
                    random_state=cfg.seed,
                    labels_key="test.labels",
                    indices_key="test.sample_indices",
                ),
                SelectSamplesByIndices(
                    data_key="test.outputs.latents",
                    indices_key="test.sample_indices",
                    output_key="test.sampled_latents",
                ),
                SelectSamplesByIndices(
                    data_key="test.labels",
                    indices_key="test.sample_indices",
                    output_key="test.sampled_labels",
                ),
                ClusteringMetrics(
                    vectors_key="test.sampled_latents",
                    labels_key="test.sampled_labels",
                    metrics_key="test.clustering_metrics",
                ),
                VectorsProjectionPlot(
                    vectors_key="test.sampled_latents",
                    labels_key="test.sampled_labels",
                    n_components=2,
                    output_key="test.projection_plot",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.classification_metrics",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.clustering_metrics",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.projection_plot",
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
