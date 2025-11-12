from datetime import datetime

from omegaconf import DictConfig


from ..base import Experiment
from .. import ExperimentFactory

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
from idspy.src.idspy.builtins.step.data.adjust import (
    DropNulls,
    RareClassFilter,
    DFToNumpy,
    Filter,
    Clip,
)
from idspy.src.idspy.builtins.step.data.sample import (
    ComputeIndicesByLabel,
    SelectSamplesByIndices,
)
from idspy.src.idspy.builtins.step.data.map import FrequencyMap, LabelMap, ColumnMap
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

from idspy.src.idspy.builtins.step.nn.torch.engine.tensor import CatTensors, ToArray
from idspy.src.idspy.builtins.step.nn.torch.engine.epoch import (
    TrainOneEpoch,
    ValidateOneEpoch,
)

from idspy.src.idspy.builtins.step.nn.torch.engine.early_stopping import EarlyStopping
from idspy.src.idspy.builtins.step.nn.torch.model.io import (
    LoadModelWeights,
    SaveModelWeights,
)
from idspy.src.idspy.builtins.step.metric.classification import (
    UnsupervisedClassificationMetrics,
)
from idspy.src.idspy.builtins.step.metric.clustering import ClusteringMetrics
from idspy.src.idspy.builtins.step.metric.projection import VectorsProjectionPlot

from idspy.src.idspy.builtins.step.log.tensorboard import TBLogger, TBWeightsLogger
from idspy.src.idspy.builtins.step.log.profiler import DataFrameProfiler

from idspy.src.idspy.builtins.step.ml.cluster.algorithms import (
    KMeans,
    GaussianMixture,
    HDBSCAN,
)


@ExperimentFactory.register()
class UnsupervisedClassifier(Experiment):

    def __init__(self, cfg: DictConfig, storage: DictStorage) -> None:
        super().__init__(cfg, storage)

        if cfg.experiment.exclude_background:
            if cfg.experiment.benign_tag is None:
                raise ValueError(
                    "benign_tag must be specified for the experiment when exclude_background is True."
                )

    def preprocessing(self) -> None:
        cfg = self.cfg
        storage = self.storage

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
                LabelMap(
                    benign_tag=(
                        cfg.experiment.benign_tag
                        if cfg.experiment.exclude_background
                        else cfg.data.benign_tag
                    ),
                    target_col=f"binary_{cfg.data.label_column}",
                ),
                ColumnMap(
                    source_col=cfg.data.label_column,
                    target_col=f"multi_{cfg.data.label_column}",
                ),
            ],
            name="fit_aware_pipeline",
            bus=bus,
            storage=storage,
        )

        query = None
        if cfg.experiment.exclude_background:
            if cfg.experiment.malicious_tag is not None:
                query = f"{cfg.data.label_column} == '{cfg.experiment.benign_tag}' or {cfg.data.label_column} == '{cfg.experiment.malicious_tag}'"
            else:
                query = f"{cfg.data.label_column} != '{cfg.data.benign_tag}'"

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
                    query=query,
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

    def training(self) -> None:
        cfg = self.cfg
        storage = self.storage

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
                Filter(
                    df_key="train.data",
                    query=(
                        f"{cfg.data.label_column} == '{cfg.experiment.benign_tag}'"
                        if cfg.experiment.exclude_background
                        else f"{cfg.data.label_column} == '{cfg.data.benign_tag}'"
                    ),
                ),
                Filter(
                    df_key="val.data",
                    query=(
                        f"{cfg.data.label_column} == '{cfg.experiment.benign_tag}'"
                        if cfg.experiment.exclude_background
                        else f"{cfg.data.label_column} == '{cfg.data.benign_tag}'"
                    ),
                ),
                BuildModel(model_name=cfg.model.name, model_args=cfg.model.args),
                BuildLoss(loss_name=cfg.loss.name, loss_args=cfg.loss.args),
                BuildOptimizer(
                    optimizer_name=cfg.optimizer.name,
                    optimizer_args=cfg.optimizer.args,
                    loss_key="loss_fn",
                ),
                BuildDataset(
                    df_key="train.data",
                    dataset_key="train.dataset",
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

    def testing(self) -> None:
        cfg = self.cfg
        storage = self.storage

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
                    output_key="test.binary_labels",
                    cols=f"binary_{cfg.data.label_column}",
                ),
                DFToNumpy(
                    df_key="test.data",
                    output_key="test.multi_labels",
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
                ),
                BuildDataLoader(
                    dataloader_args=cfg.loops.test.dataloader,
                    dataset_key="test.dataset",
                    dataloader_key="test.dataloader",
                ),
                BuildLoss(
                    loss_name=cfg.loss.name,
                    loss_args={
                        "reduction": "none",
                        "learnable_weight": False,
                        "numerical_sigma": 0.5691,
                        "categorical_sigma": 0.1,
                    },
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
                    losses_key="test.losses",
                    loss_fn_key="loss_fn",
                    save_outputs=True,
                ),
                CatTensors(
                    tensors_key="test.losses",
                    output_key="test.predictions_tensor",
                ),
                ToArray(
                    tensor_key="test.predictions_tensor",
                    output_key="test.predictions",
                ),
                TBLogger(log_dir=self.log_dir, subject_key="test.metrics"),
                UnsupervisedClassificationMetrics(
                    labels_key="test.binary_labels",
                    predictions_key="test.predictions",
                    metrics_key="test.classification_metrics",
                ),
                ComputeIndicesByLabel(
                    sample_size=10000,
                    stratify=True,
                    random_state=cfg.seed,
                    labels_key="test.multi_labels",
                    indices_key="test.multi_sample_indices",
                ),
                ComputeIndicesByLabel(
                    sample_size=10000,
                    stratify=True,
                    random_state=cfg.seed,
                    labels_key="test.binary_labels",
                    indices_key="test.binary_sample_indices",
                ),
                SelectSamplesByIndices(
                    data_key="test.outputs.z",
                    indices_key="test.multi_sample_indices",
                    output_key="test.multi_sampled_z",
                ),
                SelectSamplesByIndices(
                    data_key="test.multi_labels",
                    indices_key="test.multi_sample_indices",
                    output_key="test.multi_sampled_labels",
                ),
                SelectSamplesByIndices(
                    data_key="test.outputs.z",
                    indices_key="test.binary_sample_indices",
                    output_key="test.binary_sampled_z",
                ),
                SelectSamplesByIndices(
                    data_key="test.binary_labels",
                    indices_key="test.binary_sample_indices",
                    output_key="test.binary_sampled_labels",
                ),
                ClusteringMetrics(
                    vectors_key="test.multi_sampled_z",
                    labels_key="test.multi_sampled_labels",
                    metrics_key="test.multi_latent_metrics",
                ),
                ClusteringMetrics(
                    vectors_key="test.binary_sampled_z",
                    labels_key="test.binary_sampled_labels",
                    metrics_key="test.binary_latent_metrics",
                ),
                VectorsProjectionPlot(
                    vectors_key="test.multi_sampled_z",
                    labels_key="test.multi_sampled_labels",
                    n_components=2,
                    output_key="test.multi_projection_plot",
                ),
                VectorsProjectionPlot(
                    vectors_key="test.binary_sampled_z",
                    labels_key="test.binary_sampled_labels",
                    n_components=2,
                    output_key="test.binary_projection_plot",
                ),
                GaussianMixture(
                    n_clusters=7,
                    data_key="test.multi_sampled_z",
                    output_key="test.gm_labels",
                ),
                ClusteringMetrics(
                    vectors_key="test.multi_sampled_z",
                    labels_key="test.gm_labels",
                    metrics_key="test.gm_metrics",
                ),
                VectorsProjectionPlot(
                    vectors_key="test.multi_sampled_z",
                    labels_key="test.gm_labels",
                    n_components=2,
                    output_key="test.gm_projection_plot",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.classification_metrics",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.multi_latent_metrics",
                    secondary_prefix="multi_z",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.binary_latent_metrics",
                    secondary_prefix="binary_z",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.binary_projection_plot",
                    secondary_prefix="binary_z",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.multi_projection_plot",
                    secondary_prefix="multi_z",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.gm_metrics",
                    secondary_prefix="gm",
                ),
                TBLogger(
                    log_dir=self.log_dir,
                    subject_key="test.gm_projection_plot",
                    secondary_prefix="gm",
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
