from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class InstantiationConfig:
    _target_: str
    config: Any = None
    args: Any = None
    kwargs: Any = None


@dataclass
class ObjectReferenceConfig:
    _target_: str


@dataclass
class ImageShape:
    channels: int = 3
    width: int = 224
    height: int = 224


@dataclass
class ModalityConfig:
    image: bool = True
    audio: bool = True
    video: bool = True
    text: bool = True
    name: str = ""


@dataclass
class DatasetDirectoryConfig:
    train: Optional[str] = None
    val: Optional[str] = None
    test: Optional[str] = None


@dataclass
class DataLoaderConfig:
    batch_size: int = 2
    persistent_workers: bool = False
    pin_memory: bool = True
    prefetch_factor: int = 2
    num_workers: int = 2
    shuffle_train: bool = True
    shuffle_eval: bool = False
    train_start_index: int = 0
    val_start_index: int = 0
    test_start_index: int = 0
    train_num_samples: Optional[int] = 200000000
    val_num_samples: Optional[int] = None
    test_num_samples: Optional[int] = None
    use_dummy_dataloader: bool = False


@dataclass
class DatasetConfig:
    dataset_dir_config: DatasetDirectoryConfig
    using_pre_sampled_split: bool = False
    dataset_size_identifier: str = "base"
    dataset_name: str = "base"
    modality_config: ModalityConfig = ModalityConfig()
    rescan_paths: bool = False
    num_video_frames_per_datapoint: int = 10
    num_audio_frames_per_datapoint: int = 88200
    num_audio_sample_rate: int = 44100
    image_shape: ImageShape = ImageShape(channels=3, width=224, height=224)
    text_context_length: int = 77


@dataclass
class AutoCLIPResNetConfig:
    vision_layers: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    embedding_output_features: int = 512
    image_resolution: int = 224
    vision_width: int = 16


@dataclass
class AutoConv1DTransformersConfig:
    embedding_output_features: int = 512
    resnet_num_filters: int = 64
    resnet_num_stages: int = 3
    resnet_num_blocks: int = 4
    resnet_kernel_size: int = 3
    resnet_adaptive_pool_output_features: int = 4096
    resnet_dilated: bool = True
    grid_patch_size: int = 11025
    transformer_num_filters: int = 32
    transformer_num_layers: int = 12
    transformer_num_heads: int = 8
    transformer_dim_feedforward: int = 128


@dataclass
class AutoCLIPTextTransformerConfig:
    transformer_num_filters: int = 32
    transformer_num_layers: int = 12
    transformer_num_heads: int = 8
    transformer_dim_feedforward: int = 128
    vocab_size: int = 49408
    context_length: int = 77
    embedding_output_features: int = 512


@dataclass
class AutoCLIPVisionTransformerConfig:
    embedding_output_features: int = 512
    image_resolution: Tuple[int] = (176, 288)
    grid_patch_size: int = 16
    transformer_num_filters: int = 768
    transformer_num_layers: int = 12
    transformer_num_heads: int = 12
    transformer_dim_feedforward: int = 3072


@dataclass
class AutoAveragerConfig:
    embedding_output_features: int = 512
    dim: int = 1


@dataclass
class AutoVideoTransformersConfig:
    transformer_num_filters: int = 512
    transformer_num_layers: int = 12
    transformer_num_heads: int = 8
    transformer_dim_feedforward: int = 2048
    embedding_output_features: int = 512
    dim: int = 1


@dataclass
class TrainerModes:
    fit: bool = True
    test: bool = True
    predict: bool = True


@dataclass
class TrainerConfig:
    _target_ = "pytorch_lightning.trainer.Trainer"
    logger: Any = None
    checkpoint_callback: Optional[bool] = (None,)
    enable_checkpointing: bool = (True,)
    callbacks: Any = None
    default_root_dir: Optional[str] = (None,)
    gradient_clip_val: Any = None
    gradient_clip_algorithm: Optional[str] = (None,)
    process_position: int = (0,)
    num_nodes: int = (1,)
    num_processes: int = (1,)
    devices: Any = None
    gpus: Any = None
    auto_select_gpus: bool = (False,)
    tpu_cores: Any = None
    ipus: Optional[int] = (None,)
    log_gpu_memory: Optional[str] = (None,)  # TODO: Remove in 1.7
    progress_bar_refresh_rate: Optional[int] = (None,)  # TODO: remove in v1.7
    enable_progress_bar: bool = (True,)
    overfit_batches: Any = None
    track_grad_norm: Any = None
    check_val_every_n_epoch: int = (1,)
    fast_dev_run: Any = None
    accumulate_grad_batches: Any = None
    max_epochs: Optional[int] = (None,)
    min_epochs: Optional[int] = (None,)
    max_steps: int = (-1,)
    min_steps: Optional[int] = (None,)
    max_time: Any = None
    limit_train_batches: Any = None
    limit_val_batches: Any = None
    limit_test_batches: Any = None
    limit_predict_batches: Any = None
    val_check_interval: Any = None
    flush_logs_every_n_steps: Optional[int] = (None,)
    log_every_n_steps: int = (50,)
    accelerator: Any = None
    strategy: Any = None
    sync_batchnorm: bool = (False,)
    precision: Any = None
    enable_model_summary: bool = (True,)
    weights_summary: Optional[str] = ("top",)
    weights_save_path: Optional[str] = (None,)
    num_sanity_val_steps: int = (2,)
    resume_from_checkpoint: Any = None
    profiler: Any = None
    benchmark: bool = (False,)
    deterministic: bool = (False,)
    reload_dataloaders_every_n_epochs: int = (0,)
    reload_dataloaders_every_epoch: bool = (False,)
    auto_lr_find: Any = None
    replace_sampler_ddp: bool = (True,)
    detect_anomaly: bool = (False,)
    auto_scale_batch_size: Any = None
    prepare_data_per_node: Optional[bool] = (None,)
    plugins: Any = None
    amp_backend: str = ("native",)
    amp_level: Optional[str] = (None,)
    move_metrics_to_cpu: bool = (False,)
    multiple_trainloader_mode: str = ("max_size_cycle",)
    stochastic_weight_avg: bool = (False,)
    terminate_on_nan: Optional[bool] = (None,)


@dataclass
class ModusPrimeConfig:
    image_embedding_config: InstantiationConfig = InstantiationConfig(
        _target_="tali.models.auto_builder.transformers.AutoCLIPResNet",
        config=AutoCLIPResNetConfig(),
    )
    audio_embedding_config: InstantiationConfig = InstantiationConfig(
        _target_="tali.models.auto_builder.transformers.AutoConv1DTransformers",
        config=AutoConv1DTransformersConfig(),
    )
    text_embedding_config: InstantiationConfig = InstantiationConfig(
        _target_="tali.models.auto_builder.transformers.AutoCLIPTextTransformer",
        config=AutoCLIPTextTransformerConfig(),
    )
    video_embedding_config: InstantiationConfig = InstantiationConfig(
        _target_="tali.models.auto_builder.transformers.AutoAverager",
        config=AutoAveragerConfig,
    )
    optimizer: InstantiationConfig = InstantiationConfig(
        _target_="torch.optim.Adam", config={"lr": 0.01}
    )
    lr_scheduler: InstantiationConfig = InstantiationConfig(
        _target_="torch.optim.lr_scheduler.CosineAnnealingLR",
        config=dict(eta_min=0, verbose=False),
    )
    name: str = "modus_prime"
    embedding_output_features: int = 512
    logit_scale: float = 1.0


class ExperimentStatus:
    def __init__(self, status_string="new"):
        # should be one of new, continued-full, continued-minimal
        assert any(
            status_string == valid_entry
            for valid_entry in ("new", "continued-full", "continued-minimal")
        )
        self.status_string = status_string

    def __call__(self):
        return self.status_string


@dataclass
class GoogleStorageConfig:
    local_experiments_root_dir: str
    experiment_name: str
    from_scratch: bool
    use_google_storage: bool
    bucket_name: str = "tali-experiments"
