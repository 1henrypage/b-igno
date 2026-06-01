"""
- Single training phase (encoder + decoders + NF jointly)
- Evaluation config for inversion methods
"""
from dataclasses import dataclass, asdict, field, fields
from typing import Literal, Optional, List, Dict, Any, Type, TypeVar
from pathlib import Path
import yaml
import numpy as np
import jax.numpy as jnp


def _serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, jnp.ndarray):
        return float(obj) if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    return obj


T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        if data is None:
            data = {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))


@dataclass
class ProblemConfig(BaseConfig):
    """Problem type and data paths."""
    type: str = None
    train_data: Optional[str] = None
    test_data: Optional[str] = None


@dataclass
class OptimizerConfig(BaseConfig):
    type: Literal['Adam', 'AdamW', 'RMSprop', 'SGD'] = 'Adam'
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig(BaseConfig):
    type: Optional[Literal['StepLR', 'CosineAnnealing', 'OneCycle']] = None
    # StepLR params
    step_size: Optional[int] = None
    gamma: Optional[float] = None
    # CosineAnnealing params
    eta_min: Optional[float] = None
    # OneCycle params
    pct_start: Optional[float] = 0.3
    div_factor: Optional[float] = 25.0
    final_div_factor: Optional[float] = 1e4


@dataclass
class LossWeights(BaseConfig):
    pde: float = 1.0
    data: float = 1.0
    nf: float = 1.0  # Set by trainer based on NFTrainingConfig


@dataclass
class NFTrainingConfig(BaseConfig):
    mode: Literal['joint', 'separate'] = 'joint'
    loss_weight: float = 1.0          # NF loss weight in joint mode
    # Separate-mode-only settings:
    epochs: int = 2000
    batch_size: int = 128
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(
        type='Adam', lr=1e-3, weight_decay=1e-4
    ))
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig(
        type='StepLR', step_size=400, gamma=0.333
    ))

    @classmethod
    def from_dict(cls, data: dict) -> 'NFTrainingConfig':
        if data is None:
            return cls()
        return cls(
            mode=data.get('mode', 'joint'),
            loss_weight=data.get('loss_weight', 1.0),
            epochs=data.get('epochs', 2000),
            batch_size=data.get('batch_size', 128),
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
        )


@dataclass
class IGNOTrainingConfig(BaseConfig):
    """
    Config for joint IGNO training (encoder + decoders + NF).

    The NF is trained jointly with encoder/decoders using detached latents,
    as per the original IGNO paper implementation.
    """
    epochs: int = 10000
    batch_size: int = 50
    epoch_show: int = 50

    # Loss weights; NF weight is set by trainer from nf.loss_weight
    loss_weights: LossWeights = field(default_factory=LossWeights)

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # NF training configuration
    nf: NFTrainingConfig = field(default_factory=NFTrainingConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'IGNOTrainingConfig':
        if data is None:
            return cls()
        return cls(
            epochs=data.get('epochs', 10000),
            batch_size=data.get('batch_size', 50),
            epoch_show=data.get('epoch_show', 50),
            loss_weights=LossWeights.from_dict(data.get('loss_weights', {})),
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
            nf=NFTrainingConfig.from_dict(data.get('nf')),
        )


@dataclass
class InversionConfig(BaseConfig):
    epochs: int = 500
    loss_weights: LossWeights = None
    optimizer: OptimizerConfig = None
    scheduler: SchedulerConfig = None

    @classmethod
    def from_dict(cls, data: dict) -> 'InversionConfig':
        if data is None:
            return cls()
        return cls(
            epochs=data.get('epochs', 500),
            loss_weights=LossWeights.from_dict(data.get('loss_weights', {})),
            optimizer=OptimizerConfig.from_dict(data.get('optimizer', {})),
            scheduler=SchedulerConfig.from_dict(data.get('scheduler', {})),
        )


@dataclass
class EvaluationConfig(BaseConfig):
    """Config for evaluation/inversion."""
    method: Literal['igno', 'mcmc'] = 'igno'

    batch_size: int = 200
    n_obs: int = 100
    obs_sampling: Literal['random', 'grid'] = 'random'

    # Noise (None for clean)
    snr_db: Optional[float] = None

    inversion: InversionConfig = field(default_factory=InversionConfig)
    results_dir: str = "results"

    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluationConfig':
        if data is None:
            return cls()
        return cls(
            method=data.get('method', 'igno'),
            batch_size=data.get('batch_size', 200),
            n_obs=data.get('n_obs', 100),
            obs_sampling=data.get('obs_sampling', 'random'),
            snr_db=data.get('snr_db'),
            inversion=InversionConfig.from_dict(data.get('inversion', {})),
            results_dir=data.get('results_dir', 'results'),
        )


@dataclass
class TrainingConfig(BaseConfig):
    run_name: str = None
    artifact_root: str = "runs"
    seed: int = 10086

    problem: ProblemConfig = field(default_factory=ProblemConfig)
    pretrained: Optional[Dict[str, Any]] = None

    training: IGNOTrainingConfig = field(default_factory=IGNOTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        if data is None:
            return cls()

        problem_data = data.get('problem', {})
        if isinstance(problem_data, str):
            problem_data = {'type': problem_data}

        return cls(
            run_name=data.get('run_name'),
            artifact_root=data.get('artifact_root', 'runs'),
            seed=data.get('seed', 10086),
            problem=ProblemConfig.from_dict(problem_data),
            pretrained=data.get('pretrained'),
            training=IGNOTrainingConfig.from_dict(data.get('training', {})),
            evaluation=EvaluationConfig.from_dict(data.get('evaluation', {})),
        )

    @classmethod
    def load(cls, path: Path) -> 'TrainingConfig':
        with open(Path(path), 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_pretrained_path(self) -> Optional[Path]:
        if self.pretrained is None:
            return None
        path = Path(self.pretrained.get('path', ''))
        if not path.exists():
            raise RuntimeError("Pretrained path doesn't exist.")
        if path.suffix == '.pt' or path.suffix == '.pkl':
            return path
        # Default to best.pt in weights directory
        checkpoint = self.pretrained.get('checkpoint', 'best.pt')
        return path / 'weights' / checkpoint
