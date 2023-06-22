from enum import Enum
from pathlib import Path
import os
from typing import NamedTuple


class DedupStrategy(Enum):
    SIFT_KNN = "sift_and_knn"
    SIGNATURE = "signature"


class DedupConfig(NamedTuple):
    volume: Path
    strategy: DedupStrategy


class ConfigError(Exception):
    """
    ConfigError is thrown whenever there is a configuration error.
    """


def read_dedup_cfg() -> DedupConfig:
    volume = _read_dedup_volume()
    strategy = _read_dedup_strategy()

    return DedupConfig(
        volume=volume,
        strategy=strategy,
    )


def _read_dedup_volume() -> Path:
    raw_volume = os.getenv("DEDUP_VOLUME", None)
    if raw_volume is None:
        raise ConfigError("volume is not provided")
    
    path = Path(raw_volume)
    if not path.exists():
        raise ConfigError("non-existing volume provided")
    
    return path


def _read_dedup_strategy() -> DedupStrategy:
    raw_strategy = os.getenv("DEDUP_STRATEGY", DedupStrategy.SIFT_KNN.value)

    if raw_strategy == DedupStrategy.SIFT_KNN.value:
        return DedupStrategy.SIFT_KNN
    elif raw_strategy == DedupStrategy.SIGNATURE.value:
        return DedupStrategy.SIGNATURE
    else:
        raise ConfigError(f"unknown deduplication strategy {raw_strategy} requested")
