from functools import lru_cache
from typing import NamedTuple

import livy.depinjection.config as config
import livy.dedup.store.persistent.siftknn as bruteforcestore
import livy.dedup.feature as dedupfeature
import livy.dedup.service as dedupservice


class Bundle(NamedTuple):
    dedup_svc: dedupservice.Service


@lru_cache()
def dependencies() -> Bundle:
    dedup_cfg = config.read_dedup_cfg()

    svc = _dedup_svc(dedup_cfg)

    return Bundle(
        dedup_svc=svc,
    )


def _dedup_svc(dedup_cfg: config.DedupConfig) -> dedupservice.Service:
    if dedup_cfg.strategy == config.DedupStrategy.SIFT_KNN:
        im_store = bruteforcestore.ImageStore(dedup_cfg.volume)
        sift_extractor = dedupfeature.SIFTExtractor()

        return dedupservice.BruteForceService(sift_extractor, im_store)
    
    spin_im_extractor = dedupfeature.SpinImageExtractor()
    return dedupservice.SignatureService(spin_im_extractor)
