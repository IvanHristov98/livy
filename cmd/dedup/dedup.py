import os
from pathlib import Path

import cv2 as cv
import numpy as np

import livy.dedup as dedup


def main():
    dedup_path = _dedup_im_path()

    im = _load_img(dedup_path, "apples.jpeg")
    transformed_im = _load_img(dedup_path, "apples_transformed.jpg")

    extractor = dedup.SIFTExtractor()
    dup_checker = dedup.EarthMoverChecker()

    dedup_svc = dedup.Service(extractor, dup_checker)
    is_duplicate = dedup_svc.is_duplicate(im, transformed_im)

    print("images are duplicate:", is_duplicate)


def _dedup_im_path() -> Path:
    data_path = os.getenv("DATA_PATH")   
    dedup_path = Path(data_path, "dedup")
    
    return str(dedup_path)

def _load_img(path: str, name: str) -> np.ndarray:
    im_path = Path(path, name)

    return cv.imread(str(im_path))


if __name__ == "__main__":
    main()
