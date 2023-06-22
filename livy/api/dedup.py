from fastapi import APIRouter, status, UploadFile, HTTPException
from typing import List
from pydantic import BaseModel
import uuid

import numpy as np
import cv2 as cv

import livy.depinjection as depinjection
import livy.model as model
import livy.id as id


router = APIRouter(prefix="/api/dedup")
deps = depinjection.dependencies()


@router.post("/add/{name}", status_code=status.HTTP_201_CREATED)
async def add(name: str, img: UploadFile):
    try:
        svc = deps.dedup_svc

        mat = await read_img(img)

        im_id = id.NewImage()
        im = model.Image(im_id, name, mat)
        svc.add_im(im)
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))


class TopNResponse(BaseModel):
    im_ids: List[str]


@router.post("/topn", status_code=status.HTTP_200_OK, response_model=TopNResponse)
async def topn(img: UploadFile, n: int = 5):
    try:
        svc = deps.dedup_svc 

        mat = await read_img(img)
        im_id = id.NewImage()
        im = model.Image(im_id, "", mat)

        im_ids = svc.similar_ims(im, n)
        uuids = []

        for im_id in im_ids:
            uuids.append(str(im_id))

        return TopNResponse(im_ids=uuids)
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))


async def read_img(img: UploadFile) -> cv.Mat:
    try:
        img_content = await img.read()
    finally:
        await img.close()

    jpg_as_np = np.frombuffer(img_content, dtype=np.uint8)
    return cv.imdecode(jpg_as_np, flags=cv.IMREAD_COLOR)
