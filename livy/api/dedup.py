from fastapi import APIRouter, status, UploadFile

import numpy as np
import cv2 as cv

import livy.depinjection as depinjection
import livy.model as model
import livy.id as id


router = APIRouter(prefix="/api/dedup")
deps = depinjection.dependencies()


@router.post("/add/{name}", status_code=status.HTTP_201_CREATED)
async def add(name: str, img: UploadFile):
    svc = deps.dedup_svc

    mat = await read_img(img)

    im_id = id.NewImage()
    im = model.Image(im_id, name, mat)
    svc.add_im(im)


async def read_img(img: UploadFile) -> cv.Mat:
    try:
        img_content = await img.read()
    finally:
        await img.close()

    jpg_as_np = np.frombuffer(img_content, dtype=np.uint8)
    return cv.imdecode(jpg_as_np, flags=cv.IMREAD_COLOR)
