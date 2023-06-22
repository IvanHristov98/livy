from fastapi import FastAPI

import livy.api as api # noqa


app = FastAPI()

for r in api.routers:
    app.include_router(r)
