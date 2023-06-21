import uuid
from typing import NamedTuple


class Image(NamedTuple):
    id: uuid.UUID

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: "Image"):
        return self.id == other.id

    def __str__(self) -> str:
        return str(self.id)


NoImage = Image(id=uuid.UUID(int=0))


def NewImage() -> Image:
    return Image(uuid.uuid4())
