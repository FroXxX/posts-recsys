from pydantic import BaseModel, Field


class PostResponse(BaseModel):
    id: int = Field(validation_alias='post_id')
    text: str
    topic: str

    class Config:
        from_attributes = True
        populate_by_name = True
