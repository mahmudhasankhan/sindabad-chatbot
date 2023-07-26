from pydantic import BaseModel, validator


class ChatResponse(BaseModel):
    sender: str
    message: str

    @validator("sender")
    def sender_must_be_human_or_bot(cls, v):
        if v not in ["human", "bot"]:
            raise ValueError("Sender must be Human or Bot")
        return v
