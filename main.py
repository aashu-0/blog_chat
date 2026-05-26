import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from rag import load_url, chat, delete_session

app = FastAPI(title="RAG Blog Chat")
app.mount("/static", StaticFiles(directory="static"), name="static")


# response models

class LoadRequest(BaseModel):
    url: str


class LoadResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    answer: str



@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/load", response_model=LoadResponse)
def load_blog(req: LoadRequest):
    try:
        session_id, num_chunks = load_url(req.url)
        return LoadResponse(
            session_id=session_id,
            message=f"Blog loaded and indexed into {num_chunks} chunks. You can now chat!",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load URL: {e}")


@app.post("/chat", response_model=ChatResponse)
def chat_with_blog(req: ChatRequest):
    try:
        answer = chat(req.session_id, req.question, req.history)
        return ChatResponse(answer=answer)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted and vectors removed from Pinecone."}