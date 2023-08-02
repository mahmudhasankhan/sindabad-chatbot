import uvicorn
from fastapi import FastAPI
from chain import make_chain
from schemas import ChatResponse


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello! How can I help?"}


@app.post("/chat")
async def get_chat_response(question: ChatResponse) -> dict:

    chain = make_chain("sindabad")

    try:
        response = chain({"question": question.message})
        answer = ChatResponse(sender="bot", message=response["answer"])
        user_source = response["source_documents"]
        source_dict = {}
        for document in user_source:
            source_dict[f"Page = {document.metadata['page_number']}"] = f"Text chunk: {document.page_content[:160]}...\n"
        print(source_dict)
        return answer.dict()
    except Exception as e:
        print(f"Exception {e} has ocurred")
        error_resp = ChatResponse(
            sender="bot",
            message="Sorry, something went wrong, Try Again"
        )
        return error_resp


def main():
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
