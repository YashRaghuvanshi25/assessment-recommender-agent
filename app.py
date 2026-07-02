from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import invoke
from schemas import ChatMessage, ChatRequest, ChatResponse, Recommendation


app = FastAPI(title="SHL Assessment Advisor")

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_FILE = PROJECT_ROOT / "index.html"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    return {"status": "healthy"}


ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
    "tool": ToolMessage,
}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = []

    for msg in request.messages:
        message_cls = ROLE_TO_MESSAGE[msg.role]

        if msg.role == "tool":
            messages.append(message_cls(content=msg.content, tool_call_id="tool"))
        else:
            messages.append(message_cls(content=msg.content))


    result = invoke(messages)

    final_message = result["messages"][-1]
    recommendations = None

    # Extract structured recommendations from tool output
    for message in reversed(result["messages"]):
        if isinstance(message, ToolMessage):
            try:
                import json
                payload = json.loads(message.content)
                if "recommendations" in payload:
                    recommendations = [
                        Recommendation(**item)
                        for item in payload["recommendations"]
                    ]
                    print(f"[DEBUG] Extracted {len(recommendations)} recommendations")
                break
            except Exception as e:
                print(f"[ERROR] Failed to extract recommendations: {e}")
                import traceback
                traceback.print_exc()


    return ChatResponse(
        reply=final_message.content,
        recommendations=recommendations,
    )
