from app.graph import build_workflow
from fastapi import FastAPI, HTTPException
from app.models.schemas import CATRequest, CATResponse
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage

workflow = build_workflow()
app = FastAPI()

@app.post("/ask", response_model=CATResponse)
async def ask(request: CATRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    current_state = workflow.get_state(config)
    existing_messages = current_state.values.get("conversation_messages", [])
    
    new_message = HumanMessage(content=request.user_query)
    
    input_state = {
        "passage": request.passage,
        "user_query": request.user_query,
        "conversation_messages": existing_messages + [new_message]
    }
    
    final_state = workflow.invoke(input_state, config=config)
    final_answer = final_state.get("final_answer", "No answer generated")
    print(final_state.keys())
    # print(f"conversation mesages are ----- {final_state.get("conversation_messages")}  ")
    # print(f"conversation mesages are ----- {type(final_state.get("conversation_messages"))}")

    # print("histroy",list(workflow.get_state_history(config)))

    if isinstance(final_answer, AIMessage):
      final_answer = final_answer.content

    return CATResponse(final_answer=final_answer)
    
   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# uvicorn app.main:app --reload




