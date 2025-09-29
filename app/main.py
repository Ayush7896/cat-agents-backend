# main.py
from app.graph import build_workflow
from fastapi import FastAPI, HTTPException, Depends
from app.models.schemas import CATRequest, CATResponse, CorrectionRequest
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from app.core.feedback_system import (
    check_previous_correction, 
    save_correction,
    get_all_corrections,
    create_question_hash  
)
# Import authentication from database-sharing package
from dbsharing.auth.auth import router as auth_router, get_current_user
from dbsharing.auth.auth import router as admin_router

from app.core.logging_config import setup_logging
import logging

# setup logging once
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Initialize workflow and app
workflow = build_workflow()
app = FastAPI(title="CAT VARC Backend Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routers from database-sharing
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])

# Dependency for protected endpoints
user_dependency = Annotated[dict, Depends(get_current_user)]

# Health check endpoint (public)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cat-varc-backend"}

@app.post("/ask", response_model=CATResponse)
async def ask(
    request: CATRequest,
    current_user: user_dependency
):
    """
    Protected endpoint - requires valid JWT token
    User info available in current_user dict
    """
    try:
        user_id = current_user.get("user_id")
        username = current_user.get("username")
        
        logger.info(f"User {username} (ID: {user_id}) asking question")
        logger.info(f"Question: {request.user_query}")
        
        # Check for previous corrections first
        logger.info("🔍 Checking for previous corrections...")
        previous_correction = check_previous_correction(
            passage=request.passage,
            question=request.user_query
        )
        
        if previous_correction:
            logger.info(f"✅ Found previous correction for user {username}")
            logger.info(f"Correct answer: {previous_correction['correct_answer']}")
            
            # Build detailed response with explanation
            response_text = f"**Answer: {previous_correction['correct_answer']}**\n\n"
            
            if previous_correction.get('explanation'):
                response_text += f"**Explanation:**\n{previous_correction['explanation']}\n\n"
            
            response_text += "*✓ This answer is based on previous user feedback*"
            
            return CATResponse(final_answer=response_text)
        
        logger.info("No previous correction found, proceeding with AI")
        
        # Your existing logic
        thread_id = request.thread_id
        config = {"configurable": {"thread_id": thread_id}}
        
        user_input_state = {
            "user_query": request.user_query, 
            "passage": request.passage
        }
        
        final_state = workflow.invoke(user_input_state, config=config)
        logger.debug(f"Workflow keys: {final_state.keys()}")
        
        final_answer = final_state.get("final_answer", "No answer generated")
        logger.info(f"Generated answer: {final_answer[:100]}...")
        
        return CATResponse(final_answer=final_answer)
        
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/correct-answer")
async def submit_correction(
    correction: CorrectionRequest,
    current_user: user_dependency
):
    """Submit a correction for an incorrect answer"""
    try:
        username = current_user.get("username")
        logger.info(f"📝 User {username} submitting correction")
        logger.info(f"Question: {correction.question[:100]}...")
        logger.info(f"Correct answer: {correction.correct_answer}")
        
        # Save the correction
        saved = save_correction(
            passage=correction.passage,
            question=correction.question,
            wrong_answer=correction.wrong_answer,
            correct_answer=correction.correct_answer,
            user_explanation=correction.explanation
        )
        
        if saved:
            logger.info("✅ Correction saved successfully")
            return {
                "status": "success",
                "message": "Thank you! I'll remember this correction for next time.",
                "details": {
                    "question": correction.question[:100],
                    "correct_answer": correction.correct_answer
                }
            }
        else:
            logger.error("❌ Failed to save correction")
            return {
                "status": "error",
                "message": "Failed to save correction"
            }
            
    except Exception as e:
        logger.error(f"❌ Error saving correction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))