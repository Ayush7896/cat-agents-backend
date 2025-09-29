import hashlib
import re
from typing import Dict, Optional, List
from sqlalchemy import Column, String, Text, DateTime, Integer
from datetime import datetime
import logging

from dbsharing.db.database import Base, SessionLocal

logger = logging.getLogger(__name__)


class QuestionCorrections(Base):
    __tablename__ = "question_corrections"
    __table_args__ = {"schema": "rag_chatbot"}

    id = Column(Integer, primary_key=True)
    question_hash = Column(String, unique=True)
    passage_hash = Column(String)
    original_question = Column(Text)
    original_passage = Column(Text)
    wrong_answer = Column(Text)
    correct_answer = Column(Text)
    user_explanation = Column(Text)
    correction_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


def extract_question_core(text: str) -> str:
    """Extract core question text, removing options"""
    if not text:
        return ""
    
    text = text.lower()
    
    if "options:" in text:
        text = text.split("options:")[0]
    elif "\na)" in text or "\na." in text:
        text = text.split("\na")[0]
    elif "\n1)" in text or "\n1." in text:
        text = text.split("\n1")[0]
    
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s?]', '', text)
    
    return text


def create_question_hash(question: str) -> str:
    """Create hash from question text only"""
    core = extract_question_core(question)
    return hashlib.md5(core.encode('utf-8')).hexdigest()


def calculate_similarity(text1: str, text2: str) -> float:
    """Word-level Jaccard similarity"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.split())
    words2 = set(text2.split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def check_previous_correction(passage: str, question: str) -> Optional[Dict]:
    """Check for previous correction using hash matching with fuzzy fallback"""
    db = SessionLocal()
    
    try:
        question_hash = create_question_hash(question)
        
        # Try exact hash match
        correction = db.query(QuestionCorrections).filter(
            QuestionCorrections.question_hash == question_hash
        ).first()
        
        if correction:
            return {
                "correct_answer": correction.correct_answer,
                "explanation": correction.user_explanation or "",
                "source": "user_correction",
                "correction_count": correction.correction_count,
                "correction_id": correction.id
            }
        
        # Fuzzy match fallback
        question_core = extract_question_core(question)
        all_corrections = db.query(QuestionCorrections).all()
        
        for corr in all_corrections:
            stored_core = extract_question_core(corr.original_question)
            similarity = calculate_similarity(question_core, stored_core)
            
            if similarity > 0.90:
                return {
                    "correct_answer": corr.correct_answer,
                    "explanation": corr.user_explanation or "",
                    "source": "user_correction_fuzzy",
                    "correction_count": corr.correction_count,
                    "correction_id": corr.id
                }
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking correction: {e}")
        return None
    finally:
        db.close()


def save_correction(
    passage: str, 
    question: str, 
    wrong_answer: str, 
    correct_answer: str, 
    user_explanation: str = None
) -> bool:
    """Save correction"""
    db = SessionLocal()
    
    try:
        question_hash = create_question_hash(question)
        passage_hash = hashlib.md5(passage.encode('utf-8')).hexdigest() if passage else ""
        
        existing = db.query(QuestionCorrections).filter(
            QuestionCorrections.question_hash == question_hash
        ).first()
        
        if existing:
            existing.correct_answer = correct_answer
            existing.wrong_answer = wrong_answer
            existing.correction_count += 1
            existing.updated_at = datetime.now()
            
            if user_explanation:
                existing.user_explanation = user_explanation
            if passage:
                existing.original_passage = passage
        else:
            new_correction = QuestionCorrections(
                question_hash=question_hash,
                passage_hash=passage_hash,
                original_question=question,
                original_passage=passage,
                wrong_answer=wrong_answer,
                correct_answer=correct_answer,
                user_explanation=user_explanation or ""
            )
            db.add(new_correction)
        
        db.commit()
        return True
        
    except Exception as e:
        logger.error(f"Error saving correction: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def get_all_corrections(limit: int = 100) -> List[Dict]:
    """Get all corrections"""
    db = SessionLocal()
    try:
        corrections = db.query(QuestionCorrections).order_by(
            QuestionCorrections.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": c.id,
                "question": c.original_question[:150],
                "correct_answer": c.correct_answer,
                "explanation": c.user_explanation or "No explanation",
                "count": c.correction_count,
                "created": str(c.created_at)
            }
            for c in corrections
        ]
    finally:
        db.close()


def migrate_existing_data():
    """Update existing records with new hash format"""
    db = SessionLocal()
    try:
        corrections = db.query(QuestionCorrections).all()
        
        for corr in corrections:
            new_hash = create_question_hash(corr.original_question)
            if corr.question_hash != new_hash:
                corr.question_hash = new_hash
        
        db.commit()
        logger.info(f"Migration complete: {len(corrections)} records updated")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        db.rollback()
    finally:
        db.close()