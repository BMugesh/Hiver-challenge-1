"""
Repository functions for SupportDNA Live Conversation Memory.
Provides database access for conversations, messages, AI analysis,
attempted steps, and retrieved historical evidence.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
from sqlalchemy import desc, func
from src.database.connection import get_db_session, init_db
from src.database.models import (
    Conversation,
    Message,
    ConversationAnalysis,
    AttemptedStep,
    RetrievedEvidence
)

logger = logging.getLogger("supportdna.database.repository")


def now_utc():
    return datetime.now(timezone.utc)


class ConversationRepository:
    """Repository handling all live conversation persistence."""

    def __init__(self):
        # Ensure database tables exist
        try:
            init_db()
        except Exception as e:
            logger.error(f"[Database] Error initializing database: {e}")

    def get_or_create_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Fetch existing conversation or create a new one."""
        with get_db_session() as session:
            conv = session.query(Conversation).filter_by(conversation_id=conversation_id).first()
            if not conv:
                conv = Conversation(
                    conversation_id=conversation_id,
                    status="ACTIVE",
                    created_at=now_utc(),
                    updated_at=now_utc()
                )
                session.add(conv)
                session.flush()
                logger.info(f"[Database] Created new conversation: {conversation_id}")
            return self._conv_to_dict(conv)

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation by ID."""
        with get_db_session() as session:
            conv = session.query(Conversation).filter_by(conversation_id=conversation_id).first()
            return self._conv_to_dict(conv) if conv else None

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve list of conversations ordered by most recently updated."""
        with get_db_session() as session:
            convs = session.query(Conversation).order_by(Conversation.updated_at.desc()).limit(limit).all()
            return [self._conv_to_dict(c) for c in convs]

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        message_type: str = "CUSTOMER_MESSAGE"
    ) -> Dict[str, Any]:
        """
        Save a message (USER, ASSISTANT, SYSTEM) and update conversation timestamps.
        """
        with get_db_session() as session:
            # Ensure conversation exists
            conv = session.query(Conversation).filter_by(conversation_id=conversation_id).first()
            now = now_utc()
            if not conv:
                conv = Conversation(
                    conversation_id=conversation_id,
                    status="ACTIVE",
                    created_at=now,
                    updated_at=now
                )
                session.add(conv)
                session.flush()

            msg = Message(
                conversation_id=conversation_id,
                role=role.upper(),
                content=content,
                message_type=message_type,
                created_at=now
            )
            session.add(msg)
            session.flush()

            # Update conversation timestamp
            conv.updated_at = now
            if role.upper() == "USER":
                conv.last_user_message_at = now
            elif role.upper() == "ASSISTANT":
                conv.last_assistant_message_at = now

            session.flush()
            return {
                "id": msg.id,
                "conversation_id": msg.conversation_id,
                "role": msg.role,
                "content": msg.content,
                "message_type": msg.message_type,
                "created_at": msg.created_at.isoformat()
            }

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages in chronological order.
        """
        with get_db_session() as session:
            query = session.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.created_at.asc())
            if limit:
                # If limit specified, get the last N messages chronologically
                total = query.count()
                if total > limit:
                    query = query.offset(total - limit)
            msgs = query.all()
            return [
                {
                    "id": m.id,
                    "conversation_id": m.conversation_id,
                    "role": m.role,
                    "content": m.content,
                    "message_type": m.message_type,
                    "created_at": m.created_at.isoformat()
                }
                for m in msgs
            ]

    def save_conversation_analysis(
        self,
        conversation_id: str,
        analysis_data: Dict[str, Any],
        message_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Save AI analysis per turn and update the conversation's active state.
        """
        with get_db_session() as session:
            conv = session.query(Conversation).filter_by(conversation_id=conversation_id).first()
            now = now_utc()

            analysis = ConversationAnalysis(
                conversation_id=conversation_id,
                message_id=message_id,
                business_intent=analysis_data.get("business_intent"),
                intent_confidence=analysis_data.get("intent_confidence"),
                customer_goal=analysis_data.get("customer_goal"),
                issues=analysis_data.get("issues"),
                known_information=analysis_data.get("known_information"),
                missing_information=analysis_data.get("missing_information"),
                follow_up_type=analysis_data.get("follow_up_type"),
                response_mode=analysis_data.get("response_mode"),
                decision=analysis_data.get("decision"),
                evidence_quality=analysis_data.get("evidence_quality"),
                prompt_injection_status=analysis_data.get("prompt_injection_status"),
                multi_issue_detected=bool(analysis_data.get("multi_issue_detected", False)),
                verifier_passed=bool(analysis_data.get("verifier_passed", True)),
                verifier_reason=analysis_data.get("verifier_reason"),
                created_at=now
            )
            session.add(analysis)

            # Update conversation active state
            if conv:
                if analysis_data.get("business_intent"):
                    conv.current_intent = analysis_data["business_intent"]
                if analysis_data.get("intent_confidence") is not None:
                    conv.current_intent_confidence = analysis_data["intent_confidence"]
                if analysis_data.get("customer_goal"):
                    conv.current_customer_goal = analysis_data["customer_goal"]
                if analysis_data.get("issues"):
                    conv.current_issues = analysis_data["issues"]
                if analysis_data.get("follow_up_type"):
                    conv.follow_up_type = analysis_data["follow_up_type"]

                decision = analysis_data.get("decision")
                if decision == "RESOLVED":
                    conv.resolved = True
                    conv.status = "RESOLVED"
                elif decision == "ESCALATE":
                    conv.escalated = True
                    conv.status = "ESCALATED"

                conv.updated_at = now

            session.flush()
            return {"id": analysis.id, "conversation_id": conversation_id}

    def save_attempted_step(
        self,
        conversation_id: str,
        step: str,
        source: str = "AI_RESPONSE",
        result: str = "PENDING"
    ) -> Dict[str, Any]:
        """
        Record a troubleshooting step recommended or attempted.
        If already recorded, updates result/timestamp.
        """
        with get_db_session() as session:
            # Check if similar step already exists
            existing = (
                session.query(AttemptedStep)
                .filter_by(conversation_id=conversation_id)
                .filter(func.lower(AttemptedStep.step) == step.lower().strip())
                .first()
            )
            now = now_utc()
            if existing:
                existing.result = result
                existing.source = source
                session.flush()
                return {
                    "id": existing.id,
                    "step": existing.step,
                    "result": existing.result,
                    "updated": True
                }
            else:
                att = AttemptedStep(
                    conversation_id=conversation_id,
                    step=step.strip(),
                    source=source,
                    result=result,
                    created_at=now
                )
                session.add(att)
                session.flush()
                return {
                    "id": att.id,
                    "step": att.step,
                    "result": att.result,
                    "created": True
                }

    def update_attempted_step_result(
        self,
        conversation_id: str,
        step_query: str,
        result: str
    ) -> bool:
        """
        Find an attempted step matching the query keywords and update its result (e.g. FAILED, SUCCESS).
        """
        with get_db_session() as session:
            steps = session.query(AttemptedStep).filter_by(conversation_id=conversation_id).all()
            q = step_query.lower()
            matched = False
            for s in steps:
                if any(k in q for k in s.step.lower().split() if len(k) > 3) or s.step.lower() in q:
                    s.result = result
                    matched = True
            session.flush()
            return matched

    def get_attempted_steps(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all attempted troubleshooting steps for a conversation."""
        with get_db_session() as session:
            steps = session.query(AttemptedStep).filter_by(conversation_id=conversation_id).order_by(AttemptedStep.created_at.asc()).all()
            return [
                {
                    "id": s.id,
                    "step": s.step,
                    "source": s.source,
                    "result": s.result,
                    "created_at": s.created_at.isoformat()
                }
                for s in steps
            ]

    def save_retrieved_evidence(
        self,
        conversation_id: str,
        cases: List[Dict[str, Any]],
        message_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Record historical cases retrieved for observability and auditing.
        Does NOT store embeddings or add live conversations to FAISS.
        """
        with get_db_session() as session:
            records = []
            now = now_utc()
            for c in cases:
                rec = RetrievedEvidence(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    case_id=str(c.get("case_id") or c.get("id") or "UNKNOWN"),
                    rank=int(c.get("rank", 0)),
                    similarity=float(c.get("similarity", 0.0)),
                    evidence_quality=c.get("evidence_quality", "UNKNOWN"),
                    used_in_response=bool(c.get("used_in_response", False)),
                    created_at=now
                )
                session.add(rec)
                records.append(rec)
            session.flush()
            return [{"id": r.id, "case_id": r.case_id} for r in records]

    def update_summary(self, conversation_id: str, summary: str) -> None:
        """Update conversation summary."""
        with get_db_session() as session:
            conv = session.query(Conversation).filter_by(conversation_id=conversation_id).first()
            if conv:
                conv.conversation_summary = summary
                conv.updated_at = now_utc()

    @staticmethod
    def _conv_to_dict(conv: Conversation) -> Dict[str, Any]:
        return {
            "id": conv.id,
            "conversation_id": conv.conversation_id,
            "status": conv.status,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            "current_intent": conv.current_intent,
            "current_intent_confidence": conv.current_intent_confidence,
            "current_customer_goal": conv.current_customer_goal,
            "current_issues": conv.current_issues or [],
            "follow_up_type": conv.follow_up_type,
            "resolved": conv.resolved,
            "escalated": conv.escalated,
            "conversation_summary": conv.conversation_summary,
            "last_user_message_at": conv.last_user_message_at.isoformat() if conv.last_user_message_at else None,
            "last_assistant_message_at": conv.last_assistant_message_at.isoformat() if conv.last_assistant_message_at else None,
        }
