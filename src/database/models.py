"""
SQLAlchemy database models for SupportDNA Live Conversation Memory.
Defines 5 logical tables:
  1. conversations
  2. messages
  3. conversation_analysis
  4. attempted_steps
  5. retrieved_evidence
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    JSON
)
from sqlalchemy.orm import relationship
from src.database.connection import Base


def utc_now():
    return datetime.now(timezone.utc)


class Conversation(Base):
    """
    Represents one live customer conversation session.
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(64), unique=True, nullable=False, index=True)
    status = Column(String(32), default="ACTIVE", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    current_intent = Column(String(64), nullable=True)
    current_intent_confidence = Column(Float, nullable=True)
    current_customer_goal = Column(Text, nullable=True)
    current_issues = Column(JSON, nullable=True)  # List of identified issue strings

    follow_up_type = Column(String(64), nullable=True)
    resolved = Column(Boolean, default=False, nullable=False)
    escalated = Column(Boolean, default=False, nullable=False)

    conversation_summary = Column(Text, nullable=True)
    last_user_message_at = Column(DateTime, nullable=True)
    last_assistant_message_at = Column(DateTime, nullable=True)

    # Relationships
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    analyses = relationship(
        "ConversationAnalysis",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    attempted_steps = relationship(
        "AttemptedStep",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    retrieved_evidence = relationship(
        "RetrievedEvidence",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )


class Message(Base):
    """
    Stores every message in the live conversation with chronological ordering.
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        String(64),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    role = Column(String(16), nullable=False)  # "USER", "ASSISTANT", "SYSTEM"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)
    message_type = Column(String(32), nullable=False)  # "CUSTOMER_MESSAGE", "AI_RESPONSE", "SYSTEM_EVENT"

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conv_created", "conversation_id", "created_at"),
    )


class ConversationAnalysis(Base):
    """
    Stores what the AI understood/determined for each important user turn.
    Allows inspection of how the agent reasoned without reconstructing the pipeline.
    """
    __tablename__ = "conversation_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        String(64),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    message_id = Column(
        Integer,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    business_intent = Column(String(64), nullable=True)
    intent_confidence = Column(Float, nullable=True)
    customer_goal = Column(Text, nullable=True)
    issues = Column(JSON, nullable=True)
    known_information = Column(JSON, nullable=True)
    missing_information = Column(JSON, nullable=True)

    follow_up_type = Column(String(64), nullable=True)
    response_mode = Column(String(64), nullable=True)
    decision = Column(String(32), nullable=True)
    evidence_quality = Column(String(32), nullable=True)
    prompt_injection_status = Column(String(32), nullable=True)
    multi_issue_detected = Column(Boolean, default=False, nullable=False)
    verifier_passed = Column(Boolean, default=True, nullable=False)
    verifier_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    conversation = relationship("Conversation", back_populates="analyses")


class AttemptedStep(Base):
    """
    Tracks troubleshooting steps already attempted in the current conversation
    so the agent does not repeat them blindly.
    """
    __tablename__ = "attempted_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        String(64),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    step = Column(Text, nullable=False)
    source = Column(String(32), nullable=False)  # "AI_RESPONSE", "USER_INPUT"
    result = Column(String(32), nullable=False)  # "PENDING", "FAILED", "SUCCESS"
    created_at = Column(DateTime, default=utc_now, nullable=False)

    conversation = relationship("Conversation", back_populates="attempted_steps")


class RetrievedEvidence(Base):
    """
    Records which historical FAISS cases were retrieved for each turn
    for observability and evaluation.
    """
    __tablename__ = "retrieved_evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        String(64),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    message_id = Column(
        Integer,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    case_id = Column(String(64), nullable=False)
    rank = Column(Integer, nullable=False)
    similarity = Column(Float, nullable=False)
    evidence_quality = Column(String(32), nullable=True)
    used_in_response = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    conversation = relationship("Conversation", back_populates="retrieved_evidence")
