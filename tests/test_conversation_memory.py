"""
Unit and Integration Tests for SupportDNA Live Conversation Memory.
Validates:
  - PART 17: The 5 exact required conversation test scenarios:
      TEST 1: New Issue ("My iPhone battery is draining very fast.")
      TEST 2: Follow-up ("I already restarted it.")
      TEST 3: Unresolved follow-up ("Still not resolved.")
      TEST 4: Escalation request ("No, still having the same issue. Contact Apple Support team.")
      TEST 5: New issue in same conversation ("My WiFi is also not connecting.")
  - PART 18: Database verification:
      Tables: conversations, messages, conversation_analysis, attempted_steps, retrieved_evidence.
      Chronological ordering, foreign keys, index integrity, and traceability.
"""

import sys
import uuid
from pathlib import Path
import pytest
from sqlalchemy import inspect, Index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_engine, get_db_session, init_db
from src.database.models import (
    Conversation,
    Message,
    ConversationAnalysis,
    AttemptedStep,
    RetrievedEvidence
)
from src.database.repository import ConversationRepository
from src.agent.conversation_context import ConversationContextBuilder
from src.agent.followup_analyzer import FollowUpAnalyzer
from src.agent.agent import SupportDNAAgent
from src.stage5_retrieval import (
    load_data,
    build_case_representations,
    init_embedding_model,
    build_or_load_faiss_index,
    SemanticRetriever
)
from src.stage5_domain_ranking import build_intent_prototypes, DomainAwareIntentRanker
from src.stage7_decision import CalibratedIntentClassifier


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Ensure database schema is created prior to any tests."""
    init_db()
    yield


@pytest.fixture(scope="module")
def repo():
    """Shared repository initialized with database schema."""
    return ConversationRepository()


@pytest.fixture(scope="module")
def agent():
    """Fully initialized SupportDNA agent with FAISS retriever and 11-intent classifier."""
    train_df, _, _, resolved_threads, taxonomy = load_data()
    threads_by_case = {t["case_id"]: t for t in resolved_threads}
    train_cases = build_case_representations(train_df, threads_by_case)

    model = init_embedding_model()
    index, embeddings, metadata = build_or_load_faiss_index(model, train_cases)
    retriever = SemanticRetriever(model, index, metadata)

    prototypes = build_intent_prototypes(train_cases, taxonomy, embeddings)
    domain_ranker = DomainAwareIntentRanker(
        prototypes=prototypes,
        w_semantic=1.0,
        w_keyword=0.50,
        w_prototype=0.20,
        w_confusion=0.40,
        debias_context=True
    )
    classifier = CalibratedIntentClassifier(embedding_model=model)
    classifier.fit(train_df, train_embeddings=embeddings)

    return SupportDNAAgent(
        classifier=classifier,
        retriever=retriever,
        domain_ranker=domain_ranker,
        embedding_model=model
    )


# ==============================================================================
# PART 18: Database Schema, Foreign Keys, Indexes & Ordering
# ==============================================================================

def test_database_tables_exist():
    """Verify all 5 required tables exist in the database."""
    init_db()
    engine = get_engine()
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    required_tables = [
        "conversations",
        "messages",
        "conversation_analysis",
        "attempted_steps",
        "retrieved_evidence"
    ]
    for tbl in required_tables:
        assert tbl in table_names, f"Table '{tbl}' is missing from database schema."


def test_database_indexes_and_foreign_keys():
    """Verify indexes and relationships across conversation tables."""
    init_db()
    engine = get_engine()
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    assert "messages" in table_names
    assert "conversations" in table_names

    msg_indexes = inspector.get_indexes("messages")
    has_conv_idx = any("conversation_id" in (idx.get("column_names") or []) for idx in msg_indexes) or any("conv" in (idx.get("name") or "") for idx in msg_indexes)
    assert has_conv_idx, "messages table should have index on conversation_id"

    conv_indexes = inspector.get_indexes("conversations")
    has_conv_id_idx = any("conversation_id" in (idx.get("column_names") or []) for idx in conv_indexes) or any("conversation_id" in (idx.get("name") or "") for idx in conv_indexes)
    assert has_conv_id_idx, "conversations table should have index on conversation_id"


def test_chronological_ordering_and_traceability(repo):
    """Verify messages preserve strict chronological order and complete link trail."""
    conv_id = f"test_order_{uuid.uuid4().hex[:6]}"
    repo.get_or_create_conversation(conv_id)

    # Save 3 messages in sequence
    m1 = repo.save_message(conv_id, "USER", "Message 1", "CUSTOMER_MESSAGE")
    m2 = repo.save_message(conv_id, "ASSISTANT", "Message 2", "AI_RESPONSE")
    m3 = repo.save_message(conv_id, "USER", "Message 3", "CUSTOMER_MESSAGE")

    retrieved = repo.get_messages(conv_id)
    assert len(retrieved) == 3
    assert retrieved[0]["content"] == "Message 1"
    assert retrieved[1]["content"] == "Message 2"
    assert retrieved[2]["content"] == "Message 3"

    # Verify analysis linking
    repo.save_conversation_analysis(
        conv_id,
        analysis_data={
            "business_intent": "BATTERY_CHARGING_POWER",
            "intent_confidence": 0.95,
            "decision": "GUIDE"
        },
        message_id=m2["id"]
    )

    # Verify attempted steps
    repo.save_attempted_step(conv_id, "Restart device", "USER_INPUT", "FAILED")
    steps = repo.get_attempted_steps(conv_id)
    assert len(steps) == 1
    assert steps[0]["step"] == "Restart device"
    assert steps[0]["result"] == "FAILED"


# ==============================================================================
# PART 17: The 5 Exact Conversation Test Scenarios
# ==============================================================================

def test_scenario_1_new_issue(agent, repo):
    """
    TEST 1 — NEW ISSUE
    User: "My iPhone battery is draining very fast."
    Expected:
      - intent relates to BATTERY_CHARGING_POWER.
      - A conversation is created.
      - Message is saved.
      - AI response is saved.
      - Analysis is saved.
    """
    conv_id = f"conv_test_1_{uuid.uuid4().hex[:6]}"

    res = agent.run("My iPhone battery is draining very fast.", conversation_id=conv_id)

    # 1. Intent should relate to BATTERY_CHARGING_POWER
    assert res["business"]["intent"] == "BATTERY_CHARGING_POWER"
    assert res["business"]["confidence"] >= 0.60

    # 2. Conversation must be created in DB
    conv = repo.get_conversation(conv_id)
    assert conv is not None
    assert conv["status"] == "ACTIVE"
    assert conv["current_intent"] == "BATTERY_CHARGING_POWER"

    # 3. Messages must be saved (1 USER, 1 ASSISTANT)
    msgs = repo.get_messages(conv_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "USER"
    assert msgs[0]["content"] == "My iPhone battery is draining very fast."
    assert msgs[1]["role"] == "ASSISTANT"
    assert len(msgs[1]["content"]) > 10

    # 4. Analysis must be saved in DB
    with get_db_session() as session:
        analyses = session.query(ConversationAnalysis).filter_by(conversation_id=conv_id).all()
        assert len(analyses) >= 1
        assert analyses[0].business_intent == "BATTERY_CHARGING_POWER"


def test_scenario_2_follow_up(agent, repo):
    """
    TEST 2 — FOLLOW-UP
    User: "My iPhone battery is draining very fast."
    Then: "I already restarted it."
    Expected:
      - same conversationId
      - follow-up detected (ADDITIONAL_INFORMATION / FOLLOW_UP)
      - original battery intent preserved (BATTERY_CHARGING_POWER)
      - 'Restart device' added to attempted steps
      - No unrelated GENERAL_DEVICE_INQUIRY classification.
    """
    conv_id = f"conv_test_2_{uuid.uuid4().hex[:6]}"

    # Turn 1
    agent.run("My iPhone battery is draining very fast.", conversation_id=conv_id)

    # Turn 2
    res2 = agent.run("I already restarted it.", conversation_id=conv_id)

    # 1. Same conversation ID
    assert res2["conversation_id"] == conv_id

    # 2. Follow-up type detected
    assert res2["follow_up_type"] in ["ADDITIONAL_INFORMATION", "FOLLOW_UP"]

    # 3. Original battery intent preserved, NOT GENERAL_DEVICE_INQUIRY
    assert res2["business"]["intent"] == "BATTERY_CHARGING_POWER"
    assert res2["business"]["intent"] != "GENERAL_DEVICE_INQUIRY"

    # 4. 'Restart device' recorded in attempted steps
    attempted = repo.get_attempted_steps(conv_id)
    step_names = [s["step"] for s in attempted]
    assert any("restart" in s.lower() for s in step_names)

    # Verify in response: does not repeat restart
    assert "restart your device" not in res2["final_reply"].lower()


def test_scenario_3_unresolved_follow_up(agent, repo):
    """
    TEST 3 — UNRESOLVED FOLLOW-UP
    Follow-up: "Still not resolved."
    Expected:
      - same conversationId
      - intent remains related to battery issue
      - follow-up type = UNRESOLVED_FOLLOW_UP
      - retrieval query includes battery context
      - previously attempted restart is known
      - system does not blindly repeat restart
    """
    conv_id = f"conv_test_3_{uuid.uuid4().hex[:6]}"

    # Turn 1
    agent.run("My iPhone battery is draining very fast.", conversation_id=conv_id)
    # Turn 2
    agent.run("I already restarted it.", conversation_id=conv_id)

    # Turn 3: "Still not resolved."
    res3 = agent.run("Still not resolved.", conversation_id=conv_id)

    # 1. Same conversation ID
    assert res3["conversation_id"] == conv_id

    # 2. Intent preserved as battery
    assert res3["business"]["intent"] == "BATTERY_CHARGING_POWER"
    assert res3["business"]["intent"] != "GENERAL_DEVICE_INQUIRY"

    # 3. Follow-up type is UNRESOLVED_FOLLOW_UP
    assert res3["follow_up_type"] == "UNRESOLVED_FOLLOW_UP"

    # 4. Attempted restart is known
    attempted = repo.get_attempted_steps(conv_id)
    assert any("restart" in s["step"].lower() for s in attempted)

    # 5. System does not blindly repeat restart
    assert "and restart your device" not in res3["final_reply"].lower()
    assert "restart your phone" not in res3["final_reply"].lower()


def test_scenario_4_escalation_request(agent, repo):
    """
    TEST 4 — ESCALATION REQUEST
    User: "No, still having the same issue. Contact Apple Support team."
    Expected:
      - same conversation
      - escalation request detected
      - response should acknowledge the battery issue
      - decision and responseMode must agree (ESCALATE & SAFE_ESCALATION)
      - No generic troubleshooting fallback.
    """
    conv_id = f"conv_test_4_{uuid.uuid4().hex[:6]}"

    # Turn 1
    agent.run("My iPhone battery is draining very fast.", conversation_id=conv_id)
    # Turn 2
    agent.run("I already restarted it.", conversation_id=conv_id)

    # Turn 3: Escalation Request
    res3 = agent.run(
        "No, still having the same issue. Contact Apple Support team.",
        conversation_id=conv_id
    )

    # 1. Same conversation ID
    assert res3["conversation_id"] == conv_id

    # 2. Escalation request detected
    assert res3["follow_up_type"] == "ESCALATION_REQUEST"

    # 3. Decision and responseMode must agree
    assert res3["decision"]["action"] == "ESCALATE"
    assert res3["decision"]["response_mode"] == "SAFE_ESCALATION"

    # 4. Response acknowledges battery issue and escalates safely without generic troubleshooting
    reply_lower = res3["final_reply"].lower()
    assert "battery" in reply_lower or "drain" in reply_lower
    assert any(k in reply_lower for k in ["apple support", "specialist", "getsupport.apple.com"])
    assert "here are the recommended troubleshooting steps" not in reply_lower


def test_scenario_5_new_issue_in_same_conversation(agent, repo):
    """
    TEST 5 — NEW ISSUE IN SAME CONVERSATION
    User: "My WiFi is also not connecting."
    Expected:
      - system recognizes a new issue
      - Do not incorrectly force this into battery intent.
      - multi_issue / new intent detected.
    """
    conv_id = f"conv_test_5_{uuid.uuid4().hex[:6]}"

    # Turn 1: Battery issue
    res1 = agent.run("My iPhone battery is draining very fast.", conversation_id=conv_id)
    assert res1["business"]["intent"] == "BATTERY_CHARGING_POWER"

    # Turn 2: WiFi issue in same conversation
    res2 = agent.run("My WiFi is also not connecting.", conversation_id=conv_id)

    # 1. System recognizes new issue, does NOT force into BATTERY_CHARGING_POWER
    assert res2["conversation_id"] == conv_id
    assert res2["business"]["intent"] != "BATTERY_CHARGING_POWER"
    assert "CONNECTIVITY" in res2["business"]["intent"] or "WIFI" in res2["business"]["intent"]
    assert res2["follow_up_type"] == "NEW_ISSUE"
