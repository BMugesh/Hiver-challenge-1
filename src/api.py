"""
Backend API for AppleSupport Grounded Support Agent
===================================================
Provides high-performance REST API endpoints for the React frontend:
- POST /api/support/analyze: Runs end-to-end 8-stage pipeline inference on user query
- GET  /api/health: Health check and pipeline readiness status

Deterministic, non-LLM safety enforcement, fully connected to:
- Stage 4: Calibrated Intent Classifier
- Stage 5: Dense FAISS Retrieval + Domain-Aware Intent Ranking + Multi-Issue Detection
- Stage 6: Grounded Response Generation + Independent Claim Verifier + Prompt Injection Defense
- Stage 7: Deterministic AUTO-HANDLE vs. ESCALATE Decision Engine (Frozen Policy)
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure offline deterministic execution
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage5_retrieval import (
    load_data,
    build_case_representations,
    SemanticRetriever,
    init_embedding_model,
    build_or_load_faiss_index
)
from src.stage5_domain_ranking import (
    build_intent_prototypes,
    DomainAwareIntentRanker
)
from src.stage5_multi_issue import detect_multi_issue
from src.stage6_grounded_reply import (
    GroundedReplyGenerator,
    IndependentGroundingVerifier,
    run_stage6,
    clean_twitter_noise,
    check_prompt_injection,
    extract_resolution_pattern
)
from src.stage7_decision import (
    CalibratedIntentClassifier,
    DecisionPolicy,
    DecisionEngine,
    EscalationReasonCode
)
from src.agent.agent import SupportDNAAgent
from src.agent.security_detector import detect_security_intent

app = FastAPI(
    title="AppleSupport Grounded Support Agent API",
    description="Deterministic Grounded Support Agent Inference Pipeline API",
    version="1.0.0"
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global singleton state for fast pipeline inference
class PipelineState:
    def __init__(self):
        self.is_ready = False
        self.train_df = None
        self.threads_by_case = {}
        self.taxonomy = {}
        self.model = None
        self.index = None
        self.embeddings = None
        self.metadata = []
        self.retriever = None
        self.domain_ranker = None
        self.classifier = None
        self.generator = None
        self.verifier = None
        self.engine = None
        self.policy = None

    def initialize(self):
        print("Initializing Grounded Support Agent Pipeline...")
        self.train_df, _, _, resolved_threads, self.taxonomy = load_data()
        self.threads_by_case = {t["case_id"]: t for t in resolved_threads}

        train_cases = build_case_representations(self.train_df, self.threads_by_case)

        # 1. Embedding Model & FAISS Index
        self.model = init_embedding_model()
        self.index, self.embeddings, self.metadata = build_or_load_faiss_index(self.model, train_cases)
        self.retriever = SemanticRetriever(self.model, self.index, self.metadata)

        # 2. Stage 5 Domain-Aware Intent Ranker
        prototypes = build_intent_prototypes(train_cases, self.taxonomy, self.embeddings)
        self.domain_ranker = DomainAwareIntentRanker(
            prototypes=prototypes,
            w_semantic=1.0,
            w_keyword=0.50,
            w_prototype=0.20,
            w_confusion=0.40,
            debias_context=True
        )

        # 3. Stage 4 Calibrated Intent Classifier
        self.classifier = CalibratedIntentClassifier(embedding_model=self.model)
        self.classifier.fit(self.train_df, train_embeddings=self.embeddings)

        # 4. Stage 6 Generator & Verifier
        self.generator = GroundedReplyGenerator()
        self.verifier = IndependentGroundingVerifier()

        # 5. Stage 7 Deterministic Decision Policy
        self.policy = DecisionPolicy(
            min_similarity=0.65,
            min_intent_confidence=0.60,
            min_intent_alignment=0.66,
            enforce_grounding_gate=True,
            policy_version="stage7_v1"
        )
        self.engine = DecisionEngine(policy=self.policy)

        # 6. Integrated 9-Stage Knowledge-Driven Agent
        self.agent = SupportDNAAgent(
            classifier=self.classifier,
            retriever=self.retriever,
            domain_ranker=self.domain_ranker,
            embedding_model=self.model
        )

        self.is_ready = True
        print("Pipeline and SupportDNA Agent successfully initialized and ready for inference.")


pipeline = PipelineState()


@app.on_event("startup")
def startup_event():
    if not pipeline.is_ready:
        pipeline.initialize()


# Request / Response Schemas
class QueryRequest(BaseModel):
    message: str = Field(..., description="Customer support inquiry text")
    conversationId: Optional[str] = Field(None, description="Stable session ID for live conversation memory")


class EvidenceCase(BaseModel):
    caseId: str
    intentId: str
    similarity: float
    resolutionStatus: str
    resolvedSummary: Optional[str] = None
    responseText: Optional[str] = None


class AnalysisResponse(BaseModel):
    customerMessage: str
    reply: str
    intent: str
    intentConfidence: float
    top3Intents: List[Dict[str, Any]]
    evidence: List[EvidenceCase]
    resolutionPattern: str
    multiIssue: Dict[str, Any]
    promptInjection: Dict[str, Any]
    groundingCheck: Dict[str, Any]
    decision: str
    reason: str
    reasonCode: str
    isCustomerSafe: bool
    conversationId: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def health_check():
    return {
        "status": "ok" if pipeline.is_ready else "initializing",
        "service": "AppleSupport Grounded Support Agent",
        "pipeline_stages": 8,
        "policy_version": pipeline.policy.policy_version if pipeline.policy else "stage7_v1"
    }


@app.post("/api/support/analyze", response_model=AnalysisResponse)
def analyze_support_query(req: QueryRequest):
    if not pipeline.is_ready:
        pipeline.initialize()

    raw_query = req.message.strip()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Inquiry message cannot be empty.")

    # 1. Parallel Security Intent & Prompt Injection Detector
    sec_analysis = detect_security_intent(raw_query)
    injection_detected = sec_analysis.is_prompt_injection
    injection_attack_type = sec_analysis.security_intent if injection_detected else "NONE"
    sanitized_query = raw_query

    # 2. Multi-Issue Detection
    multi_issue_res = detect_multi_issue(sanitized_query, prototypes=pipeline.domain_ranker.prototypes)
    is_multi_issue = multi_issue_res.get("multi_issue", False)

    # 3. Stage 4 Intent Classification & Calibrated Probability
    top_3_predictions = pipeline.classifier.predict_top_k(sanitized_query, k=3)
    pred_intent = top_3_predictions[0][0]
    intent_confidence = float(top_3_predictions[0][1])

    top3_intents = [
        {"intent": p[0], "confidence": round(float(p[1]), 4)}
        for p in top_3_predictions
    ]

    # 4. Stage 5 FAISS Semantic Candidate Retrieval (Top-10 Pool)
    raw_top10 = pipeline.retriever.retrieve(sanitized_query, top_k=10)
    query_emb = pipeline.model.encode(
        [sanitized_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )[0].astype(np.float32)

    # 5. Stage 5 Domain-Aware Intent Ranking & Evidence Alignment
    ranking_res = pipeline.domain_ranker.rank_candidates(
        query_text=sanitized_query,
        query_embedding=query_emb,
        candidate_cases=raw_top10
    )
    ranked_cases = ranking_res["ranked_cases"]
    top_intent_domain = ranking_res["top1_intent"] if ranking_res.get("top1_intent") else pred_intent

    # Select Top-3 evidence cases aligned with dominant domain or semantic pool
    aligned_evidence = [c for c in ranked_cases if c.get("intent_id") == top_intent_domain][:3]
    if len(aligned_evidence) < 3:
        for c in ranked_cases:
            if c not in aligned_evidence:
                aligned_evidence.append(c)
            if len(aligned_evidence) >= 3:
                break
    final_evidence = aligned_evidence[:3]

    # 6. Stage 6 Grounded Reply Generation & Claim Verification
    resolution_pattern = extract_resolution_pattern(pred_intent, final_evidence)
    stage6_output = run_stage6(
        customer_message=sanitized_query,
        intent=pred_intent,
        intent_confidence=intent_confidence,
        retrieved_cases=final_evidence,
        resolution_pattern=resolution_pattern,
        top_k=3,
        generator=pipeline.generator,
        verifier=pipeline.verifier
    )

    draft_reply = stage6_output["draft_reply"]
    grounding_check = stage6_output["grounding_check"]
    gen_output = stage6_output["generator_output"]

    # If adversarial prompt injection was intercepted, neutralize draft response
    if injection_detected:
        grounding_check["grounding_pass"] = False
        grounding_check["severity"] = "HIGH"
        grounding_check["unsupported_claims"].append(f"Adversarial prompt-injection attack detected ({injection_attack_type}).")

    # 7. Stage 7 Deterministic AUTO-HANDLE vs. ESCALATE Decision Policy
    decision_obj = pipeline.engine.evaluate(
        customer_message=sanitized_query,
        intent=pred_intent,
        intent_confidence=intent_confidence,
        retrieved_evidence=final_evidence,
        draft_reply=draft_reply,
        grounding_check=grounding_check,
        generator_output=gen_output,
        policy_override=pipeline.policy
    )

    # If multi-issue detected and similarity is marginal, escalate safely
    if is_multi_issue and decision_obj["decision"] == "AUTO_HANDLE":
        if decision_obj["retrieval"]["top_similarity"] < 0.85:
            decision_obj["decision"] = "ESCALATE"
            decision_obj["reason_code"] = EscalationReasonCode.INTENT_EVIDENCE_MISMATCH
            decision_obj["reason"] = "Multiple distinct customer issues detected. Escalation recommended for human handling."
            decision_obj["is_customer_safe"] = False

    # Format Evidence list for frontend
    evidence_items = []
    for c in final_evidence:
        cid = c.get("case_id", "CASE_UNKNOWN")
        thread_info = pipeline.threads_by_case.get(cid, {})
        resp_text = c.get("support_response") or c.get("response_text", "")
        summary = c.get("outcome") or c.get("resolution_summary") or thread_info.get("resolution_summary")
        evidence_items.append(EvidenceCase(
            caseId=cid,
            intentId=c.get("intent_id", "GENERAL_DEVICE_INQUIRY"),
            similarity=round(float(c.get("similarity", 0.0)), 4),
            resolutionStatus=c.get("resolution_status", "CLEARLY_RESOLVED"),
            resolvedSummary=summary,
            responseText=resp_text[:160] + "..." if len(resp_text) > 160 else resp_text
        ))

    resolution_pattern_str = resolution_pattern

    # Format Decision display
    # Generate or reuse stable conversation session ID
    import uuid
    conv_id = req.conversationId or f"conv_{uuid.uuid4().hex[:8]}"

    # Use SupportDNAAgent verified response with live conversation memory
    final_reply = decision_obj["draft_reply"]
    agent_res = None
    if pipeline.agent:
        try:
            agent_res = pipeline.agent.run(raw_query, conversation_id=conv_id)
            final_reply = agent_res["final_reply"]
            if agent_res.get("business"):
                pred_intent = agent_res["business"]["intent"]
                intent_confidence = agent_res["business"]["confidence"]
            if agent_res.get("decision"):
                act = agent_res["decision"]["action"]
                final_decision_str = "AUTO-HANDLE" if act in ["AUTO_HANDLE", "GUIDE"] else "ESCALATE"
        except Exception as e:
            print(f"Agent execution error: {e}")

    analysis_dict = {
        "intent": pred_intent,
        "confidence": round(intent_confidence, 4),
        "customerGoal": agent_res.get("business", {}).get("customer_goal", "Resolve inquiry") if agent_res else "Resolve inquiry",
        "followUpType": agent_res.get("follow_up_type", "NEW_ISSUE") if agent_res else "NEW_ISSUE",
        "evidenceQuality": agent_res.get("evidence_quality", {}).get("overall_quality", "MODERATE") if agent_res else "MODERATE",
        "resolutionPattern": resolution_pattern_str,
        "promptInjectionStatus": agent_res.get("security", {}).get("security_intent", "NONE") if agent_res else injection_attack_type,
        "decision": final_decision_str,
        "responseMode": agent_res.get("decision", {}).get("response_mode", "DIRECT_GUIDANCE") if agent_res else "DIRECT_GUIDANCE",
        "verifierPassed": agent_res.get("verification", {}).get("verification_pass", True) if agent_res else True,
        "attemptedSteps": agent_res.get("attempted_steps", []) if agent_res else [],
        "agentDiagnostics": agent_res or {}
    }

    return AnalysisResponse(
        customerMessage=raw_query,
        reply=final_reply,
        intent=pred_intent,
        intentConfidence=round(intent_confidence, 4),
        top3Intents=top3_intents,
        evidence=evidence_items,
        resolutionPattern=resolution_pattern_str,
        multiIssue={
            "detected": is_multi_issue,
            "issues": [
                {"name": dom, "keywords": multi_issue_res.get("symptom_keywords", {}).get(dom, [])}
                for dom in multi_issue_res.get("symptom_domains", [])
            ] if is_multi_issue else []
        },
        promptInjection={
            "detected": injection_detected,
            "attackType": injection_attack_type if injection_detected else "NONE"
        },
        groundingCheck={
            "status": "PASS" if grounding_check.get("grounding_pass") and grounding_check.get("severity") == "NONE" else "FAIL",
            "severity": grounding_check.get("severity", "NONE"),
            "unsupportedClaims": grounding_check.get("unsupported_claims", [])
        },
        decision=final_decision_str,
        reason=decision_obj["reason"],
        reasonCode=decision_obj["reason_code"],
        isCustomerSafe=decision_obj["is_customer_safe"],
        conversationId=conv_id,
        analysis=analysis_dict
    )


@app.get("/api/conversations")
def get_conversations(limit: int = 50):
    """List recent conversation sessions."""
    if pipeline.agent and pipeline.agent.repository:
        try:
            return pipeline.agent.repository.list_conversations(limit=limit)
        except Exception as e:
            return []
    return []


@app.get("/api/conversations/{conversation_id}")
def get_conversation_details(conversation_id: str):
    """Get full message history and attempted steps for a conversation."""
    if pipeline.agent and pipeline.agent.repository:
        try:
            conv = pipeline.agent.repository.get_conversation(conversation_id)
            messages = pipeline.agent.repository.get_messages(conversation_id)
            steps = pipeline.agent.repository.get_attempted_steps(conversation_id)
            return {
                "conversation": conv,
                "messages": messages,
                "attempted_steps": steps
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=404, detail="Repository not initialized")


@app.get("/api/evaluation/metrics")
def get_evaluation_metrics():
    """Retrieve verified offline Stage 8 and Agent Knowledge metrics."""
    metrics_csv = PROJECT_ROOT / "reports" / "stage8_final_metrics.csv"
    agent_csv = PROJECT_ROOT / "reports" / "agent_knowledge_metrics.csv"
    
    stage8_records = []
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        stage8_records = df.to_dict(orient="records")
        
    agent_records = {}
    if agent_csv.exists():
        df_agent = pd.read_csv(agent_csv)
        if not df_agent.empty:
            agent_records = df_agent.iloc[0].to_dict()

    return {
        "status": "success",
        "stage8_metrics": stage8_records,
        "agent_knowledge_metrics": agent_records,
        "dataset_summary": {
            "test_set_size": 1189,
            "golden_set_size": 200,
            "faiss_index_size": 5545,
            "intent_classes": 11
        }
    }


@app.post("/api/agent/run")
def run_agent_endpoint(req: QueryRequest):
    """
    Direct endpoint to the integrated 9-stage knowledge-driven SupportDNAAgent.
    Returns full diagnostics: query understanding, risk analysis, evidence quality,
    decision plan, response plan, response verification, and trustworthy status.
    """
    if not pipeline.is_ready:
        pipeline.initialize()
    raw_query = req.message.strip()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Inquiry message cannot be empty.")
    return pipeline.agent.run(raw_query, conversation_id=req.conversationId)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
