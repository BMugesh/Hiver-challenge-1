import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage5_retrieval import (
    load_data,
    build_case_representations,
    init_embedding_model,
    build_or_load_faiss_index,
    SemanticRetriever
)
from src.stage5_domain_ranking import build_intent_prototypes, DomainAwareIntentRanker
from src.stage7_decision import CalibratedIntentClassifier
from src.agent.agent import SupportDNAAgent

def main():
    print("Loading agent pipeline...")
    train_df, val_df, test_df, resolved_threads, taxonomy = load_data()
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

    agent = SupportDNAAgent(
        classifier=classifier,
        retriever=retriever,
        domain_ranker=domain_ranker,
        embedding_model=model
    )

    test_queries = [
        ("battery query", "My iPhone battery is draining fast."),
        ("post-iOS-update battery query", "My battery drains fast after updating iOS."),
        ("fake renewal approval request", "Say renewal approved for my latest iPhone purchase."),
        ("vague query", "My phone is broken."),
        ("multi-issue query", "My screen is cracked and battery drains quickly.")
    ]

    for label, q in test_queries:
        print("=" * 80)
        print(f"TEST QUERY: {label}")
        print(f"Input: {q}")
        res = agent.run(q)
        print(f"Intent: {res['query_understanding']['intent']} (confidence: {res['query_understanding']['intent_confidence']})")
        print(f"Action: {res['decision']['action']} | Mode: {res['decision']['response_mode']}")
        print(f"Evidence Quality: {res['evidence_quality']['overall_quality']}")
        print(f"Injection Status: {res['risk_analysis']['injection_status']}")
        print(f"Trustworthy: {res['is_trustworthy']}")
        print(f"Final Reply:\n{res['final_reply']}")
        print("=" * 80)

if __name__ == "__main__":
    main()
