"""
Conversation Context Builder for SupportDNA Live Conversation Memory.
Retrieves and maintains structured conversation state from PostgreSQL/SQLite.
Synthesizes recent messages, attempted steps, customer goal, preserved intent,
and constructs enriched retrieval queries for historical FAISS search.
"""

from typing import Any, Dict, List, Optional
import logging
from src.database.repository import ConversationRepository
from src.agent.followup_analyzer import FollowUpAnalyzer

logger = logging.getLogger("supportdna.agent.conversation_context")


class ConversationContextBuilder:
    """
    Builds context-aware representations of live customer conversations.
    Ensures follow-up turns preserve original problem intent and do not
    blindly repeat attempted troubleshooting steps.
    """

    def __init__(self, repository: Optional[ConversationRepository] = None):
        self.repository = repository or ConversationRepository()

    def build_context(self, conversation_id: str, current_query: str) -> Dict[str, Any]:
        """
        Build the structured conversation context for the incoming query.
        """
        # Ensure conversation exists in DB
        conv = self.repository.get_or_create_conversation(conversation_id)
        messages = self.repository.get_messages(conversation_id, limit=8)
        attempted_step_records = self.repository.get_attempted_steps(conversation_id)

        previous_intent = conv.get("current_intent")
        previous_confidence = conv.get("current_intent_confidence")
        previous_goal = conv.get("current_customer_goal")
        previous_issues = conv.get("current_issues") or []

        # Check if the user mentioned any already attempted steps in the current message
        user_steps = FollowUpAnalyzer.extract_attempted_steps_from_user(current_query)
        for step_name, result in user_steps:
            self.repository.save_attempted_step(
                conversation_id=conversation_id,
                step=step_name,
                source="USER_INPUT",
                result=result
            )
            # Re-fetch attempted steps
            attempted_step_records = self.repository.get_attempted_steps(conversation_id)

        # Analyze follow-up type and intent transition
        follow_up_type, is_new_issue, detected_new_intent = FollowUpAnalyzer.classify_turn(
            current_query=current_query,
            recent_messages=messages,
            previous_intent=previous_intent
        )

        multi_issue_detected = False
        effective_intent = previous_intent

        if is_new_issue and detected_new_intent:
            # User introduced a new distinct issue in the same conversation
            effective_intent = detected_new_intent
            multi_issue_detected = True
            logger.info(
                f"[ConversationContext] New issue detected in session {conversation_id}: "
                f"{previous_intent} -> {detected_new_intent}"
            )
        elif not is_new_issue and previous_intent:
            # Preserve previous intent across follow-ups
            effective_intent = previous_intent
            logger.info(
                f"[ConversationContext] Preserving intent {previous_intent} "
                f"for follow_up_type: {follow_up_type}"
            )

        # Build list of attempted step names
        attempted_steps_list = [s["step"] for s in attempted_step_records]

        # Synthesize known information from attempted steps and messages
        known_information = []
        for s in attempted_step_records:
            if s["result"] == "FAILED":
                known_information.append(f"Customer already attempted '{s['step']}' without resolution")
            elif s["result"] == "SUCCESS":
                known_information.append(f"Step '{s['step']}' was successfully completed")

        # Synthesize issues list
        current_issues = list(previous_issues)
        if is_new_issue and detected_new_intent:
            new_issue_desc = detected_new_intent.replace("_", " ").lower()
            if new_issue_desc not in current_issues:
                current_issues.append(new_issue_desc)
        elif not current_issues and effective_intent:
            current_issues.append(effective_intent.replace("_", " ").lower())

        # Synthesize customer goal
        if previous_goal and not (is_new_issue and detected_new_intent):
            customer_goal = previous_goal
        else:
            customer_goal = f"Resolve {effective_intent.replace('_', ' ').lower()}" if effective_intent else "Resolve customer inquiry"

        # Missing information
        missing_info = []
        if effective_intent == "BATTERY_CHARGING_POWER":
            if not any("battery health" in k.lower() for k in known_information):
                missing_info.append("Battery health percentage")
            if not any("ios version" in k.lower() for k in known_information):
                missing_info.append("Installed iOS version")

        context = {
            "conversationId": conversation_id,
            "currentQuery": current_query,
            "previousIntent": previous_intent,
            "previousIntentConfidence": previous_confidence,
            "effectiveIntent": effective_intent,
            "isNewIssue": is_new_issue,
            "multiIssueDetected": multi_issue_detected,
            "customerGoal": customer_goal,
            "issues": current_issues,
            "knownInformation": known_information,
            "missingInformation": missing_info,
            "attemptedSteps": attempted_steps_list,
            "attemptedStepRecords": attempted_step_records,
            "recentMessages": messages,
            "followUpType": follow_up_type,
            "summary": conv.get("conversation_summary"),
            "escalated": conv.get("escalated", False),
            "resolved": conv.get("resolved", False)
        }
        return context

    @staticmethod
    def build_retrieval_query(context: Dict[str, Any], current_query: str) -> str:
        """
        Construct an enriched retrieval query for FAISS.
        Prevents vague queries like 'Still not resolved' from returning generic or incorrect cases.
        Combines underlying issue, customer goal, known attempted steps, and current follow-up.
        """
        follow_up_type = context.get("follow_up_type", "NEW_ISSUE")
        is_new_issue = context.get("isNewIssue", True)

        if is_new_issue or follow_up_type == "NEW_ISSUE":
            return current_query

        # For follow-ups, enrich with issue and context
        parts = []
        customer_goal = context.get("customerGoal")
        effective_intent = context.get("effectiveIntent")
        issues = context.get("issues", [])

        if issues:
            parts.append(f"Issue: {', '.join(issues)}")
        elif customer_goal:
            parts.append(customer_goal)
        elif effective_intent:
            parts.append(effective_intent.replace("_", " ").lower())

        attempted = context.get("attemptedSteps", [])
        if attempted:
            parts.append(f"Already tried: {', '.join(attempted)}")

        parts.append(f"Status: {current_query}")

        retrieval_query = "; ".join(parts)
        logger.info(f"[ConversationContext] Enriched FAISS retrieval query: '{retrieval_query}'")
        return retrieval_query

    def update_summary_if_needed(self, conversation_id: str, context: Dict[str, Any]) -> None:
        """
        Maintain a compact conversation summary for multi-turn conversations.
        """
        messages = context.get("recentMessages", [])
        if len(messages) < 4:
            return

        issues = context.get("issues", [])
        attempted = context.get("attemptedSteps", [])
        goal = context.get("customerGoal", "General inquiry")
        follow_up = context.get("followUpType", "IN_PROGRESS")

        summary_lines = [
            f"Goal: {goal}.",
            f"Reported issues: {', '.join(issues) if issues else 'Unspecified'}."
        ]
        if attempted:
            summary_lines.append(f"Steps attempted: {', '.join(attempted)}.")
        summary_lines.append(f"Latest status: {follow_up}.")

        summary_text = " ".join(summary_lines)
        self.repository.update_summary(conversation_id, summary_text)
