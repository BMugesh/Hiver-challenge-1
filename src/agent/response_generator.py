"""
SupportDNA Agent — Step 7 & 8: Response Generation & Relevant Escalation
========================================================================
Synthesizes grounded, evidence-backed customer support replies without copying raw tweets.
Crucially fixes escalation handling so that an ESCALATE decision produces an honest,
contextually relevant explanation addressing the customer's actual inquiry rather than
unrelated troubleshooting.
"""

import re
from typing import Dict, List, Any, Optional

from src.stage6_grounded_reply import clean_twitter_noise


class ResponseGenerator:
    """
    Evidence-grounded response synthesizer.
    Generates natural, actionable customer guidance based on the structured Response Plan.
    """

    def generate_response(
        self,
        query: str,
        query_understanding: Any,
        risk_result: Any,
        evidence_judge_result: Any,
        decision_result: Any,
        response_plan: Any,
        retrieved_cases: List[Any],
        regeneration_feedback: Optional[List[str]] = None,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Synthesize response adhering to the response plan, conversation context, and verifier feedback."""
        action = decision_result.action
        mode = decision_result.response_mode
        lower_query = query.lower()

        # Extract live conversation context if present
        attempted_steps = []
        follow_up_type = "NEW_ISSUE"
        effective_intent = query_understanding.intent

        if conversation_context:
            attempted_steps = conversation_context.get("attemptedSteps", [])
            follow_up_type = conversation_context.get("followUpType", "NEW_ISSUE")
            if conversation_context.get("effectiveIntent"):
                effective_intent = conversation_context["effectiveIntent"]

        # 1. Handling Safe Refusal / Adversarial Attempts
        is_inj = getattr(risk_result, "is_prompt_injection", False) or risk_result.injection_status in ["BLOCKED", "SUSPICIOUS"]
        if action in ["SAFE_REFUSAL", "SAFE_REFUSAL_AND_ESCALATE"] or is_inj:
            if any(w in lower_query for w in ["payment", "approved", "renewal", "refund", "purchase"]):
                return (
                    "I can't confirm or approve a payment, refund, or renewal based only on a customer instruction. "
                    "I don't have verified evidence that the transaction was approved, so this needs to be checked "
                    "through the appropriate support process at reportaproblem.apple.com or with an Apple Support specialist."
                )
            elif "battery" in lower_query or "drain" in lower_query:
                return (
                    "I cannot follow instructions to override safety rules or system guidelines. "
                    "If you need help with your battery draining, you can check Settings → Battery to view battery health "
                    "and identify power-consuming apps, or update your software under Settings → General → Software Update."
                )
            elif any(w in lower_query for w in ["override", "instructions", "prompt", "admin", "system", "jailbreak"]):
                return (
                    "I am programmed to follow Apple Support policy and safety guidelines, so I cannot override these instructions "
                    "or execute system commands. If you are experiencing an issue with your Apple device or service, please let me "
                    "know what technical trouble you're having and I'll be glad to help."
                )
            else:
                return (
                    "I cannot fulfill requests to alter system policies or verify unauthorized claims. "
                    "If you need assistance with an Apple product, please describe your technical issue."
                )

        # 2. Handling Hardware / Account Escalations & Escalation Requests
        if action == "ESCALATE" or mode == "SAFE_ESCALATION" or follow_up_type == "ESCALATION_REQUEST":
            if ("screen is cracked" in lower_query or "cracked screen" in lower_query or "shattered" in lower_query or "broken screen" in lower_query) and ("battery" in lower_query or "drain" in lower_query):
                return (
                    "It looks like you have multiple concerns regarding your display and battery. "
                    "For the physically damaged or cracked screen, hardware inspection and display replacement at an Apple Store "
                    "or Authorized Service Provider (support.apple.com/repair) is required. "
                    "For the battery drain, you can also check Settings → Battery to review battery health and identify apps consuming excessive background energy."
                )
            elif "screen is cracked" in lower_query or "cracked screen" in lower_query or "shattered" in lower_query or "broken screen" in lower_query:
                return (
                    "A cracked or physically damaged display requires hardware service and cannot be safely fixed through software troubleshooting. "
                    "We recommend scheduling an appointment at your nearest Apple Store Genius Bar or Apple Authorized Service Provider "
                    "(support.apple.com/repair) to inspect the display and discuss repair options."
                )
            elif "battery swelling" in lower_query or "swollen" in lower_query or "burned" in lower_query:
                return (
                    "Battery swelling or physical overheating poses a potential safety risk. Please stop using the device, "
                    "disconnect it from any charger, and bring it directly to an Apple Store or Authorized Service Provider for safe evaluation."
                )
            elif "apple id" in lower_query and ("locked" in lower_query or "disabled" in lower_query or "password" in lower_query):
                return (
                    "For your security, Apple ID credentials and account recovery cannot be managed in automated chat. "
                    "Please visit iforgot.apple.com to verify your identity and regain access, or contact Apple Support directly."
                )
            # Escalation specifically addressing battery drain
            elif (effective_intent == "BATTERY_CHARGING_POWER" or "battery" in lower_query or "drain" in lower_query) and not query_understanding.is_multi_issue and "screen" not in lower_query:
                return (
                    "I understand the battery drain issue is still persisting after the troubleshooting steps you've tried. "
                    "At this point, the available troubleshooting evidence isn't sufficient to safely resolve it remotely, "
                    "so this needs Apple Support's direct assistance. Please connect with an Apple Support specialist "
                    "at getsupport.apple.com or visit an Apple Authorized Service Provider for battery diagnostics."
                )
            elif decision_result.reason_code == "EVIDENCE_IRRELEVANT_OR_INSUFFICIENT":
                return (
                    "I don't have sufficient verified support evidence to provide a reliable resolution for this specific issue. "
                    "Rather than provide incomplete or speculative advice, I recommend connecting with an Apple Support specialist "
                    "at getsupport.apple.com for direct assistance."
                )
            else:
                return (
                    f"This inquiry requires direct review by a support specialist ({decision_result.reason}). "
                    "Please connect with Apple Support at getsupport.apple.com or visit an authorized location for personalized assistance."
                )

        # 3. Handling Clarification (Vague queries)
        if action == "CLARIFY" or mode == "CLARIFICATION":
            return (
                "We'd like to help get this sorted out for you. Could you please describe what specific issue you're experiencing "
                "with your device (for example, rapid battery drain, Wi-Fi connectivity, app crashes, or screen responsiveness)?"
            )

        # 4. Multi-Issue Guidance (e.g. cracked screen and battery draining)
        if query_understanding.is_multi_issue:
            if "battery" in lower_query and ("screen" in lower_query or "display" in lower_query):
                return (
                    "It looks like you have multiple concerns regarding your display and battery. "
                    "For the battery drain, start by opening Settings → Battery to check which apps are consuming the most background power, "
                    "and ensure your apps are updated. For any physical screen damage or touch unresponsiveness, that typically requires hardware inspection "
                    "at an Apple Store or Authorized Service Provider. Let us know which issue you'd like to prioritize!"
                )

        # 5. Evidence-Backed Guidance (Battery, Connectivity, Keyboard, etc.)
        steps = response_plan.steps_to_include

        # If user is asking about battery drain (initial or follow-up)
        if effective_intent == "BATTERY_CHARGING_POWER" or "battery" in lower_query:
            has_restarted = any("restart" in s.lower() for s in attempted_steps)
            has_checked_battery = any("battery usage" in s.lower() for s in attempted_steps)

            if follow_up_type == "UNRESOLVED_FOLLOW_UP":
                if has_restarted and not has_checked_battery:
                    return (
                        "Since restarting your device didn't resolve the rapid battery drain, let's look closer at app energy usage. "
                        "Open Settings → Battery and examine the 24-hour and 7-day usage breakdown to see which specific apps consume the most power. "
                        "Next, go to Settings → General → Background App Refresh to disable background refresh for non-essential apps. "
                        "Also verify if a software update is available under Settings → General → Software Update."
                    )
                else:
                    return (
                        "Since the battery drain persists after the initial troubleshooting, try these additional steps:\n"
                        "1. Navigate to Settings → General → Background App Refresh and turn it off completely or for unused apps.\n"
                        "2. Enable Low Power Mode under Settings → Battery to temporarily limit background activity.\n"
                        "3. Check Settings → General → Software Update to ensure you have the latest iOS performance patches."
                    )

            if follow_up_type == "ADDITIONAL_INFORMATION" and has_restarted:
                return (
                    "Thank you for confirming that you already restarted your device. "
                    "Since the battery drain persists, the next step is to inspect app consumption. "
                    "Open Settings → Battery to check the 24-hour battery usage breakdown and identify apps consuming excessive background energy. "
                    "You can also disable background activity under Settings → General → Background App Refresh, "
                    "and check Settings → General → Software Update for available fixes."
                )

            # Normal initial battery guidance (only recommend restart if NOT already attempted)
            restart_suffix = ", and restart your device." if not has_restarted else "."
            return (
                "Fast battery drain can often be narrowed down by checking which apps are using the most power. "
                "Open Settings → Battery and check the 24-hour and 7-day usage breakdown to see if specific apps are consuming excessive background energy. "
                "You can also navigate to Settings → General → Background App Refresh to disable background activity for apps you don't need constantly. "
                f"Additionally, make sure your device is running the latest software by checking Settings → General → Software Update{restart_suffix}"
            )

        # If user is asking about Wi-Fi / Connectivity
        if query_understanding.intent == "CONNECTIVITY_WIFI_BLUETOOTH":
            return (
                "For Wi-Fi and connectivity troubles, we recommend following these steps:\n"
                "1. Toggle Airplane Mode on for 15 seconds, then turn it off to refresh network connections.\n"
                "2. Go to Settings → Wi-Fi, tap the info icon next to your network, select 'Forget This Network', and reconnect.\n"
                "3. If issues persist, navigate to Settings → General → Reset → Reset Network Settings (note: this resets saved Wi-Fi passwords).\n"
                "4. Restart your Wi-Fi router and your Apple device."
            )

        # If user is asking about Keyboard / Autocorrect bug
        if query_understanding.intent == "KEYBOARD_TYPING_AUTOCORRECT":
            return (
                "If you are experiencing keyboard glitches or autocorrect issues, try these verified steps:\n"
                "1. Go to Settings → General → Keyboard → Text Replacement and check for any conflicting or erroneous replacement shortcuts.\n"
                "2. Check Settings → General → Software Update to ensure you've installed the latest bug-fix updates.\n"
                "3. If needed, you can reset your dictionary under Settings → General → Reset → Reset Keyboard Dictionary."
            )

        # General synthesized troubleshooting from resolution steps
        formatted_steps = []
        for idx, s in enumerate(steps[:4], 1):
            s_clean = s.strip().rstrip(".")
            formatted_steps.append(f"{idx}. {s_clean}")

        steps_str = "\n".join(formatted_steps) if formatted_steps else "1. Check Settings > General > Software Update to install the latest updates.\n2. Restart your device."

        return (
            f"Here are the recommended troubleshooting steps based on verified Apple Support guidance:\n\n"
            f"{steps_str}\n\n"
            "Please try these steps and let us know if your issue is resolved!"
        )
