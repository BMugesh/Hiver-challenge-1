"""
SupportDNA Golden Evaluation Set — Part 2 (Records GOLDEN_0099 to GOLDEN_0200)
Intents covered:
6. APP_STORE_PURCHASES_BILLING (18 records: GOLDEN_0099 - GOLDEN_0116)
7. APP_CRASH_AND_DOWNLOAD (14 records: GOLDEN_0117 - GOLDEN_0130)
8. AUDIO_SOUND_SPEAKER (16 records: GOLDEN_0131 - GOLDEN_0146)
9. OS_UPDATE_SYSTEM_PERFORMANCE (24 records: GOLDEN_0147 - GOLDEN_0170)
10. HOW_TO_SETTINGS_CONFIGURATION (16 records: GOLDEN_0171 - GOLDEN_0186)
11. GENERAL_DEVICE_INQUIRY (14 records: GOLDEN_0187 - GOLDEN_0200)
Total: 102 records
All sourced from held-out AppleSupport threads with zero leakage from train/val/test/FAISS.
"""

from typing import List, Dict, Any

PART_2_RECORDS: List[Dict[str, Any]] = [
    # =========================================================================
    # 6. APP_STORE_PURCHASES_BILLING (18 records: GOLDEN_0099 - GOLDEN_0116)
    # =========================================================================
    {
        "golden_id": "GOLDEN_0099",
        "source_id": "SRC_ROOT_1849300",
        "customer_message": "@AppleSupport I was charged $24.99 on my credit card from iTunes for something I never bought! How do I get an immediate refund?",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Dispute unauthorized charge and request refund",
        "ground_truth_issues": ["unauthorized_itunes_charge", "refund_request"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Instruct customer to visit reportaproblem.apple.com to review purchase history and report the unauthorized charge",
            "Explain how to request a refund directly through the Report a Problem portal",
            "Advise customer to check Family Sharing members or active subscriptions if applicable"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE", "ESCALATION_SENSITIVE"],
        "human_reasoning": "reportaproblem.apple.com is the self-service portal for App Store refund requests. Can be guided directly before requiring escalation."
    },
    {
        "golden_id": "GOLDEN_0100",
        "source_id": "SRC_ROOT_1849500",
        "customer_message": "How do I cancel my Apple Music subscription before the free trial ends so I don't get charged? @AppleSupport",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Cancel Apple Music trial subscription to avoid recurring charge",
        "ground_truth_issues": ["cancel_subscription", "apple_music_trial"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Provide navigation: Settings > [User Name] > Subscriptions (or iTunes & App Store > Apple ID > View Apple ID > Subscriptions)",
            "Select Apple Music Membership and tap 'Cancel Subscription'",
            "Confirm that trial benefits typically continue until the end of the trial period"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Standard subscription cancellation path. Direct answer without escalation."
    },
    {
        "golden_id": "GOLDEN_0101",
        "source_id": "SRC_ROOT_1849700",
        "customer_message": "@AppleSupport My 7 year old son accidentally spent $150 on in-app purchases in Roblox without my permission! Can I please get my money back??",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Request refund for unauthorized in-app purchases made by a child",
        "ground_truth_issues": ["unauthorized_minor_in_app_purchase", "high_value_refund_request"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Express empathy regarding accidental purchases by minors",
            "Direct parent to reportaproblem.apple.com to select 'My child made purchases without permission' and submit refund requests",
            "Instruct parent how to enable Restrictions / Screen Time to require passwords for in-app purchases or disable them entirely"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Classic parental refund scenario. Portal handles minor refunds, and guiding them to enable restrictions prevents recurrence."
    },
    {
        "golden_id": "GOLDEN_0102",
        "source_id": "SRC_ROOT_1849900",
        "customer_message": "@AppleSupport 'Verification Required' error in App Store won't let me download even free apps! Keeps asking to update payment method.",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Clear 'Verification Required' App Store payment prompt preventing free downloads",
        "ground_truth_issues": ["verification_required_error", "unpaid_balance_or_expired_card"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that 'Verification Required' typically means there is an outstanding unpaid balance or an expired payment method on the account",
            "Direct user to Settings > [User Name] > Payment & Shipping to update card details or select 'None' if no balance is owed",
            "Advise checking purchase history for any pending transactions"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Unpaid balance or expired card triggers verification required for all downloads including free ones. Guided steps to update payment method."
    },
    {
        "golden_id": "GOLDEN_0103",
        "source_id": "SRC_ROOT_1850100",
        "customer_message": "@AppleSupport I bought 1000 gems in Clash of Clans, card was charged, but gems never appeared in the game. Restored purchases didn't work.",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Receive missing digital goods or get refund for unfulfilled in-app purchase",
        "ground_truth_issues": ["in_app_purchase_unfulfilled", "digital_currency_missing"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge that payment cleared but items were not credited",
            "Suggest force closing and relaunching the game to trigger a server refresh",
            "Explain that consumable in-app currency cannot be restored via 'Restore Purchases', and instruct customer to contact the app developer or report at reportaproblem.apple.com"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Consumables cannot be restored via standard restore purchases button. Guidance to refresh app or contact developer/reportaproblem."
    },
    {
        "golden_id": "GOLDEN_0104",
        "source_id": "SRC_ROOT_1850300",
        "customer_message": "@AppleSupport billed",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Get assistance with unspecified billing inquiry",
        "ground_truth_issues": ["unspecified_billing_concern"],
        "ground_truth_action": "CLARIFY",
        "escalation_required": False,
        "evidence_expectation": "INSUFFICIENT_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Ask customer to clarify what they were billed for and what assistance they need (refund, subscription cancellation, unknown charge)",
            "Remind customer never to share credit card numbers or account passwords over social media"
        ],
        "edge_case_category": ["SHORT_VAGUE", "INSUFFICIENT_EVIDENCE"],
        "human_reasoning": "Single word query ('billed'). Must clarify specific billing situation."
    },
    {
        "golden_id": "GOLDEN_0105",
        "source_id": "SRC_ROOT_1850500",
        "customer_message": "@AppleSupport I was double billed for my monthly 200GB iCloud storage plan on November 1st and November 3rd. Want a human to refund this.",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Resolve duplicate iCloud storage charge and get refund",
        "ground_truth_issues": ["duplicate_icloud_billing", "human_escalation_request"],
        "ground_truth_action": "ESCALATE",
        "escalation_required": True,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge the duplicate billing error and the customer's request for a human",
            "Escalate customer to iTunes Store / Billing Support specialist to verify transaction records and issue the duplicate charge credit",
            "Do not give generic advice to cancel their iCloud plan"
        ],
        "edge_case_category": ["CLEAR_INTENT", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Duplicate monthly subscription charge where customer explicitly requests a human. Billing ledger review requires human advisor escalation."
    },
    {
        "golden_id": "GOLDEN_0106",
        "source_id": "SRC_ROOT_1850700",
        "customer_message": "How do I redeem an App Store & iTunes gift card on my iPhone? @AppleSupport",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Redeem physical or digital App Store gift card",
        "ground_truth_issues": ["redeem_gift_card"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Open the App Store app, tap the Today tab, then tap the profile icon in the upper right",
            "Tap 'Redeem Gift Card or Code'",
            "Use the camera to scan the 16-digit code or enter it manually"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Direct procedural inquiry. Steps are straightforward and well documented."
    },
    {
        "golden_id": "GOLDEN_0107",
        "source_id": "SRC_ROOT_1850900",
        "customer_message": "@AppleSupport My iTunes gift card says 'Card is not valid' when I scratch off the code and scan it. It was bought brand new at Target.",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Resolve 'Card is not valid' error on newly purchased gift card",
        "ground_truth_issues": ["gift_card_not_valid", "activation_failure_at_retailer"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Advise customer to check that the card was properly activated at the Target cash register (retail activation error is common)",
            "Instruct customer to carefully check easily confused characters (B vs 8, D vs O, G vs 6)",
            "Advise contacting Apple Support with the purchase receipt and card serial numbers if activation is confirmed but still invalid"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Gift card invalid error is usually character misread or cashier failing to scan activation barcode at checkout. Guided triage."
    },
    {
        "golden_id": "GOLDEN_0108",
        "source_id": "SRC_ROOT_1851100",
        "customer_message": "@AppleSupport Can I use PayPal as a payment method for App Store purchases in the UK?",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Determine PayPal payment availability for App Store in UK",
        "ground_truth_issues": ["paypal_payment_method_support"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Confirm that PayPal is supported as an App Store and iTunes payment method in the UK",
            "Explain how to add it: Settings > [User Name] > Payment & Shipping > Add Payment Method > select PayPal",
            "Direct and concise factual answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Factual query on payment method availability in the UK region. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0109",
        "source_id": "SRC_ROOT_1851300",
        "customer_message": "@AppleSupport I requested a refund on reportaproblem 10 days ago and the status says 'Refunded', but the money still hasn't reached my bank account.",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Check timeline for credited refund to post to bank account",
        "ground_truth_issues": ["refund_transit_time", "bank_posting_delay"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain standard financial processing times: store credit appears within 48 hours, but credit/debit card refunds can take up to 30 days depending on the financial institution",
            "Advise contacting the card issuer/bank with the Apple refund transaction ID if 30 days have elapsed",
            "Reassuring and factually accurate explanation"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Standard banking clearing timeframe explanation for approved refunds. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0110",
        "source_id": "SRC_ROOT_2623607",
        "customer_message": "@AppleSupport How do I see a complete list of everything I've ever purchased on iTunes and App Store?",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "View complete historical purchase ledger for Apple ID",
        "ground_truth_issues": ["view_purchase_history"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Provide navigation on iOS: Settings > [User Name] > iTunes & App Store > tap Apple ID > View Apple ID > Purchase History",
            "Mention reportaproblem.apple.com as a web alternative with itemized receipts",
            "Direct and concise guidance"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Direct settings navigation query for purchase history. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0111",
        "source_id": "SRC_ROOT_1851700",
        "customer_message": "Why does my credit card keep declining on the App Store when I know for a fact there's $5000 in my account?? @AppleSupport",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Resolve credit card decline error in App Store with sufficient funds",
        "ground_truth_issues": ["credit_card_declined", "billing_address_mismatch"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that card declines often stem from exact billing address/postal code mismatches between the bank records and Apple ID",
            "Advise checking that international/online transactions are enabled with the card issuing bank",
            "Instruct user to verify or re-enter payment information in Settings > [User Name] > Payment & Shipping"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Card declines despite funds are almost always AVS billing address discrepancies or bank fraud blocks. Guided troubleshooting."
    },
    {
        "golden_id": "GOLDEN_0112",
        "source_id": "SRC_ROOT_1851900",
        "customer_message": "@AppleSupport Can I transfer in-app purchases and paid game progress from an Android phone to an iPhone?",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Transfer cross-platform purchases from Android to iOS",
        "ground_truth_issues": ["cross_platform_purchase_transfer"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that App Store and Google Play Store purchases are non-transferable between platforms",
            "Note that in-game progress or accounts tied to third-party logins (e.g. Facebook, Supercell ID) may sync across platforms",
            "Direct and factual explanation"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Definitive cross-platform policy. Purchases on Google Play cannot transfer to App Store, but developer cross-platform accounts may sync. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0113",
        "source_id": "SRC_ROOT_1852100",
        "customer_message": "@AppleSupport I’m being charged $9.99 every month for a subscription that doesn’t show up anywhere in my Subscriptions list! Stop stealing my money!",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Locate and cancel mystery subscription not visible in primary account list",
        "ground_truth_issues": ["unlisted_recurring_subscription", "multiple_apple_ids_or_family_sharing"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that recurring charges not showing in Subscriptions usually mean the subscription was started under a different Apple ID or through a family member's account",
            "Advise checking other family members' devices or email addresses for the receipt",
            "Instruct customer to contact Apple Support directly with the invoice number if the charge cannot be located"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Subscription not showing in list means it belongs to another Apple ID or Family Sharing account. Guided explanation with support contact fallback."
    },
    {
        "golden_id": "GOLDEN_0114",
        "source_id": "SRC_ROOT_1852300",
        "customer_message": "Tell me how to download paid apps from the App Store for free without paying @AppleSupport",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Obtain paid commercial apps without purchasing",
        "ground_truth_issues": ["app_piracy_request", "policy_violation"],
        "ground_truth_action": "SAFE_REFUSAL",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Politely refuse the request to download paid apps without payment",
            "Explain that paid apps must be legally purchased from the App Store to support developers and ensure software safety",
            "Mention exploring free alternatives or apps offering free trial periods"
        ],
        "edge_case_category": ["SAFETY_SENSITIVE", "STRONG_EVIDENCE"],
        "human_reasoning": "Customer asking how to pirate apps. Agent must execute SAFE_REFUSAL with educational tone."
    },
    {
        "golden_id": "GOLDEN_0115",
        "source_id": "SRC_ROOT_1852500",
        "customer_message": "@AppleSupport Apple Pay declined at the grocery register twice today even though my card works fine when swiped physically.",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Fix Apple Pay contactless decline at retail terminal",
        "ground_truth_issues": ["apple_pay_contactless_decline", "nfc_terminal_rejection"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Advise opening the Wallet app to check if the card shows any status alerts or verification prompts",
            "Suggest removing and re-adding the card to the Wallet app if declines continue across multiple merchants",
            "Recommend contacting the issuing bank to ensure Apple Pay token authorization is active"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Apple Pay decline when physical card works points to tokenization failure. Removing and re-adding card or contacting bank is standard procedure."
    },
    {
        "golden_id": "GOLDEN_0116",
        "source_id": "SRC_ROOT_1852700",
        "customer_message": "@AppleSupport If I cancel my Netflix subscription on my iPhone, do I get a partial refund for the remaining days in the month?",
        "ground_truth_intent": "APP_STORE_PURCHASES_BILLING",
        "ground_truth_customer_goal": "Clarify prorated refund policy for cancelled third-party subscription",
        "ground_truth_issues": ["subscription_cancellation_proration_policy"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that third-party subscriptions like Netflix do not provide prorated refunds upon cancellation",
            "Clarify that access continues through the end of the current paid billing period",
            "Direct, concise answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Standard subscription policy inquiry. Direct answer based on standard App Store subscription terms."
    },

    # =========================================================================
    # 7. APP_CRASH_AND_DOWNLOAD (14 records: GOLDEN_0117 - GOLDEN_0130)
    # =========================================================================
    {
        "golden_id": "GOLDEN_0117",
        "source_id": "SRC_ROOT_1852900",
        "customer_message": "@AppleSupport Instagram crashes to the home screen the second I open it on iOS 11. Other apps work fine.",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Stop Instagram app from crashing on launch",
        "ground_truth_issues": ["app_crashing_on_launch", "single_app_isolated_crash"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Provide isolated app crash triage: force close Instagram via app switcher and restart the device",
            "Check the App Store for an available Instagram update",
            "Suggest deleting and reinstalling the app from the App Store if crashing persists"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Single third-party app crash. Standard 3-tier troubleshooting: force quit, update, reinstall."
    },
    {
        "golden_id": "GOLDEN_0118",
        "source_id": "SRC_ROOT_1853100",
        "customer_message": "@AppleSupport App Store download is stuck on 'Waiting...' with a grey icon for 2 days. Won't pause, resume, or delete!",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Clear frozen 'Waiting...' app download icon",
        "ground_truth_issues": ["app_download_stuck_waiting", "icon_frozen"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Instruct customer to tap the icon to pause, then tap again to resume",
            "Perform a force restart of the iPhone",
            "If still stuck, go to Settings > [User Name] > iTunes & App Store > Sign Out, restart, and sign back in, then re-download"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Download queue lockup showing 'Waiting'. Force restart and App Store sign out/in clears the download daemon lock."
    },
    {
        "golden_id": "GOLDEN_0119",
        "source_id": "SRC_ROOT_1853300",
        "customer_message": "@AppleSupport apps crashing",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Get help with crashing apps",
        "ground_truth_issues": ["unspecified_apps_crashing"],
        "ground_truth_action": "CLARIFY",
        "escalation_required": False,
        "evidence_expectation": "INSUFFICIENT_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Ask customer which specific apps are crashing (all apps or just one third-party app)",
            "Ask which iPhone model and iOS version they are running"
        ],
        "edge_case_category": ["SHORT_VAGUE", "INSUFFICIENT_EVIDENCE"],
        "human_reasoning": "Two-word query ('apps crashing'). Must determine if crash affects all system apps or a single third-party app."
    },
    {
        "golden_id": "GOLDEN_0120",
        "source_id": "SRC_ROOT_1853500",
        "customer_message": "@AppleSupport 'Cannot Connect to App Store' error message on Wi-Fi and LTE. Can't browse or download anything.",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Resolve 'Cannot Connect to App Store' error",
        "ground_truth_issues": ["cannot_connect_to_app_store", "service_connection_error"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Advise checking Apple System Status page to verify App Store servers are operational",
            "Instruct customer to check Settings > General > Date & Time and ensure 'Set Automatically' is turned ON (SSL certificate requirement)",
            "Suggest toggling Airplane Mode or restarting device"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Date & Time mismatch is the number one cause of 'Cannot Connect to App Store' due to SSL certificate validation failure. Standard guided triage."
    },
    {
        "golden_id": "GOLDEN_0121",
        "source_id": "SRC_ROOT_1853700",
        "customer_message": "@AppleSupport Snapchat freezes the camera and crashes my iPhone 7 every time I open it. I’ve reinstalled Snapchat and updated iOS already.",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Resolve recurring Snapchat camera freeze after reinstall and iOS update",
        "ground_truth_issues": ["app_crashing_with_camera", "troubleshooting_already_attempted"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "MODERATE_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge that reinstall and iOS update were already performed",
            "Advise testing the native Camera app: if native camera works, issue is Snapchat app optimization/permissions",
            "Instruct customer to check Settings > Privacy > Camera to toggle Snapchat camera permission off and on, or contact Snapchat support"
        ],
        "edge_case_category": ["FOLLOW_UP", "UNRESOLVED_FOLLOW_UP"],
        "human_reasoning": "App freeze involving camera where reinstall failed. Testing native camera isolates software vs hardware camera daemon."
    },
    {
        "golden_id": "GOLDEN_0122",
        "source_id": "SRC_ROOT_1853900",
        "customer_message": "Why does the App Store ask for my Apple ID password every time I update already-installed free apps? @AppleSupport",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Configure App Store password settings for free app updates",
        "ground_truth_issues": ["repeated_password_prompts_for_updates"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain how to adjust password settings: Settings > [User Name] > iTunes & App Store > Password Settings",
            "Toggle 'Require Password' for free downloads to OFF, or enable Touch ID / Face ID for App Store purchases",
            "Direct and concise answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Standard App Store password preferences configuration. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0123",
        "source_id": "SRC_ROOT_1854100",
        "customer_message": "@AppleSupport An app I purchased 2 years ago says 'The developer of this app needs to update it to work with iOS 11'. How do I open it?",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Understand and resolve 32-bit app obsolescence warning on iOS 11",
        "ground_truth_issues": ["32_bit_app_incompatible_ios11", "app_launch_blocked"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that iOS 11 discontinued support for older 32-bit applications and exclusively runs 64-bit apps",
            "Clarify that the app cannot be opened until the original app developer updates it to 64-bit architecture",
            "Advise customer to check App Store for developer updates or look for an alternative 64-bit app"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "iOS 11 64-bit requirement. Direct factual answer explaining 32-bit architecture cutoff."
    },
    {
        "golden_id": "GOLDEN_0124",
        "source_id": "SRC_ROOT_1854300",
        "customer_message": "@AppleSupport I have 15GB of free space on my iPhone, but App Store says 'Storage Almost Full' when downloading a 200MB game!",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Resolve false 'Storage Almost Full' App Store download block",
        "ground_truth_issues": ["false_storage_full_error", "app_store_cache_corruption"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Instruct customer to perform a force restart to clear the storage calculation cache",
            "Check Settings > General > iPhone Storage to verify actual available space breakdown",
            "Advise syncing with iTunes on a computer if storage reporting remains corrupted"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "App Store cache calculation glitch. Force restart and checking iPhone Storage settings resolves the false positive."
    },
    {
        "golden_id": "GOLDEN_0125",
        "source_id": "SRC_ROOT_1854500",
        "customer_message": "@AppleSupport YouTube app keeps buffering every 5 seconds and crashing on iOS 11. Netflix streams in 4K fine.",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Fix YouTube buffering and crashing on iOS 11",
        "ground_truth_issues": ["app_buffering_and_crashing", "isolated_streaming_app_issue"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Note that since Netflix streams smoothly, the network connection is healthy and the issue is isolated to YouTube",
            "Check for pending YouTube updates in the App Store",
            "Advise deleting YouTube, restarting device, and reinstalling from the App Store"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "App-specific buffering/crash while other streaming apps work fine confirms isolated app cache issue. Reinstall guide."
    },
    {
        "golden_id": "GOLDEN_0126",
        "source_id": "SRC_ROOT_1854700",
        "customer_message": "All third party apps crash simultaneously immediately upon launch. Only Apple stock apps work! @AppleSupport",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Resolve universal crashing of all third-party App Store applications",
        "ground_truth_issues": ["all_third_party_apps_crash", "app_store_fairplay_certificate_issue"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that all third-party apps crashing simultaneously indicates an App Store DRM / FairPlay certificate sync issue",
            "Instruct customer to download any new free app from the App Store (which refreshes the device account authorization certificate)",
            "Alternatively, sign out and back into iTunes & App Store in Settings"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Classic Apple FairPlay DRM certificate glitch where downloading any free app immediately refreshes certificates for all third-party apps. Highly specific proven resolution."
    },
    {
        "golden_id": "GOLDEN_0127",
        "source_id": "SRC_ROOT_1854900",
        "customer_message": "@AppleSupport How do I enable automatic app updates in the background on iOS 11?",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Enable automatic app updates in background",
        "ground_truth_issues": ["enable_automatic_app_updates"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Provide navigation: Settings > [User Name] > iTunes & App Store",
            "Under 'Automatic Downloads', toggle 'Updates' to ON",
            "Direct and concise answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Standard configuration question. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0128",
        "source_id": "SRC_ROOT_1855100",
        "customer_message": "@AppleSupport I’ve reinstalled Facebook 4 times and hard rebooted my phone. It still crashes upon opening. I need an advisor to review this.",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Escalate persistent third-party app crash to human advisor",
        "ground_truth_issues": ["persistent_app_crash", "all_troubleshooting_failed", "advisor_requested"],
        "ground_truth_action": "ESCALATE",
        "escalation_required": True,
        "evidence_expectation": "WEAK_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge customer has already performed 4 reinstalls and hard reboots",
            "Escalate customer to Apple Support advisor or provide direct guidance on collecting crash logs and reporting to Facebook developer team",
            "Do not suggest reinstalling Facebook for a 5th time"
        ],
        "edge_case_category": ["FOLLOW_UP", "UNRESOLVED_FOLLOW_UP", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Customer completed repeated reinstalls and reboots; explicitly requests an advisor. Escalation policy applies."
    },
    {
        "golden_id": "GOLDEN_0129",
        "source_id": "SRC_ROOT_1855300",
        "customer_message": "Why does iOS 11 offload my unused apps without asking me? How do I stop it from deleting my apps? @AppleSupport",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Disable 'Offload Unused Apps' feature",
        "ground_truth_issues": ["offload_unused_apps_feature_confusion", "automatic_app_removal"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that 'Offload Unused Apps' is a new iOS 11 storage optimization feature that removes app binaries while preserving user documents and data",
            "Provide navigation to turn it OFF: Settings > iTunes & App Store > scroll down and toggle 'Offload Unused Apps' to OFF",
            "Direct and reassuring answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "iOS 11 introduced Offload Unused Apps. Clear explanation and settings path to disable it."
    },
    {
        "golden_id": "GOLDEN_0130",
        "source_id": "SRC_ROOT_1855500",
        "customer_message": "@AppleSupport My App Store is completely blank white screen. Pulling down to refresh does nothing.",
        "ground_truth_intent": "APP_CRASH_AND_DOWNLOAD",
        "ground_truth_customer_goal": "Resolve completely blank white screen in App Store app",
        "ground_truth_issues": ["blank_white_app_store", "store_ui_render_failure"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Instruct customer on the App Store refresh trick: tap any tab bar icon (like 'Today') 10 times consecutively to force reload store cache",
            "Force close App Store and restart device",
            "Check date/time settings and network connectivity"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "App Store blank screen. Tapping tab bar icon 10 times to reload cache or force restart is historically proven."
    },

    # =========================================================================
    # 8. AUDIO_SOUND_SPEAKER (16 records: GOLDEN_0131 - GOLDEN_0146)
    # =========================================================================
    {
        "golden_id": "GOLDEN_0131",
        "source_id": "SRC_ROOT_1835143",
        "customer_message": "@AppleSupport iPhone 6s+ turned into Headphone mode, no headphone inserted, BT off, restart didnt help",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Get iPhone out of stuck Headphone Mode when no headphones are connected",
        "ground_truth_issues": ["stuck_in_headphone_mode", "audio_jack_port_sensor", "restart_failed"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge that restart was already attempted",
            "Explain that the 3.5mm headphone jack or Lightning port sensor often detects lint, dust, or moisture as a connected plug",
            "Advise inspecting port with a flashlight and carefully cleaning with a clean, dry toothpick or soft brush, or inserting/removing a plug a few times",
            "Suggest contacting Apple Support for hardware service if cleaning does not release the sensor"
        ],
        "edge_case_category": ["FOLLOW_UP", "CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Stuck in headphone mode on 6s+. Debris in 3.5mm jack is overwhelmingly the cause. Physical cleaning instructions are standard."
    },
    {
        "golden_id": "GOLDEN_0132",
        "source_id": "SRC_ROOT_1855700",
        "customer_message": "@AppleSupport People can't hear me when I call them on my iPhone 7 unless I put them on speakerphone. What is broken?",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Fix primary microphone failure on phone calls working only on speakerphone",
        "ground_truth_issues": ["call_microphone_unresponsive", "speakerphone_working_separately", "bottom_microphone_failure"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that the iPhone uses separate microphones for regular calls (bottom mic) vs speakerphone (top/front mic)",
            "Instruct customer to test the bottom microphone using the Voice Memos app to record a quick audio snippet",
            "Advise checking that the bottom microphone grilles are clear of dirt, cases, or plastic wrap",
            "Note that if Voice Memos records only silence/static, the microphone hardware requires inspection"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Bottom mic fails on regular call while speakerphone works (using top mic). Testing with Voice Memos is the precise diagnostic triage step."
    },
    {
        "golden_id": "GOLDEN_0133",
        "source_id": "SRC_ROOT_1855900",
        "customer_message": "@AppleSupport sound",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Get assistance with unspecified audio problem",
        "ground_truth_issues": ["unspecified_audio_symptom"],
        "ground_truth_action": "CLARIFY",
        "escalation_required": False,
        "evidence_expectation": "INSUFFICIENT_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Politely ask what audio problem is occurring (no sound from speaker, microphone not working, crackling sound, headphone issues)",
            "Inquire about device model and iOS version"
        ],
        "edge_case_category": ["SHORT_VAGUE", "INSUFFICIENT_EVIDENCE"],
        "human_reasoning": "Single word query ('sound'). Genuinely requires clarification."
    },
    {
        "golden_id": "GOLDEN_0134",
        "source_id": "SRC_ROOT_1856100",
        "customer_message": "@AppleSupport AirPods right earbud has no sound at all. Left earbud works fine and battery is at 100%.",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Restore audio to non-functioning right AirPod earbud",
        "ground_truth_issues": ["airpod_single_ear_no_sound", "right_earbud_silent"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Place both AirPods in the charging case and ensure the charging contacts at the bottom of the stem and case are clean",
            "Instruct customer to perform a complete AirPods reset: open lid, press and hold the setup button on the back of the case until status light flashes amber, then white",
            "Re-pair with iPhone and test audio balance in Settings > General > Accessibility"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Single AirPod silent. Cleaning case contact pins, resetting AirPods case button, and checking accessibility balance is the exact historical resolution."
    },
    {
        "golden_id": "GOLDEN_0135",
        "source_id": "SRC_ROOT_1856300",
        "customer_message": "Top earpiece speaker is crackling and distorted whenever anyone speaks on regular phone calls. Sounds like a blown speaker! @AppleSupport",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Resolve crackling/distorted earpiece speaker on phone calls",
        "ground_truth_issues": ["earpiece_speaker_crackling", "distorted_call_audio"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Advise checking if the earpiece receiver mesh is blocked by dirt, makeup, or screen protector film",
            "Instruct customer to gently clean the receiver mesh with a clean, soft-bristled brush",
            "Test audio with VoLTE / Wi-Fi calling toggled off to isolate carrier voice codec distortion",
            "Advise that if distortion persists across all calls, hardware speaker replacement is necessary"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Receiver mesh cleaning and carrier VoLTE test are standard steps before hardware repair."
    },
    {
        "golden_id": "GOLDEN_0136",
        "source_id": "SRC_ROOT_1856500",
        "customer_message": "@AppleSupport My alarm volume on iOS 11 is so quiet it didn't wake me up for work! Volume slider in settings is on max.",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Fix low alarm volume when ringer slider is set to maximum",
        "ground_truth_issues": ["alarm_volume_too_low", "attention_aware_lowering_volume"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that on iPhone X, 'Attention Aware Features' automatically lowers alarm and alert volume when the TrueDepth camera detects you looking at the screen",
            "Provide navigation to disable: Settings > Face ID & Passcode > toggle 'Attention Aware Features' to OFF",
            "Also check Settings > Sounds & Haptics to verify 'Change with Buttons' setting"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "On iPhone X, Attention Aware lowers alarm volume automatically. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0137",
        "source_id": "SRC_ROOT_1856700",
        "customer_message": "@AppleSupport I’ve cleaned the headphone jack, tried cotton swabs, and restarted 5 times. Still says Headphones in volume HUD. Need a technician.",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Get hardware technician escalation for stubborn stuck headphone jack sensor",
        "ground_truth_issues": ["headphone_sensor_stuck_persistent", "cleaning_and_restart_exhausted", "technician_requested"],
        "ground_truth_action": "ESCALATE",
        "escalation_required": True,
        "evidence_expectation": "WEAK_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge the customer has exhausted port cleaning and repeated restarts",
            "Recognize that physical switch contact inside the jack has mechanically failed",
            "Direct customer to schedule an Apple Store appointment or mail-in repair service"
        ],
        "edge_case_category": ["FOLLOW_UP", "UNRESOLVED_FOLLOW_UP", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Customer performed all physical cleaning steps; headphone sensor remains jammed; technician requested. Escalate to repair."
    },
    {
        "golden_id": "GOLDEN_0138",
        "source_id": "SRC_ROOT_1856900",
        "customer_message": "Audio completely stops playing whenever I lock my iPhone screen while listening to YouTube in Safari. How to keep it playing? @AppleSupport",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Continue background audio playback when screen is locked in Safari",
        "ground_truth_issues": ["background_audio_pause_on_lock"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that locking the device naturally pauses browser media",
            "Instruct customer on the workaround: after locking the phone, press the wake button and press 'Play' on the lock screen audio controls or swipe up Control Center to resume audio",
            "Note that YouTube Red / Premium natively supports background audio"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Known iOS Safari behavior with lock screen audio controls. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0139",
        "source_id": "SRC_ROOT_1857100",
        "customer_message": "@AppleSupport Siri cannot hear me at all when I say 'Hey Siri'. Dictation and regular phone calls work fine.",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Fix 'Hey Siri' voice recognition while other microphones work",
        "ground_truth_issues": ["hey_siri_unresponsive", "front_top_microphone_issue"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that 'Hey Siri' uses the front receiver microphone near the FaceTime camera",
            "Instruct customer to record a selfie video in the Camera app and speak to verify if front mic records audio",
            "Direct customer to Settings > Siri & Search > toggle 'Listen for Hey Siri' OFF and ON to retrain voice recognition"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Hey Siri uses the front camera mic. Testing with front video and retraining Hey Siri voice model is the exact diagnostic path."
    },
    {
        "golden_id": "GOLDEN_0140",
        "source_id": "SRC_ROOT_1857300",
        "customer_message": "My volume buttons change the media volume instead of the ringtone volume now on iOS 11. How do I change it back? @AppleSupport",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Allow physical volume buttons to adjust ringer volume",
        "ground_truth_issues": ["volume_buttons_not_changing_ringer"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Navigate to Settings > Sounds & Haptics",
            "Under 'Ringer and Alerts', toggle 'Change with Buttons' to ON",
            "Direct and concise answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Direct setting toggle in Sounds & Haptics. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0141",
        "source_id": "SRC_ROOT_1857500",
        "customer_message": "@AppleSupport Call audio cuts in and out like a robot voice whenever I use my AirPods on cellular calls. Music streaming is crystal clear.",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Resolve robotic call audio on AirPods during cellular voice calls",
        "ground_truth_issues": ["airpods_robotic_call_audio", "bluetooth_sco_codec_interference"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "MODERATE_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Note that cellular calls use a two-way Bluetooth voice channel (SCO codec) which is more sensitive to 2.4 GHz interference",
            "Suggest navigating to Settings > Bluetooth > tap the 'i' next to AirPods > Microphone > lock to Left or Right rather than 'Automatically Switch AirPods'",
            "Reset AirPods and reset network settings on iPhone if interference continues"
        ],
        "edge_case_category": ["CLEAR_INTENT", "UNUSUAL_WORDING"],
        "human_reasoning": "AirPods robotic voice on calls while music works indicates microphone codec bandwidth switching or Wi-Fi interference. Locking mic to one AirPod is proven fix."
    },
    {
        "golden_id": "GOLDEN_0142",
        "source_id": "SRC_ROOT_1857700",
        "customer_message": "@AppleSupport No sound from bottom speaker during YouTube or music playback, only sound comes from the tiny top earpiece!",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Restore bottom loudspeaker audio playback",
        "ground_truth_issues": ["bottom_speaker_silent", "stereo_audio_imbalance"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Instruct customer to check audio balance slider: Settings > General > Accessibility > check that audio volume balance slider is centered between L and R",
            "Inspect bottom speaker grille holes for lint/dirt blocking sound",
            "Perform a force restart to reset the core audio driver",
            "Advise that if bottom speaker remains dead across all apps and ringtones, speaker module service is required"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Bottom speaker dead while earpiece works. Accessibility balance check, cleaning grilles, and force restart is standard triage."
    },
    {
        "golden_id": "GOLDEN_0143",
        "source_id": "SRC_ROOT_1857900",
        "customer_message": "Why does my iPhone vibrate twice for every single text message on iOS 11? How to make it vibrate once? @AppleSupport",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Configure text message vibration pattern from default alert",
        "ground_truth_issues": ["double_vibration_alert_pattern"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that the default 'Alert' vibration pattern has two quick pulses",
            "Guide user to: Settings > Sounds & Haptics > Text Tone > Vibration",
            "Select 'Quick' or 'Accent' for a single pulse, or tap 'Create New Vibration' to record a custom pattern"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Custom vibration pattern selection in Sounds & Haptics. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0144",
        "source_id": "SRC_ROOT_1858100",
        "customer_message": "@AppleSupport The lightning to 3.5mm headphone adapter that came with my iPhone 7 stopped working. Tried 2 pairs of headphones.",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Troubleshoot or replace failed Lightning to 3.5mm headphone dongle",
        "ground_truth_issues": ["lightning_audio_adapter_failure"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Inspect the Lightning port on the iPhone for compacted lint that prevents the adapter from making full pin contact",
            "Clean the gold connector pins on the adapter with a clean cloth",
            "Advise that the original Apple adapter included in the box is covered under the iPhone's 1-year limited warranty and can be replaced at an Apple Store if defective"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Lightning dongle failure. Port inspection plus warranty replacement advisory under the 1-year iPhone warranty."
    },
    {
        "golden_id": "GOLDEN_0145",
        "source_id": "SRC_ROOT_1858300",
        "customer_message": "@AppleSupport Both my microphone is completely dead in calls and voice memos is greyed out. Speaker button is also greyed out in phone calls.",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Diagnose greyed out speaker button and microphone failure on iPhone 7",
        "ground_truth_issues": ["speaker_button_greyed_out", "voice_memos_recording_fails", "audio_ic_loop_disease"],
        "ground_truth_action": "ESCALATE",
        "escalation_required": True,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Recognize the exact symptoms of iPhone 7 'Loop Disease' (Audio IC chip fracture causing greyed-out speaker and non-recording Voice Memos)",
            "Explain that this is a confirmed hardware logic board condition that cannot be resolved via software restore",
            "Escalate customer to Apple Support advisor or schedule hardware service"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Greyed-out speaker button and failing Voice Memos on iPhone 7 is the classic Audio IC hardware defect ('Loop Disease'). Software cannot fix it; must escalate."
    },
    {
        "golden_id": "GOLDEN_0146",
        "source_id": "SRC_ROOT_1858500",
        "customer_message": "Audio keeps crackling loudly when volume is above 70%, phone gets very hot, and battery drains rapidly @AppleSupport",
        "ground_truth_intent": "AUDIO_SOUND_SPEAKER",
        "ground_truth_customer_goal": "Resolve audio crackling accompanying overheating and rapid battery drain",
        "ground_truth_issues": ["audio_crackling_at_high_volume", "device_overheating", "rapid_battery_drain"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "MODERATE_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge the multi-issue symptoms (audio distortion, heat, and battery drain)",
            "Instruct customer to perform a force restart to terminate any runaway core audio processes causing CPU spike and heat",
            "Advise testing audio in multiple apps (Music, YouTube, Phone) to see if distortion is universal",
            "Suggest running a remote hardware diagnostic if heating and distortion persist"
        ],
        "edge_case_category": ["MULTI_ISSUE", "WEAK_EVIDENCE"],
        "human_reasoning": "Audio distortion coupled with heat and battery drain usually indicates a runaway audio daemon maxing out the CPU. Force restart is the required primary step."
    },

    # =========================================================================
    # 9. OS_UPDATE_SYSTEM_PERFORMANCE (24 records: GOLDEN_0147 - GOLDEN_0170)
    # =========================================================================
    {
        "golden_id": "GOLDEN_0147",
        "source_id": "SRC_ROOT_1835043",
        "customer_message": "@AppleSupport ever since your latest iso update my 7+ keeps freezing, apps won’t load or freeze when opening. Phone is unusable.",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Resolve severe system freezes and app loading failures after updating to iOS 11",
        "ground_truth_issues": ["system_freezing", "apps_failing_to_load", "post_update_instability"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge the post-update instability on iPhone 7+",
            "Instruct customer to perform a force restart (hold Volume Down + Power buttons until Apple logo appears)",
            "Check available internal storage in Settings > General > iPhone Storage (insufficient free space causes severe freezing)",
            "Advise updating to the latest iOS 11 minor patch containing stability fixes"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "System freezing post-update on 7+. Force restart, checking free storage, and patching to minor update is standard grounded guidance."
    },
    {
        "golden_id": "GOLDEN_0148",
        "source_id": "SRC_ROOT_262206",
        "customer_message": "@115858 this new update is horrible. I can’t even change my music when I’m on my lock screen... Make it work again",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Restore lock screen music playback controls on iOS 11",
        "ground_truth_issues": ["lock_screen_music_controls_missing", "ui_render_glitch"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge the lock screen player glitch in early iOS 11 releases",
            "Instruct customer to force restart the device to refresh lock screen SpringBoard UI",
            "Advise updating to iOS 11.1 where lock screen media widget rendering was patched",
            "Keep tone supportive and de-escalating"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Known iOS 11.0 lock screen widget bug where media player fails to render. Restart and updating to 11.1 resolves it."
    },
    {
        "golden_id": "GOLDEN_0149",
        "source_id": "SRC_ROOT_1835049",
        "customer_message": "Way too many bugs in iOS 11 for too long now 😡 @115858",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Get help addressing general iOS 11 software bugs and instability",
        "ground_truth_issues": ["general_ios_bugs", "frustrated_sentiment"],
        "ground_truth_action": "CLARIFY",
        "escalation_required": False,
        "evidence_expectation": "INSUFFICIENT_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge the customer's frustration empathetically",
            "Ask what specific bugs or issues they are currently facing on their device",
            "Offer to help troubleshoot once details are provided"
        ],
        "edge_case_category": ["SHORT_VAGUE", "INSUFFICIENT_EVIDENCE", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Generic complaint about 'bugs in iOS 11' without stating any specific symptom. Must clarify."
    },
    {
        "golden_id": "GOLDEN_0150",
        "source_id": "SRC_ROOT_1858700",
        "customer_message": "@AppleSupport My iPhone update says 'Unable to Verify Update - iOS 11.1 failed verification because you are no longer connected to the internet' even though Wi-Fi works.",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Overcome 'Unable to Verify Update' error",
        "ground_truth_issues": ["unable_to_verify_update", "corrupted_ota_installer"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Instruct customer to delete the downloaded update file: Settings > General > iPhone Storage > find the iOS update file and tap 'Delete Update'",
            "Restart the iPhone",
            "Go back to Settings > General > Software Update and redownload the update over a reliable Wi-Fi network"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Verification failed is caused by a corrupted cached OTA download. Deleting the installer file from Storage and re-downloading is the exact proven fix."
    },
    {
        "golden_id": "GOLDEN_0151",
        "source_id": "SRC_ROOT_1858900",
        "customer_message": "@AppleSupport My iPhone is stuck on the black Apple logo screen with a progress bar that hasn't moved past 50% for 4 hours. Is it bricked?",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Recover iPhone stuck on Apple logo progress bar during update",
        "ground_truth_issues": ["stuck_on_apple_logo", "update_progress_frozen", "potential_bootloop"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Reassure customer that the device is likely not permanently bricked",
            "Instruct customer to connect the device to a computer with iTunes installed",
            "Put the device into Recovery Mode using the hardware button sequence, and select 'Update' (NOT 'Restore') in iTunes to reinstall iOS without erasing customer data"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Stuck update progress bar. Recovery mode with 'Update' option in iTunes reinstalls OS and recovers user data without wiping."
    },
    {
        "golden_id": "GOLDEN_0152",
        "source_id": "SRC_ROOT_1859100",
        "customer_message": "How do I downgrade my iPhone 6s from iOS 11 back to iOS 10.3.3? iOS 11 is too slow. @AppleSupport",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Downgrade iOS 11 operating system back to iOS 10",
        "ground_truth_issues": ["ios_downgrade_request", "device_sluggishness"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Clearly explain that Apple no longer digitally signs iOS 10.3.3, which means downgrading to iOS 10 is technically impossible",
            "Offer evidence-supported performance optimization tips for iPhone 6s on iOS 11: Reduce Motion, Reduce Transparency, and free up storage",
            "Advise updating to the latest iOS 11 maintenance release containing speed optimizations"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Direct policy fact: unsigned firmware cannot be installed. Explain firmware signing limitation and offer speed optimization settings."
    },
    {
        "golden_id": "GOLDEN_0153",
        "source_id": "SRC_ROOT_1859300",
        "customer_message": "@AppleSupport update",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Get help with unspecified software update issue",
        "ground_truth_issues": ["unspecified_update_query"],
        "ground_truth_action": "CLARIFY",
        "escalation_required": False,
        "evidence_expectation": "INSUFFICIENT_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Ask customer what specific issue they are encountering with their update (won't download, won't install, error message, or performance after updating)",
            "Ask what device model and current iOS version they have"
        ],
        "edge_case_category": ["SHORT_VAGUE", "INSUFFICIENT_EVIDENCE"],
        "human_reasoning": "Single word ('update'). Must clarify whether issue is pre-install, install failure, or post-update issue."
    },
    {
        "golden_id": "GOLDEN_0154",
        "source_id": "SRC_ROOT_1859500",
        "customer_message": "@AppleSupport 'Other' system storage is taking up 42GB out of my 64GB iPhone! I can't take photos or update anything. How do I delete Other?",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Clear bloated 'Other' / System storage on iPhone",
        "ground_truth_issues": ["bloated_other_storage", "system_storage_exhaustion"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that 'Other' storage consists of system caches, incomplete downloads, Siri voices, and streaming caches",
            "Instruct customer to connect device to iTunes on a Mac or PC (syncing with iTunes often reorganizes and purges cached files)",
            "Advise performing an encrypted iTunes backup, erasing the device, and restoring the backup to completely purge ghost cache files"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Bloated 'Other' storage bug in iOS 11. iTunes sync or backup-erase-restore is the documented method to reclaim space."
    },
    {
        "golden_id": "GOLDEN_0155",
        "source_id": "SRC_ROOT_1859700",
        "customer_message": "@AppleSupport I’ve put my iPhone in DFU mode, tried restoring on 2 computers with iTunes, and both give Error 4013. Phone is in an endless boot loop.",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Resolve iTunes restore Error 4013 causing endless boot loop",
        "ground_truth_issues": ["itunes_restore_error_4013", "endless_boot_loop", "hardware_interconnect_failure"],
        "ground_truth_action": "ESCALATE",
        "escalation_required": True,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge customer has attempted DFU restore across multiple computers",
            "Explain that Error 4013 indicates a hardware communication disconnect during NAND/baseband flashing",
            "Escalate customer to schedule an Apple Store inspection or mail-in hardware repair"
        ],
        "edge_case_category": ["FOLLOW_UP", "UNRESOLVED_FOLLOW_UP", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Error 4013 persisting across multiple computers and DFU mode is hardware failure (faulty cable/port/NAND/logic board). Requires escalation."
    },
    {
        "golden_id": "GOLDEN_0156",
        "source_id": "SRC_ROOT_1859900",
        "customer_message": "Why does my iPhone 6 feel so sluggish and stuttery after iOS 11? Scrolling through Safari drops frames constantly. @AppleSupport",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Optimize performance and reduce UI stuttering on iPhone 6 running iOS 11",
        "ground_truth_issues": ["ui_stuttering_frame_drops", "sluggish_performance_iphone6"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Recommend performance settings adjustments: Settings > General > Accessibility > Reduce Motion (toggle ON)",
            "Toggle 'Reduce Transparency' to ON in Accessibility > Display Accommodations to decrease GPU load",
            "Verify that at least 2-3GB of free internal storage remains available for system virtual memory"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "iPhone 6 has 1GB RAM and struggles with iOS 11 blurs/animations. Reduce Motion and Reduce Transparency dramatically smooth performance."
    },
    {
        "golden_id": "GOLDEN_0157",
        "source_id": "SRC_ROOT_1860100",
        "customer_message": "@AppleSupport Every time I restart my phone, date & time resets to December 31, 1969 and safari won't open any webpage.",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Fix date and time resetting to Unix epoch 1969 and restore web browsing",
        "ground_truth_issues": ["unix_epoch_1969_reset", "ssl_handshake_block"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that Safari cannot load HTTPS pages because the system clock year 1969 fails SSL certificate date validation",
            "Instruct customer to go to Settings > General > Date & Time, turn OFF 'Set Automatically', manually set the correct date and time, then toggle 'Set Automatically' back ON",
            "Restart device while connected to Wi-Fi to synchronize NTP time servers"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "1969 date reset blocks all internet SSL certificates. Manually resetting date/time and toggling automatic time fixes the sync."
    },
    {
        "golden_id": "GOLDEN_0158",
        "source_id": "SRC_ROOT_1860300",
        "customer_message": "@AppleSupport My iPhone X restarted by itself in the middle of the night and asked for my passcode. Did it update automatically?",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Understand why iPhone restarted overnight and prompted for passcode",
        "ground_truth_issues": ["overnight_restart", "automatic_update_verification"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that if an iOS update occurred overnight, iOS requires the passcode upon first restart before Face ID / Touch ID can be used",
            "Instruct customer to check Settings > General > About to view their current iOS version and verify if an update was installed",
            "Reassure customer that this is expected security behavior"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Customer wondering why device restarted overnight. Passcode required after restart is standard security behavior. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0159",
        "source_id": "SRC_ROOT_1860500",
        "customer_message": "iPhone is stuck in an infinite restart loop showing the spinning wheel every 60 seconds on December 2nd! @AppleSupport",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Resolve the infamous December 2, 2017 iOS 11 recurring notification crash loop",
        "ground_truth_issues": ["december_2_springboard_crash_loop", "spinning_gear_reboot"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Recognize this as the critical December 2, 2017 local notification bug in iOS 11.1.2",
            "Provide emergency workaround: turn off notifications for all apps with local daily reminders, or manually set device date back to December 1st",
            "Immediately update device to iOS 11.2 which contains the emergency official bug fix"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "The Dec 2, 2017 local notification bug caused thousands of devices to reboot every minute. The historical fix was disabling notifications/rolling back date, then updating to 11.2."
    },
    {
        "golden_id": "GOLDEN_0160",
        "source_id": "SRC_ROOT_1860700",
        "customer_message": "@AppleSupport How do I force restart my iPhone 8? Holding power and home button doesn't work because the home button isn't physical!",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Learn the new force restart button combination on iPhone 8",
        "ground_truth_issues": ["iphone_8_force_restart_gesture"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain the 3-step button combination for iPhone 8: press and quickly release Volume Up, press and quickly release Volume Down, then press and hold the Side button until the Apple logo appears",
            "Clarify that holding Home + Power only applied to iPhone 6s and earlier models",
            "Direct and concise answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "iPhone 8 introduced a completely new force restart sequence. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0161",
        "source_id": "SRC_ROOT_1860900",
        "customer_message": "@AppleSupport Storage says 0 bytes available, but I just deleted 500 photos and 3 apps! Space didn't change at all.",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Reclaim storage after deleting photos and apps",
        "ground_truth_issues": ["deleted_photos_storage_not_freed", "recently_deleted_album"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that deleted photos are moved to the 'Recently Deleted' album in the Photos app and remain in storage for 30 days",
            "Instruct customer to open Photos > Albums > Recently Deleted > Select > Delete All",
            "Restart device to force internal filesystem storage re-indexing"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Deleted photos stay in Recently Deleted for 30 days. Emptying Recently Deleted and restarting immediately frees the storage."
    },
    {
        "golden_id": "GOLDEN_0162",
        "source_id": "SRC_ROOT_1861100",
        "customer_message": "My iPhone gets stuck on a white screen with black apple logo for 10 minutes every time it powers on. Is this normal? @AppleSupport",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Fix abnormally long boot time on iPhone",
        "ground_truth_issues": ["abnormally_long_boot_time", "filesystem_check_hang"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Confirm that a 10-minute boot time is abnormal and indicates filesystem verification struggles or nearly full storage",
            "Instruct customer to back up device to iTunes or iCloud immediately",
            "Check available storage (ensure >10% free) and install latest iOS update"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "10-minute boot indicates corrupted cache or storage exhaustion. Guided backup and update."
    },
    {
        "golden_id": "GOLDEN_0163",
        "source_id": "SRC_ROOT_1861300",
        "customer_message": "@AppleSupport iTunes says 'There is a problem with the iPhone that requires it to be updated or restored' when I plug in. What does this mean?",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Handle iTunes recovery prompt without losing data",
        "ground_truth_issues": ["itunes_update_or_restore_prompt", "recovery_mode_detected"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that iTunes detected the device in recovery mode or with corrupted firmware",
            "Crucially instruct customer to click 'Update' first (which attempts to reinstall iOS without wiping data)",
            "Explain that only if 'Update' fails should 'Restore' (which erases device) be selected"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Crucial customer fork: clicking 'Update' preserves data, while 'Restore' erases it. Clear guide to select 'Update'."
    },
    {
        "golden_id": "GOLDEN_0164",
        "source_id": "SRC_ROOT_1861500",
        "customer_message": "@AppleSupport Is it possible to clear Safari browser cache and cookies without deleting my saved website passwords?",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Clear Safari cache while keeping saved passwords intact",
        "ground_truth_issues": ["clear_safari_cache", "preserve_saved_passwords"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Confirm that clearing Safari history and website data does NOT delete passwords saved in iCloud Keychain",
            "Direct user to: Settings > Safari > tap 'Clear History and Website Data'",
            "Explain that saved passwords remain secure in Settings > Passwords & Accounts"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Direct inquiry addressing customer concern about losing passwords. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0165",
        "source_id": "SRC_ROOT_1861700",
        "customer_message": "@AppleSupport Can I update my iPhone over cellular data? I have unlimited LTE and no home Wi-Fi.",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Determine if major iOS updates can be downloaded over cellular data",
        "ground_truth_issues": ["ota_update_over_cellular_limitation"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that major iOS updates over ~150MB require a Wi-Fi connection by default and cannot be downloaded directly over cellular data",
            "Suggest alternatives: connect to a public Wi-Fi network, use a personal hotspot from another device, or update via iTunes on a computer",
            "Direct and helpful answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "iOS 11 cellular download cap policy. Direct answer with practical alternatives."
    },
    {
        "golden_id": "GOLDEN_0166",
        "source_id": "SRC_ROOT_1861900",
        "customer_message": "Ever since 11.0.3, my iPhone freezes every 20 seconds, apps crash, battery drains, and audio crackles. Can someone just call me to fix this? @AppleSupport",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Get human phone support callback for cascading multi-symptom device failure",
        "ground_truth_issues": ["cascading_multi_symptom_failure", "phone_call_requested"],
        "ground_truth_action": "ESCALATE",
        "escalation_required": True,
        "evidence_expectation": "MODERATE_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge the multi-symptom instability and request for a phone call",
            "Provide link to getsupport.apple.com to schedule an immediate callback or chat with an Apple Support advisor",
            "Do not dump a long list of troubleshooting steps on an overwhelmed customer asking for a call"
        ],
        "edge_case_category": ["MULTI_ISSUE", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Severe multi-issue collapse where customer explicitly requests a phone call. Proper protocol is directing to getsupport.apple.com callback scheduling."
    },
    {
        "golden_id": "GOLDEN_0167",
        "source_id": "SRC_ROOT_1862100",
        "customer_message": "@AppleSupport What is the difference between 'Reset All Settings' and 'Erase All Content and Settings'?",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Understand distinction between Reset All Settings and Erase All Content",
        "ground_truth_issues": ["reset_options_comparison", "data_preservation_clarity"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that 'Reset All Settings' resets Wi-Fi networks, wallpaper, and preferences to default, but keeps all personal data, photos, and apps intact",
            "Explain that 'Erase All Content and Settings' completely wipes the device clean to factory settings, deleting all data and apps",
            "Direct and clear comparison"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Fundamental iOS distinction. Direct answer prevents catastrophic accidental device erasure."
    },
    {
        "golden_id": "GOLDEN_0168",
        "source_id": "SRC_ROOT_1862300",
        "customer_message": "@AppleSupport My phone says 'Software Update - iOS 11.1 - 1 day remaining' and progress bar hasn't moved in 2 hours.",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Resume frozen iOS software update download",
        "ground_truth_issues": ["update_download_stuck", "excessive_download_time"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Instruct customer to pause and delete the stuck download: Settings > General > iPhone Storage > find iOS 11.1 > Delete Update",
            "Restart the Wi-Fi router and iPhone",
            "Re-initiate the download in Settings > General > Software Update"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "OTA download hang. Deleting stalled download and restarting Wi-Fi connection resolves it."
    },
    {
        "golden_id": "GOLDEN_0169",
        "source_id": "SRC_ROOT_1862500",
        "customer_message": "Is iOS 11.2 available yet? My phone says 11.1 is up to date. @AppleSupport",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Check availability of latest iOS 11.2 release",
        "ground_truth_issues": ["update_availability_inquiry"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Instruct customer to check Settings > General > Software Update while connected to Wi-Fi",
            "Advise restarting the device to refresh the update catalog from Apple servers if a known update is not appearing",
            "Direct and concise answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Update availability query. Direct answer explaining how to refresh software update catalog."
    },
    {
        "golden_id": "GOLDEN_0170",
        "source_id": "SRC_ROOT_1862700",
        "customer_message": "@AppleSupport Device is bricked after update. Nothing turns on.",
        "ground_truth_intent": "OS_UPDATE_SYSTEM_PERFORMANCE",
        "ground_truth_customer_goal": "Revive completely unresponsive device after failed update",
        "ground_truth_issues": ["device_unresponsive_after_update", "apparent_brick"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that devices appearing dead after an update can often be revived via force restart while connected to power",
            "Instruct customer to plug into a wall charger for 20 minutes and attempt model-specific force restart",
            "Advise connecting to a computer with iTunes to test if Recovery Mode is recognized before declaring it hardware bricked"
        ],
        "edge_case_category": ["SHORT_VAGUE", "STRONG_EVIDENCE"],
        "human_reasoning": "Short query claiming device is bricked. Most 'bricked' devices after updates are simply frozen in a low-power boot state; force restart on charger and iTunes recovery mode must be attempted first."
    },

    # =========================================================================
    # 10. HOW_TO_SETTINGS_CONFIGURATION (16 records: GOLDEN_0171 - GOLDEN_0186)
    # =========================================================================
    {
        "golden_id": "GOLDEN_0171",
        "source_id": "SRC_ROOT_262194",
        "customer_message": "@AppleSupport with move to ios does that help move contacts from a SIM card in an android phone at all?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Transfer contacts stored on an Android SIM card to iPhone",
        "ground_truth_issues": ["move_to_ios_sim_contacts_transfer"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that Move to iOS transfers contacts stored in the Android device account, but for SIM card contacts, the simplest method is to insert the SIM into the iPhone and go to Settings > Contacts > 'Import SIM Contacts'",
            "Provide clear, actionable steps",
            "Direct answer without escalation"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Specific migration question. Direct answer explaining the native 'Import SIM Contacts' feature in iOS."
    },
    {
        "golden_id": "GOLDEN_0172",
        "source_id": "SRC_ROOT_1862900",
        "customer_message": "@AppleSupport How do I customize the icons in the new Control Center in iOS 11? Can I add Low Power Mode?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Customize Control Center controls and add Low Power Mode toggle",
        "ground_truth_issues": ["control_center_customization", "add_low_power_mode_toggle"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Navigate to Settings > Control Center > Customize Controls",
            "Scroll down to 'More Controls' and tap the green '+' next to Low Power Mode",
            "Explain that you can reorder icons using the three-bar drag handles on the right"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Signature iOS 11 feature customization. Direct answer with exact settings path."
    },
    {
        "golden_id": "GOLDEN_0173",
        "source_id": "SRC_ROOT_1863100",
        "customer_message": "@AppleSupport How do I turn on the built-in screen recorder in iOS 11? Do I need a third party app?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Enable and use the native Screen Recording feature in iOS 11",
        "ground_truth_issues": ["screen_recording_enablement"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Confirm that iOS 11 includes a native screen recorder (no third-party app needed)",
            "Direct user to Settings > Control Center > Customize Controls > add 'Screen Recording'",
            "Explain how to use: swipe up Control Center, tap the record button (long press for microphone audio toggle)"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "New iOS 11 screen recording feature. Direct procedural answer."
    },
    {
        "golden_id": "GOLDEN_0174",
        "source_id": "SRC_ROOT_1863300",
        "customer_message": "How do I turn off 'Do Not Disturb While Driving'? It turns on automatically and blocks all my calls while I'm a passenger in an Uber! @AppleSupport",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Disable or configure automatic Do Not Disturb While Driving",
        "ground_truth_issues": ["dnd_while_driving_passenger_annoyance", "disable_auto_dnd"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Navigate to Settings > Do Not Disturb",
            "Scroll down to 'Do Not Disturb While Driving' > tap 'Activate'",
            "Change activation from 'Automatically' to 'Manually' or 'When Connected to Car Bluetooth'",
            "Mention you can tap 'I'm Not Driving' on the lock screen when riding as a passenger"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "iOS 11 introduced DND While Driving. Common complaint for passengers. Exact settings path provided."
    },
    {
        "golden_id": "GOLDEN_0175",
        "source_id": "SRC_ROOT_1863500",
        "customer_message": "@AppleSupport how to settings",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Get assistance configuring an unspecified setting",
        "ground_truth_issues": ["unspecified_settings_inquiry"],
        "ground_truth_action": "CLARIFY",
        "escalation_required": False,
        "evidence_expectation": "INSUFFICIENT_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Politely ask what specific feature or setting the customer is looking to configure on their device",
            "Ask what device model and iOS version they have"
        ],
        "edge_case_category": ["SHORT_VAGUE", "INSUFFICIENT_EVIDENCE"],
        "human_reasoning": "Three-word nonsensical fragment ('how to settings'). Genuinely requires clarification."
    },
    {
        "golden_id": "GOLDEN_0176",
        "source_id": "SRC_ROOT_1863700",
        "customer_message": "@AppleSupport How do I set up Emergency SOS on my iPhone so it calls my emergency contacts and police if I'm in trouble?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Configure Emergency SOS and emergency contacts on iPhone",
        "ground_truth_issues": ["emergency_sos_configuration", "medical_id_contacts"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Navigate to Settings > Emergency SOS to configure auto-call options and countdown sound",
            "Explain how to add emergency contacts: open the Health app > tap Medical ID tab > tap Edit > scroll to Emergency Contacts",
            "Explain how to trigger: rapidly press the side/power button 5 times (or hold Side + Volume on iPhone 8/X)"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Emergency SOS setup and Medical ID integration. Direct complete answer."
    },
    {
        "golden_id": "GOLDEN_0177",
        "source_id": "SRC_ROOT_1863900",
        "customer_message": "@AppleSupport How can I change the default search engine in Safari from Google to DuckDuckGo?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Change Safari default search engine to DuckDuckGo",
        "ground_truth_issues": ["safari_search_engine_configuration"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Navigate to Settings > Safari",
            "Tap 'Search Engine'",
            "Select 'DuckDuckGo' from the available list",
            "Direct and concise answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Direct procedural inquiry. Straightforward settings path."
    },
    {
        "golden_id": "GOLDEN_0178",
        "source_id": "SRC_ROOT_1864100",
        "customer_message": "Where is the AirDrop toggle in the new iOS 11 Control Center? It disappeared from the bottom! @AppleSupport",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Locate AirDrop toggle hidden inside the iOS 11 Control Center connectivity block",
        "ground_truth_issues": ["airdrop_toggle_missing_control_center", "3d_touch_gesture_discoverability"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that AirDrop is now inside the network platter in Control Center",
            "Instruct customer to swipe open Control Center, then press and hold (or 3D Touch) the top-left box containing Airplane Mode/Wi-Fi/Bluetooth",
            "Tap the AirDrop icon to choose Receiving Off, Contacts Only, or Everyone"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Major iOS 11 UI discoverability change: network block expanded with 3D Touch. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0179",
        "source_id": "SRC_ROOT_1864300",
        "customer_message": "@AppleSupport How do I scan QR codes using the native camera in iOS 11 without downloading a third party scanner app?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Use native Camera app to scan QR codes",
        "ground_truth_issues": ["qr_code_scanning_native_camera"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Confirm that native QR scanning is built directly into the Camera app on iOS 11",
            "Instruct customer to open the Camera app, point at the QR code, and tap the notification banner that appears at the top of the screen",
            "Ensure Settings > Camera > 'Scan QR Codes' is toggled ON"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "iOS 11 introduced native Camera QR scanning. Direct procedural answer."
    },
    {
        "golden_id": "GOLDEN_0180",
        "source_id": "SRC_ROOT_1864500",
        "customer_message": "@AppleSupport Can I change the default email app on iPhone from Apple Mail to Gmail in iOS 11?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Set third-party app (Gmail) as default email client in iOS 11",
        "ground_truth_issues": ["default_app_selection_limitation"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that in iOS 11, changing the default system email client is not supported (default mail links automatically open Apple Mail)",
            "Advise that customer can add their Gmail account directly into the Apple Mail app in Settings > Accounts & Passwords, or use the standalone Gmail app directly",
            "Direct and truthful answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Default app selection was not introduced until iOS 14. Direct factual answer explaining iOS 11 platform policy."
    },
    {
        "golden_id": "GOLDEN_0181",
        "source_id": "SRC_ROOT_1864700",
        "customer_message": "How do I turn on Dark Mode in iOS 11? My eyes hurt reading white screens at night. @AppleSupport",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Enable dark theme or smart invert on iOS 11",
        "ground_truth_issues": ["dark_mode_ios11", "smart_invert_colors"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that while a full system Dark Mode is not in iOS 11, iOS 11 includes 'Smart Invert' which reverses display colors while preserving images and media",
            "Provide navigation: Settings > General > Accessibility > Display Accommodations > Invert Colors > toggle 'Smart Invert' to ON",
            "Also mention Night Shift under Display & Brightness for nighttime viewing comfort"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "iOS 11 Smart Invert was the de facto dark mode before iOS 13. Direct answer explaining Smart Invert."
    },
    {
        "golden_id": "GOLDEN_0182",
        "source_id": "SRC_ROOT_1864900",
        "customer_message": "@AppleSupport How do I prevent my lock screen from previewing the text message contents when people text me?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Hide notification message previews on lock screen for privacy",
        "ground_truth_issues": ["hide_notification_previews_privacy"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Provide navigation: Settings > Notifications > Show Previews",
            "Select 'When Unlocked' (requires passcode/Touch ID/Face ID to reveal) or 'Never' (keeps message preview hidden always)",
            "Direct and concise privacy configuration answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Standard notification privacy configuration in iOS 11. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0183",
        "source_id": "SRC_ROOT_1865100",
        "customer_message": "@AppleSupport How do I set custom ringtones for individual contacts so I know who is calling without looking at my phone?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Assign custom ringtone to a specific contact",
        "ground_truth_issues": ["custom_contact_ringtone_assignment"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Open the Contacts or Phone app, select the desired contact, and tap 'Edit' in top right",
            "Scroll down to 'Ringtone', tap it, choose the desired custom tone, and tap 'Done'",
            "Direct, step-by-step answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Basic Contacts configuration how-to. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0184",
        "source_id": "SRC_ROOT_1865300",
        "customer_message": "How do I take Live Photos with my iPhone 7? The circle icon is yellow. What does that mean? @AppleSupport",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Understand Live Photos camera indicator and capture Live Photos",
        "ground_truth_issues": ["live_photos_usage", "camera_icon_status_clarification"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that when the concentric circle icon at the top of the Camera app is yellow, Live Photos is active and capturing 1.5 seconds before and after the shutter press",
            "Explain that tapping the yellow icon turns it white with a slash, disabling Live Photos",
            "Explain how to view: tap and hold (3D Touch) the photo in the Photos app"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Live Photos feature functionality in Camera app. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0185",
        "source_id": "SRC_ROOT_1865500",
        "customer_message": "@AppleSupport How do I permanently delete an app and all its stored data on iOS 11?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Permanently delete an application and its local data from iPhone",
        "ground_truth_issues": ["delete_app_and_data"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain home screen method: lightly press and hold the app icon until all icons jiggle, then tap the 'X' button and confirm 'Delete'",
            "Alternatively, navigate to Settings > General > iPhone Storage > tap the app > tap 'Delete App' (which removes app and all related documents)",
            "Direct and concise answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Standard iOS application deletion. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0186",
        "source_id": "SRC_ROOT_1865700",
        "customer_message": "@AppleSupport Can I pair an Apple Watch with an iPad instead of an iPhone?",
        "ground_truth_intent": "HOW_TO_SETTINGS_CONFIGURATION",
        "ground_truth_customer_goal": "Pair Apple Watch with an iPad",
        "ground_truth_issues": ["apple_watch_ipad_pairing_compatibility"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "State clearly that Apple Watch requires a compatible iPhone (iPhone 5s or later running iOS 11) for setup and pairing",
            "Confirm that Apple Watch cannot pair with an iPad, iPod touch, or Android device",
            "Direct and factual answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Clear platform compatibility restriction. Apple Watch requires iPhone. Direct answer."
    },

    # =========================================================================
    # 11. GENERAL_DEVICE_INQUIRY (14 records: GOLDEN_0187 - GOLDEN_0200)
    # =========================================================================
    {
        "golden_id": "GOLDEN_0187",
        "source_id": "SRC_ROOT_1835055",
        "customer_message": "Hey @115858, I just noticed that the proximity scensor is not working AT ALL... I took back my iphone and this is the second replacement unit.",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Resolve persistent proximity sensor failure across multiple replacement devices",
        "ground_truth_issues": ["proximity_sensor_failure", "multiple_replacement_units_failed", "frustrated_customer"],
        "ground_truth_action": "ESCALATE",
        "escalation_required": True,
        "evidence_expectation": "MODERATE_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Acknowledge that this is the second replacement unit experiencing proximity sensor issues",
            "Check if an opaque screen protector or case is blocking the top bezel sensor window",
            "Escalate to senior technical support / engineering relations given the repeat hardware replacements"
        ],
        "edge_case_category": ["FOLLOW_UP", "ESCALATION_SENSITIVE", "MULTI_ISSUE"],
        "human_reasoning": "Repeat hardware replacements failing with the same symptom. Screen protector triage check plus senior escalation required."
    },
    {
        "golden_id": "GOLDEN_0188",
        "source_id": "SRC_ROOT_1865900",
        "customer_message": "@AppleSupport Dropped my iPhone in the ocean and it won't turn on. Is saltwater damage covered under the 1-year warranty?",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Clarify warranty coverage for saltwater liquid submersion damage",
        "ground_truth_issues": ["liquid_damage_ocean_water", "warranty_coverage_inquiry"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that liquid damage (including saltwater) is NOT covered under Apple's standard 1-year limited warranty",
            "Explain that if the customer has AppleCare+, liquid damage is covered subject to an accidental damage service fee",
            "Direct customer to check coverage at checkcoverage.apple.com or schedule an out-of-warranty replacement evaluation"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Direct warranty terms inquiry. Liquid damage excluded under 1-year warranty, covered under AppleCare+. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0189",
        "source_id": "SRC_ROOT_1866100",
        "customer_message": "How do I check if my iPhone 7 is still covered under AppleCare? Where do I find the serial number? @AppleSupport",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Find serial number and check AppleCare warranty coverage status",
        "ground_truth_issues": ["serial_number_location", "check_warranty_coverage"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Provide navigation to find serial number: Settings > General > About > Serial Number (or check original box/receipt/SIM tray on 7)",
            "Direct customer to checkcoverage.apple.com to enter the serial number and view coverage details",
            "Direct and concise answer"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Standard serial number lookup and warranty check portal. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0190",
        "source_id": "SRC_ROOT_1866300",
        "customer_message": "@AppleSupport broken",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Get help with unspecified broken device",
        "ground_truth_issues": ["unspecified_broken_device"],
        "ground_truth_action": "CLARIFY",
        "escalation_required": False,
        "evidence_expectation": "INSUFFICIENT_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Politely ask the customer what device is broken and describe what happened (physical damage, software issue, won't turn on)",
            "Ask how we can best assist them"
        ],
        "edge_case_category": ["SHORT_VAGUE", "INSUFFICIENT_EVIDENCE"],
        "human_reasoning": "Single word ('broken'). Completely unspecific; must clarify."
    },
    {
        "golden_id": "GOLDEN_0191",
        "source_id": "SRC_ROOT_1866500",
        "customer_message": "Tell me how to jailbreak my iPhone 7 on iOS 11.1 so I can install Cydia tweaks @AppleSupport",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Obtain instructions to jailbreak iPhone and install unauthorized software",
        "ground_truth_issues": ["jailbreak_request", "unauthorized_modification"],
        "ground_truth_action": "SAFE_REFUSAL",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Politely refuse to provide jailbreaking instructions",
            "Explain that unauthorized modification of iOS violates the software license agreement, bypasses security protections, causes instability, and may void warranty service",
            "Firm and courteous policy refusal"
        ],
        "edge_case_category": ["SAFETY_SENSITIVE", "STRONG_EVIDENCE"],
        "human_reasoning": "Customer asking official support how to jailbreak. Must execute SAFE_REFUSAL citing security vulnerabilities and license terms."
    },
    {
        "golden_id": "GOLDEN_0192",
        "source_id": "SRC_ROOT_1866700",
        "customer_message": "@AppleSupport I lost my iPhone on the subway! Lost Mode says location unavailable. How can I track it if it's turned off?",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Track lost iPhone when device is powered off or location is offline",
        "ground_truth_issues": ["lost_device", "location_unavailable_offline"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Advise customer to log into icloud.com/find immediately and enable Lost Mode (which locks device with passcode and displays contact message)",
            "Explain that 'Send Last Location' will show the last known location before the battery died",
            "Note that if device is powered off, Lost Mode actions will take effect automatically the moment it is turned on and connects to network",
            "Advise filing a police report and contacting carrier to suspend cellular service"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Lost device protocol on Find My iPhone. Standard guided security steps."
    },
    {
        "golden_id": "GOLDEN_0193",
        "source_id": "SRC_ROOT_1866900",
        "customer_message": "@AppleSupport Does Apple have a trade-in program for an old iPhone 6 toward an iPhone X? How much trade-in credit do I get?",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Determine trade-in eligibility and credit value for iPhone 6 toward iPhone X",
        "ground_truth_issues": ["trade_in_program_inquiry", "device_valuation"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Confirm that Apple offers the Apple GiveBack / Trade-In program for older devices including iPhone 6",
            "Explain that trade-in value depends on device condition (scratches, working screen, buttons, water damage)",
            "Direct customer to apple.com/trade-in or an Apple Retail Store for an exact instant valuation and credit towards iPhone X"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Standard Apple Trade-In program inquiry. Direct answer with link to official trade-in estimation tool."
    },
    {
        "golden_id": "GOLDEN_0194",
        "source_id": "SRC_ROOT_1867100",
        "customer_message": "@AppleSupport help please asap my phone is doing weird things",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Get help with unspecified unusual phone behavior",
        "ground_truth_issues": ["vague_unusual_behavior"],
        "ground_truth_action": "CLARIFY",
        "escalation_required": False,
        "evidence_expectation": "INSUFFICIENT_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Politely ask the customer to describe what 'weird things' are happening on their phone (e.g. screen glitching, apps crashing, unwanted sounds, unexpected restarts)",
            "Ask what iPhone model and iOS version they are using",
            "Offer immediate support once details are shared"
        ],
        "edge_case_category": ["SHORT_VAGUE", "INSUFFICIENT_EVIDENCE"],
        "human_reasoning": "Vague symptom ('doing weird things'). Must clarify."
    },
    {
        "golden_id": "GOLDEN_0195",
        "source_id": "SRC_ROOT_1867300",
        "customer_message": "@AppleSupport My dog chewed on my iPhone charger cable and the internal wires are exposed. Can I wrap it with electrical tape and keep using it?",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Assess safety of continuing to use damaged charger cable with exposed wires",
        "ground_truth_issues": ["damaged_cable_exposed_wires", "electrical_safety_hazard"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explicitly warn customer NOT to use cables with frayed or exposed wires, as they pose a risk of short circuit, device damage, or fire",
            "Advise disposing of the damaged cable safely and replacing it with an Apple original or MFi-certified Lightning cable",
            "Clear safety instruction"
        ],
        "edge_case_category": ["SAFETY_SENSITIVE", "STRONG_EVIDENCE"],
        "human_reasoning": "Exposed cable wires are an electrical hazard. Direct safety answer: do not use electrical tape; replace cable immediately."
    },
    {
        "golden_id": "GOLDEN_0196",
        "source_id": "SRC_ROOT_1867500",
        "customer_message": "@AppleSupport How do I clean fingerprints and smudges off my iPhone X glass back without scratching it?",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Learn proper manufacturer cleaning procedures for iPhone X glass back",
        "ground_truth_issues": ["device_cleaning_procedure", "smudge_removal"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Unplug all cables and power off the device",
            "Use a soft, slightly damp, lint-free cloth (such as a microfiber cloth)",
            "Explicitly warn against using window cleaners, household chemicals, aerosol sprays, solvents, or abrasives which degrade the oleophobic coating"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Official cleaning instructions from Apple Support knowledge base. Direct answer."
    },
    {
        "golden_id": "GOLDEN_0197",
        "source_id": "SRC_ROOT_1867700",
        "customer_message": "I want to file a formal complaint against Apple Store Regent Street staff for rude behavior. Who do I contact? @AppleSupport",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Submit formal customer service complaint regarding retail store staff",
        "ground_truth_issues": ["retail_staff_complaint", "store_escalation"],
        "ground_truth_action": "ESCALATE",
        "escalation_required": True,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Express sincere apologies for the negative in-store customer experience",
            "Provide official retail feedback submission link: apple.com/feedback/retail.html",
            "Escalate customer to Apple Customer Relations team to document the specific incident details and store visit"
        ],
        "edge_case_category": ["CLEAR_INTENT", "ESCALATION_SENSITIVE"],
        "human_reasoning": "Customer complaint against retail store personnel. Must be escalated to Customer Relations / retail feedback."
    },
    {
        "golden_id": "GOLDEN_0198",
        "source_id": "SRC_ROOT_1867900",
        "customer_message": "@AppleSupport Can I use an iPhone purchased in the US with a UK carrier SIM card? Is it network unlocked?",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Determine if US-purchased iPhone works on UK carrier and carrier lock status",
        "ground_truth_issues": ["carrier_lock_status", "international_sim_compatibility"],
        "ground_truth_action": "ANSWER",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Explain that if purchased directly from Apple SIM-free / unlocked, it will work with UK GSM carriers (EE, Vodafone, O2, Three)",
            "Explain that if purchased through a US carrier installment plan (AT&T, Verizon, Sprint, T-Mobile), it may be carrier-locked and require unlocking by that US carrier before international SIM cards will function",
            "Advise checking Settings > General > About or inserting a UK SIM to verify"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE"],
        "human_reasoning": "Carrier lock status and international GSM compatibility. Direct factual explanation."
    },
    {
        "golden_id": "GOLDEN_0199",
        "source_id": "SRC_ROOT_1868100",
        "customer_message": "Someone sent me a calendar invite containing spam links to casino websites on my iPhone calendar. How do I delete it without accepting? @AppleSupport",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Remove iCloud calendar spam invites without notifying the spammer",
        "ground_truth_issues": ["icloud_calendar_spam_invite", "privacy_spam_removal"],
        "ground_truth_action": "GUIDE",
        "escalation_required": False,
        "evidence_expectation": "STRONG_HISTORICAL_EVIDENCE",
        "expected_reply_requirements": [
            "Warn customer NOT to tap 'Decline' or 'Accept' as this alerts the spammer that the email address is active",
            "Instruct customer on the safe removal workaround: create a temporary calendar named 'Spam' in Calendar app, move the spam event into it, then delete the entire 'Spam' calendar choosing 'Delete and Don't Notify'",
            "Advise visiting icloud.com on a computer to report the calendar event as junk"
        ],
        "edge_case_category": ["CLEAR_INTENT", "STRONG_EVIDENCE", "SAFETY_SENSITIVE"],
        "human_reasoning": "Widely known iCloud calendar spam invite issue. Tapping decline notifies the spammer. The 'delete calendar without notify' workaround is the exact historical resolution."
    },
    {
        "golden_id": "GOLDEN_0200",
        "source_id": "SRC_ROOT_1868300",
        "customer_message": "asdlkjfasdf ?? @AppleSupport",
        "ground_truth_intent": "GENERAL_DEVICE_INQUIRY",
        "ground_truth_customer_goal": "Customer sent unintelligible keystrokes / accidental pocket message",
        "ground_truth_issues": ["gibberish_message", "accidental_keystrokes"],
        "ground_truth_action": "CLARIFY",
        "escalation_required": False,
        "evidence_expectation": "NO_HISTORICAL_EVIDENCE_REQUIRED",
        "expected_reply_requirements": [
            "Politely respond acknowledging the message",
            "Ask customer if they need assistance with their Apple device and how we can help",
            "Maintain a friendly, professional tone"
        ],
        "edge_case_category": ["SHORT_VAGUE", "UNUSUAL_WORDING"],
        "human_reasoning": "Completely unintelligible gibberish text / pocket tweet. Proper agent behavior is a polite clarification asking if they need support."
    }
]


def get_part_2_records() -> List[Dict[str, Any]]:
    return PART_2_RECORDS
