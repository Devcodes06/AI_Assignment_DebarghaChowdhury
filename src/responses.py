from pathlib import Path

labels_path = Path(__file__).parent.parent / "model" / "labels.txt"
with open(labels_path, "r", encoding="utf-8") as _f:
    _CATEGORIES = [line.strip() for line in _f if line.strip()]

SUGGESTED_REPLIES = {
    "Login Issue": (
        "Please use the 'Forgot Password' option on the login page to reset your password. "
        "If you still cannot sign in after resetting, contact the support team with your "
        "registered email address so we can unlock your account."
    ),
    "Application Error": (
        "Thank you for reporting this error. Our engineering team is investigating the issue. "
        "Please try refreshing the page or clearing your browser cache, and let us know if the "
        "problem continues after that."
    ),
    "Report": (
        "We can help you generate the report you need. Please let us know which report type and "
        "date range you require, and we will send it to your registered email address."
    ),
    "Account Update": (
        "We can update your account details for you. Please reply with the specific information "
        "you would like changed, and we will confirm once the update has been made."
    ),
    "Performance": (
        "We are sorry the application is running slowly for you. Please try clearing your cache "
        "and restarting the app. If the issue persists, let us know what you were doing when the "
        "slowdown occurred so we can investigate."
    ),
}


def get_suggested_reply(category: str) -> str:
    if category in SUGGESTED_REPLIES:
        return SUGGESTED_REPLIES[category]
    return (
        "Thank you for contacting support. We have received your ticket and will get back to you "
        "shortly. Please let us know if you have any additional details to share."
    )