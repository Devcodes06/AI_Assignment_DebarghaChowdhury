import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

CATEGORIES = ["Login Issue", "Application Error", "Report", "Account Update", "Performance"]

OPENERS = [
    "Hi team,", "Hello,", "Urgent:", "Please help.", "Good morning,", "",
    "I need help,", "Quick question:", "Hey,", "Hi there,", "Hello team,", "Dear support,"
]

CLOSERS = [
    "Please check.", "This is urgent.", "Thanks.", "Waiting for your reply.",
    "It started yesterday.", "", "Regards,", "Appreciate your help."
]

CORE_PHRASES = {
    "Login Issue": [
        "I forgot my password and cannot login",
        "my account is locked after too many attempts",
        "the OTP is not coming to my phone",
        "password reset link is not working",
        "unable to sign in with my credentials",
        "login page keeps showing invalid username",
        "two-factor authentication code is not accepted",
        "I am locked out of my account",
        "sign in keeps failing with wrong password error",
        "cannot access my account after password change",
        "login session expires immediately after signing in",
        "the login button does nothing when I click it",
        "I keep getting account locked message",
        "password recovery email never arrives",
        "cannot log in from my mobile device",
    ],
    "Application Error": [
        "the app shows an error while saving data",
        "I get a 500 internal server error on submit",
        "the application crashes when I click save",
        "the page shows something went wrong",
        "upload fails with an unknown error",
        "the app freezes when processing my request",
        "error code 404 appears on every page",
        "the checkout process fails at payment step",
        "I receive a null pointer exception on load",
        "the form will not submit due to validation error",
        "application closes unexpectedly after update",
        "server returns a timeout error on request",
        "the sync feature is not working properly",
        "I get a database connection error on login",
        "the print function throws a runtime error",
    ],
    "Report": [
        "please help me generate my monthly sales report",
        "I need the attendance report for last quarter",
        "the export to Excel is not giving the full report",
        "need a summary report for this week",
        "report totals look different from the dashboard",
        "the PDF report download is failing",
        "I cannot schedule the weekly report",
        "report filters are not applying correctly",
        "the scheduled report email was not sent",
        "need a custom report for Q3 results",
        "report generation takes too long and times out",
        "the chart data does not match the report table",
        "I need to export the report to CSV format",
        "the report preview shows blank pages",
        "missing data in the monthly revenue report",
    ],
    "Account Update": [
        "I need to change my registered mobile number",
        "please update my email address on the account",
        "I want to correct the spelling of my name",
        "change my billing address",
        "update my profile photo and designation",
        "I need to update my shipping address",
        "please change my preferred language setting",
        "I want to update my emergency contact details",
        "change my account currency preference",
        "I need to correct my date of birth",
        "please update my job title and department",
        "I want to change my notification email",
        "update my residential address on file",
        "I need to add a secondary email address",
        "please update my marital status in the profile",
    ],
    "Performance": [
        "the application is very slow today",
        "pages take too long to load",
        "the dashboard keeps freezing for a long time",
        "it takes minutes to open the report page",
        "the system lags during peak hours",
        "search results are slow to appear",
        "the mobile app is sluggish on my device",
        "file upload is extremely slow today",
        "the homepage takes forever to render",
        "the application responds slowly after login",
        "database queries are running very slowly",
        "the video feed buffers constantly",
        "I experience high latency on the server",
        "the autocomplete feature is lagging",
        "the system hangs when processing large files",
    ],
}

AMBIGUOUS_PHRASES = {
    "Login Issue": [
        "I cannot login and the app shows an error",
        "the login page is very slow to respond",
        "my account is locked and the report will not load",
        "I forgot my password and the system is lagging",
        "login fails when the application crashes",
        "cannot sign in and pages take too long to load",
        "the login form shows an unknown error",
        "account locked after app update error",
    ],
    "Application Error": [
        "the report page crashes when I try to save",
        "application error while generating the monthly report",
        "the app shows an error during account update",
        "login error appears on the report screen",
        "upload fails when trying to update profile",
        "the system crashes on the login page",
        "error while saving the attendance report",
        "application freezes during account email update",
    ],
    "Report": [
        "I need help generating the report but the system is slow",
        "the report export fails with a login error",
        "report page takes too long to load",
        "cannot generate report due to application crash",
        "report generation error after account update",
        "the report dashboard keeps freezing",
        "report file upload fails with unknown error",
        "report not sending due to slow system",
    ],
    "Account Update": [
        "need to update my email because the login OTP goes to the old one",
        "account update fails when the app shows an error",
        "I want to change my password and login is blocked",
        "update my profile but the system is very slow",
        "cannot update account due to report error",
        "email change fails because login page crashes",
        "account update form shows application error",
        "profile update slow during peak hours",
    ],
    "Performance": [
        "the report page is very slow to open",
        "login page takes minutes to load due to lag",
        "application error causes system slowness",
        "report generation slows down the whole system",
        "account update page freezes frequently",
        "the login process is extremely slow today",
        "system lags when opening the report module",
        "performance issue during account email update",
    ],
}

TYPO_MAP = {
    "password": "pasword",
    "application": "aplication",
    "report": "reprot",
    "account": "acount",
    "email": "emai",
    "login": "logoin",
}


def add_light_noise(text):
    r = random.random()
    if r < 0.25:
        for good, bad in TYPO_MAP.items():
            if good in text:
                text = text.replace(good, bad, 1)
                break
    elif r < 0.5:
        text = "".join(c.upper() if random.random() < 0.4 else c.lower() for c in text)
    elif r < 0.75:
        text = text + "!!!"
    else:
        text = text.replace(" ", "  ", 1)
    return text


def generate_rows(n_per_category):
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_category):
            is_ambiguous = random.random() < 0.06
            if is_ambiguous:
                core = random.choice(AMBIGUOUS_PHRASES[cat])
            else:
                core = random.choice(CORE_PHRASES[cat])
            opener = random.choice(OPENERS)
            closer = random.choice(CLOSERS)
            parts = [p for p in [opener, core, closer] if p]
            description = " ".join(parts)
            rows.append({
                "ticket_description": description,
                "category": cat,
            })
    return rows


def apply_label_noise(rows, noise_rate=0.02):
    flip_count = 0
    for row in rows:
        if random.random() < noise_rate:
            wrong = random.choice([c for c in CATEGORIES if c != row["category"]])
            row["category"] = wrong
            flip_count += 1
    return flip_count


def apply_content_noise(rows, noise_rate=0.08):
    noise_count = 0
    for row in rows:
        if random.random() < noise_rate:
            row["ticket_description"] = add_light_noise(row["ticket_description"])
            noise_count += 1
    return noise_count


def add_defects(rows):
    n = len(rows)
    defect_counts = {"duplicates": 0, "empty_priority": 0, "empty_status": 0, "empty_description": 0}

    n_dup = max(1, round(n * 0.03))
    dup_indices = random.sample(range(n), n_dup)
    for idx in dup_indices:
        rows.append(dict(rows[idx]))
    defect_counts["duplicates"] = n_dup

    n_pri = max(1, round(n * 0.02))
    pri_indices = random.sample(range(n), n_pri)
    for idx in pri_indices:
        rows[idx]["priority"] = ""
    defect_counts["empty_priority"] = n_pri

    n_sta = max(1, round(n * 0.015))
    sta_indices = random.sample(range(n), n_sta)
    for idx in sta_indices:
        rows[idx]["status"] = ""
    defect_counts["empty_status"] = n_sta

    n_desc = max(1, round(n * 0.01))
    desc_indices = random.sample(range(n), n_desc)
    for idx in desc_indices:
        rows[idx]["ticket_description"] = ""
    defect_counts["empty_description"] = n_desc

    return defect_counts


def assign_priority_status(rows):
    priority_weights = {
        "Login Issue": ([0.60, 0.30, 0.10], ["High", "Medium", "Low"]),
        "Application Error": ([0.55, 0.35, 0.10], ["High", "Medium", "Low"]),
        "Performance": ([0.50, 0.35, 0.15], ["High", "Medium", "Low"]),
        "Report": ([0.15, 0.45, 0.40], ["High", "Medium", "Low"]),
        "Account Update": ([0.15, 0.40, 0.45], ["High", "Medium", "Low"]),
    }
    status_weights = ([0.35, 0.25, 0.40], ["Open", "In Progress", "Closed"])

    for row in rows:
        if "priority" not in row or row["priority"] == "":
            weights, choices = priority_weights[row["category"]]
            row["priority"] = np.random.choice(choices, p=weights)
        if "status" not in row or row["status"] == "":
            weights, choices = status_weights
            row["status"] = np.random.choice(choices, p=weights)


def main():
    project_root = Path(__file__).parent.parent
    out_path = project_root / "data" / "raw_tickets_synthetic.csv"

    n_per_category = 400
    rows = generate_rows(n_per_category)

    noise_count = apply_content_noise(rows, noise_rate=0.08)

    label_noise_count = apply_label_noise(rows, noise_rate=0.02)

    defect_counts = add_defects(rows)

    assign_priority_status(rows)

    random.shuffle(rows)

    for i, row in enumerate(rows):
        row["ticket_id"] = f"T{i+1:04d}"

    df = pd.DataFrame(rows, columns=["ticket_id", "ticket_description", "category", "priority", "status"])

    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Generated {len(df)} rows")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Content noise applied to: {noise_count} rows")
    print(f"Label noise (wrong labels): {label_noise_count} rows")
    print(f"Defects added: {defect_counts}")


if __name__ == "__main__":
    main()