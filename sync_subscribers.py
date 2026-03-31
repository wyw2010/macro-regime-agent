"""Sync Formspree submissions into subscribers.json."""
import json
import os
import urllib.request

SUBSCRIBERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subscribers.json")
FORMSPREE_ENDPOINT = "https://formspree.io/f/xeepkpvp"


def load_subscribers() -> list[str]:
    if not os.path.exists(SUBSCRIBERS_PATH):
        return []
    with open(SUBSCRIBERS_PATH) as f:
        data = json.load(f)
    return [e.strip().lower() for e in data.get("subscribers", []) if e.strip()]


def fetch_formspree_submissions(api_key: str) -> list[str]:
    """Fetch all submissions from Formspree API."""
    url = f"{FORMSPREE_ENDPOINT}/submissions"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    emails = []
    for sub in data.get("submissions", []):
        email = sub.get("email", "").strip().lower()
        if email and "@" in email:
            emails.append(email)
    return emails


def sync():
    api_key = os.getenv("FORMSPREE_API_KEY", "").strip()
    if not api_key:
        print("FORMSPREE_API_KEY not set, skipping sync.")
        return False

    existing = load_subscribers()
    existing_set = set(existing)

    try:
        new_emails = fetch_formspree_submissions(api_key)
    except Exception as e:
        print(f"WARNING: Could not fetch Formspree submissions: {e}")
        return False

    added = []
    for email in new_emails:
        if email not in existing_set:
            existing.append(email)
            existing_set.add(email)
            added.append(email)

    if added:
        with open(SUBSCRIBERS_PATH, "w") as f:
            json.dump({"subscribers": existing}, f, indent=2)
            f.write("\n")
        print(f"Added {len(added)} new subscriber(s): {', '.join(added)}")
        return True
    else:
        print("No new subscribers to add.")
        return False


if __name__ == "__main__":
    sync()
