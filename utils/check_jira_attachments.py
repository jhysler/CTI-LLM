#!/usr/bin/env python3
"""Verify every attachment referenced in issues/*.json exists on disk under attachment/<id>/."""
import json
import sys
from pathlib import Path

jira_dir = Path(sys.argv[1])
missing = []

for issue_path in (jira_dir / "issues").glob("*.json"):
    issue = json.loads(issue_path.read_text())
    for att in issue["fields"].get("attachment") or []:
        att_dir = jira_dir / "attachment" / att["id"]
        # On-disk filenames use '+' for spaces (form-encoded upload name);
        # the JSON 'filename' field has literal spaces. Check both.
        if not (att_dir / att["filename"]).exists() and not (att_dir / att["filename"].replace(" ", "+")).exists():
            missing.append(f"{issue_path.stem}: {att['id']}/{att['filename']}")

if missing:
    print(f"MISSING {len(missing)} attachments:")
    print("\n".join(missing))
else:
    print("All referenced attachments present.")
