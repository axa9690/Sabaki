from __future__ import annotations

from typing import Dict, Optional


def list_labels(service) -> Dict[str, str]:
    """Return mapping: label_name -> label_id"""
    resp = service.users().labels().list(userId="me").execute()
    labels = resp.get("labels", [])
    return {l["name"]: l["id"] for l in labels}


def ensure_label(service, name: str) -> str:
    """Ensure a Gmail label exists and return its label_id."""
    existing = list_labels(service)
    if name in existing:
        return existing[name]

    body = {
        "name": name,
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show",
    }
    created = service.users().labels().create(userId="me", body=body).execute()
    return created["id"]


def ensure_labels(service, names: list[str]) -> Dict[str, str]:
    """Ensure all labels exist. Return mapping name -> id."""
    out = {}
    for n in names:
        out[n] = ensure_label(service, n)
    return out


def apply_labels(service, msg_id: str, add_label_ids: list[str], remove_label_ids: Optional[list[str]] = None):
    body = {"addLabelIds": add_label_ids, "removeLabelIds": remove_label_ids or []}
    return service.users().messages().modify(userId="me", id=msg_id, body=body).execute()
