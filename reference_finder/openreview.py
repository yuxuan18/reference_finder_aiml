"""Public accepted-paper rosters for ICLR and COLM using OpenReview APIs 1/2.

API v2 acceptance: https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc
Legacy submissions require a venue-issued decision, never a review rating.
"""
import json
import re
from urllib.parse import urlencode

from .proceedings import collecting, complete, read, result
from .storage import normalize_doi

PAGE_SIZE = 100


def value(content, name, default=None):
    item = content.get(name, default)
    return item.get("value", default) if isinstance(item, dict) else item


def scope(venue, year):
    if venue not in {"iclr", "colm"}:
        raise ValueError("OpenReview collector supports ICLR and COLM")
    if venue == "colm":
        return "https://api2.openreview.net", f"colmweb.org/COLM/{year}/Conference", 2
    group = f"ICLR.cc/{year}/" + ("conference" if year == 2017 else "Conference")
    return ("https://api.openreview.net" if year <= 2023 else "https://api2.openreview.net",
            group, 1 if year <= 2023 else 2)


def acceptance(note, group, version):
    """Return True/False, or None when legacy acceptance cannot be established."""
    content = note.get("content") or {}
    if not isinstance(content, dict):
        raise ValueError("Malformed OpenReview note content")
    if note.get("ddate") or value(content, "withdrawn", False):
        return False
    venueid = value(content, "venueid")
    if venueid and venueid != group:
        return False
    if version == 2:
        return True if venueid == group else None
    # API 1 often assigns the conference venueid to rejected submissions too.
    # Only a venue-issued decision establishes acceptance for legacy notes.
    decisions = []
    details = note.get("details") or {}
    if not isinstance(details, dict) or not isinstance(details.get("directReplies", []), list):
        raise ValueError("Malformed OpenReview decision details")
    for reply in details.get("directReplies", []):
        if not isinstance(reply, dict) or not isinstance(reply.get("content", {}), dict):
            raise ValueError("Malformed OpenReview decision reply")
        invitation = reply.get("invitation", "")
        if not isinstance(invitation, str) or not invitation.startswith(group + "/"):
            continue
        decision_field = "decision"
        if group == "ICLR.cc/2019/Conference" and invitation.endswith("/Meta_Review"):
            # This edition published final decisions as AC recommendations.
            # Its 502 accepted IDs match the independent DBLP main roster.
            if not any(re.fullmatch(re.escape(group) + r"/Paper\d+/Area_Chair\d*", signature)
                       for signature in reply.get("signatures", []) if isinstance(signature, str)):
                continue
            decision_field = "recommendation"
        elif not re.search(r"/(?:decision|acceptance|acceptance_decision)$", invitation, re.I):
            continue
        if reply.get("ddate"):
            continue
        decision = value(reply.get("content") or {}, decision_field, "")
        if isinstance(decision, str) and decision:
            decisions.append((reply.get("tmdate") or reply.get("cdate") or 0, decision))
    if not decisions:
        return None
    decision = max(decisions, key=lambda item: item[0])[1]
    return bool(re.match(r"^Accept\b", decision, re.I)) and not bool(re.search(r"workshop", decision, re.I))


def collect_openreview(venue, year, fetch):
    base, group, version = scope(venue, year)
    output = result()
    if venue == "colm" and year < 2024:
        output.update(status="not_applicable", note="COLM began in 2024.")
        return output
    with collecting(output):
        offset, expected, seen, uncertain, malformed = 0, None, set(), 0, 0
        while True:
            params = {"limit": PAGE_SIZE, "sort": "number:asc"}
            if offset:
                params["offset"] = offset
            elif version == 2:
                # API 2 omits the total unless requested, and count queries
                # must not include an offset (including zero).
                params["count"] = "true"
            if version == 2:
                params["content.venueid"] = group
            else:
                params["invitation"] = group + ("/-/submission" if year == 2017 else "/-/Blind_Submission")
                params["details"] = "directReplies"
            payload = json.loads(read(fetch, output, base + "/notes?" + urlencode(params)))
            if not isinstance(payload, dict) or not isinstance(payload.get("notes"), list):
                raise ValueError("OpenReview response is missing notes")
            count, notes = payload.get("count", expected), payload["notes"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("OpenReview response is missing a valid total count")
            if expected is not None and expected != count:
                raise ValueError("OpenReview count changed during pagination; rerun with --refresh")
            expected = count
            for note in notes:
                if not isinstance(note, dict) or not isinstance(note.get("id"), str) or not note["id"]:
                    raise ValueError("Malformed OpenReview note")
                if note["id"] in seen:
                    raise ValueError("OpenReview pagination repeated a note")
                seen.add(note["id"])
                accepted = acceptance(note, group, version)
                if accepted is None:
                    uncertain += 1
                if not accepted:
                    continue
                content = note.get("content") or {}
                title = value(content, "title", "")
                authors = value(content, "authors", [])
                abstract = value(content, "abstract")
                # Some accepted 2017 notes use a plain author string instead
                # of the list used by newer submission forms.
                if version == 1 and isinstance(authors, str):
                    authors = [name.strip() for name in re.split(r"[,;\n]", authors) if name.strip()]
                if (not isinstance(title, str) or not title.strip() or not isinstance(authors, list)
                        or any(not isinstance(a, str) for a in authors)):
                    malformed += 1
                    continue
                output["records"].append({
                    "external_id": note["id"], "title": title.strip(), "year": year,
                    "authors": authors, "doi": normalize_doi(value(content, "doi")),
                    "url": "https://openreview.net/forum?" + urlencode({"id": note.get("forum") or note["id"]}),
                    "abstract": abstract.strip() if isinstance(abstract, str) and abstract.strip() else None,
                })
            offset += len(notes)
            if offset >= expected:
                if offset != expected:
                    raise ValueError("OpenReview returned more notes than its total count")
                break
            if not notes:
                raise ValueError("OpenReview pagination ended before its total count")
        complete(output)
        if uncertain or malformed:
            output.update(status="incomplete", note=(
                f"Excluded {uncertain} notes without verified acceptance and {malformed} accepted notes "
                "with missing/invalid title or authors. Other valid papers were retained; review required."))
    return output
