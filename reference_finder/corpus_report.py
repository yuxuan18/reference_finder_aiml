"""Export a searchable local roster and full per-paper abstract availability."""
from __future__ import annotations

import csv
from datetime import date
import html
import json
from pathlib import Path

from .storage import now


def status_for(coverage, year):
    runs = coverage["runs"]
    good = [r for r in runs if r["status"] == "complete"]
    if not good:
        if coverage["listed"] or any(r["status"] == "incomplete" for r in runs):
            return "incomplete"
        return "unavailable / not indexed" if runs else "not attempted"
    status = "current year; provisional" if year >= date.today().year else "source lists retrieved"
    if any(r["status"] == "incomplete" for r in runs):
        status += "; another source incomplete"
    return status


def export_corpus(store, venues, start, end, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    keys = [v.key for v in venues]
    papers = list(store.papers(keys, start, end))
    coverage = []
    for venue in venues:
        for year in range(start, end + 1):
            row = store.coverage(venue.key, year)
            row.update(venue=venue.key, year=year)
            row["status"] = "not applicable; conference not yet established" if year < venue.first_year else status_for(row, year)
            coverage.append(row)
    fields = ["venue", "event_year", "title", "authors", "doi", "has_abstract", "abstract_source", "abstract", "url", "roster_sources", "evidence_urls", "review_notes", "paper_id"]
    with (output / "papers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in papers:
            writer.writerow({"venue": p["venue"], "event_year": p["event_year"], "title": p["title"],
                "authors": "; ".join(json.loads(p["authors_json"])), "doi": p["doi"], "has_abstract": int(bool(p["abstract"])),
                "abstract_source": p["abstract_source"], "abstract": p["abstract"], "url": p["url"],
                "roster_sources": p["roster_sources"], "evidence_urls": " | ".join(dict.fromkeys(url for e in p["evidence"]
                    for url in [e["url"], *e.get("review_evidence_urls", [])] if url)),
                "review_notes": " | ".join(e["review_note"] for e in p["evidence"] if e.get("review_note")), "paper_id": p["id"]})
    exclusion_fields = ["paper_id", "venue", "year", "title", "section", "evidence_url", "reason", "reviewed_at"]
    marks = ",".join("?" for _ in keys)
    with (output / "excluded_papers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=exclusion_fields)
        writer.writeheader()
        for row in store.db.conn.execute(f"""SELECT x.paper_id,x.venue,x.year,p.title,x.section,
                x.evidence_url,x.reason,x.reviewed_at FROM scope_exclusions x JOIN papers p ON p.id=x.paper_id
                WHERE x.venue IN ({marks}) AND x.year BETWEEN ? AND ? ORDER BY x.venue,x.year,p.id""", (*keys, start, end)):
            writer.writerow(dict(row))
    with (output / "coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["venue", "year", "listed_papers", "with_abstract", "without_abstract", "api_only_unverified", "excluded_program_entries", "status", "source_counts"])
        for r in coverage:
            writer.writerow([r["venue"], r["year"], r["listed"], r["abstracts"], r["listed"] - r["abstracts"], r["unverified_api"], r["excluded"], r["status"],
                             "; ".join(f"{s['provider']}={s['record_count']} ({s['status']})" for s in r["runs"])])
    (output / "coverage.json").write_text(json.dumps({"generated_at": now(), "coverage": coverage}, ensure_ascii=False, indent=2))
    compact = [{"venue": p["venue"], "year": p["event_year"], "title": p["title"], "doi": p["doi"],
                "abstract": bool(p["abstract"]), "source": p["abstract_source"] or "", "url": p["url"],
                "evidence": p["evidence"]} for p in papers]
    payload = json.dumps(compact, ensure_ascii=False).replace("<", "\\u003c")
    table_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in [r["venue"], r["year"], r["listed"], r["abstracts"], r["listed"] - r["abstracts"], r["unverified_api"], r["excluded"], r["status"], "; ".join(f"{s['provider']}: {s['record_count']} ({s['status']})" for s in r["runs"])]) + "</tr>" for r in coverage)
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Conference paper inventory</title><style>
body{font:15px/1.55 system-ui,sans-serif;margin:32px auto;padding:0 24px;max-width:1450px;color:#1e293b;background:#f8fafc}h1{font-size:30px}h2{margin-top:32px}a{color:#1459a6}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #e2e8f0;vertical-align:top}th{background:#e8edf3;position:sticky;top:0}input,select,button{font:inherit;padding:9px;border:1px solid #b7c3d2;border-radius:5px}input{min-width:320px}.scroll{overflow:auto;max-height:540px}.note{background:#e9f0fb;padding:16px;border-left:4px solid #2869ae}.ok{color:#147442}.missing{color:#a33}small{color:#64748b}#results td:nth-child(3){width:55%}
</style><h1>Conference paper inventory</h1><p class="note">This list enumerates main-proceedings entries from DBLP and official conference sources, covering ICLR, ICML, NeurIPS, AAAI, and COLM. Abstract availability is separate from paper enumeration. API-only records remain in SQLite for review and are excluded from roster totals. Current-year lists are provisional. “Source lists retrieved” describes the source scan, not an independent guarantee that a publisher has indexed every paper. For AAAI, confirmed IAAI/EAAI and ancillary programs are excluded using official issue section evidence; main and special research tracks are included. Excluded records remain preserved in SQLite. Other venues follow their published main proceedings; attached workshops are excluded.</p>
<p><strong>__TOTAL__ conference-paper entries · __ABSTRACTS__ with abstracts · __MISSING__ without abstracts</strong></p>
<p><a href="papers.csv">Download every paper and abstract as CSV</a> · <a href="coverage.csv">Download counts</a> · <a href="coverage.json">Source status and provenance</a> · <a href="excluded_papers.csv">Excluded programs and evidence</a></p>
<h2>Coverage by conference and event year</h2><div class="scroll"><table><thead><tr><th>Conference</th><th>Event year</th><th>Listed</th><th>With abstract</th><th>No abstract</th><th>API-only / unverified</th><th>Excluded programs</th><th>Status</th><th>Source counts</th></tr></thead><tbody>__COVERAGE__</tbody></table></div>
<h2>Every listed paper</h2><p><input id="query" type="search" placeholder="Search titles…" aria-label="Search titles"> <select id="venue"><option value="">All conferences</option></select> <select id="year"><option value="">All years</option></select> <select id="available"><option value="">Any abstract status</option><option value="yes">Has abstract</option><option value="no">Missing abstract</option></select></p><p id="count"></p><table><thead><tr><th>Conference</th><th>Year</th><th>Title</th><th>Abstract</th><th>Source</th></tr></thead><tbody id="results"></tbody></table><p><button id="more">Show 200 more</button></p>
<script>const papers=__PAPERS__;let shown=200;const $=id=>document.getElementById(id);for(const name of ['venue','year'])for(const v of [...new Set(papers.map(p=>p[name]))].sort()){const o=document.createElement('option');o.value=v;o.textContent=v;$(name).append(o)}
function render(){const q=$('query').value.toLowerCase(),v=$('venue').value,y=$('year').value,a=$('available').value;const rows=papers.filter(p=>(!q||p.title.toLowerCase().includes(q))&&(!v||p.venue===v)&&(!y||String(p.year)===y)&&(!a||p.abstract===(a==='yes')));$('count').textContent=`${rows.length.toLocaleString()} papers · ${rows.filter(p=>p.abstract).length.toLocaleString()} have abstracts · showing ${Math.min(shown,rows.length).toLocaleString()}`;$('results').replaceChildren();for(const p of rows.slice(0,shown)){const tr=document.createElement('tr');for(const value of [p.venue,p.year,p.title,p.abstract?'Available':'Missing',p.source]){const td=document.createElement('td');td.textContent=value;tr.append(td)}const url=p.evidence.find(e=>e.url)?.url||p.url;if(url&&/^https?:\\/\\//.test(url)){const link=document.createElement('a');link.href=url;link.textContent=p.title;link.target='_blank';link.rel='noopener';tr.children[2].replaceChildren(link)}for(const e of p.evidence.filter(e=>e.review_note)){const note=document.createElement('small');note.style.display='block';note.textContent=e.review_note;tr.children[2].append(note)}tr.children[3].className=p.abstract?'ok':'missing';$('results').append(tr)}$('more').hidden=shown>=rows.length}for(const name of ['query','venue','year','available'])$(name).addEventListener('input',()=>{shown=200;render()});$('more').onclick=()=>{shown+=200;render()};render();</script></html>"""
    (output / "index.html").write_text(page.replace("__TOTAL__", f"{len(papers):,}").replace("__ABSTRACTS__", f"{sum(bool(p['abstract']) for p in papers):,}").replace("__MISSING__", f"{sum(not p['abstract'] for p in papers):,}").replace("__COVERAGE__", table_rows).replace("__PAPERS__", payload), encoding="utf-8")
    print(f"\n{'Venue':<17} {'Year':>4} {'Listed':>7} {'Abstracts':>10} {'No abstract':>12} {'Excluded':>9}  Status")
    for r in coverage:
        print(f"{r['venue']:<17} {r['year']:>4} {r['listed']:>7} {r['abstracts']:>10} {r['listed']-r['abstracts']:>12} {r['excluded']:>9}  {r['status']}")
    print(f"\nExported {len(papers):,} conference-paper entries; {sum(bool(p['abstract']) for p in papers):,} have abstracts.")
    print(f"Searchable inventory: {(output / 'index.html').resolve()}")
    return coverage
