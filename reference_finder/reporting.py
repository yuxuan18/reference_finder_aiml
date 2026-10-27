"""Read-only coverage and import progress summaries.

Provider checkpoints describe API work completed, not a verified proceedings
roster. Keep that distinction visible even when every checkpoint is finished.
"""

from .importer import job_key


def report(db, venues, start, end):
    """Print stored counts and progress under the current import rules."""
    venues = tuple(venues)
    years = range(start, end + 1)
    print("\nStored coverage (discovered records only; absent papers are not counted):")
    print(f"{'Venue':<17} {'Year':>4} {'Papers':>8} {'Abstracts':>10} {'No abstract':>11} {'No abstract/no DOI':>18}")
    for venue in venues:
        for year in years:
            if year < venue.first_year:
                print(f"{venue.key:<17} {year:>4} not applicable; conference not yet established")
                continue
            row = db.coverage(venue.key, year)
            print(f"{venue.key:<17} {year:>4} {row['papers']:>8} {row['abstracts']:>10} "
                  f"{row['papers'] - row['abstracts']:>11} {row['missing_without_doi']:>18}")

    checkpoints = {row["job"]: bool(row["done"])
                   for row in db.conn.execute("SELECT job, done FROM checkpoints")}

    def finished(stage, venue, **kwargs):
        return checkpoints.get(job_key(stage, venue, **kwargs), False)

    s2_total = sum(year >= venue.first_year for venue in venues for year in years)
    s2_done = sum(finished("s2-discover", venue.key, year=year)
                  for venue in venues for year in years if year >= venue.first_year)
    source_total = source_done = works_total = works_done = 0
    pending_source_venues = []
    no_source_venues = []
    for venue in venues:
        queries = venue.openalex_queries
        completed = sum(finished("oa-sources", venue.key, query=query) for query in queries)
        source_total += len(queries)
        source_done += completed
        sources = db.conn.execute(
            "SELECT source_id FROM openalex_sources WHERE venue=?", (venue.key,)
        ).fetchall()
        if completed < len(queries):
            pending_source_venues.append(venue.key)
        elif not sources:
            no_source_venues.append(venue.key)
        works_total += len(sources) * sum(year >= venue.first_year for year in years)
        works_done += sum(finished("oa-discover", venue.key, year=year, source=source["source_id"])
                          for source in sources for year in years if year >= venue.first_year)

    print(f"\nSemantic Scholar discovery: {s2_done}/{s2_total} conference-years finished; "
          f"{s2_total - s2_done} unfinished or not run under current import rules.")
    print(f"OpenAlex source searches: {source_done}/{source_total} finished; "
          f"venues with unfinished searches: {', '.join(pending_source_venues) or 'none'}.")
    print(f"OpenAlex discovery: {works_done}/{works_total} known-source/year jobs finished; "
          f"{works_total - works_done} unfinished or not run.")
    print("OpenAlex no matching source after finished searches: "
          f"{', '.join(no_source_venues) or 'none'}.")
    print("Finished API jobs do not establish complete proceedings coverage; "
          "no matching source does not mean no papers exist.")
