"""
Sample r/fitness posts from different time periods via Arctic Shift
to check what the removal/deletion rate looks like per year.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, UTC
from fitness_reddit_analyzer.arctic import ArcticShiftClient

client = ArcticShiftClient()

# Sample months: Jan of 2019, 2020, 2021, 2022, 2023, 2024
sample_months = [
    (2019, 6), (2020, 6), (2021, 6), (2022, 6), (2023, 6), (2024, 1),
]

print(f"{'Period':<12} {'Total':>6} {'Removed':>8} {'Deleted':>8} {'Neither':>8} {'%Surv':>6}")
print("-" * 60)

for year, month in sample_months:
    start = int(datetime(year, month, 1, tzinfo=UTC).timestamp())
    # Just sample ~2 weeks to avoid rate limits
    end = start + (14 * 86400)
    
    all_posts = []
    cursor = start
    # Paginate to get a decent sample (up to 500 posts)
    for _ in range(5):
        page = client.search_posts("fitness", after=cursor, before=end, limit=100)
        if not page.items:
            break
        all_posts.extend(page.items)
        next_cursor = max(int(p["created_utc"]) for p in page.items if p.get("created_utc"))
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    
    if not all_posts:
        print(f"{year}-{month:02d}      -- no data --")
        continue
    
    total = len(all_posts)
    removed = sum(1 for p in all_posts if p.get("removed_by_category") or p.get("removed_by"))
    deleted = sum(1 for p in all_posts 
                  if (p.get("author") in ("[deleted]", "[removed]", None)) 
                  and not (p.get("removed_by_category") or p.get("removed_by")))
    neither = total - removed - deleted
    surv_pct = (neither / total * 100) if total else 0
    
    print(f"{year}-{month:02d}      {total:>6} {removed:>8} {deleted:>8} {neither:>8} {surv_pct:>5.1f}%")

    # Show a few examples of surviving posts from this period
    survivors = [p for p in all_posts 
                 if not (p.get("removed_by_category") or p.get("removed_by"))
                 and p.get("author") not in ("[deleted]", "[removed]", None)]
    if survivors[:3]:
        for s in survivors[:3]:
            selftext = (s.get("selftext") or "")[:60].replace("\n", " ")
            status = "bot" if (s.get("author","").lower() == "automoderator" or s.get("author","").lower().endswith("bot")) else "human"
            print(f"  └─ [{status}] score={s.get('score',0):>4} comments={s.get('num_comments',0):>3} title={s.get('title','')[:50]}")
