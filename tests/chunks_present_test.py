from pathlib import Path
for coach in ["alex_hormozi","dan_martell","sam_ovens"]:
    p = Path("data/processed")/coach/"chunks.jsonl"
    print(coach, "exists:", p.exists(), "size:", p.stat().st_size if p.exists() else "NA")
