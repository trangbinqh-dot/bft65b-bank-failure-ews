from pathlib import Path
from collections import defaultdict
import pyarrow.parquet as pq

files = sorted(Path("data_raw/fdic/financials").glob("*.parquet"))
types = defaultdict(lambda: defaultdict(list))
for f in files:
    schema = pq.read_schema(f)
    for name, typ in zip(schema.names, schema.types):
        types[name][str(typ)].append(f.stem.replace("REPDTE=", ""))

print(f"Files scanned: {len(files)}")
n_bad = 0
for col, d in types.items():
    if len(d) > 1:
        n_bad += 1
        print(f"\n{col}: {len(d)} different types")
        for t, qs in d.items():
            print(f"   {t:<10} {len(qs):>3} quarters  (e.g. {qs[0]} ... {qs[-1]})")
print(f"\nColumns with inconsistent type: {n_bad} of {len(types)}")

