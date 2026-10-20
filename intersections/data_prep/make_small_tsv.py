import pandas as pd

TSV = "data/out"
OUT = "data/tiles_small.tsv"
PER_TYPE = 3 
SEP = ","

df = pd.read_csv(TSV, sep="\t", dtype={"tile_id": str})
type_lists = df["types"].str.split(SEP)
all_types = sorted({t for ts in type_lists for t in ts})

keep = set()
for t in all_types:
    has_t = df[type_lists.map(lambda ts: t in ts)]
    keep.update(has_t.sample(min(PER_TYPE, len(has_t)), random_state=0).index)
keep.update(df.index[type_lists.map(len) > 1][:1])

small = df.loc[sorted(keep)]
small.to_csv(OUT, sep="\t", index=False)
print(small["types"].str.split(SEP).explode().value_counts())