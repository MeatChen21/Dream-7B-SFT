# file: inspect_s1k.py
from datasets import load_dataset
ds = load_dataset("open-r1/s1K-1.1")
print(ds)
ex = ds["train"][0]
for m in ex["messages"]:
    print(m["role"].upper()+":", m["content"][:160].replace("\n"," ") + ("..." if len(m["content"])>160 else ""))