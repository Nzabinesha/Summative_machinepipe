import pandas as pd

rows = []
for n in (1, 2, 3):
    df = pd.read_csv(f"results/containers_{n}_stats.csv")
    r = df[df["Name"] == "/predict"].iloc[0]
    rows.append({"containers": n, "requests": int(r["Request Count"]),
                 "failures": int(r["Failure Count"]),
                 "avg_ms": round(r["Average Response Time"], 1),
                 "median_ms": r["Median Response Time"], "p95_ms": r["95%"],
                 "req_per_sec": round(r["Requests/s"], 1)})
print(pd.DataFrame(rows).to_string(index=False))