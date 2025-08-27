# tiny.py  — mini argparse demo
import argparse

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True, help="Path to CSV")                       # required
p.add_argument("--model", choices=["kmeans", "hier", "dbscan"], default="kmeans") # pick one
p.add_argument("--id-col", default="country")                                     # optional
p.add_argument("--exclude", nargs="*", default=[])                                # 0+ values -> list
args = p.parse_args()

print("csv     =", args.csv)
print("model   =", args.model)
print("id_col  =", args.id_col)
print("exclude =", args.exclude)
