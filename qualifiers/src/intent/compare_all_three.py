import csv
import os
from collections import Counter, defaultdict

def main():
    root = "D:/EAI-Kaggle"
    ascii_v1 = os.path.join(root, "qualifiers/weights_pure_ascii/pure_ascii_intent_preds.csv")
    ascii_v2 = os.path.join(root, "qualifiers/weights_pure_ascii_v2/pure_ascii_v2_intent_preds.csv")
    champ_file = os.path.join(root, "intent_champion_submission.csv")
    test_csv = os.path.join(root, "qualifiers/data/intent_test.csv")

    questions = {}
    if os.path.exists(test_csv):
        with open(test_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                questions[r["question_id"]] = r["question"]

    def load_preds(p):
        d = {}
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    d[r["sample_id"]] = r["prediction"]
        return d

    v1_preds = load_preds(ascii_v1)
    v2_preds = load_preds(ascii_v2)
    ch_preds = load_preds(champ_file)

    common = [qid for qid in v2_preds if qid in ch_preds]
    total = len(common)
    print(f"Total evaluated test questions: {total:,}")

    # Agreement V2 vs Champion
    matches_v2_ch = sum(1 for q in common if v2_preds[q] == ch_preds[q])
    pct_v2_ch = (matches_v2_ch / total) * 100
    print(f"\n[★ AGREEMENT] ASCII v2 vs Champion: {matches_v2_ch}/{total} ({pct_v2_ch:.2f}%)")

    # Agreement V1 vs V2
    if v1_preds:
        matches_v1_v2 = sum(1 for q in common if v1_preds[q] == v2_preds[q])
        print(f"[★ AGREEMENT] ASCII v1 vs ASCII v2 : {matches_v1_v2}/{total} ({(matches_v1_v2/total)*100:.2f}%)")

    # Distributions
    labels = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
    v1_dist = Counter(v1_preds.values())
    v2_dist = Counter(v2_preds.values())
    ch_dist = Counter(ch_preds.values())

    print("\n--- CLASS DISTRIBUTION COMPARISON ---")
    print(f"{'Class':<15} | {'ASCII v1':<10} | {'ASCII v2':<10} | {'Champion':<10}")
    print("-" * 55)
    for l in labels:
        print(f"{l:<15} | {v1_dist.get(l, 0):<10} | {v2_dist.get(l, 0):<10} | {ch_dist.get(l, 0):<10}")

    # Check the 28 'Tinh tong so' cases from v1
    print("\n--- CHECKING THE 28 'Tinh tong so' CASES (Did V2 fix them?) ---")
    fixed_count = 0
    for qid in common:
        qtext = questions.get(qid, "")
        if "tinh tong so" in qtext.lower() or "so luong" in qtext.lower():
            if v1_preds.get(qid) == "EXIST" and v2_preds[qid] == "COUNT":
                fixed_count += 1
                if fixed_count <= 5:
                    print(f"  [FIXED -> COUNT] [{qid}] \"{qtext}\"")

    print(f"Total 'tinh tong so' / 'so luong' cases fixed to COUNT in ASCII v2: {fixed_count}")

if __name__ == "__main__":
    main()
