import json
with open("eval/test_set/questions.json") as f:
    q = json.load(f)
print(f"{len(q)} questions loaded")
print("Categories:", sorted(set(x["category"] for x in q)))
print("Charts referenced:", sorted(set(x["chart_id"] for x in q)))
print("Expected refusals:", sum(1 for x in q if x.get("expected_refusal")))