from pipeline import query

tests = [
    ("Give me the starting 11 for both teams", "303731"),
    ("Who scored the goals and who assisted them", "303731"),
    ("Give me the starting 11 for both teams", "303516"),
    ("Who scored in this match", "303516"),
    ("Give me the starting 11 for both teams", "303532"),
    ("Who scored in this match", "303532"),
]

for question, match_id in tests:
    print(f"\nMatch {match_id} | Q: {question}")
    print(query(question, match_id=match_id))
    print("-" * 60)