import csv

file = open("semantic_search_practice_dataset.csv", "r", encoding="utf-8")

reader = csv.DictReader(file)

data = list(reader)

keyword = input("enter a keyword to search: ")

found = 0

found_description = 0
for row in data:
    if keyword in row["title"].lower():
        found = found + 1
    if keyword in row["text"].lower():
        found_description = found_description + 1

print(f"Total search result for the given keyword: {found} Title")
print(f"Total search result for the given keyword: {found_description} texts")

file.close()