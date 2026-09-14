import csv

file = open("semantic_search_practice_dataset.csv", "r", encoding="utf-8")

reader = csv.DictReader(file)

data = list(reader)


# -------------------------
# LIST
# -------------------------

print("Number of documents:", len(data))

print("\nFirst document:")
print(data[0])

titles = [row["title"] for row in data]

print("\nFirst 5 titles:")
print(titles[:5])


# -------------------------
# SET
# -------------------------

categories = [row["category"] for row in data]

unique_categories = set(categories)

print("\nAll categories:")
print(unique_categories)

print("\nNumber of unique categories:")
print(len(unique_categories))


# -------------------------
# FILTERING
# -------------------------

ml_documents = [
    row for row in data
    if row["category"] == "Machine Learning"
]

print("\nMachine Learning documents:")
print(len(ml_documents))


# -------------------------
# TUPLE
# -------------------------

first = data[0]

document_tuple = (
    first["id"],
    first["title"],
    first["category"]
)

print("\nTuple:")
print(document_tuple)


# Tuple unpacking

doc_id, title, category = document_tuple

print("\nUnpacked tuple:")
print(doc_id)
print(title)
print(category)


# -------------------------
# KEYWORDS SET
# -------------------------

all_keywords = set()

for row in data:
    keywords = row["keywords"].split(",")

    for keyword in keywords:
        all_keywords.add(keyword.strip())


print("\nNumber of unique keywords:")
print(len(all_keywords))

print("\nDoes Python basics exist?")

if "Python basics" in all_keywords:
    print("Yes")
else:
    print("No")



file.close()

print(len(data))
print(data[0])