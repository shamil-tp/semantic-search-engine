import csv

file = open("semantic_search_practice_dataset.csv", "r", encoding="utf-8")

reader = csv.DictReader(file)

data = list(reader)

keyword = input("enter a keyword to search")