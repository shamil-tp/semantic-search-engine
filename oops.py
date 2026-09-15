import csv
class Document:
    def __init__(self, id, title, category, text, keywords):
        self.id = id
        self.title = title
        self.category = category
        self.text = text
        self.keywords = keywords

    # 1. Display basic information
    def display(self):
        print("ID:", self.id)
        print("Title:", self.title)
        print("Category:", self.category)

    # 2. Display the complete document
    def show_full(self):
        print("ID:", self.id)
        print("Title:", self.title)
        print("Category:", self.category)
        print("Text:", self.text)
        print("Keywords:", self.keywords)

    # 3. Check whether a word exists in the document
    def contains(self, word):
        return word.lower() in self.text.lower()

    # 4. Get the number of words
    def word_count(self):
        return len(self.text.split())

    # 5. Get keywords as a list
    def get_keywords(self):
        return [keyword.strip() for keyword in self.keywords.split(",")]

    # 6. Check whether a keyword exists
    def has_keyword(self, keyword):
        return keyword.lower() in self.keywords.lower()

    # 7. Return a short preview
    def preview(self, length=50):
        return self.text[:length] + "..."

    # 8. Change the category
    def change_category(self, new_category):
        self.category = new_category
     


file = open("semantic_search_practice_dataset.csv", "r", encoding="utf-8")

reader = csv.DictReader(file)
documents = []

for i in reader:
    document = Document(i["id"],i["title"],i["category"],i["text"],i["keywords"])
    documents.append(document)
while True:

    print("\n===== DOCUMENT MENU =====")
    print("1. Show first document")
    print("2. Search documents")
    print("3. Show documents by category")
    print("4. Show document keywords")
    print("5. Show document word count")
    print("6. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":

        documents[0].show_full()

    elif choice == "2":

        query = input("Enter search word: ")

        found = False

        for document in documents:

            if document.contains(query):
                print("\nID:", document.id)
                print("Title:", document.title)
                print("Category:", document.category)

                found = True

        if not found:
            print("No documents found.")

    elif choice == "3":

        category = input("Enter category: ")

        found = False

        for document in documents:

            if document.category.lower() == category.lower():

                print("\nID:", document.id)
                print("Title:", document.title)

                found = True

        if not found:
            print("No documents found.")

    elif choice == "4":

        doc_id = input("Enter document ID: ")

        for document in documents:

            if document.id == doc_id:

                print("\nTitle:", document.title)
                print("Keywords:", document.get_keywords())

                break

        else:
            print("Document not found.")

    elif choice == "5":

        doc_id = input("Enter document ID: ")

        for document in documents:

            if document.id == doc_id:

                print("\nTitle:", document.title)
                print("Word count:", document.word_count())

                break

        else:
            print("Document not found.")

    elif choice == "6":

        print("Exiting...")
        break

    else:

        print("Invalid choice. Please try again.")


# for i in reader:
#     print(i)

file.close()