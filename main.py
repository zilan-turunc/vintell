from features.text_to_fashion import suggest_fashion_items

if __name__ == "__main__":
    description = input("Describe the item or style you're looking for:\n> ")
    suggestion = suggest_fashion_items(description)
    print("\nSuggested Items:\n" + suggestion)
