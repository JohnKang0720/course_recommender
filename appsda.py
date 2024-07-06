import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

def most_significant_word(phrase):
    # Process the phrase using spaCy
    doc = nlp(phrase)
    
    # Initialize variables to track the most significant word
    max_significance = -1
    significant_word = None
    
    # Iterate over each token in the processed document
    for token in doc:
        # Check if the token's dependency relation indicates it's an important word (you can adjust this condition as needed)
        # print(token.text, token.dep_)
        if token.dep_ in ['nsubj', 'dobj', 'ROOT', 'compound']:  # Example dependency relations indicating importance
            print(token.text, token.dep_)
            pass
    
    return significant_word

# Example phrase
phrase = "Data structures and other stuff"

# Find the most significant word in the phrase using dependency parsing
important_word = most_significant_word(phrase)
print(f"The most significant word in the phrase '{phrase}' is '{important_word}'")
