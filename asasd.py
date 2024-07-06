# import spacy

# # Load the small English model
# nlp = spacy.load("en_core_web_sm")

# def most_significant_word(phrase):
#     # Process the phrase using spaCy
#     doc = nlp(phrase)
    
#     # Initialize variables to track the most significant word
#     max_similarity = -1
#     significant_word = None
    
#     # Iterate over each token in the processed document
#     for token in doc:
#         # Check if the token has a vector representation (is in the vocabulary)
#         if token.has_vector:
#             # Calculate the average similarity of the token with all other tokens in the document
#             similarity = sum(token.similarity(other) for other in doc) / len(doc)
            
#             # Update significant_word if current token's similarity is higher
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 significant_word = token.text
    
#     return significant_word

# # Example phrase
# phrase = "Visualization methods"

# # Find the most significant word in the phrase using word embeddings
# important_word = most_significant_word(phrase)
# print(f"The most significant word in the phrase '{phrase}' is '{important_word}'")
