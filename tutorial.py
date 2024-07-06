from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
from nltk.corpus import stopwords
import re
from nltk.probability import FreqDist
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import euclidean_distances
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from nltk.stem import WordNetLemmatizer

nlp = spacy.load('en_core_web_sm')
nltk.download('wordnet')
df = pd.read_csv("./movies_metadata.csv", low_memory=False)
tf = TfidfVectorizer(stop_words='english') #count vectorizer, alternatively
df["overview"] = df["overview"].fillna("")



str1="Fundamental computation and program structures. Continuing systematic program design from CPSC 103. [3-2-0] Prerequisite: CPSC 103. "
str2="Fundamental program and computation structures. Introductory programming skills. Computation as a tool for information processing, simulation and modelling, and interacting with the world. [3-3-0]"
str3="Overview of database systems, ER models, logical database design and normalization, formal relational query languages, SQL and other commercial languages,data warehouses, special topics. [3-0-1] Prerequisite: CPSC 221.  "
str4="Computation as a tool for systematic problem solving in non-computer-science disciplines. Introductory programming skills. Not for credit for students who have credit for, or exemption from, or are concurrently taking CPSC 110 or APSC 160. No programming experience expected. [3-0-1]"
matrix = tf.fit_transform([str4])

arr = [str1, str2, str3]


# print(pd.DataFrame(matrix.toarray(), columns=tf.get_feature_names_out()))
print(tf.get_feature_names_out())
# indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# def return_similar(title):
#     movie = indices[title]
#     movie_vector = pd.DataFrame(cosine_sim[movie], columns=["score"]).reset_index().sort_values(by=["score"], ascending=False).iloc[1:11]
#     recommendations = df["title"].iloc[movie_vector["index"]]
#     return recommendations

df = pd.read_csv("./coursedata/course_data.csv")

searched = set()

#this takes all course codes
for i in range(1,20):
    text = nlp(df["description"].iloc[i])
    # print(text)
    for j in text.ents:
        if j.label_ == "ORG" and len(j.text) == 4:
            searched.add(j.text)
    

# print(searched)

#handling technical terms - regex

#array of all technical words from GPT
technical_words = [
    "Computational Thinking",
    "Systematic Program Design",
    "Introductory programming skills",
    "Computation as a tool",
    "Information processing",
    "Simulation and modelling",
    "Interacting with the world",
    "Models of Computation",
    "Boolean algebra",
    "Logic circuits",
    "Proof techniques",
    "Functions",
    "Sequential circuits",
    "Sets and relations",
    "Finite state machines",
    "Sequential instruction execution",
    "Programming, Problem Solving, and Algorithms",
    "Algorithmic problems",
    "Modern programming language",
    "Applied algorithms",
    "Voronoi Diagrams",
    "Markov Chains",
    "Bin Packing",
    "Graph Search",
    "Software Construction",
    "Robust software components",
    "Software design",
    "Computational models",
    "Data structures",
    "Debugging and testing",
    "Introduction to Computer Systems",
    "Software architecture",
    "Operating systems",
    "IO architectures",
    "Application software",
    "Computing hardware",
    "Disks and networks",
    "Basic Algorithms and Data Structures",
    "Algorithm analysis methods",
    "Searching and sorting algorithms",
    "Graphs and concurrency",
    "Data Structures and Algorithms for Electrical Engineers",
    "Procedural programming",
    "Sorting and searching algorithms",
    "Lists, trees, and hash tables",
    "Scripting languages",
    "File input/output",
    "Basics of Computer Systems",
    "Critical sections",
    "Deadlock avoidance",
    "Performance",
    "Disks and networks",
    "Numerical Computation for Algebraic Problems",
    "Numerical techniques",
    "Linear systems",
    "Round-off errors",
    "Norms and condition number",
    "Iterative techniques",
    "Eigenvalue problems",
    "Nonlinear equations",
    "Numerical Approximation and Discretization",
    "Interpolation and approximation",
    "Splines and least squares data fitting",
    "Numerical differentiation and integration",
    "Initial value ordinary differential equations",
    "Introduction to Relational Databases",
    "Database systems",
    "ER models",
    "Logical database design",
    "Normalization",
    "Relational query languages",
    "SQL",
    "Data warehouses",
    "Advanced Relational Databases",
    "Physical database design",
    "Indexing",
    "External mergesort",
    "Query processing and optimization",
    "Transaction processing",
    "Concurrency control",
    "Numerical Linear Algebra",
    "Computational linear algebra",
    "Orthogonal transformations",
    "Linear equations",
    "Eigenproblem",
    "Linear least squares",
    "Symmetric eigenproblem",
    "QR method",
    "Sparse matrices",
    "Advanced Methods for Human Computer Interaction",
    "Interface design",
    "Task analysis",
    "Analytic and empirical evaluation methods",
    "HCI research frontiers",
    "Algorithms in Bioinformatics",
    "Sequence alignment",
    "Phylogenetic tree reconstruction",
    "Prediction of RNA and protein structure",
    "Gene finding",
    "Sequence annotation",
    "Biomolecular computing",
    "Introduction to Visualization",
    "Static and interactive visualizations",
    "Visualization methods",
    "Design and implementation of 3D shapes"
]



# print(extract_tech_words(technical_words))


def most_significant_word(phrases):
    # Process the phrase using spaCy
    res = []
    for i in phrases:
        doc = nlp(i)

        significant_word = " "
        arr=[]
        for token in doc:
            # Check if the token's dependency relation indicates it's an important word (you can adjust this condition as needed)
            # s = PorterStemmer()
            # print(token.text, token.dep_)
            if token.dep_ in ['ROOT', 'compound', 'amod', 'nobj', 'pobj', 'nsubj']:  # Example dependency relations indicating importance
                arr.append(token.text)
        res.append(significant_word.join(arr))
    return res

words = list(most_significant_word(technical_words))
# print(words)
# tf.fit_transform(words)
#155
# print(len(tf.get_feature_names_out()))


# def most_important_word(phrase):
#     tokens = word_tokenize(phrase)
#     freq_dist = FreqDist(tokens)
#     most_common_word = freq_dist.max()
#     return most_common_word

# phrase = "This is a short phrase with some words."
# important_word = most_important_word(phrase)
# print(f"The most important word in the phrase '{phrase}' is '{important_word}'")
