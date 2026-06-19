from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = {
    "Avengers": "action sci-fi superhero marvel",
    "John Wick": "action crime thriller",
    "Interstellar": "sci-fi space future",
    "The Matrix": "sci-fi artificial intelligence future",
    "The Conjuring": "horror ghost supernatural scary",
    "Home Alone": "comedy family funny",
    "Mr. Bean": "comedy humor funny",
    "Titanic": "romance love drama"
}

print("🎬 MOVIE RECOMMENDATION SYSTEM")
print("\nAvailable Genres:")
print("- Action")
print("- Comedy")
print("- Horror")
print("- Sci-Fi")
print("- Romance")

user_interest = input("\nEnter your preferred movie genre: ").lower()

documents = [user_interest] + list(movies.values())

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)

scores = similarity_scores[0]

movie_scores = list(zip(movies.keys(), scores))
movie_scores.sort(key=lambda x: x[1], reverse=True)

print("\n🎥 Recommended Movies:\n")

found = False

for movie, score in movie_scores:
    if score > 0:
        print(f"{movie} (Similarity Score: {score:.2f})")
        found = True

if not found:
    print("No matching movies found.")