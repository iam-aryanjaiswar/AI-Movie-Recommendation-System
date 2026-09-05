import pickle
import pandas as pd

from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

with open("artifacts/movie_dict.pkl", "rb") as file:
    movies = pickle.load(file)

movies = pd.DataFrame(movies)

with open("artifacts/vectorizer.pkl", "rb") as file:
    cv = pickle.load(file)

vectors = cv.transform(movies["tags"])

print("Vector shape:", vectors.shape)

knn = NearestNeighbors(
    n_neighbors=10,
    metric="cosine",
    algorithm="brute"
)

knn.fit(vectors)


def recommend(movie_title):

    movie_index = None

    for index, title in movies["title"].items():

        if title.lower() == movie_title.lower():
            movie_index = index
            break

    if movie_index is None:
        return []

    movie_vector = vectors[movie_index]

    distances, indices = knn.kneighbors(movie_vector)

    recommendations = []

    for i in indices[0][1:]:

        recommendations.append({
            "id": movies["movie_id"][i],
            "title": movies["title"][i],
            "rating": movies["vote_average"][i],
            "year": movies["year"][i]
        })

    return recommendations


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    selected_movie = ""

    movie_titles = movies["title"].tolist()

    if request.method == "POST":

        selected_movie = request.form.get("movie", "").strip()

        if selected_movie:
            recommendations = recommend(selected_movie)

    return render_template(
        "index.html",
        recommendations=recommendations,
        selected_movie=selected_movie,
        movie_titles=movie_titles
    )


if __name__ == "__main__":
    app.run()
