import pandas as pd
import ast
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

movies = movies[
    [
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
        "release_date",
        "vote_average"
    ]
]

movies.dropna(inplace=True)


movies["year"] = pd.to_datetime(
    movies["release_date"],
    errors="coerce"
).dt.year

def convert(text):
    result = []

    for item in ast.literal_eval(text):
        result.append(item["name"])

    return result


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)


#top 3 actors
def convert_cast(text):
    result = []

    for item in ast.literal_eval(text)[:3]:
        result.append(item["name"])

    return result


movies["cast"] = movies["cast"].apply(convert_cast)

# GET DIRECTOR
def fetch_director(text):
    result = []

    for item in ast.literal_eval(text):

        if item["job"] == "Director":
            result.append(item["name"])
            break

    return result


movies["crew"] = movies["crew"].apply(fetch_director)

movies["overview"] = movies["overview"].apply(
    lambda x: x.split() if isinstance(x, str) else []
)

def remove_space(items):
    return [item.replace(" ", "") for item in items]


movies["genres"] = movies["genres"].apply(remove_space)
movies["keywords"] = movies["keywords"].apply(remove_space)
movies["cast"] = movies["cast"].apply(remove_space)
movies["crew"] = movies["crew"].apply(remove_space)


# creating all in one tag 
movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)


movies["tags"] = movies["tags"].apply(
    lambda x: " ".join(x)
)


new_df = movies[
    [
        "movie_id",
        "title",
        "year",
        "vote_average",
        "tags"
    ]
].copy()


cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)



vectors = cv.fit_transform(new_df["tags"])


print("Vector shape:", vectors.shape)



os.makedirs("artifacts", exist_ok=True)


with open("artifacts/movie_dict.pkl", "wb") as file:
    pickle.dump(new_df.to_dict(), file)


with open("artifacts/vectorizer.pkl", "wb") as file:
    pickle.dump(cv, file)



print("\nData preparation completed!")

print("\nGenerated files:")
print("1. artifacts/movie_dict.pkl")
print("2. artifacts/vectorizer.pkl")
