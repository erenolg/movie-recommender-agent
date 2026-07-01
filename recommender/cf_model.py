import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringModel:
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.movies = None
        self.ratings = None
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.movie_title_to_id = None
        self.movie_id_to_title = None

    def load_data(self):
        """Load MovieLens data from CSV files."""
        self.movies = pd.read_csv(f"{self.data_path}/movies.csv")
        self.ratings = pd.read_csv(f"{self.data_path}/ratings.csv")

        # Title lookup dictionaries
        self.movie_title_to_id = dict(
            zip(self.movies["title"].str.lower(), self.movies["movieId"])
        )
        self.movie_id_to_title = dict(
            zip(self.movies["movieId"], self.movies["title"])
        )

        print(f"Loaded {len(self.movies)} movies and {len(self.ratings)} ratings.")

    def build_matrix(self):
        """Build user-item matrix and compute item-item cosine similarity."""
        # Pivot ratings into user-item matrix
        # Rows = users, Columns = movies, Values = ratings (0 if not rated)
        self.user_item_matrix = self.ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

        print(f"User-item matrix shape: {self.user_item_matrix.shape}")

        # Compute item-item similarity
        # Transpose so rows = movies, then compute cosine similarity between movies
        item_matrix = self.user_item_matrix.T
        self.item_similarity_matrix = pd.DataFrame(
            cosine_similarity(item_matrix),
            index=item_matrix.index,
            columns=item_matrix.index
        )

        print("Item similarity matrix built.")

    def get_similar_movies(self, movie_title: str, n: int = 20) -> list[dict]:
        """Get N most similar movies to a given title based on CF similarity."""
        movie_title_lower = movie_title.lower()

        # Fuzzy title match - find closest match if exact match fails
        if movie_title_lower not in self.movie_title_to_id:
            matches = [
                t for t in self.movie_title_to_id.keys()
                if movie_title_lower in t
            ]
            if not matches:
                return []
            movie_title_lower = matches[0]

        movie_id = self.movie_title_to_id[movie_title_lower]

        if movie_id not in self.item_similarity_matrix.index:
            return []

        # Get similarity scores for this movie against all others
        similarity_scores = self.item_similarity_matrix[movie_id].sort_values(
            ascending=False
        )

        # Exclude the movie itself (similarity = 1.0 with itself)
        similarity_scores = similarity_scores.drop(movie_id, errors="ignore")

        # Take top N
        top_movies = similarity_scores.head(n)

        # Build result with metadata
        results = []
        for mid, score in top_movies.items():
            movie_row = self.movies[self.movies["movieId"] == mid]
            if movie_row.empty:
                continue
            results.append({
                "movie_id": int(mid),
                "title": movie_row["title"].values[0],
                "genres": movie_row["genres"].values[0],
                "similarity_score": round(float(score), 4)
            })

        return results

    def get_movie_details(self, movie_id: int) -> dict:
        """Get metadata for a specific movie by ID."""
        row = self.movies[self.movies["movieId"] == movie_id]
        if row.empty:
            return {}
        avg_rating = self.ratings[
            self.ratings["movieId"] == movie_id
        ]["rating"].mean()

        return {
            "movie_id": int(movie_id),
            "title": row["title"].values[0],
            "genres": row["genres"].values[0],
            "avg_rating": round(float(avg_rating), 2) if not np.isnan(avg_rating) else None
        }

    def get_popular_movies(self, n: int = 20, min_ratings: int = 50) -> list[dict]:
        """Get top N popular movies by average rating with minimum rating threshold."""
        rating_stats = self.ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count")
        ).reset_index()

        # Filter by minimum ratings to avoid obscure movies with one 5-star review
        popular = rating_stats[
            rating_stats["num_ratings"] >= min_ratings
        ].sort_values("avg_rating", ascending=False).head(n)

        results = []
        for _, row in popular.iterrows():
            movie_row = self.movies[self.movies["movieId"] == row["movieId"]]
            if movie_row.empty:
                continue
            results.append({
                "movie_id": int(row["movieId"]),
                "title": movie_row["title"].values[0],
                "genres": movie_row["genres"].values[0],
                "avg_rating": round(float(row["avg_rating"]), 2),
                "num_ratings": int(row["num_ratings"])
            })

        return results