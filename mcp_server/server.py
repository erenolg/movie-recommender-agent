import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from recommender.cf_model import CollaborativeFilteringModel

model = CollaborativeFilteringModel(data_path="data/ml-latest-small")
model.load_data()
model.build_matrix()

mcp = FastMCP("movie-recommender")


@mcp.tool()
def get_similar_movies(movie_title: str, n: int = 20) -> str:
    """Get movies similar to a given title using collaborative filtering.
    Returns a candidate pool with genre and similarity metadata for further filtering.
    Use this when the user mentions a specific movie they liked.
    """
    import json
    results = model.get_similar_movies(movie_title, n=n)
    if not results:
        return json.dumps({"error": f"Movie '{movie_title}' not found in database."})
    return json.dumps(results)

@mcp.tool()
def search_movies_by_genre(genre: str, n: int = 20) -> str:
    """Search for popular movies filtered by a specific genre.
    Use this when the user explicitly asks for movies of a specific genre.
    Genre should be one of: Action, Adventure, Animation, Children, Comedy,
    Crime, Documentary, Drama, Fantasy, Horror, Mystery, Romance,
    Sci-Fi, Thriller, War, Western.
    """
    import json
    # existing filtering logic stays the same, just change return
    filtered = model.movies[
        model.movies["genres"].str.contains(genre, case=False, na=False)
    ]
    if filtered.empty:
        return json.dumps({"error": f"No movies found for genre '{genre}'."})

    rating_stats = model.ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()

    merged = filtered.merge(rating_stats, on="movieId", how="left")
    merged = merged[merged["num_ratings"] >= 20].sort_values(
        "avg_rating", ascending=False
    ).head(n)

    results = [
        {
            "movie_id": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
            "avg_rating": round(float(row["avg_rating"]), 2),
            "num_ratings": int(row["num_ratings"])
        }
        for _, row in merged.iterrows()
    ]
    return json.dumps(results)

@mcp.tool()
def get_popular_movies(n: int = 20, min_ratings: int = 50) -> str:
    """Get top rated popular movies with a minimum number of ratings.
    Use min_ratings=50 for this dataset. Do not use values above 200,
    the dataset is small. Use this when the user wants general recommendations
    without a specific reference movie. Returns a JSON list of movies.
    """
    import json
    results = model.get_popular_movies(n=n, min_ratings=min_ratings)
    return json.dumps(results)

@mcp.tool()
def get_movie_details(movie_id: int) -> dict:
    """Get detailed metadata for a specific movie by its ID.
    Use this to fetch more information about a movie before presenting it to the user.
    """
    details = model.get_movie_details(movie_id)
    if not details:
        return {"error": f"Movie ID {movie_id} not found."}
    return details

@mcp.tool()
def search_movie_by_title(title: str) -> dict:
    """Search for a movie by title and return its details including average rating.
    Use this when the user asks about ratings or details of a specific movie by name.
    """
    title_lower = title.lower()
    matches = model.movies[
        model.movies["title"].str.lower().str.contains(title_lower, na=False)
    ]
    if matches.empty:
        return {"error": f"No movie found matching '{title}'"}

    # Take the best match
    row = matches.iloc[0]
    movie_id = int(row["movieId"])
    details = model.get_movie_details(movie_id)
    return details


if __name__ == "__main__":
    mcp.run()