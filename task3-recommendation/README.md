# Movie Recommendation Engine

A content-based recommendation system that uses TF-IDF vectorization and cosine similarity
to match movies to user preferences. Supports hybrid scoring, cold-start users, negative
feedback, and outputs a measured precision score before any interaction begins.

## How to run

```bash
cd task3-recommendation
pip install -r ../requirements.txt

python recommender.py
```

The system runs a leave-one-out evaluation on startup, then guides you through selecting
liked movies and genre preferences to generate personalized recommendations.

## Technical highlights

- **TF-IDF + cosine similarity**: each movie gets a weighted term vector from its genres,
  director, cast, and description  -  similarity is computed in that space, not by genre if-else
- **Precision@5 evaluation**: for each movie, checks whether TF-IDF similarity surfaces
  at least one title sharing 2+ genres in its top-5. Runs across all 50 movies on startup
  and prints the score (0.960) before any user interaction
- **Cold-start**: if the user hasn't seen any movies from the list, it falls back to pure
  genre preference scoring instead of crashing or returning garbage
- **Negative feedback**: typing a result number removes it and pulls in the next best match
- **Hybrid scoring**: content similarity weighted at 70%, genre preference at 30%  -  adjusts
  to 0/100 automatically when there are no liked movies to compute similarity from

## Sample output

```
System Evaluation -- Mean Precision@5: 0.960 (tested on 50 movies)

  1. Dune (2021)
     Genres: Action|Adventure|Drama|Sci-Fi
     Match score: 33.9%  |  Why: matches your genres: Sci-Fi | similar to Interstellar

  2. Avengers: Endgame (2019)
     Genres: Action|Adventure|Sci-Fi
     Match score: 33.7%  |  Why: matches your genres: Sci-Fi | similar to Interstellar
```

## Dataset

50 hand-curated films spanning diverse genres (1972-2023), stored in `data/movies.csv`.
Each movie includes title, year, genres, director, lead cast, and a description used
for TF-IDF vectorization.

## Dependencies

`scikit-learn` | `pandas` | `numpy`
