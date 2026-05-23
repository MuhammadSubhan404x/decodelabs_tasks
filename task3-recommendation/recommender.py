import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "movies.csv"


class MovieDataset:
    def __init__(self, csv_path=DATA_PATH):
        self.df = pd.read_csv(csv_path)
        self.df["content"] = (
            self.df["genres"].str.replace("|", " ", regex=False) + " "
            + self.df["director"] + " "
            + self.df["cast"].str.replace("|", " ", regex=False) + " "
            + self.df["description"]
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])
        self.sim_matrix = cosine_similarity(self.tfidf_matrix)

    def genre_set(self, idx):
        return set(self.df.iloc[idx]["genres"].split("|"))

    def title(self, idx):
        return self.df.iloc[idx]["title"]

    def year(self, idx):
        return self.df.iloc[idx]["year"]

    def genres_str(self, idx):
        return self.df.iloc[idx]["genres"]


class UserProfile:
    def __init__(self):
        self.liked_indices = []

    def add_liked(self, idx):
        if idx not in self.liked_indices:
            self.liked_indices.append(idx)

    def preference_vector(self, tfidf_matrix):
        if not self.liked_indices:
            return None
        vecs = tfidf_matrix[self.liked_indices]
        return np.asarray(vecs.mean(axis=0))


class HybridRecommender:
    def __init__(self, dataset):
        self.ds = dataset

    def recommend(self, profile, genre_prefs,
                  top_n=5, exclude_seen=True, exclude_extra=None, offset=0):
        n = len(self.ds.df)
        cold_start = len(profile.liked_indices) == 0
        skip = set(profile.liked_indices if exclude_seen else [])
        if exclude_extra:
            skip.update(exclude_extra)

        if cold_start:
            w_content, w_genre = 0.0, 1.0
        else:
            w_content, w_genre = 0.7, 0.3

        if not cold_start:
            pref_vec = profile.preference_vector(self.ds.tfidf_matrix)
            content_scores = cosine_similarity(pref_vec, self.ds.tfidf_matrix)[0]
        else:
            content_scores = np.zeros(n)

        genre_scores = np.zeros(n)
        if genre_prefs:
            pref_set = set(g.strip().title() for g in genre_prefs)
            for i in range(n):
                movie_genres = self.ds.genre_set(i)
                overlap = len(pref_set & movie_genres)
                genre_scores[i] = overlap / max(len(pref_set), 1)

        final_scores = w_content * content_scores + w_genre * genre_scores

        candidates = [(i, final_scores[i]) for i in range(n) if i not in skip]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[offset: offset + top_n]

    def explain(self, candidate_idx, profile, genre_prefs):
        parts = []
        movie_genres = self.ds.genre_set(candidate_idx)
        if genre_prefs:
            pref_set = set(g.strip().title() for g in genre_prefs)
            matching = pref_set & movie_genres
            if matching:
                parts.append(f"matches your genres: {', '.join(sorted(matching))}")
        if profile.liked_indices:
            sims = [
                (self.ds.sim_matrix[candidate_idx][i], self.ds.title(i))
                for i in profile.liked_indices
            ]
            sims.sort(reverse=True)
            if sims[0][0] > 0.05:
                parts.append(f"similar to {sims[0][1]}")
        return " | ".join(parts) if parts else "based on overall content profile"


def evaluate_precision_at_k(dataset, k=5):
    n = len(dataset.df)
    hits = 0
    total = 0

    for i in range(n):
        sim_row = dataset.sim_matrix[i].copy()
        sim_row[i] = 0.0
        top_k = np.argsort(sim_row)[::-1][:k]
        genres_i = dataset.genre_set(i)
        for j in top_k:
            genres_j = dataset.genre_set(j)
            if len(genres_i & genres_j) >= 2:
                hits += 1
                break
        total += 1

    precision = hits / total if total > 0 else 0.0
    print(f"System Evaluation -- Mean Precision@{k}: {precision:.3f} "
          f"(tested on {n} movies, threshold: >=2 matching genres)")
    return precision


def display_recommendations(recs, dataset, recommender, profile, genre_prefs):
    print()
    for rank, (idx, score) in enumerate(recs, 1):
        title = dataset.title(idx)
        year = dataset.year(idx)
        genres = dataset.genres_str(idx)
        explanation = recommender.explain(idx, profile, genre_prefs)
        print(f"  {rank}. {title} ({year})")
        print(f"     Genres: {genres}")
        print(f"     Match score: {score * 100:.1f}%  |  Why: {explanation}")
        print()


def run():
    print("=" * 55)
    print("  Movie Recommendation Engine")
    print("=" * 55 + "\n")

    ds = MovieDataset()
    evaluate_precision_at_k(ds, k=5)

    recommender = HybridRecommender(ds)
    profile = UserProfile()

    print()
    seen_movies = input("Have you seen any movies from our list? (y/n): ").strip().lower()

    if seen_movies == "y":
        print("\nAvailable movies:\n")
        for i, row in ds.df.iterrows():
            print(f"  {i + 1:>3}. {row['title']} ({row['year']})  [{row['genres']}]")

        print()
        picks_raw = input("Enter numbers of movies you liked (e.g. 1,5,12): ").strip()
        for p in picks_raw.split(","):
            p = p.strip()
            if p.isdigit():
                idx = int(p) - 1
                if 0 <= idx < len(ds.df):
                    profile.add_liked(idx)
                    print(f"  Added: {ds.title(idx)}")

    genre_input = input("\nPreferred genres (e.g. Action, Drama, Sci-Fi  -  or Enter to skip): ").strip()
    if genre_input:
        sep = "," if "," in genre_input else None
        genre_prefs = [g.strip().title() for g in (genre_input.split(sep) if sep else genre_input.split()) if g.strip()]
    else:
        genre_prefs = []

    if not profile.liked_indices and not genre_prefs:
        print("\nNo preferences given. Showing popular picks across genres.\n")
        genre_prefs = ["Drama", "Action", "Sci-Fi"]

    offset = 0
    excluded = []

    while True:
        recs = recommender.recommend(profile, genre_prefs, top_n=5,
                                     offset=offset, exclude_seen=True,
                                     exclude_extra=excluded)

        if not recs:
            print("No more recommendations available.")
            break

        print(f"\nTop recommendations (showing {offset + 1}-{offset + len(recs)}):")
        display_recommendations(recs, ds, recommender, profile, genre_prefs)

        print("Options: [m] show more  [number] mark not interested  [q] quit")
        action = input("Your choice: ").strip().lower()

        if action == "q":
            break
        elif action == "m":
            offset += 5
        elif action.isdigit():
            pick = int(action) - 1
            if 0 <= pick < len(recs):
                excluded_idx = recs[pick][0]
                excluded.append(excluded_idx)
                print(f"  Noted: '{ds.title(excluded_idx)}' removed from recommendations.")
            else:
                print("  Invalid number.")
        else:
            print("  Didn't catch that. Try 'm', a number, or 'q'.")

    print("\nThanks for using the Movie Recommendation Engine.\n")


if __name__ == "__main__":
    run()
