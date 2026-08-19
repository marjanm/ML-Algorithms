"""
Recommendation Systems — Demo
================================
Powers Netflix, Amazon, Spotify. Demonstrates:
  1. User-User Collaborative Filtering  — find similar users, recommend what they liked
  2. Item-Item Collaborative Filtering  — find similar items to what you already liked
  3. Matrix Factorisation (SVD)         — latent factors, the Netflix Prize approach
  4. Content-Based Filtering            — recommend based on item features
  5. Hybrid approach                    — combine collaborative + content signals

Uses a synthetic movie-rating dataset (no external data needed).

Run:
    python recommendation_demo.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

np.random.seed(42)


def create_movie_dataset():
    """Synthetic movie ratings: 50 users × 30 movies with realistic sparsity."""
    n_users, n_movies = 50, 30

    movie_names = [
        "The Matrix", "Inception", "Interstellar", "Titanic", "The Notebook",
        "Avengers", "Spider-Man", "Iron Man", "The Godfather", "Goodfellas",
        "Toy Story", "Finding Nemo", "Up", "Alien", "Blade Runner",
        "La La Land", "Whiplash", "Parasite", "Joker", "The Dark Knight",
        "Frozen", "Moana", "Coco", "Shrek", "Pulp Fiction",
        "Fight Club", "Se7en", "Memento", "Arrival", "Dune",
    ]

    # genres per movie (for content-based)
    genres = {
        "The Matrix": ["Sci-Fi", "Action"], "Inception": ["Sci-Fi", "Thriller"],
        "Interstellar": ["Sci-Fi", "Drama"], "Titanic": ["Romance", "Drama"],
        "The Notebook": ["Romance", "Drama"], "Avengers": ["Action", "Sci-Fi"],
        "Spider-Man": ["Action", "Sci-Fi"], "Iron Man": ["Action", "Sci-Fi"],
        "The Godfather": ["Crime", "Drama"], "Goodfellas": ["Crime", "Drama"],
        "Toy Story": ["Animation", "Family"], "Finding Nemo": ["Animation", "Family"],
        "Up": ["Animation", "Family"], "Alien": ["Sci-Fi", "Horror"],
        "Blade Runner": ["Sci-Fi", "Thriller"], "La La Land": ["Romance", "Musical"],
        "Whiplash": ["Drama", "Music"], "Parasite": ["Thriller", "Drama"],
        "Joker": ["Drama", "Thriller"], "The Dark Knight": ["Action", "Thriller"],
        "Frozen": ["Animation", "Family"], "Moana": ["Animation", "Family"],
        "Coco": ["Animation", "Family"], "Shrek": ["Animation", "Comedy"],
        "Pulp Fiction": ["Crime", "Thriller"], "Fight Club": ["Drama", "Thriller"],
        "Se7en": ["Crime", "Thriller"], "Memento": ["Thriller", "Mystery"],
        "Arrival": ["Sci-Fi", "Drama"], "Dune": ["Sci-Fi", "Action"],
    }

    # user archetypes for realistic patterns
    archetypes = {
        "sci_fi_fan": {"Sci-Fi": 1.5, "Action": 1.0, "Thriller": 0.8},
        "romance_fan": {"Romance": 1.5, "Drama": 1.0, "Musical": 0.8},
        "action_fan": {"Action": 1.5, "Sci-Fi": 0.8, "Thriller": 0.8},
        "animation_fan": {"Animation": 1.5, "Family": 1.2, "Comedy": 0.8},
        "crime_fan": {"Crime": 1.5, "Thriller": 1.0, "Drama": 0.8},
    }
    arch_names = list(archetypes.keys())

    ratings = np.zeros((n_users, n_movies))
    user_archetypes = []

    for u in range(n_users):
        arch = arch_names[u % len(arch_names)]
        user_archetypes.append(arch)
        prefs = archetypes[arch]
        for m, movie in enumerate(movie_names):
            if np.random.rand() < 0.4:  # 60% sparsity
                continue
            base = 3.0
            for g in genres[movie]:
                base += prefs.get(g, -0.2) * 0.8
            rating = base + np.random.normal(0, 0.5)
            ratings[u, m] = np.clip(round(rating * 2) / 2, 1.0, 5.0)  # 0.5 increments

    ratings_df = pd.DataFrame(ratings, columns=movie_names,
                               index=[f"User_{i}" for i in range(n_users)])

    all_genres = sorted(set(g for gs in genres.values() for g in gs))
    genre_matrix = np.zeros((n_movies, len(all_genres)))
    for m, movie in enumerate(movie_names):
        for g in genres[movie]:
            genre_matrix[m, all_genres.index(g)] = 1.0
    genre_df = pd.DataFrame(genre_matrix, index=movie_names, columns=all_genres)

    return ratings_df, genre_df, user_archetypes


def user_user_cf(ratings_df, target_user, k=5, n_recs=5):
    """User-User Collaborative Filtering: find k most similar users, predict ratings."""
    R = ratings_df.values.copy()
    target_idx = ratings_df.index.get_loc(target_user)
    target_ratings = R[target_idx]

    # mean-centre each user's ratings (only non-zero)
    user_means = np.zeros(R.shape[0])
    R_centered = R.copy()
    for i in range(R.shape[0]):
        mask = R[i] > 0
        if mask.sum() > 0:
            user_means[i] = R[i, mask].mean()
            R_centered[i, mask] -= user_means[i]
        R_centered[i, ~mask] = 0

    # cosine similarity between target and all users
    sims = cosine_similarity(R_centered[target_idx:target_idx+1], R_centered)[0]
    sims[target_idx] = -1  # exclude self

    # top-k similar users
    top_k = np.argsort(sims)[::-1][:k]
    top_k_sims = sims[top_k]

    # predict ratings for unrated movies
    predictions = {}
    for m in range(R.shape[1]):
        if target_ratings[m] > 0:
            continue
        num, den = 0.0, 0.0
        for j, sim in zip(top_k, top_k_sims):
            if R[j, m] > 0:
                num += sim * R_centered[j, m]
                den += abs(sim)
        if den > 0:
            predictions[ratings_df.columns[m]] = user_means[target_idx] + num / den

    recs = sorted(predictions.items(), key=lambda x: -x[1])[:n_recs]
    return recs, top_k, top_k_sims


def item_item_cf(ratings_df, target_user, k=5, n_recs=5):
    """Item-Item Collaborative Filtering: find similar items to what user liked."""
    R = ratings_df.values.copy()
    target_idx = ratings_df.index.get_loc(target_user)
    target_ratings = R[target_idx]

    # item similarity (columns are items)
    R_items = R.T.copy()
    # mean-centre per item
    for i in range(R_items.shape[0]):
        mask = R_items[i] > 0
        if mask.sum() > 0:
            R_items[i, mask] -= R_items[i, mask].mean()
        R_items[i, ~mask] = 0

    item_sims = cosine_similarity(R_items)

    predictions = {}
    for m in range(R.shape[1]):
        if target_ratings[m] > 0:
            continue
        rated_items = np.where(target_ratings > 0)[0]
        sims_to_m = item_sims[m, rated_items]
        top_k_idx = np.argsort(sims_to_m)[::-1][:k]
        top_items = rated_items[top_k_idx]
        top_sims = sims_to_m[top_k_idx]

        num = np.sum(top_sims * target_ratings[top_items])
        den = np.sum(np.abs(top_sims))
        if den > 0:
            predictions[ratings_df.columns[m]] = num / den

    recs = sorted(predictions.items(), key=lambda x: -x[1])[:n_recs]
    return recs


def matrix_factorisation_svd(ratings_df, n_factors=5, n_recs=5):
    """Matrix Factorisation via truncated SVD — the Netflix Prize approach."""
    R = ratings_df.values.copy()
    mask = R > 0

    # fill missing with user mean
    R_filled = R.copy()
    for i in range(R.shape[0]):
        m = R[i, mask[i]].mean() if mask[i].sum() > 0 else 3.0
        R_filled[i, ~mask[i]] = m

    # SVD decomposition
    U, sigma, Vt = np.linalg.svd(R_filled, full_matrices=False)
    U_k = U[:, :n_factors]         # user latent factors
    S_k = np.diag(sigma[:n_factors])
    Vt_k = Vt[:n_factors, :]       # item latent factors

    R_pred = U_k @ S_k @ Vt_k
    R_pred = np.clip(R_pred, 1.0, 5.0)

    # evaluation: RMSE on known ratings
    errors = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if mask[i, j]:
                errors.append((R[i, j] - R_pred[i, j]) ** 2)
    rmse = np.sqrt(np.mean(errors))

    # variance explained by top factors
    total_var = np.sum(sigma ** 2)
    explained = np.cumsum(sigma[:n_factors] ** 2) / total_var

    return R_pred, rmse, sigma, explained, U_k, Vt_k


def content_based(ratings_df, genre_df, target_user, n_recs=5):
    """Content-Based Filtering: build user profile from genres of liked movies."""
    target_idx = ratings_df.index.get_loc(target_user)
    target_ratings = ratings_df.values[target_idx]

    # build user profile: weighted average of genre vectors for rated movies
    rated_mask = target_ratings > 0
    rated_indices = np.where(rated_mask)[0]

    if len(rated_indices) == 0:
        return []

    weights = target_ratings[rated_indices] - 3.0  # centre around neutral
    user_profile = genre_df.values[rated_indices].T @ weights
    user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-8)

    # score unrated movies by similarity to user profile
    predictions = {}
    for m in range(len(target_ratings)):
        if target_ratings[m] > 0:
            continue
        movie_vec = genre_df.values[m]
        score = np.dot(user_profile, movie_vec)
        predictions[ratings_df.columns[m]] = score

    recs = sorted(predictions.items(), key=lambda x: -x[1])[:n_recs]
    return recs


def main():
    lines = [
        "=" * 70,
        "  RECOMMENDATION SYSTEMS  —  Demo",
        "=" * 70, "",
    ]

    ratings_df, genre_df, user_archetypes = create_movie_dataset()
    n_ratings = (ratings_df.values > 0).sum()
    total = ratings_df.shape[0] * ratings_df.shape[1]
    sparsity = 1 - n_ratings / total

    lines += [
        f"  Dataset: {ratings_df.shape[0]} users × {ratings_df.shape[1]} movies",
        f"  Ratings: {n_ratings} / {total} ({sparsity:.1%} sparse)",
        f"  Rating scale: 1.0 – 5.0 (0.5 increments)",
        "",
    ]

    target = "User_0"  # sci-fi fan archetype
    target_idx = ratings_df.index.get_loc(target)
    rated = ratings_df.values[target_idx]
    liked = [(ratings_df.columns[i], rated[i]) for i in range(len(rated)) if rated[i] >= 4.0]
    lines.append(f"  Target user: {target} (archetype: {user_archetypes[target_idx]})")
    lines.append(f"  Movies they liked (≥ 4.0):")
    for movie, rating in sorted(liked, key=lambda x: -x[1]):
        lines.append(f"    {rating:.1f} ★  {movie}")

    # ── 1. User-User CF ──
    lines += ["", "  ── 1. User-User Collaborative Filtering ──"]
    uu_recs, top_k_users, top_k_sims = user_user_cf(ratings_df, target, k=5)
    lines.append(f"  Most similar users:")
    for u_idx, sim in zip(top_k_users, top_k_sims):
        lines.append(f"    {ratings_df.index[u_idx]} (sim={sim:.3f}, archetype={user_archetypes[u_idx]})")
    lines.append(f"  Recommendations:")
    for movie, score in uu_recs:
        lines.append(f"    {score:.2f} ★  {movie}")

    # ── 2. Item-Item CF ──
    lines += ["", "  ── 2. Item-Item Collaborative Filtering ──"]
    ii_recs = item_item_cf(ratings_df, target, k=5)
    lines.append(f"  Recommendations:")
    for movie, score in ii_recs:
        lines.append(f"    {score:.2f} ★  {movie}")

    # ── 3. Matrix Factorisation (SVD) ──
    lines += ["", "  ── 3. Matrix Factorisation (SVD) ──"]
    R_pred, rmse, sigma, explained, U_k, Vt_k = matrix_factorisation_svd(ratings_df, n_factors=5)
    lines.append(f"  Latent factors: 5")
    lines.append(f"  RMSE on known ratings: {rmse:.4f}")
    lines.append(f"  Variance explained by top factors: {explained[-1]:.1%}")

    # SVD recs for target user
    svd_preds = {}
    for m in range(ratings_df.shape[1]):
        if ratings_df.values[target_idx, m] == 0:
            svd_preds[ratings_df.columns[m]] = R_pred[target_idx, m]
    svd_recs = sorted(svd_preds.items(), key=lambda x: -x[1])[:5]
    lines.append(f"  Recommendations:")
    for movie, score in svd_recs:
        lines.append(f"    {score:.2f} ★  {movie}")

    # singular value spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(len(sigma[:15])), sigma[:15], color="steelblue")
    axes[0].set_xlabel("Factor")
    axes[0].set_ylabel("Singular Value")
    axes[0].set_title("Singular Value Spectrum")
    cum_var = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
    axes[1].plot(range(1, len(cum_var[:15]) + 1), cum_var[:15], marker="o", color="coral")
    axes[1].axhline(y=0.9, linestyle="--", color="gray", alpha=0.5)
    axes[1].set_xlabel("Number of Factors")
    axes[1].set_ylabel("Cumulative Variance Explained")
    axes[1].set_title("How Many Factors Do We Need?")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "svd_factors.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] → plots/svd_factors.png")

    # ── 4. Content-Based ──
    lines += ["", "  ── 4. Content-Based Filtering ──"]
    cb_recs = content_based(ratings_df, genre_df, target)
    lines.append(f"  Recommendations (based on genre profile):")
    for movie, score in cb_recs:
        lines.append(f"    score={score:.3f}  {movie}")

    # ── 5. Method comparison ──
    lines += ["", "  ── 5. Comparison of Approaches ──", ""]
    lines.append(f"  {'Method':35s} | Top recommendation")
    lines.append(f"  {'-'*35}-+--{'─'*25}")
    if uu_recs: lines.append(f"  {'User-User CF':35s} | {uu_recs[0][0]} ({uu_recs[0][1]:.2f})")
    if ii_recs: lines.append(f"  {'Item-Item CF':35s} | {ii_recs[0][0]} ({ii_recs[0][1]:.2f})")
    if svd_recs: lines.append(f"  {'Matrix Factorisation (SVD)':35s} | {svd_recs[0][0]} ({svd_recs[0][1]:.2f})")
    if cb_recs: lines.append(f"  {'Content-Based':35s} | {cb_recs[0][0]} ({cb_recs[0][1]:.3f})")

    # ── Heatmap of ratings ──
    fig, ax = plt.subplots(figsize=(14, 8))
    R_display = ratings_df.values[:15].copy()
    R_display[R_display == 0] = np.nan
    im = ax.imshow(R_display, cmap="RdYlGn", aspect="auto", vmin=1, vmax=5)
    ax.set_xticks(range(ratings_df.shape[1]))
    ax.set_xticklabels(ratings_df.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(15))
    ax.set_yticklabels([f"{ratings_df.index[i]} ({user_archetypes[i][:6]})" for i in range(15)], fontsize=8)
    ax.set_title("User-Movie Rating Matrix (first 15 users, grey = unrated)")
    plt.colorbar(im, ax=ax, label="Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rating_matrix.png"), dpi=150)
    plt.close()
    lines.append(f"\n  [saved] → plots/rating_matrix.png")

    # user embedding from SVD
    fig, ax = plt.subplots(figsize=(8, 6))
    arch_colors = {"sci_fi_fan": "blue", "romance_fan": "red", "action_fan": "green",
                   "animation_fan": "orange", "crime_fan": "purple"}
    for i, arch in enumerate(user_archetypes):
        ax.scatter(U_k[i, 0], U_k[i, 1], c=arch_colors[arch], s=50, alpha=0.7,
                   label=arch if i < 5 else "")
    ax.set_xlabel("Latent Factor 1")
    ax.set_ylabel("Latent Factor 2")
    ax.set_title("User Embeddings from SVD (users with similar taste cluster)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "user_embeddings.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] → plots/user_embeddings.png")

    lines += [
        "", "  ── Key Takeaways ──",
        "    • User-User CF: good for small user bases, captures taste directly",
        "    • Item-Item CF: scales better (items < users), more stable over time",
        "    • Matrix Factorisation: best accuracy, discovers latent factors, Netflix Prize winner",
        "    • Content-Based: no cold-start for items (uses features), but limited discovery",
        "    • Hybrid: production systems combine multiple signals (Netflix, Spotify, YouTube)",
        "",
        "  ── Cold Start Problem ──",
        "    • New user: no ratings → CF fails. Use content-based or popularity fallback.",
        "    • New item: no ratings → CF fails. Use content features or explore/exploit.",
        "", "=" * 70,
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    main()
