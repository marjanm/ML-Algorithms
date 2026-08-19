"""
Cold Start Strategy Demo
==========================
How to handle new users/items with zero interaction history.

Part 1 — The Problem:
    Collaborative filtering fails completely for new users (no ratings to
    find similar users from). Show RMSE = very high for cold users.

Part 2 — Strategies:
    a) Popularity fallback — recommend the most popular items
    b) Content-based bootstrapping — use item features instead of ratings
    c) Hybrid — blend CF scores (when available) with content scores

Part 3 — Onboarding Simulation:
    Track how fast each strategy improves as the user rates 1, 3, 5, 10 items.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output_cold_start.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def create_dataset():
    """Synthetic movie dataset with ratings + content features."""
    np.random.seed(42)

    n_users = 50
    n_items = 30
    n_genres = 5
    genre_names = ["Action", "Comedy", "Drama", "SciFi", "Romance"]

    # Item content features (genre memberships, 0-1)
    item_features = np.random.dirichlet(np.ones(n_genres), size=n_items)

    # User preferences (latent genre preferences)
    user_prefs = np.random.dirichlet(np.ones(n_genres) * 0.5, size=n_users)

    # Generate ratings: users rate items based on preference alignment + noise
    ratings = np.zeros((n_users, n_items))
    mask = np.zeros((n_users, n_items), dtype=bool)  # which ratings exist

    for u in range(n_users):
        n_rated = np.random.randint(5, 20)
        rated_items = np.random.choice(n_items, n_rated, replace=False)
        for i in rated_items:
            alignment = np.dot(user_prefs[u], item_features[i])
            rating = 1 + 4 * alignment + np.random.normal(0, 0.5)
            ratings[u, i] = np.clip(rating, 1, 5)
            mask[u, i] = True

    item_names = [f"Movie_{i}" for i in range(n_items)]
    return ratings, mask, item_features, user_prefs, genre_names, item_names


def popularity_recommend(ratings, mask, n_rec=5):
    """Recommend the most popular (highest avg rating) items."""
    avg_ratings = np.zeros(ratings.shape[1])
    for i in range(ratings.shape[1]):
        rated = ratings[mask[:, i], i]
        avg_ratings[i] = rated.mean() if len(rated) > 0 else 0
    return np.argsort(-avg_ratings)[:n_rec], avg_ratings


def content_based_recommend(user_ratings, user_mask, item_features, n_rec=5):
    """Recommend based on item content similarity to user's liked items."""
    if user_mask.sum() == 0:
        return np.array([]), np.zeros(item_features.shape[0])

    # Build user profile from rated items (weighted by rating)
    rated_idx = np.where(user_mask)[0]
    weights = user_ratings[rated_idx] - 3  # center around 3
    user_profile = np.average(item_features[rated_idx], axis=0, weights=np.abs(weights) + 0.1)
    user_profile = user_profile.reshape(1, -1)

    # Score all items by similarity to user profile
    scores = cosine_similarity(user_profile, item_features)[0]
    scores[user_mask] = -1  # exclude already rated
    return np.argsort(-scores)[:n_rec], scores


def cf_recommend(ratings, mask, user_idx, n_rec=5):
    """User-user collaborative filtering."""
    user_ratings = ratings[user_idx]
    user_mask = mask[user_idx]

    if user_mask.sum() < 2:
        return np.array([]), np.zeros(ratings.shape[1])

    # Find similar users
    sims = np.zeros(ratings.shape[0])
    for u in range(ratings.shape[0]):
        if u == user_idx:
            continue
        common = user_mask & mask[u]
        if common.sum() < 2:
            continue
        r1 = ratings[user_idx, common]
        r2 = ratings[u, common]
        if r1.std() == 0 or r2.std() == 0:
            continue
        sims[u] = np.corrcoef(r1, r2)[0, 1]

    # Predict ratings from top-k similar users
    top_k = 10
    top_users = np.argsort(-sims)[:top_k]
    top_sims = sims[top_users]

    pred_scores = np.zeros(ratings.shape[1])
    for i in range(ratings.shape[1]):
        if user_mask[i]:
            continue
        weighted_sum = 0
        sim_sum = 0
        for u, s in zip(top_users, top_sims):
            if s > 0 and mask[u, i]:
                weighted_sum += s * ratings[u, i]
                sim_sum += s
        pred_scores[i] = weighted_sum / sim_sum if sim_sum > 0 else 0

    pred_scores[user_mask] = -1
    return np.argsort(-pred_scores)[:n_rec], pred_scores


def evaluate_recommendations(recs, true_prefs, item_features):
    """How well do recommendations match the user's latent preferences?"""
    if len(recs) == 0:
        return 0.0
    rec_features = item_features[recs]
    alignment = np.array([np.dot(true_prefs, f) for f in rec_features])
    return alignment.mean()


def run_cold_start_demo():
    log("COLD START STRATEGY DEMO")
    log("=" * 60)

    ratings, mask, item_features, user_prefs, genre_names, item_names = create_dataset()

    log(f"\n  Dataset: {ratings.shape[0]} users × {ratings.shape[1]} items")
    log(f"  Sparsity: {1 - mask.mean():.1%} missing")
    log(f"  Genres: {', '.join(genre_names)}")

    # ═══════════════════════════════════════════════════════
    # Part 1: The Problem
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 1: COLD START FAILURE")
    log("=" * 60)

    # Simulate a new user (no ratings)
    new_user_prefs = user_prefs[0]  # we know their true preferences
    new_user_ratings = np.zeros(ratings.shape[1])
    new_user_mask = np.zeros(ratings.shape[1], dtype=bool)

    cf_recs, _ = cf_recommend(ratings, mask, 0)
    log(f"\n  New user with 0 ratings:")
    log(f"  CF recommendation: {'FAILS — no data to find similar users' if len(cf_recs) == 0 else cf_recs}")

    # Existing user with many ratings
    rich_user = np.argmax(mask.sum(axis=1))
    cf_recs_rich, _ = cf_recommend(ratings, mask, rich_user)
    rich_alignment = evaluate_recommendations(cf_recs_rich, user_prefs[rich_user], item_features)
    log(f"\n  Existing user with {mask[rich_user].sum()} ratings:")
    log(f"  CF top-5: {cf_recs_rich}")
    log(f"  Preference alignment: {rich_alignment:.3f}")

    # ═══════════════════════════════════════════════════════
    # Part 2: Strategies
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 2: COLD START STRATEGIES")
    log("=" * 60)

    # Strategy A: Popularity
    pop_recs, pop_scores = popularity_recommend(ratings, mask, n_rec=5)
    pop_alignment = evaluate_recommendations(pop_recs, new_user_prefs, item_features)
    log(f"\n  A) Popularity fallback:")
    log(f"     Top-5: {[item_names[i] for i in pop_recs]}")
    log(f"     Alignment: {pop_alignment:.3f}")
    log(f"     → Works for everyone but not personalized")

    # Strategy B: Content-based (after 3 ratings)
    # Simulate user rates 3 items they'd naturally like
    sim_rated_items = np.argsort(-item_features @ new_user_prefs)[:3]
    sim_ratings = np.zeros(ratings.shape[1])
    sim_mask = np.zeros(ratings.shape[1], dtype=bool)
    for i in sim_rated_items:
        sim_ratings[i] = 4.0 + np.random.normal(0, 0.3)
        sim_mask[i] = True

    cb_recs, cb_scores = content_based_recommend(sim_ratings, sim_mask, item_features, n_rec=5)
    cb_alignment = evaluate_recommendations(cb_recs, new_user_prefs, item_features)
    log(f"\n  B) Content-based (after rating {sim_mask.sum()} items):")
    log(f"     Top-5: {[item_names[i] for i in cb_recs]}")
    log(f"     Alignment: {cb_alignment:.3f}")
    log(f"     → Uses item features, doesn't need other users")

    # Strategy C: Hybrid
    log(f"\n  C) Hybrid (blend popularity + content):")
    alpha = 0.6  # weight for content
    hybrid_scores = alpha * cb_scores + (1 - alpha) * (pop_scores / pop_scores.max())
    hybrid_scores[sim_mask] = -1
    hybrid_recs = np.argsort(-hybrid_scores)[:5]
    hybrid_alignment = evaluate_recommendations(hybrid_recs, new_user_prefs, item_features)
    log(f"     Top-5: {[item_names[i] for i in hybrid_recs]}")
    log(f"     Alignment: {hybrid_alignment:.3f}")

    log(f"\n  Summary:")
    log(f"  {'Strategy':<30} {'Alignment':>10}")
    log(f"  {'-' * 42}")
    log(f"  {'Popularity (0 ratings)':<30} {pop_alignment:>10.3f}")
    log(f"  {'Content-based (3 ratings)':<30} {cb_alignment:>10.3f}")
    log(f"  {'Hybrid (3 ratings)':<30} {hybrid_alignment:>10.3f}")

    # ═══════════════════════════════════════════════════════
    # Part 3: Onboarding — How Fast Do Strategies Improve?
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 3: ONBOARDING — IMPROVEMENT WITH MORE RATINGS")
    log("=" * 60)

    n_ratings_steps = [0, 1, 2, 3, 5, 8, 12, 15]

    # Items sorted by how much the user would like them
    preferred_order = np.argsort(-item_features @ new_user_prefs)

    pop_curve = []
    cb_curve = []
    cf_curve = []

    for n_rated in n_ratings_steps:
        # Simulate rating n items
        rated_items = preferred_order[:n_rated] if n_rated > 0 else []
        sim_r = np.zeros(ratings.shape[1])
        sim_m = np.zeros(ratings.shape[1], dtype=bool)
        for i in rated_items:
            sim_r[i] = 3.5 + np.random.normal(0, 0.5)
            sim_m[i] = True

        # Popularity (constant)
        p_recs, _ = popularity_recommend(ratings, mask, n_rec=5)
        pop_curve.append(evaluate_recommendations(p_recs, new_user_prefs, item_features))

        # Content-based
        if n_rated > 0:
            c_recs, _ = content_based_recommend(sim_r, sim_m, item_features, n_rec=5)
            cb_curve.append(evaluate_recommendations(c_recs, new_user_prefs, item_features))
        else:
            cb_curve.append(0)

        # CF (inject into rating matrix)
        if n_rated >= 2:
            temp_ratings = ratings.copy()
            temp_mask = mask.copy()
            new_uid = 0
            temp_ratings[new_uid] = sim_r
            temp_mask[new_uid] = sim_m
            c_recs_cf, _ = cf_recommend(temp_ratings, temp_mask, new_uid, n_rec=5)
            if len(c_recs_cf) > 0:
                cf_curve.append(evaluate_recommendations(c_recs_cf, new_user_prefs, item_features))
            else:
                cf_curve.append(0)
        else:
            cf_curve.append(0)

    log(f"\n  {'# Ratings':>10} | {'Popularity':>10} | {'Content':>10} | {'CF':>10}")
    log(f"  {'-' * 47}")
    for i, n in enumerate(n_ratings_steps):
        log(f"  {n:>10} | {pop_curve[i]:>10.3f} | {cb_curve[i]:>10.3f} | {cf_curve[i]:>10.3f}")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Strategy comparison bar chart
    ax = axes[0]
    strategies = ["Popularity\n(0 ratings)", "Content-Based\n(3 ratings)", "Hybrid\n(3 ratings)",
                  f"CF\n({mask[rich_user].sum()} ratings)"]
    alignments = [pop_alignment, cb_alignment, hybrid_alignment, rich_alignment]
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"]
    bars = ax.bar(strategies, alignments, color=colors, width=0.5)
    ax.set_ylabel("Preference Alignment (higher = better)")
    ax.set_title("Cold Start Strategies vs Established User")
    for b, a in zip(bars, alignments):
        ax.text(b.get_x() + b.get_width()/2, a + 0.005, f"{a:.3f}", ha="center", fontweight="bold")

    # Plot 2: Onboarding curve
    ax = axes[1]
    ax.plot(n_ratings_steps, pop_curve, "s--", label="Popularity", color="#95a5a6", linewidth=2)
    ax.plot(n_ratings_steps, cb_curve, "o-", label="Content-Based", color="#3498db", linewidth=2)
    ax.plot(n_ratings_steps, cf_curve, "^-", label="Collaborative Filtering", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Number of Ratings from New User")
    ax.set_ylabel("Recommendation Quality (alignment)")
    ax.set_title("How Fast Does Each Strategy Improve?")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cold_start.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/cold_start.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
Cold start types:
  • New user  — no interaction history → can't find similar users
  • New item  — no one has rated it → can't compute item-item similarity

Strategies:
  ┌─────────────────────────┬──────────────────────────────────────┐
  │ Popularity fallback     │ Safe default. Not personalized.      │
  │ Content-based bootstrap │ Uses item metadata. Works with 1-3   │
  │                         │ ratings. No other users needed.      │
  │ Hybrid                  │ Blend CF + content. Best overall.    │
  │ Onboarding questions    │ "Pick 5 genres you like" — Netflix   │
  │                         │ does this on signup.                 │
  │ Exploration (bandit)    │ Show diverse items to learn prefs    │
  │                         │ quickly. Explore-exploit tradeoff.   │
  └─────────────────────────┴──────────────────────────────────────┘

Real-world: Netflix's signup screen asks you to pick genres → instant
content-based profile → CF kicks in after ~20 ratings.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output_cold_start.txt")


if __name__ == "__main__":
    run_cold_start_demo()
