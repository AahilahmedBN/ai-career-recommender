"""
============================================================
  HYBRID AI-BASED CAREER RECOMMENDATION SYSTEM
  Clustering + Collaborative Filtering + Neural Networks
============================================================
  Author  : [Your Name]
  Subject : Machine Learning Lab Project
  Tools   : Python, scikit-learn, pandas, matplotlib
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

np.random.seed(42)

# ============================================================
#  SECTION 1 — DATASET
#  700 students, 14 careers, 7 interest domains.
#  Skill ranges are TIGHTLY defined per career so the MLP
#  can learn clear boundaries between careers.
# ============================================================

print("=" * 62)
print("   HYBRID AI-BASED CAREER RECOMMENDATION SYSTEM")
print("=" * 62)
print("\n[1/6] Generating synthetic student dataset...")

# interest → list of careers in that domain
DOMAIN_CAREERS = {
    "AI":         ["AI/ML Engineer",       "Data Scientist"],
    "Tech":       ["Software Developer",   "Cybersecurity Analyst"],
    "Design":     ["UX/UI Designer",       "Graphic Designer"],
    "Business":   ["Product Manager",      "Business Analyst",    "Marketing Specialist"],
    "Science":    ["Research Scientist",   "Biomedical Engineer"],
    "Healthcare": ["Clinical Psychologist","Nutritionist"],
    "Media":      ["Content Creator",      "Journalist"],
}

# Tight, distinct score profiles per career
# Keys: math, creativity, communication, technical, leadership
CAREER_PROFILES = {
    # ── AI ────────────────────────────────────────────────
    "AI/ML Engineer":        {"math":(8,10),"creativity":(3,5), "communication":(2,5),"technical":(8,10),"leadership":(1,4)},
    "Data Scientist":        {"math":(8,10),"creativity":(4,6), "communication":(5,7),"technical":(7,9), "leadership":(3,5)},
    # ── Tech ──────────────────────────────────────────────
    "Software Developer":    {"math":(5,8), "creativity":(3,5), "communication":(2,5),"technical":(8,10),"leadership":(1,4)},
    "Cybersecurity Analyst": {"math":(6,8), "creativity":(3,5), "communication":(3,6),"technical":(8,10),"leadership":(2,5)},
    # ── Design ────────────────────────────────────────────
    "UX/UI Designer":        {"math":(2,5), "creativity":(8,10),"communication":(7,9),"technical":(5,7), "leadership":(3,6)},
    "Graphic Designer":      {"math":(1,4), "creativity":(9,10),"communication":(3,6),"technical":(2,5), "leadership":(1,4)},
    # ── Business ──────────────────────────────────────────
    "Product Manager":       {"math":(5,7), "creativity":(6,8), "communication":(8,10),"technical":(4,6),"leadership":(8,10)},
    "Business Analyst":      {"math":(7,9), "creativity":(2,5), "communication":(7,9),"technical":(4,7), "leadership":(6,8)},
    "Marketing Specialist":  {"math":(2,5), "creativity":(7,9), "communication":(9,10),"technical":(1,4),"leadership":(7,9)},
    # ── Science ───────────────────────────────────────────
    "Research Scientist":    {"math":(9,10),"creativity":(4,6), "communication":(2,5),"technical":(7,9), "leadership":(1,3)},
    "Biomedical Engineer":   {"math":(8,10),"creativity":(4,6), "communication":(3,6),"technical":(7,9), "leadership":(2,5)},
    # ── Healthcare ────────────────────────────────────────
    "Clinical Psychologist": {"math":(4,7), "creativity":(5,7), "communication":(9,10),"technical":(1,4),"leadership":(5,7)},
    "Nutritionist":          {"math":(5,7), "creativity":(5,7), "communication":(8,10),"technical":(2,5),"leadership":(4,6)},
    # ── Media ─────────────────────────────────────────────
    "Content Creator":       {"math":(1,4), "creativity":(8,10),"communication":(9,10),"technical":(3,6),"leadership":(3,6)},
    "Journalist":            {"math":(3,6), "creativity":(7,9), "communication":(9,10),"technical":(1,4),"leadership":(4,7)},
}

# Build interest lookup: career → domain
CAREER_TO_DOMAIN = {}
for domain, careers in DOMAIN_CAREERS.items():
    for c in careers:
        CAREER_TO_DOMAIN[c] = domain

records = []
for career, p in CAREER_PROFILES.items():
    domain = CAREER_TO_DOMAIN[career]
    for _ in range(50):
        records.append({
            "interest":      domain,
            "math_score":    round(np.clip(np.random.normal(np.mean(p["math"]),        0.6), 0, 10), 1),
            "creativity":    round(np.clip(np.random.normal(np.mean(p["creativity"]),  0.6), 0, 10), 1),
            "communication": round(np.clip(np.random.normal(np.mean(p["communication"]),0.6),0, 10), 1),
            "technical":     round(np.clip(np.random.normal(np.mean(p["technical"]),   0.6), 0, 10), 1),
            "leadership":    round(np.clip(np.random.normal(np.mean(p["leadership"]),  0.6), 0, 10), 1),
            "career":        career,
        })

df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

VALID_INTERESTS = sorted(DOMAIN_CAREERS.keys())
SKILL_COLS      = ["math_score","creativity","communication","technical","leadership"]

print(f"   ✓ Dataset         : {len(df)} records, {len(CAREER_PROFILES)} careers")
print(f"   ✓ Interest domains: {', '.join(VALID_INTERESTS)}")

# ============================================================
#  SECTION 2 — PREPROCESSING
#  Skills only (no interest column in feature matrix).
#  Interest is used to ROUTE the student to the right
#  per-domain MLP — not as a model input.
# ============================================================

print("\n[2/6] Preprocessing data...")

le_career = LabelEncoder()
df["career_enc"] = le_career.fit_transform(df["career"])

# Global scaler (fit on all 700 rows, same scale for everyone)
scaler   = MinMaxScaler()
X_all    = scaler.fit_transform(df[SKILL_COLS])
df_scaled = pd.DataFrame(X_all, columns=SKILL_COLS)
df_scaled["career"]   = df["career"].values
df_scaled["interest"] = df["interest"].values

print(f"   ✓ MinMaxScaler    : 5 skill features → [0, 1]")
print(f"   ✓ Interest used as routing key (not model input)")

# ============================================================
#  SECTION 3 — CLUSTERING (K-Means)
#  K=7 (one cluster per domain). Trained on skills only.
#  Elbow Method used to justify K.
# ============================================================

print("\n[3/6] Running K-Means Clustering...")

inertias = []
K_range  = range(2, 15)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_all)
    inertias.append(km.inertia_)

K_FINAL       = 7
kmeans        = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_all)

# Name each cluster by dominant interest domain + fixed archetype label
# This ensures an AI student with high communication doesn't get labeled
# "Business / Product" just because their skill shape resembles that cluster.
DOMAIN_ARCHETYPES = {
    "AI":         "Analytical Innovator",
    "Tech":       "Technical Builder",
    "Design":     "Creative Designer",
    "Business":   "Strategic Leader",
    "Science":    "Research Mind",
    "Healthcare": "Care Professional",
    "Media":      "Content Communicator",
}
cluster_names = {}
for c in range(K_FINAL):
    sub     = df[df["cluster"] == c]
    dom_int = sub["interest"].mode()[0]
    cluster_names[c] = f"{dom_int} — {DOMAIN_ARCHETYPES[dom_int]}"

df["cluster_name"] = df["cluster"].map(cluster_names)

print(f"   ✓ K={K_FINAL} clusters selected via Elbow Method")
for c, name in cluster_names.items():
    print(f"     Cluster {c} [{name}]: {(df['cluster']==c).sum()} students")

# ============================================================
#  SECTION 4 — RECOMMENDATION ENGINE (Collaborative Filtering)
#  Cosine similarity on skill vectors only.
#  Strictly filtered to same interest domain → guarantees
#  recommendations are always within the chosen domain.
# ============================================================

print("\n[4/6] Building Recommendation Engine...")

def recommend_careers(user_skills_scaled, user_interest, top_n=3):
    """
    Finds top_n careers by cosine similarity.
    Only considers students with the same interest domain.
    Falls back to closest-skill match within domain if needed.
    """
    # Filter to same domain
    mask        = df["interest"] == user_interest
    domain_rows = df[mask].index.tolist()
    domain_X    = X_all[domain_rows]

    sims        = cosine_similarity([user_skills_scaled], domain_X)[0]
    top_idx     = np.argsort(sims)[::-1][:50]
    top_careers = df.loc[[domain_rows[i] for i in top_idx], "career"]
    voted       = top_careers.value_counts().head(top_n).index.tolist()

    # If domain is small (e.g. Healthcare has only 2 careers), pad with rest
    if len(voted) < top_n:
        all_domain_careers = df[mask]["career"].unique().tolist()
        for c in all_domain_careers:
            if c not in voted:
                voted.append(c)
            if len(voted) == top_n:
                break

    return voted[:top_n]

print(f"   ✓ CF strictly domain-locked — only recommends within chosen interest")

# ============================================================
#  SECTION 5 — PER-DOMAIN MLP (Neural Network)
#  KEY FIX: Instead of one global 14-class MLP, we train a
#  separate MLP for each interest domain.
#
#  Why this works:
#    A Science student's scores are compared only against
#    Research Scientist and Biomedical Engineer profiles.
#    The MLP never sees Business or Media careers at all.
#    This eliminates cross-domain contamination completely.
#
#  Architecture per domain MLP:
#    Input(5) → Dense(64) → Dense(32) → Output(N_careers_in_domain)
# ============================================================

print("\n[5/6] Training per-domain MLP Neural Networks...")

domain_mlps    = {}   # domain → fitted MLPClassifier
domain_encoders = {}  # domain → LabelEncoder for careers in that domain

overall_correct = 0
overall_total   = 0

for domain in VALID_INTERESTS:
    mask     = df["interest"] == domain
    X_dom    = X_all[mask.values]
    careers  = df[mask]["career"].values

    le_dom   = LabelEncoder()
    y_dom    = le_dom.fit_transform(careers)
    domain_encoders[domain] = le_dom

    n_classes = len(le_dom.classes_)

    if n_classes < 2:
        # Only 1 career in domain — no need for classifier
        domain_mlps[domain] = None
        print(f"   ✓ {domain:<12}: {n_classes} career — skipped (single class)")
        continue

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_dom, y_dom, test_size=0.2, random_state=42, stratify=y_dom
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate_init=0.001,
        alpha=0.001,
    )
    mlp.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, mlp.predict(X_te))
    domain_mlps[domain] = mlp
    overall_correct += int(acc * len(y_te))
    overall_total   += len(y_te)

    print(f"   ✓ {domain:<12}: {n_classes} careers | Accuracy: {acc*100:.0f}%  "
          f"({', '.join(le_dom.classes_)})")

overall_acc = overall_correct / overall_total * 100
print(f"\n   ✓ Overall weighted accuracy: {overall_acc:.1f}%")

# ============================================================
#  SECTION 6 — INTERACTIVE STUDENT INPUT LOOP
# ============================================================

def get_score(prompt):
    while True:
        try:
            v = float(input(prompt))
            if 0 <= v <= 10:
                return round(v, 1)
            print("      ✗ Enter a value between 0 and 10.")
        except ValueError:
            print("      ✗ Invalid. Enter a number like 7 or 8.5")

def get_interest():
    """Case-insensitive interest input."""
    lookup = {k.lower(): k for k in VALID_INTERESTS}
    hint   = " / ".join(VALID_INTERESTS)
    while True:
        raw = input(f"  Interest ({hint}) : ").strip().lower()
        if raw == "exit":
            return "exit"
        if raw in lookup:
            return lookup[raw]
        print(f"      ✗ Choose one of: {', '.join(VALID_INTERESTS)}")

# ── Skill-fit checker ────────────────────────────────────
# CAREER_PROFILES already defined above with (min, max) per skill.
# We check if the student meets the minimum threshold for each career.
# Threshold = lower bound of the profile range (the minimum requirement).
# If a key skill is below minimum, we flag it.

SKILL_KEYS = ["math_score", "creativity", "communication", "technical", "leadership"]
PROFILE_SKILL_KEYS = ["math", "creativity", "communication", "technical", "leadership"]

def check_skill_fit(user_raw_scores, career_name):
    """
    Returns (fit: bool, warnings: list of strings).
    A career is 'fit' if the student meets the minimum for every skill.
    Minimum = lower bound of that career's profile range.
    """
    profile  = CAREER_PROFILES[career_name]
    warnings_out = []
    fit = True
    for sk, pk in zip(SKILL_KEYS, PROFILE_SKILL_KEYS):
        min_req = profile[pk][0]          # lower bound of profile range
        actual  = user_raw_scores[sk]
        gap     = min_req - actual
        if gap > 1.5:                     # tolerance of 1.5 points
            warnings_out.append(
                f"{sk.replace('_score','').replace('_',' ').title()}: "
                f"you have {actual:.1f}, career needs ~{min_req:.0f}+"
            )
            fit = False
    return fit, warnings_out

def run_pipeline(user):
    # Scale skills
    raw_skills  = np.array([[user["math_score"], user["creativity"],
                              user["communication"], user["technical"],
                              user["leadership"]]])
    scaled      = scaler.transform(raw_skills)[0]
    domain      = user["interest"]

    # A — Cluster
    cid         = kmeans.predict([scaled])[0]
    # Use the student's actual chosen domain for the archetype label.
    # K-Means clusters on raw skill shape and can cross domains
    # (e.g. high-communication AI student lands in Business cluster).
    # The domain archetype always reflects what the student said they want.
    cname       = f"{domain} — {DOMAIN_ARCHETYPES[domain]}"

    # B — CF (domain-locked)
    recs        = recommend_careers(scaled, domain)

    # C — Per-domain MLP
    mlp         = domain_mlps[domain]
    le_dom      = domain_encoders[domain]

    if mlp is not None:
        proba   = mlp.predict_proba([scaled])[0]
        top_idx = np.argsort(proba)[::-1]
        mlp_results = [(le_dom.classes_[i], proba[i]*100) for i in top_idx]
    else:
        # Single-career domain
        mlp_results = [(le_dom.classes_[0], 100.0)]

    # ── Print ─────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)

    print(f"\n  ► Student Archetype  (K-Means Cluster)")
    print(f"      Cluster {cid} — {cname}")

    print(f"\n  ► Top Recommendations  (Collaborative Filtering)")
    for i, c in enumerate(recs, 1):
        fit, skill_gaps = check_skill_fit(user, c)
        tag = "" if fit else "  ⚠ skill gap"
        print(f"      {i}. {c}{tag}")
        if not fit:
            for gap in skill_gaps:
                print(f"           ↳ {gap}")

    print(f"\n  ► Career Success Probabilities  (MLP — {domain} domain)")
    for name, pct in mlp_results:
        fit, skill_gaps = check_skill_fit(user, name)
        bar = "█" * int(pct / 4)
        fit_tag = "" if fit else "  ⚠"
        print(f"      {name:<25} {pct:5.1f}%  {bar}{fit_tag}")

    # Best match = highest MLP probability with no skill gap (if possible)
    best_fit = next(
        ((n, p) for n, p in mlp_results if check_skill_fit(user, n)[0]),
        mlp_results[0]   # fallback to top if none fit
    )
    best_name, best_pct = best_fit
    _, best_gaps = check_skill_fit(user, best_name)

    print("\n" + "─" * 62)
    print(f"  BEST MATCH  : {best_name}")
    print(f"  CONFIDENCE  : {best_pct:.1f}%")
    print(f"  ARCHETYPE   : {cname}")
    if best_gaps:
        print(f"\n  ⚠  SKILL GAPS TO WORK ON:")
        for g in best_gaps:
            print(f"     • {g}")
    print("=" * 62)

    return scaled

# ── Main Loop ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  [6/6]  CAREER RECOMMENDATION — LIVE INPUT")
print("=" * 62)
print(f"  Domains  : {' / '.join(VALID_INTERESTS)}")
print( "  Scoring  : 0 (none) → 10 (expert)  |  'exit' to quit\n")

last_vec = None

while True:
    print("\n  ── New Student ──────────────────────────────────────")
    domain = get_interest()
    if domain == "exit":
        print("\n  Generating dashboard...\n")
        break

    user = {
        "interest":      domain,
        "math_score":    get_score("  Math / Analytical    (0-10) : "),
        "creativity":    get_score("  Creativity           (0-10) : "),
        "communication": get_score("  Communication        (0-10) : "),
        "technical":     get_score("  Technical / Coding   (0-10) : "),
        "leadership":    get_score("  Leadership           (0-10) : "),
    }

    last_vec = run_pipeline(user)

    again = input("\n  ► Another student? (yes / no) : ").strip().lower()
    if again not in ["yes", "y"]:
        print("\n  Generating dashboard...\n")
        break

if last_vec is None:
    last_vec = scaler.transform([[9, 5, 4, 9, 3]])[0]

# ============================================================
#  SECTION 7 — VISUALIZATION DASHBOARD
# ============================================================

print("  Building visualization dashboard...")

fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

ACCENT = "#00d4aa"
COLORS = ["#00d4aa","#ff6b6b","#ffd93d","#6bcb77","#4d96ff","#ff9f43","#a29bfe"]
BG, TEXT, GRID_C = "#161b22", "#e6edf3", "#21262d"
plt.rcParams.update({"text.color":TEXT,"axes.labelcolor":TEXT,
                     "xtick.color":TEXT,"ytick.color":TEXT})

# A — Elbow Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(BG)
ax1.plot(list(K_range), inertias, color=ACCENT, lw=2.5, marker="o",
         markerfacecolor="white", markeredgecolor=ACCENT, ms=7)
ax1.axvline(x=K_FINAL, color="#ff6b6b", ls="--", alpha=0.8, lw=1.5, label=f"K={K_FINAL}")
ax1.set_title("Elbow Method — Optimal K", color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("K"); ax1.set_ylabel("Inertia (WCSS)")
ax1.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT)
ax1.grid(color=GRID_C, ls="--", alpha=0.5)
for sp in ax1.spines.values(): sp.set_edgecolor(GRID_C)

# B — PCA Cluster Scatter
ax2 = fig.add_subplot(gs[0, 1:])
ax2.set_facecolor(BG)
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_all)
for cid, col in zip(range(K_FINAL), COLORS):
    m = df["cluster"] == cid
    ax2.scatter(X_pca[m,0], X_pca[m,1], c=col,
                label=cluster_names[cid], alpha=0.5, s=30, edgecolors="none")
up = pca.transform([last_vec])
ax2.scatter(up[0,0], up[0,1], c="white", s=300, marker="*",
            zorder=6, edgecolors=ACCENT, lw=2, label="★ You")
ax2.set_title("K-Means Clusters — PCA 2D Projection",
              color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax2.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT, fontsize=7, loc="upper right")
ax2.grid(color=GRID_C, ls="--", alpha=0.4)
for sp in ax2.spines.values(): sp.set_edgecolor(GRID_C)

# C — Career Distribution per Cluster
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_facecolor(BG)
cc  = df.groupby(["cluster_name","career"]).size().unstack(fill_value=0)
bot = np.zeros(len(cc))
tc  = plt.cm.tab20(np.linspace(0, 1, len(cc.columns)))
for col, c in zip(cc.columns, tc):
    ax3.bar(cc.index, cc[col], bottom=bot, color=c, label=col, alpha=0.85)
    bot += cc[col].values
ax3.set_title("Career Distribution Across Clusters",
              color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax3.set_xlabel("Cluster"); ax3.set_ylabel("Students")
ax3.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT, fontsize=6.5,
           loc="upper right", ncol=2)
ax3.tick_params(axis="x", rotation=25)
ax3.grid(axis="y", color=GRID_C, ls="--", alpha=0.5)
for sp in ax3.spines.values(): sp.set_edgecolor(GRID_C)

# D — MLP probabilities for last student
# Re-derive interest from last vec for display
# Just show global top-6 career softmax by running all domain MLPs
all_probs = {}
for dom, mlp_d in domain_mlps.items():
    if mlp_d is None:
        le_d  = domain_encoders[dom]
        all_probs[le_d.classes_[0]] = 1.0 / len(CAREER_PROFILES)
        continue
    le_d   = domain_encoders[dom]
    pv     = mlp_d.predict_proba([last_vec])[0]
    for i, p in enumerate(pv):
        all_probs[le_d.classes_[i]] = p / len(DOMAIN_CAREERS[dom])

sorted_careers = sorted(all_probs, key=all_probs.get, reverse=True)[:7]
bval = [all_probs[c]*100 for c in sorted_careers]
bcol = [ACCENT] + ["#4d96ff"] * (len(sorted_careers)-1)

ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor(BG)
bars = ax4.barh(sorted_careers[::-1], bval[::-1], color=bcol[::-1],
                edgecolor="none", height=0.6)
for b, v in zip(bars, bval[::-1]):
    ax4.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
             f"{v:.1f}%", va="center", ha="left", fontsize=8, color=TEXT)
ax4.set_title("MLP Career Probabilities (Last Student)",
              color=TEXT, fontsize=11, fontweight="bold", pad=10)
ax4.set_xlabel("Probability (%)"); ax4.set_xlim(0, max(bval)+15)
ax4.grid(axis="x", color=GRID_C, ls="--", alpha=0.5)
for sp in ax4.spines.values(): sp.set_edgecolor(GRID_C)

fig.suptitle("HYBRID AI-BASED CAREER RECOMMENDATION SYSTEM",
             fontsize=16, fontweight="bold", color=ACCENT, y=0.98)

plt.savefig("career_recommendation_output.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("  ✓ Dashboard saved as 'career_recommendation_output.png'")
print("  ✓ All done.\n")