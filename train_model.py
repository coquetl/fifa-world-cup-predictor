"""
train_model.py — Match Result Predictor
Coupe du Monde 2026 — Modèle de prédiction ML
-----------------------------------------------
Stack : XGBoost natif + numpy + pandas + scipy + joblib
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import json

# Force UTF-8 pour la console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_MATCH  = os.path.join(BASE_DIR, "FIFA World Cup 2026 Match Data (Unofficial)")
DATA_HIST   = os.path.join(BASE_DIR, "International football results from 1872 to 2026")
DATA_PLAYER = os.path.join(BASE_DIR, "FIFA World Cup 2026 Player Performance Dataset")
DATA_BASE   = os.path.join(BASE_DIR, "WC2026 Match Probability Baseline Dataset")
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_PATH  = os.path.join(BASE_DIR, "model.joblib")
PRED_PATH   = os.path.join(DATA_DIR, "predictions.csv")

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("  MATCH RESULT PREDICTOR — Entraînement du modèle (Refactored)")
print("  Stack : XGBoost + numpy + Top 11 + Forme + Knockout")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
print("\n[1/5] Chargement des données...")

teams_df    = pd.read_csv(os.path.join(DATA_MATCH, "teams.csv"))
matches_df  = pd.read_csv(os.path.join(DATA_MATCH, "matches.csv"))
baseline_df = pd.read_csv(os.path.join(DATA_BASE, "future_match_probabilities_baseline.csv"))

# Chargement de l'historique filtré aux 10 dernières années
results_df = pd.read_csv(os.path.join(DATA_HIST, "results.csv"))
results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")
# 10 ans avant le 17 juin 2026 => 17 juin 2016
results_df = results_df[results_df["date"] >= "2016-06-17"].copy()

# Tirs au but pour détecter les match éliminatoires
shootouts_df = pd.read_csv(os.path.join(DATA_HIST, "shootouts.csv"))
shootouts_df["date"] = pd.to_datetime(shootouts_df["date"], errors="coerce")
shootout_keys = set()
for _, row in shootouts_df.iterrows():
    if not pd.isna(row["date"]):
        dt_str = row["date"].strftime("%Y-%m-%d")
        shootout_keys.add((dt_str, str(row["home_team"]), str(row["away_team"])))
        shootout_keys.add((dt_str, str(row["away_team"]), str(row["home_team"])))

print("   Chargement des stats joueurs (fichier ~17 Mo)...")
player_df = pd.read_csv(
    os.path.join(DATA_PLAYER, "fifa_world_cup_2026_player_performance.csv"),
    low_memory=False
)

print(f"   ✓ {len(results_df):,} matchs historiques (depuis 2016-06-17)")
print(f"   ✓ {len(baseline_df)} paires Elo baseline")
print(f"   ✓ {len(player_df):,} observations joueurs")

# ─────────────────────────────────────────────
# 2. AGRÉGATION DES STATS JOUEURS (TOP 11)
# ─────────────────────────────────────────────
print("\n[2/5] Agrégation des stats des joueurs sur le TOP 11 par équipe...")

# Trier les joueurs par équipe et note individuelle (player_rating) de manière décroissante
player_sorted = player_df.sort_values(by=["team", "player_rating"], ascending=[True, False])
# Conserver uniquement le Top 11 par équipe
top_11_players = player_sorted.groupby("team").head(11)

player_agg = top_11_players.groupby("team").agg(
    avg_rating      = ("player_rating",           "mean"),
    avg_xg          = ("expected_goals_xg",        "mean"),
    avg_top_speed   = ("top_speed_kmh",            "mean"),
    avg_stamina     = ("stamina_score",            "mean"),
    avg_pass_acc    = ("pass_accuracy",            "mean"),
    avg_offensive   = ("offensive_contribution",   "mean"),
    avg_defensive   = ("defensive_contribution",   "mean"),
    avg_creativity  = ("creativity_score",         "mean"),
    avg_clutch      = ("clutch_performance_score", "mean"),
).reset_index()

player_map = player_agg.set_index("team").to_dict("index")
print(f"   ✓ Stats agrégées (Top 11) pour {len(player_agg)} équipes")

# ─────────────────────────────────────────────
# 3. CONSTRUCTIONS DU DATASET D'ENTRAÎNEMENT CHRONOLOGIQUE
# ─────────────────────────────────────────────
print("\n[3/5] Calcul Elo et forme chronologiques...")

LABEL_MAP     = {"W": 2, "D": 1, "L": 0}
LABEL_MAP_INV = {2: "W", 1: "D", 0: "L"}

def outcome(hs, as_):
    if hs > as_: return "W"
    elif hs == as_: return "D"
    return "L"

TEAM_TRANSLATION_FR_TO_EN = {
    'Mexique': 'Mexico', 'Afrique du Sud': 'South Africa', 'République de Corée': 'South Korea', 'Tchéquie': 'Czech Republic',
    'Canada': 'Canada', 'Bosnie-Herzégovine': 'Bosnia and Herzegovina', 'Qatar': 'Qatar', 'Suisse': 'Switzerland',
    'Brésil': 'Brazil', 'Maroc': 'Morocco', 'Haïti': 'Haiti', 'Écosse': 'Scotland',
    'États-Unis': 'United States', 'Paraguay': 'Paraguay', 'Australie': 'Australia', 'Turquie': 'Turkey',
    'Allemagne': 'Germany', 'Curaçao': 'Curaçao', 'Côte d’Ivoire': 'Ivory Coast', 'Équateur': 'Ecuador',
    'Pays-Bas': 'Netherlands', 'Japon': 'Japan', 'Suède': 'Sweden', 'Tunisie': 'Tunisia',
    'Belgique': 'Belgium', 'Égypte': 'Egypt', 'Iran': 'Iran', 'Nouvelle-Zélande': 'New Zealand',
    'Espagne': 'Spain', 'Cap Vert': 'Cape Verde', 'Arabie Saoudite': 'Saudi Arabia', 'Uruguay': 'Uruguay',
    'France': 'France', 'Sénégal': 'Senegal', 'Irak': 'Iraq', 'Norvège': 'Norway',
    'Argentine': 'Argentina', 'Algérie': 'Algeria', 'Autriche': 'Austria', 'Jordanie': 'Jordan',
    'Portugal': 'Portugal', 'RD Congo': 'DR Congo', 'Ouzbékistan': 'Uzbekistan', 'Colombie': 'Colombia',
    'Angleterre': 'England', 'Croatie': 'Croatia', 'Ghana': 'Ghana', 'Panama': 'Panama'
}

def team_has_fotmob_stats(team_name):
    """Retourne True si l'équipe a des données FotMob réelles (non utilisé dans les features du modèle)."""
    eng_name = TEAM_TRANSLATION_FR_TO_EN.get(team_name, team_name)
    return eng_name in player_map

def is_knockout_match(row, shootout_keys):
    dt_str = row["date"].strftime("%Y-%m-%d")
    if (dt_str, row["home_team"], row["away_team"]) in shootout_keys:
        return 1.0
    
    tourney = str(row["tournament"])
    dt = row["date"]
    
    # Tournois majeurs : phase à élimination directe
    if tourney == "FIFA World Cup":
        if (pd.Timestamp("2018-06-30") <= dt <= pd.Timestamp("2018-07-15")) or \
           (pd.Timestamp("2022-12-03") <= dt <= pd.Timestamp("2022-12-18")):
            return 1.0
    elif tourney == "UEFA Euro":
        if (pd.Timestamp("2016-06-25") <= dt <= pd.Timestamp("2016-07-10")) or \
           (pd.Timestamp("2021-06-26") <= dt <= pd.Timestamp("2021-07-11")) or \
           (pd.Timestamp("2024-06-29") <= dt <= pd.Timestamp("2024-07-14")):
            return 1.0
    elif tourney == "Copa América":
        if (pd.Timestamp("2016-06-16") <= dt <= pd.Timestamp("2016-06-26")) or \
           (pd.Timestamp("2019-06-27") <= dt <= pd.Timestamp("2019-07-07")) or \
           (pd.Timestamp("2021-07-02") <= dt <= pd.Timestamp("2021-07-10")) or \
           (pd.Timestamp("2024-07-04") <= dt <= pd.Timestamp("2024-07-14")):
            return 1.0
            
    if "play-off" in tourney.lower():
        return 1.0
    return 0.0

# Trier l'historique chronologiquement pour calculer correctement l'Elo et la Forme
results_df = results_df.sort_values("date").copy()

elo_ratings = {}
tournament_stats = {} # (tourney, year, team) -> {"points": 0, "played": 0}
last_match_date = {} # team -> pd.Timestamp

def get_elo(t):
    return elo_ratings.get(t, 1500.0)

def exp_prob(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def get_confederation(country_name):
    if not country_name or not isinstance(country_name, str):
        return 'OTHER'
    c = country_name.lower().strip()
    if c in {
        'usa', 'united states', 'mexico', 'mexique', 'canada', 'panama', 'haiti', 'haïti', 
        'curacao', 'curaçao', 'costa rica', 'honduras', 'jamaica', 'el salvador', 
        'trinidad and tobago', 'guatemala', 'cuba', 'martinique', 'guadeloupe'
    }:
        return 'CONCACAF'
    if c in {
        'brazil', 'brésil', 'argentina', 'argentine', 'colombia', 'colombie', 
        'uruguay', 'ecuador', 'équateur', 'paraguay', 'chile', 'peru', 'venezuela', 'bolivia'
    }:
        return 'CONMEBOL'
    if c in {
        'germany', 'allemagne', 'france', 'spain', 'espagne', 'england', 'angleterre', 
        'italy', 'italie', 'netherlands', 'pays-bas', 'belgium', 'belgique', 
        'portugal', 'croatia', 'croatie', 'switzerland', 'suisse', 'turkey', 'turquie', 
        'sweden', 'suède', 'norway', 'norvège', 'austria', 'autriche', 'scotland', 'écosse', 
        'czech republic', 'tchéquie', 'bosnia and herzegovina', 'bosnie-herzégovine', 
        'wales', 'ukraine', 'poland', 'denmark', 'russia', 'greece', 'serbia'
    }:
        return 'UEFA'
    if c in {
        'south africa', 'afrique du sud', 'morocco', 'maroc', 'senegal', 'sénégal', 
        'egypt', 'égypte', 'tunisia', 'tunisie', 'algeria', 'algérie', 'ghana', 
        'dr congo', 'rd congo', 'cape verde', 'cap vert', 'cabo verde', 'ivory coast', 
        'côte d’ivoire', 'côte d\'ivoire', 'nigeria', 'cameroon', 'mali', 'guinea'
    }:
        return 'CAF'
    if c in {
        'japan', 'japon', 'south korea', 'république de corée', 'korea republic', 
        'iran', 'ir iran', 'saudi arabia', 'arabie saoudite', 'qatar', 'iraq', 'irak', 
        'australia', 'australie', 'jordan', 'jordanie', 'uzbekistan', 'ouzbékistan', 
        'china', 'syria', 'vietnam', 'thailand', 'oman', 'india'
    }:
        return 'AFC'
    if c in {'new zealand', 'nouvelle-zélande', 'fiji', 'tahiti', 'solomon islands'}:
        return 'OFC'
    return 'OTHER'

FEATURES = [
    "home_elo", "away_elo", "elo_diff", "is_neutral", "is_knockout",
    "home_form", "away_form", "form_diff",
    "h_rest", "a_rest", "h_travel", "a_travel"
]

rows = []
for _, row in results_df.iterrows():
    ht, at = row["home_team"], row["away_team"]
    hs, as_ = row["home_score"], row["away_score"]
    if pd.isna(hs) or pd.isna(as_):
        continue
    hs, as_ = int(hs), int(as_)
    
    # 1. Elo actuel
    eh, ea = get_elo(ht), get_elo(at)
    
    # 2. Forme intra-tournoi actuelle
    tourney = row["tournament"]
    year = row["date"].year
    
    h_stats = tournament_stats.get((tourney, year, ht), {"points": 0, "played": 0})
    a_stats = tournament_stats.get((tourney, year, at), {"points": 0, "played": 0})
    
    h_form = h_stats["points"] / h_stats["played"] if h_stats["played"] > 0 else 0.0
    a_form = a_stats["points"] / a_stats["played"] if a_stats["played"] > 0 else 0.0
    form_diff = h_form - a_form
    
    # 3. is_knockout
    is_ko = is_knockout_match(row, shootout_keys)
    
    # 4. Rest days
    date_curr = row["date"]
    h_prev_date = last_match_date.get(ht)
    a_prev_date = last_match_date.get(at)
    
    h_rest = (date_curr - h_prev_date).days if h_prev_date else 10.0
    a_rest = (date_curr - a_prev_date).days if a_prev_date else 10.0
    h_rest = float(min(h_rest, 10.0))
    a_rest = float(min(a_rest, 10.0))
    
    last_match_date[ht] = date_curr
    last_match_date[at] = date_curr
    
    # 5. Travel Factor
    host_country = row.get("country", "")
    host_conf = get_confederation(host_country)
    h_conf = get_confederation(ht)
    a_conf = get_confederation(at)
    h_travel = 1.0 if h_conf == host_conf and host_conf != 'OTHER' else 0.0
    a_travel = 1.0 if a_conf == host_conf and host_conf != 'OTHER' else 0.0
    
    label = LABEL_MAP[outcome(hs, as_)]
    
    rows.append([
        eh, ea, eh - ea, int(row["neutral"] == True or row["neutral"] == "TRUE"), is_ko,
        h_form, a_form, form_diff,
        h_rest, a_rest, h_travel, a_travel,
        label
    ])
    
    # Mettre à jour Elo
    sa = 1 if hs > as_ else (0.5 if hs == as_ else 0)
    sb = 1 - sa
    k = 32
    elo_ratings[ht] = eh + k * (sa - exp_prob(eh, ea))
    elo_ratings[at] = ea + k * (sb - exp_prob(ea, eh))
    
    # Mettre à jour Forme
    h_pts = 3 if hs > as_ else (1 if hs == as_ else 0)
    a_pts = 3 if hs < as_ else (1 if hs == as_ else 0)
    tournament_stats[(tourney, year, ht)] = {"points": h_stats["points"] + h_pts, "played": h_stats["played"] + 1}
    tournament_stats[(tourney, year, at)] = {"points": a_stats["points"] + a_pts, "played": a_stats["played"] + 1}

# Enrichissement avec le dataset de joueurs (matchs WC2026 d'entraînement)
# Note: le dataset d'entraînement WC2026 synthétique a été retiré car il créait un biais
# (les résultats fictifs corrompaient les prédictions pour les matchs avec has_stats=1)
# Le modèle entraîne uniquement sur l'historique réel de 9500+ matchs.


data = np.array(rows, dtype=np.float32)
np.random.seed(42)
np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1].astype(int)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"   ✓ Dataset d'entraînement : {len(X):,} matchs")

# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT XGBOOST
# ─────────────────────────────────────────────
print("\n[4/5] Entraînement XGBoost (multi:softprob)...")

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURES)

params = {
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "max_depth":        5,
    "eta":              0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "seed":             42,
    "nthread":          -1,
}

evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train"), (dtest, "test")],
    evals_result=evals_result,
    verbose_eval=100,
)

# Métriques
proba_test = model.predict(dtest).reshape(-1, 3)
y_pred     = np.argmax(proba_test, axis=1)
accuracy   = float(np.mean(y_pred == y_test))
print(f"\n   ✓ Accuracy test : {accuracy:.2%}")

# Rapport par classe
for label_str, label_int in LABEL_MAP.items():
    mask  = (y_test == label_int)
    if mask.sum() == 0:
        continue
    tp    = float(np.sum((y_pred == label_int) & mask))
    fp    = float(np.sum((y_pred == label_int) & ~mask))
    fn    = float(np.sum((y_pred != label_int) & mask))
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"   [{label_str}] Précision: {prec:.2%}  Rappel: {rec:.2%}  F1: {f1:.2%}")

# Sauvegarde globale dans model.joblib
joblib.dump({
    "model":            model,
    "features":         FEATURES,
    "label_map":        LABEL_MAP,
    "label_map_inv":    LABEL_MAP_INV,
    "accuracy":         accuracy,
    "X_train_hist":     X,  # Sauvegarde de tout le dataset historique d'entraînement
    "y_train_hist":     y,
    "player_map":       player_map,
    "base_elo_ratings": elo_ratings,
    "last_historical_match_dates": last_match_date,
}, MODEL_PATH)
print(f"\n   ✓ Modèle et données historiques sauvegardés → {MODEL_PATH}")

# ─────────────────────────────────────────────
# 5. GÉNÉRATION DES PRÉDICTIONS INITIALES WC2026
# ─────────────────────────────────────────────
print("\n[5/5] Génération des prédictions initiales...")

# Charger le mappage français
with open(os.path.join(DATA_DIR, "arbre_phases_finales.json"), encoding="utf-8") as f:
    arbre_final = json.load(f)

# Charger les poules initiales
poules_df = pd.read_csv(os.path.join(DATA_DIR, "matchs_poules.csv"))

# Remplir l'Elo baseline pour les prédictions
elo_baseline = {}
for _, row in baseline_df.iterrows():
    if not pd.isna(row.get("home_elo")):
        elo_baseline[row["home_team"]] = float(row["home_elo"])
    if not pd.isna(row.get("away_elo")):
        elo_baseline[row["away_team"]] = float(row["away_elo"])

def get_base_elo(team_name):
    eng_name = TEAM_TRANSLATION_FR_TO_EN.get(team_name, team_name)
    return elo_baseline.get(eng_name, elo_ratings.get(eng_name, 1500.0))

predictions = []
for _, row in poules_df.iterrows():
    ht = row["home_team"]
    at = row["away_team"]
    
    eh = elo_baseline.get(TEAM_TRANSLATION_FR_TO_EN.get(ht, ht), elo_ratings.get(TEAM_TRANSLATION_FR_TO_EN.get(ht, ht), 1500.0))
    ea = elo_baseline.get(TEAM_TRANSLATION_FR_TO_EN.get(at, at), elo_ratings.get(TEAM_TRANSLATION_FR_TO_EN.get(at, at), 1500.0))
    
    h_travel = 1.0 if get_confederation(TEAM_TRANSLATION_FR_TO_EN.get(ht, ht)) == "CONCACAF" else 0.0
    a_travel = 1.0 if get_confederation(TEAM_TRANSLATION_FR_TO_EN.get(at, at)) == "CONCACAF" else 0.0
    
    # Les formes initiales sont à 0.0, is_knockout = 0.0 (poules), rest_days = 10.0
    feat = np.array([[eh, ea, eh - ea, 1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, h_travel, a_travel]], dtype=np.float32)
    dm = xgb.DMatrix(feat, feature_names=FEATURES)
    prob = model.predict(dm)[0]
    
    p_home_win = round(float(prob[2]) * 100, 1)
    p_draw     = round(float(prob[1]) * 100, 1)
    p_away_win = round(float(prob[0]) * 100, 1)
    
    predictions.append({
        "match_id":        int(row["match_id"]),
        "match_number":    int(row["match_number"]),
        "stage":           row["stage"],
        "home_team":       ht,
        "home_code":       row["home_code"],
        "home_flag":       row["home_flag"],
        "away_team":       at,
        "away_code":       row["away_code"],
        "away_flag":       row["away_flag"],
        "group":           row["group"],
        "date":            row["date"],
        "home_elo":        round(eh, 0),
        "away_elo":        round(ea, 0),
        "p_home_win":      p_home_win,
        "p_draw":          p_draw,
        "p_away_win":      p_away_win,
        "insight":         "Match initial non joué. Le modèle prédit les forces en présence.",
        "home_score_real": "",
        "away_score_real": "",
        "result_real":     "",
        "shootout_winner": ""
    })

pd.DataFrame(predictions).to_csv(PRED_PATH, index=False, encoding="utf-8-sig")
print(f"   ✓ {len(predictions)} prédictions initiales enregistrées dans {PRED_PATH}")

# ─────────────────────────────────────────────
# 6. SAUVEGARDE DU GRAPHIQUE DES IMPORTANCES
# ─────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    print("\n[6/6] Génération du graphique d'importance des features...")
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, importance_type="weight", ax=ax, grid=False)
    plt.title("Importance des Caractéristiques (F-score)")
    plt.tight_layout()
    importance_img_path = os.path.join(BASE_DIR, "feature_importance.png")
    plt.savefig(importance_img_path, dpi=300)
    print(f"   ✓ Graphique sauvegardé → {importance_img_path}")
except Exception as e:
    print(f"\n   ✗ Impossible de générer le graphique d'importance : {e}")

print("\n" + "=" * 60)
print("  OFFLINE TRAINING TERMINÉ AVEC SUCCÈS")
print("=" * 60)
