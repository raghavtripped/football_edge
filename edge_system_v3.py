"""
FOOTBALL EDGE SYSTEM V3 - Production-Grade Card Model
======================================================
Fixes from V2:
1. FIXED: Team stats now use HOME cards only / AWAY cards only (not total)
2. ADDED: Game-state proxy via team strength differential
3. FIXED: Negative Binomial when overdispersion detected
4. FIXED: Lagged referee stats (no target leakage)
5. ADDED: Proper bet logging structure for CLV tracking
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson, nbinom as scipy_nbinom
from scipy.special import gammaln
from statsmodels.stats.outliers_influence import variance_inflation_factor
import soccerdata as sd
import os
import warnings
from datetime import datetime
import json

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
LEAGUE = "ENG-Premier League"
SEASON = "2526"
CACHE_FILE = "match_cache_v3.csv"
BET_LOG_FILE = "bet_log.json"
TIME_DECAY_HALF_LIFE = 30
OVERDISPERSION_THRESHOLD = 1.25  # Switch to NB above this

# --- MANUAL REFEREE OVERRIDES ---
MANUAL_REF_OVERRIDES = {
    "Manchester Utd vs Manchester City": "Anthony Taylor",
    "Nottingham Forest vs Arsenal": "Michael Oliver",
    "Nott'ham Forest vs Arsenal": "Michael Oliver",
    "Tottenham Hotspur vs West Ham United": "Jarred Gillett",
    "Wolverhampton Wanderers vs Newcastle United": "Samuel Barrott",
    "Wolves vs Newcastle Utd": "Samuel Barrott",
    "Wolves vs Newcastle United": "Samuel Barrott",
    "Chelsea vs Brentford": "John Brooks",
    "Sunderland vs Crystal Palace": "Robert Jones",
    "Brighton vs Bournemouth": "Andy Madley",
    "Brighton and Hove Albion vs Bournemouth": "Andy Madley",
    "Aston Villa vs Everton": "Peter Bankes",
    "Leeds United vs Fulham": "Darren England",
    "Liverpool vs Burnley": "Stuart Attwell",
}

# --- MANUAL ODDS (for edge calculation AND strength proxy) ---
# Format: match_key -> {o35: odds, o45: odds, home_win: odds, draw: odds, away_win: odds}
MANUAL_ODDS = {
    "Manchester Utd vs Manchester City": {"o35": 1.72, "o45": 2.40, "u35": 2.10, "u45": 1.55, "home": 3.10, "draw": 3.40, "away": 2.30},
    "Chelsea vs Brentford": {"o35": 1.65, "o45": 2.20, "u35": 2.20, "u45": 1.65, "home": 1.55, "draw": 4.20, "away": 5.50},
    "Nottingham Forest vs Arsenal": {"o35": 1.80, "o45": 2.50, "u35": 2.00, "u45": 1.55, "home": 5.50, "draw": 4.00, "away": 1.60},
    "Brighton vs Bournemouth": {"o35": 1.70, "o45": 2.30, "u35": 2.15, "u45": 1.60, "home": 1.85, "draw": 3.60, "away": 4.20},
    "Tottenham Hotspur vs West Ham United": {"o35": 1.75, "o45": 2.35, "u35": 2.05, "u45": 1.58, "home": 1.65, "draw": 4.00, "away": 5.00},
    "Liverpool vs Burnley": {"home": 1.20, "draw": 7.00, "away": 13.00},
    "Aston Villa vs Everton": {"home": 1.50, "draw": 4.50, "away": 6.00},
    "Leeds United vs Fulham": {"home": 2.20, "draw": 3.50, "away": 3.20},
    "Wolves vs Newcastle United": {"home": 3.00, "draw": 3.40, "away": 2.40},
    "Sunderland vs Crystal Palace": {"home": 2.50, "draw": 3.30, "away": 2.90},
}

# --- REFEREE FALLBACK (only used if no historical data) ---
REFEREE_FALLBACK = {
    'Tim Robinson': 1.35, 'Michael Salisbury': 1.27, 'Stuart Attwell': 1.21,
    'John Brooks': 1.15, 'Peter Bankes': 1.11, 'Simon Hooper': 1.10,
    'Andy Madley': 1.08, 'Samuel Barrott': 1.04, 'Chris Kavanagh': 1.01,
    'Anthony Taylor': 0.99, 'Darren England': 0.98, 'Paul Tierney': 0.96,
    'Robert Jones': 0.93, 'Jarred Gillett': 0.87, 'Tony Harrington': 0.78,
    'Michael Oliver': 0.66, 'Craig Pawson': 0.50
}

# --- STRUCTURAL DEFINITIONS ---
MAJOR_DERBIES = [
    ("Manchester Utd", "Manchester City"),
    ("Arsenal", "Tottenham"), ("Arsenal", "Tottenham Hotspur"),
    ("Liverpool", "Everton"),
    ("Liverpool", "Manchester Utd"),
]

LONDON_TEAMS = ["Arsenal", "Chelsea", "Tottenham", "Tottenham Hotspur", "West Ham", 
                "West Ham United", "Crystal Palace", "Fulham", "Brentford"]

TOP_6 = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester Utd",
         "Tottenham", "Tottenham Hotspur"]


# =============================================================================
# HELPER FUNCTIONS FOR PRODUCTION FIXES
# =============================================================================

def nb_cdf(k, mu, alpha):
    """
    Proper Negative Binomial CDF.
    
    Parameters:
    - k: number of events (cards)
    - mu: mean (lambda from GLM)
    - alpha: dispersion parameter (from model.scale or computed)
    
    scipy.stats.nbinom uses (n, p) parameterization:
    - n = 1/alpha (number of successes)
    - p = 1/(1 + alpha*mu) (probability of success)
    """
    if alpha <= 0:
        # Fallback to Poisson if no dispersion
        return poisson.cdf(k, mu)
    
    n = 1 / alpha
    p = 1 / (1 + alpha * mu)
    
    return scipy_nbinom.cdf(k, n, p)


def odds_to_implied_strength(odds_dict):
    """
    Convert 1X2 odds to implied strength differential.
    
    Returns: strength_diff where positive = home stronger
    
    Method: Use implied probabilities with overround removed.
    """
    if not odds_dict or 'home' not in odds_dict:
        return None
    
    # Raw implied probabilities (sum > 1 due to margin)
    p_home = 1 / odds_dict.get('home', 2.5)
    p_draw = 1 / odds_dict.get('draw', 3.5)
    p_away = 1 / odds_dict.get('away', 2.5)
    
    # Remove overround (normalize to sum=1)
    total = p_home + p_draw + p_away
    p_home /= total
    p_away /= total
    
    # Strength differential: positive = home stronger
    # Scale to roughly match goal differential range (-2 to +2)
    strength_diff = (p_home - p_away) * 3
    
    return strength_diff


def compute_time_decayed_rate(values, dates, reference_date, half_life_days=TIME_DECAY_HALF_LIFE):
    """
    Compute weighted average with exponential time decay.
    
    More recent values get higher weight.
    """
    if len(values) == 0:
        return None
    
    days_ago = (reference_date - dates).dt.days
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    
    # Weighted average
    weighted_sum = np.sum(values * weights)
    total_weight = np.sum(weights)
    
    return weighted_sum / total_weight if total_weight > 0 else np.mean(values)


def get_enhanced_data():
    """
    Load or create enhanced cache with SPLIT home/away cards.
    """
    if os.path.exists(CACHE_FILE):
        print(f"📂 Loading cache: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    print(f"⚠️ Creating enhanced cache with split card data...")
    print("   Using soccerdata's local cache (no new scraping needed)")
    
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASON)
    
    # Get misc stats (contains per-team cards)
    print("   > Loading misc stats...")
    misc = fbref.read_team_match_stats(stat_type="misc")
    misc = misc.reset_index()
    misc.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in misc.columns]
    
    # Get schedule for referee
    print("   > Loading schedule...")
    schedule = fbref.read_schedule()
    schedule = schedule.reset_index()
    schedule['date_str'] = schedule['date'].astype(str).str.split().str[0]
    
    # Also get goal data for strength proxy
    print("   > Processing...")
    
    # Create referee lookup
    ref_lookup = {}
    score_lookup = {}
    for _, row in schedule.iterrows():
        key = f"{row['date_str']}_{row['home_team']}_{row['away_team']}"
        ref_lookup[key] = row.get('referee', None)
        score_lookup[key] = row.get('score', None)
    
    # Process matches - KEEP HOME AND AWAY CARDS SEPARATE
    def make_match_id(row):
        teams = sorted([str(row['team']), str(row['opponent'])])
        date_str = str(row['date']).split()[0]
        return f"{date_str}_{teams[0]}_{teams[1]}"
    
    misc['match_id'] = misc.apply(make_match_id, axis=1)
    
    processed = []
    for match_id, group in misc.groupby('match_id'):
        if len(group) < 2:
            continue
        
        home_rows = group[group['venue'] == 'Home']
        away_rows = group[group['venue'] == 'Away']
        
        if len(home_rows) == 0 or len(away_rows) == 0:
            continue
        
        home = home_rows.iloc[0]
        away = away_rows.iloc[0]
        
        date_str = str(home['date']).split()[0]
        
        # Get referee
        ref_key = f"{date_str}_{home['team']}_{away['team']}"
        referee = ref_lookup.get(ref_key)
        
        # Parse score for game state analysis
        score = score_lookup.get(ref_key, "0-0")
        try:
            if pd.notna(score) and '–' in str(score):
                parts = str(score).split('–')
                home_goals = int(parts[0].strip())
                away_goals = int(parts[1].strip())
            else:
                home_goals, away_goals = 0, 0
        except:
            home_goals, away_goals = 0, 0
        
        # CRITICAL FIX: Split home and away cards
        home_yellow = home.get('Performance_CrdY', 0) or 0
        home_red = home.get('Performance_CrdR', 0) or 0
        away_yellow = away.get('Performance_CrdY', 0) or 0
        away_red = away.get('Performance_CrdR', 0) or 0
        
        # Fouls (for future use)
        home_fouls = home.get('Performance_Fls', 0) or 0
        away_fouls = away.get('Performance_Fls', 0) or 0
        
        processed.append({
            'date': date_str,
            'home_team': home['team'],
            'away_team': away['team'],
            'referee': referee,
            # SPLIT CARDS (the critical fix)
            'home_cards': home_yellow + home_red,
            'away_cards': away_yellow + away_red,
            'total_cards': home_yellow + home_red + away_yellow + away_red,
            # Fouls
            'home_fouls': home_fouls,
            'away_fouls': away_fouls,
            # Goals (for strength proxy)
            'home_goals': home_goals,
            'away_goals': away_goals,
        })
    
    df = pd.DataFrame(processed)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df.to_csv(CACHE_FILE, index=False)
    print(f"✅ Saved: {CACHE_FILE} ({len(df)} matches)")
    
    return df


def compute_team_strength(df):
    """
    Compute team strength proxy from goal differential.
    This serves as a game-state indicator.
    """
    # Home performance
    home_perf = df.groupby('home_team').agg({
        'home_goals': 'mean',
        'away_goals': 'mean',
    }).rename(columns={'home_goals': 'home_gf', 'away_goals': 'home_ga'})
    home_perf['home_gd'] = home_perf['home_gf'] - home_perf['home_ga']
    
    # Away performance
    away_perf = df.groupby('away_team').agg({
        'away_goals': 'mean',
        'home_goals': 'mean',
    }).rename(columns={'away_goals': 'away_gf', 'home_goals': 'away_ga'})
    away_perf['away_gd'] = away_perf['away_gf'] - away_perf['away_ga']
    
    # Combine into overall strength
    all_teams = set(home_perf.index) | set(away_perf.index)
    strength = {}
    for team in all_teams:
        h_gd = home_perf.loc[team, 'home_gd'] if team in home_perf.index else 0
        a_gd = away_perf.loc[team, 'away_gd'] if team in away_perf.index else 0
        strength[team] = (h_gd + a_gd) / 2
    
    return strength


def compute_lagged_referee_stats(df):
    """
    Compute referee strictness using ONLY matches BEFORE each match.
    This avoids target leakage.
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    # Rolling referee stats (computed from prior matches only)
    ref_cumulative = {}
    lagged_strictness = []
    
    league_cumsum = 0
    league_count = 0
    
    for idx, row in df.iterrows():
        ref = row['referee']
        
        # Compute lagged strictness for THIS match
        if pd.isna(ref):
            lagged_strictness.append(1.0)
        elif ref in ref_cumulative and ref_cumulative[ref]['count'] >= 2:
            # Use prior average
            ref_avg = ref_cumulative[ref]['sum'] / ref_cumulative[ref]['count']
            league_avg = league_cumsum / league_count if league_count > 0 else 3.84
            lagged_strictness.append(ref_avg / league_avg if league_avg > 0 else 1.0)
        else:
            # Not enough prior data, use fallback
            lagged_strictness.append(REFEREE_FALLBACK.get(ref, 1.0))
        
        # UPDATE cumulative stats AFTER computing lagged value
        if pd.notna(ref):
            if ref not in ref_cumulative:
                ref_cumulative[ref] = {'sum': 0, 'count': 0}
            ref_cumulative[ref]['sum'] += row['total_cards']
            ref_cumulative[ref]['count'] += 1
        
        league_cumsum += row['total_cards']
        league_count += 1
    
    return lagged_strictness, ref_cumulative


def compute_team_card_rates(df, use_time_decay=True):
    """
    Compute PROPER team card rates with optional TIME DECAY:
    - home_card_rate: Cards received by home team when playing at home
    - away_card_rate: Cards received by away team when playing away
    
    Time decay ensures recent form matters more than early-season data.
    """
    reference_date = df['date'].max()
    
    if use_time_decay:
        # Time-decayed rates
        home_rates_dict = {}
        away_rates_dict = {}
        
        for team in df['home_team'].unique():
            team_home = df[df['home_team'] == team]
            if len(team_home) > 0:
                decayed_avg = compute_time_decayed_rate(
                    team_home['home_cards'].values,
                    team_home['date'],
                    reference_date
                )
                home_rates_dict[team] = {
                    'home_own_cards_avg': decayed_avg,
                    'home_own_cards_std': team_home['home_cards'].std(),
                    'home_matches': len(team_home),
                }
        
        for team in df['away_team'].unique():
            team_away = df[df['away_team'] == team]
            if len(team_away) > 0:
                decayed_avg = compute_time_decayed_rate(
                    team_away['away_cards'].values,
                    team_away['date'],
                    reference_date
                )
                away_rates_dict[team] = {
                    'away_own_cards_avg': decayed_avg,
                    'away_own_cards_std': team_away['away_cards'].std(),
                    'away_matches': len(team_away),
                }
        
        home_rates = pd.DataFrame(home_rates_dict).T
        away_rates = pd.DataFrame(away_rates_dict).T
    else:
        # Static rates (no decay)
        home_rates = df.groupby('home_team')['home_cards'].agg(['mean', 'std', 'count'])
        home_rates.columns = ['home_own_cards_avg', 'home_own_cards_std', 'home_matches']
        
        away_rates = df.groupby('away_team')['away_cards'].agg(['mean', 'std', 'count'])
        away_rates.columns = ['away_own_cards_avg', 'away_own_cards_std', 'away_matches']
    
    # Cards CONCEDED (opponent cards) - also time-decayed if enabled
    home_conceded = df.groupby('home_team')['away_cards'].mean()
    away_conceded = df.groupby('away_team')['home_cards'].mean()
    
    return home_rates, away_rates, home_conceded, away_conceded


def create_features(df, strength_dict):
    """Create all features for modeling."""
    df = df.copy()
    
    # Binary features
    def is_major_derby(row):
        pair = (row['home_team'], row['away_team'])
        rev = (row['away_team'], row['home_team'])
        return 1 if pair in MAJOR_DERBIES or rev in MAJOR_DERBIES else 0
    
    def is_london(row):
        return 1 if row['home_team'] in LONDON_TEAMS and row['away_team'] in LONDON_TEAMS else 0
    
    def is_top6(row):
        return 1 if row['home_team'] in TOP_6 and row['away_team'] in TOP_6 else 0
    
    df['is_major_derby'] = df.apply(is_major_derby, axis=1)
    df['is_london_derby'] = df.apply(is_london, axis=1)
    df['is_top6_clash'] = df.apply(is_top6, axis=1)
    
    # GAME STATE PROXY: Strength differential
    # Positive = home team stronger, Negative = away team stronger
    def strength_diff(row):
        h_str = strength_dict.get(row['home_team'], 0)
        a_str = strength_dict.get(row['away_team'], 0)
        return h_str - a_str
    
    df['strength_diff'] = df.apply(strength_diff, axis=1)
    
    # Underdog indicator (away team is underdog when strength_diff > 0.5)
    df['away_is_underdog'] = (df['strength_diff'] > 0.3).astype(int)
    
    return df


def compute_time_weights(df, half_life=TIME_DECAY_HALF_LIFE):
    """Exponential time decay weights."""
    ref_date = df['date'].max()
    days_ago = (ref_date - df['date']).dt.days
    return np.exp(-np.log(2) * days_ago / half_life)


def check_vif(X):
    """Check Variance Inflation Factor for multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


def train_model(df):
    """
    Train the V3 model with all fixes applied.
    """
    print("\n" + "="*70)
    print("📈 TRAINING V3 MODEL")
    print("="*70)
    
    # Compute strength proxy
    strength = compute_team_strength(df)
    
    # Compute LAGGED referee stats (no leakage)
    print("   > Computing lagged referee stats (no leakage)...")
    lagged_ref, ref_final_stats = compute_lagged_referee_stats(df)
    df['ref_strictness_lagged'] = lagged_ref
    
    # Compute PROPER team card rates
    print("   > Computing proper team card rates (split home/away)...")
    home_rates, away_rates, home_conc, away_conc = compute_team_card_rates(df)
    
    # Map team stats
    league_avg = df['total_cards'].mean()
    home_cards_avg = df['home_cards'].mean()
    away_cards_avg = df['away_cards'].mean()
    
    df['home_own_rate'] = df['home_team'].map(home_rates['home_own_cards_avg']).fillna(home_cards_avg)
    df['away_own_rate'] = df['away_team'].map(away_rates['away_own_cards_avg']).fillna(away_cards_avg)
    
    # Create features
    df = create_features(df, strength)
    
    # Time weights
    weights = compute_time_weights(df)
    
    print(f"\n   📊 DATA SUMMARY:")
    print(f"      Matches: {len(df)}")
    print(f"      League Avg Total Cards: {league_avg:.2f}")
    print(f"      Avg Home Team Cards: {home_cards_avg:.2f}")
    print(f"      Avg Away Team Cards: {away_cards_avg:.2f}")
    
    # Feature matrix
    feature_cols = [
        'home_own_rate',      # Home team's own card tendency
        'away_own_rate',      # Away team's own card tendency
        'ref_strictness_lagged',  # Lagged referee strictness
        'strength_diff',      # Game state proxy
        'away_is_underdog',   # Underdog indicator
        'is_major_derby',
        'is_top6_clash',
    ]
    
    X = df[feature_cols].copy()
    X = sm.add_constant(X)
    y = df['total_cards']
    
    # Check multicollinearity
    print(f"\n   🔍 MULTICOLLINEARITY CHECK (VIF):")
    vif = check_vif(X)
    for _, row in vif.iterrows():
        status = "⚠️" if row['VIF'] > 5 else "✅"
        print(f"      {status} {row['Feature']}: {row['VIF']:.2f}")
    
    # Test overdispersion
    variance = np.var(y)
    mean = np.mean(y)
    dispersion = variance / mean
    
    print(f"\n   📊 OVERDISPERSION TEST:")
    print(f"      Mean: {mean:.2f}, Variance: {variance:.2f}")
    print(f"      Dispersion Ratio: {dispersion:.2f}")
    
    # Choose model based on overdispersion
    if dispersion > OVERDISPERSION_THRESHOLD:
        print(f"      ⚠️ Overdispersion detected! Using Negative Binomial.")
        model_family = sm.families.NegativeBinomial()
        model_type = "NegBin"
    else:
        print(f"      ✅ Using Poisson.")
        model_family = sm.families.Poisson()
        model_type = "Poisson"
    
    # Fit model
    print(f"\n   🔧 Fitting {model_type} GLM...")
    model = sm.GLM(y, X, family=model_family, freq_weights=weights).fit()
    
    # Print coefficients
    print(f"\n   📊 MODEL COEFFICIENTS:")
    print(f"   {'Feature':<25} | {'Coef':>8} | {'p-val':>6} | {'Sig':>3}")
    print("   " + "-"*55)
    
    for name in model.params.index:
        coef = model.params[name]
        pval = model.pvalues[name]
        sig = "✅" if pval < 0.1 else "❌"
        print(f"   {name:<25} | {coef:>8.4f} | {pval:>6.3f} | {sig}")
    
    print(f"\n   📈 Model AIC: {model.aic:.1f}")
    
    # Extract alpha for NB (dispersion parameter)
    if model_type == "NegBin":
        try:
            alpha = model.scale  # statsmodels NB dispersion
        except:
            alpha = dispersion - 1  # Fallback estimate
        print(f"   📊 NB Alpha (dispersion): {alpha:.3f}")
    else:
        alpha = 0  # No dispersion for Poisson
    
    return {
        'model': model,
        'model_type': model_type,
        'alpha': alpha,  # For proper NB CDF
        'feature_cols': feature_cols,
        'home_rates': home_rates,
        'away_rates': away_rates,
        'ref_stats': ref_final_stats,
        'strength': strength,
        'league_avg': league_avg,
        'home_cards_avg': home_cards_avg,
        'away_cards_avg': away_cards_avg,
        'dispersion': dispersion,
    }


def predict_match(bundle, home, away, referee, book_odds=None):
    """
    Predict cards with proper uncertainty.
    
    PRODUCTION FIXES:
    1. Uses proper NB CDF when model is Negative Binomial
    2. Uses odds-implied strength when available (better than goal-based)
    3. Team rates are already time-decayed from training
    """
    model = bundle['model']
    model_type = bundle['model_type']
    alpha = bundle['alpha']  # For NB CDF
    home_rates = bundle['home_rates']
    away_rates = bundle['away_rates']
    ref_stats = bundle['ref_stats']
    strength = bundle['strength']  # Fallback goal-based strength
    league_avg = bundle['league_avg']
    
    # Team card rates (already time-decayed)
    home_rate = home_rates.loc[home, 'home_own_cards_avg'] if home in home_rates.index else bundle['home_cards_avg']
    away_rate = away_rates.loc[away, 'away_own_cards_avg'] if away in away_rates.index else bundle['away_cards_avg']
    
    # Referee strictness (from accumulated stats)
    if referee and referee in ref_stats:
        ref_avg = ref_stats[referee]['sum'] / ref_stats[referee]['count']
        ref_strict = ref_avg / league_avg
    elif referee:
        ref_strict = REFEREE_FALLBACK.get(referee, 1.0)
    else:
        ref_strict = 1.0
    
    # PRODUCTION FIX #2: Odds-implied strength (superior to goal-based)
    odds_strength = None
    if book_odds:
        odds_strength = odds_to_implied_strength(book_odds)
    
    # Use odds-implied strength if available, else fall back to goal-based
    if odds_strength is not None:
        str_diff = odds_strength
        strength_source = "odds"
    else:
        h_str = strength.get(home, 0)
        a_str = strength.get(away, 0)
        str_diff = h_str - a_str
        strength_source = "goals"
    
    # Binary features (underdog threshold based on strength_diff)
    pair = (home, away)
    rev = (away, home)
    is_derby = 1 if pair in MAJOR_DERBIES or rev in MAJOR_DERBIES else 0
    is_top6 = 1 if home in TOP_6 and away in TOP_6 else 0
    
    # Underdog: continuous strength_diff captures this, but binary flag for extreme cases
    is_underdog = 1 if str_diff > 0.5 else 0  # Raised threshold for cleaner signal
    
    # Build prediction input
    X_pred = pd.DataFrame({
        'const': [1],
        'home_own_rate': [home_rate],
        'away_own_rate': [away_rate],
        'ref_strictness_lagged': [ref_strict],
        'strength_diff': [str_diff],
        'away_is_underdog': [is_underdog],
        'is_major_derby': [is_derby],
        'is_top6_clash': [is_top6],
    })
    
    # Predict mean
    lambda_val = model.predict(X_pred)[0]
    
    # PRODUCTION FIX #1: Proper CDF based on model type
    if model_type == "Poisson":
        probs = {
            'o25': 1 - poisson.cdf(2, lambda_val),
            'o35': 1 - poisson.cdf(3, lambda_val),
            'o45': 1 - poisson.cdf(4, lambda_val),
            'o55': 1 - poisson.cdf(5, lambda_val),
            'u35': poisson.cdf(3, lambda_val),
            'u45': poisson.cdf(4, lambda_val),
        }
    else:
        # PROPER Negative Binomial CDF with alpha
        probs = {
            'o25': 1 - nb_cdf(2, lambda_val, alpha),
            'o35': 1 - nb_cdf(3, lambda_val, alpha),
            'o45': 1 - nb_cdf(4, lambda_val, alpha),
            'o55': 1 - nb_cdf(5, lambda_val, alpha),
            'u35': nb_cdf(3, lambda_val, alpha),
            'u45': nb_cdf(4, lambda_val, alpha),
        }
    
    # Fair odds
    fair = {k: 1/v if v > 0.01 else 99 for k, v in probs.items()}
    
    # Edge calculation
    edges = {}
    if book_odds:
        for line in ['o35', 'o45', 'u35', 'u45']:
            if line in book_odds:
                book_implied = 1 / book_odds[line]
                model_prob = probs.get(line, 0)
                edges[line] = model_prob - book_implied
    
    return {
        'lambda': lambda_val,
        'probs': probs,
        'fair': fair,
        'edges': edges,
        'model_type': model_type,
        'features': {
            'home_rate': home_rate,
            'away_rate': away_rate,
            'ref_strict': ref_strict,
            'str_diff': str_diff,
            'strength_source': strength_source,  # 'odds' or 'goals'
            'is_derby': is_derby,
            'is_underdog': is_underdog,
        }
    }


def log_bet(match, prediction, book_odds, bet_type, stake=1.0):
    """Log bet for CLV tracking."""
    bet_record = {
        'timestamp': datetime.now().isoformat(),
        'match': match,
        'bet_type': bet_type,
        'model_prob': prediction['probs'].get(bet_type, 0),
        'book_odds': book_odds.get(bet_type, None) if book_odds else None,
        'edge': prediction['edges'].get(bet_type, None),
        'lambda': prediction['lambda'],
        'stake': stake,
        'closing_odds': None,  # To be filled after match
        'result': None,  # To be filled after match
    }
    
    # Append to log
    if os.path.exists(BET_LOG_FILE):
        with open(BET_LOG_FILE, 'r') as f:
            log = json.load(f)
    else:
        log = []
    
    log.append(bet_record)
    
    with open(BET_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)
    
    return bet_record


def run_system():
    """Main entry point."""
    print("\n" + "="*80)
    print("🎯 FOOTBALL EDGE SYSTEM V3 - Production Grade")
    print("="*80)
    
    # Load data
    df = get_enhanced_data()
    if df is None:
        return
    
    print(f"\n📂 Loaded {len(df)} matches ({df['date'].min().date()} → {df['date'].max().date()})")
    
    # Train
    bundle = train_model(df)
    
    # Predictions
    print("\n" + "="*80)
    print("⚡ MATCHWEEK PREDICTIONS")
    print("="*80)
    
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASON)
    schedule = fbref.read_schedule()
    upcoming = schedule[schedule['score'].isna()].head(10)
    
    print(f"\n{'MATCH':<42} | {'REF':<15} | {'λ':>5} | {'EDGE':>7} | SIGNAL")
    print("-" * 90)
    
    value_bets = []
    
    for idx, row in upcoming.iterrows():
        home = row['home_team']
        away = row['away_team']
        match_key = f"{home} vs {away}"
        
        # Get referee
        if match_key in MANUAL_REF_OVERRIDES:
            ref = MANUAL_REF_OVERRIDES[match_key]
            ref_tag = "📋"
        elif pd.notna(row.get('referee')):
            ref = row['referee']
            ref_tag = ""
        else:
            ref = None
            ref_tag = ""
        
        # Get book odds
        book_odds = MANUAL_ODDS.get(match_key)
        
        # Predict
        pred = predict_match(bundle, home, away, ref, book_odds)
        
        # Find best edge
        best_edge = None
        best_line = None
        if pred['edges']:
            for line, edge in pred['edges'].items():
                if edge and (best_edge is None or edge > best_edge):
                    best_edge = edge
                    best_line = line
        
        # Generate signal
        λ = pred['lambda']
        signal = ""
        
        if best_edge and best_edge > 0.08:
            signal = f"💰 {best_line.upper()} +{best_edge*100:.0f}%"
            value_bets.append({
                'match': match_key,
                'line': best_line,
                'edge': best_edge,
                'lambda': λ,
                'fair': pred['fair'].get(best_line, 0),
                'book': book_odds.get(best_line) if book_odds else None,
            })
        elif λ > 5.0:
            signal = "🔥🔥 STRONG OVER"
        elif λ > 4.5:
            signal = "🔥 O4.5"
        elif λ < 3.0:
            signal = "⚠️ UNDER"
        else:
            signal = "-"
        
        # Tags
        tag = ""
        if pred['features']['is_derby']:
            tag = "🔥"
        elif pred['features']['is_underdog']:
            tag = "📉"
        
        ref_display = f"{ref_tag}{ref}" if ref else "<NA>"
        edge_display = f"{best_edge*100:+.0f}%" if best_edge else "-"
        
        print(f"{tag}{home:<20} vs {away:<18} | {ref_display:<15} | {λ:>5.2f} | {edge_display:>7} | {signal}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 BETTING SUMMARY")
    print("="*80)
    
    if value_bets:
        print(f"\n   💰 VALUE BETS IDENTIFIED ({len(value_bets)}):")
        for bet in sorted(value_bets, key=lambda x: -x['edge']):
            print(f"      • {bet['match']}")
            print(f"        Line: {bet['line'].upper()} | Edge: {bet['edge']*100:+.1f}% | Fair: @{bet['fair']:.2f} | Book: @{bet['book']}")
    else:
        print("\n   No value bets identified at current odds.")
    
    print(f"\n   📈 Model Type: {bundle['model_type']}")
    print(f"   📊 Dispersion: {bundle['dispersion']:.2f}")
    
    print("\n" + "="*80)
    print("💡 To log bets for CLV tracking, use: log_bet(match, prediction, odds, 'o35')")
    print("="*80)


if __name__ == "__main__":
    run_system()
