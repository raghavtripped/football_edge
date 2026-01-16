"""
FOOTBALL EDGE SYSTEM V2 - Enhanced Multi-Variable Card Model
=============================================================
Improvements over V1:
- Team-level card rates (home & away separated)
- Learned binary features (derby, top6) - not hardcoded magnitudes
- Time decay weighting for recent form
- Overdispersion testing
- Odds comparison for edge detection
- Proper statistical framework
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
import soccerdata as sd
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
LEAGUE = "ENG-Premier League"
SEASON = "2526"
CACHE_FILE = "match_cache.csv"  # Use existing cache from V1
TIME_DECAY_HALF_LIFE = 30  # Days

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

# --- MANUAL ODDS INPUT (for edge calculation) ---
MANUAL_ODDS = {
    "Manchester Utd vs Manchester City": {"o35_cards": 1.72, "o45_cards": 2.40},
    "Chelsea vs Brentford": {"o35_cards": 1.65, "o45_cards": 2.20},
    "Nottingham Forest vs Arsenal": {"o35_cards": 1.80, "o45_cards": 2.50},
    "Brighton vs Bournemouth": {"o35_cards": 1.70, "o45_cards": 2.30},
    "Tottenham Hotspur vs West Ham United": {"o35_cards": 1.75, "o45_cards": 2.35},
}

# --- REFEREE FALLBACK (computed from data takes priority) ---
REFEREE_FALLBACK = {
    'Tim Robinson': 1.35, 'Michael Salisbury': 1.27, 'Stuart Attwell': 1.21,
    'John Brooks': 1.15, 'Peter Bankes': 1.11, 'Simon Hooper': 1.10,
    'Andy Madley': 1.08, 'Samuel Barrott': 1.04, 'Chris Kavanagh': 1.01,
    'Anthony Taylor': 0.99, 'Darren England': 0.98, 'Paul Tierney': 0.96,
    'Robert Jones': 0.93, 'Jarred Gillett': 0.87, 'Tony Harrington': 0.78,
    'Michael Oliver': 0.66, 'Craig Pawson': 0.50
}

# --- STRUCTURAL DEFINITIONS (for binary features) ---
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


def load_data():
    """Load and prepare the cached data."""
    if not os.path.exists(CACHE_FILE):
        print(f"❌ Cache file not found: {CACHE_FILE}")
        print("   Run edge_system.py first to generate the cache.")
        return None
    
    df = pd.read_csv(CACHE_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_time_weights(df, half_life_days=TIME_DECAY_HALF_LIFE):
    """Apply exponential time decay - recent matches weighted more heavily."""
    reference_date = df['date'].max()
    days_ago = (reference_date - df['date']).dt.days
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    return weights


def create_features(df):
    """
    Create all features for the model.
    Binary features are flags (0/1), not hardcoded multipliers.
    The model learns the coefficients.
    """
    df = df.copy()
    
    # --- BINARY FEATURES (learned, not hardcoded) ---
    
    # Major derby
    def is_major_derby(row):
        pair = (row['home'], row['away'])
        rev_pair = (row['away'], row['home'])
        for d in MAJOR_DERBIES:
            if pair == d or rev_pair == d:
                return 1
        return 0
    
    # London derby
    def is_london_derby(row):
        return 1 if row['home'] in LONDON_TEAMS and row['away'] in LONDON_TEAMS else 0
    
    # Top 6 clash
    def is_top6_clash(row):
        return 1 if row['home'] in TOP_6 and row['away'] in TOP_6 else 0
    
    df['is_major_derby'] = df.apply(is_major_derby, axis=1)
    df['is_london_derby'] = df.apply(is_london_derby, axis=1)
    df['is_top6_clash'] = df.apply(is_top6_clash, axis=1)
    
    return df


def compute_team_stats(df):
    """
    Compute team-level card tendencies from actual data.
    Returns dictionaries for lookup.
    """
    # Home team stats (when playing at home)
    home_stats = df.groupby('home').agg({
        'total_cards': ['mean', 'std', 'count']
    })
    home_stats.columns = ['home_cards_avg', 'home_cards_std', 'home_matches']
    
    # Away team stats (when playing away)  
    away_stats = df.groupby('away').agg({
        'total_cards': ['mean', 'std', 'count']
    })
    away_stats.columns = ['away_cards_avg', 'away_cards_std', 'away_matches']
    
    return home_stats, away_stats


def compute_referee_stats(df):
    """Compute referee card averages from actual data."""
    ref_stats = df.groupby('referee').agg({
        'total_cards': ['mean', 'std', 'count']
    })
    ref_stats.columns = ['ref_cards_avg', 'ref_cards_std', 'ref_matches']
    
    # Compute strictness relative to league average
    league_avg = df['total_cards'].mean()
    ref_stats['strictness'] = ref_stats['ref_cards_avg'] / league_avg
    
    return ref_stats, league_avg


def test_overdispersion(y):
    """
    Test if data shows overdispersion (variance > mean).
    For Poisson, variance should equal mean.
    """
    mean = np.mean(y)
    variance = np.var(y)
    dispersion = variance / mean
    
    print(f"\n   📊 OVERDISPERSION TEST:")
    print(f"      Mean:     {mean:.3f}")
    print(f"      Variance: {variance:.3f}")
    print(f"      Ratio:    {dispersion:.3f}")
    
    if dispersion > 1.3:
        print(f"      ⚠️  Significant overdispersion detected!")
        print(f"         Consider Negative Binomial or quasi-Poisson.")
        return True
    elif dispersion > 1.1:
        print(f"      ⚡ Mild overdispersion - Poisson acceptable but monitor.")
        return False
    else:
        print(f"      ✅ Poisson assumption holds well.")
        return False


def train_model(df, use_time_weights=True):
    """
    Train the enhanced multi-variable Poisson model.
    """
    print("\n" + "="*70)
    print("📈 TRAINING ENHANCED MODEL")
    print("="*70)
    
    # Get team and referee statistics
    home_stats, away_stats = compute_team_stats(df)
    ref_stats, league_avg = compute_referee_stats(df)
    
    print(f"\n   📊 DATA SUMMARY:")
    print(f"      Matches: {len(df)}")
    print(f"      Teams: {len(home_stats)}")
    print(f"      Referees: {len(ref_stats)}")
    print(f"      League Avg Cards: {league_avg:.2f}")
    
    # Create features
    df = create_features(df)
    
    # Map team stats to each match
    df['home_cards_rate'] = df['home'].map(home_stats['home_cards_avg'])
    df['away_cards_rate'] = df['away'].map(away_stats['away_cards_avg'])
    
    # Map referee strictness
    def get_ref_strictness(ref):
        if pd.isna(ref):
            return 1.0
        if ref in ref_stats.index:
            return ref_stats.loc[ref, 'strictness']
        return REFEREE_FALLBACK.get(ref, 1.0)
    
    df['ref_strictness'] = df['referee'].apply(get_ref_strictness)
    
    # Fill NaN with averages
    df['home_cards_rate'] = df['home_cards_rate'].fillna(league_avg)
    df['away_cards_rate'] = df['away_cards_rate'].fillna(league_avg)
    
    # Compute time weights
    if use_time_weights:
        weights = compute_time_weights(df)
        print(f"\n   ⏱️  Time decay applied (half-life: {TIME_DECAY_HALF_LIFE} days)")
    else:
        weights = np.ones(len(df))
    
    # Build feature matrix
    feature_cols = [
        'home_cards_rate',
        'away_cards_rate',
        'ref_strictness',
        'is_major_derby',
        'is_london_derby', 
        'is_top6_clash',
    ]
    
    X = df[feature_cols].copy()
    X = sm.add_constant(X)
    y = df['total_cards']
    
    # Test overdispersion
    is_overdispersed = test_overdispersion(y)
    
    # Fit Poisson GLM with time weights
    print(f"\n   🔧 Fitting Poisson GLM with {len(feature_cols)} features...")
    model = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=weights).fit()
    
    # Print coefficients
    print(f"\n   📊 MODEL COEFFICIENTS:")
    print(f"   {'Feature':<20} | {'Coef':>8} | {'Exp(Coef)':>10} | Interpretation")
    print("   " + "-"*65)
    
    for name, coef in model.params.items():
        exp_coef = np.exp(coef)
        if name == 'const':
            interp = f"Base rate"
        elif 'cards_rate' in name:
            interp = f"+1 avg → ×{exp_coef:.2f} cards"
        elif 'strictness' in name:
            interp = f"+0.1 strict → ×{np.exp(coef*0.1):.2f} cards"
        elif coef > 0:
            interp = f"Adds ~{(exp_coef-1)*league_avg:.1f} cards"
        else:
            interp = f"Reduces ~{(1-exp_coef)*league_avg:.1f} cards"
        
        print(f"   {name:<20} | {coef:>8.4f} | {exp_coef:>10.4f} | {interp}")
    
    print(f"\n   📈 Model AIC: {model.aic:.1f}")
    print(f"   📈 Model Deviance: {model.deviance:.1f}")
    
    # Check if binary features are significant
    print(f"\n   🎯 BINARY FEATURE ANALYSIS:")
    for feat in ['is_major_derby', 'is_london_derby', 'is_top6_clash']:
        coef = model.params[feat]
        pval = model.pvalues[feat]
        sig = "✅ Significant" if pval < 0.1 else "❌ Not significant"
        print(f"      {feat}: coef={coef:.3f}, p={pval:.3f} → {sig}")
    
    return {
        'model': model,
        'feature_cols': feature_cols,
        'home_stats': home_stats,
        'away_stats': away_stats,
        'ref_stats': ref_stats,
        'league_avg': league_avg,
    }


def predict_match(bundle, home, away, referee, book_odds=None):
    """
    Predict total cards for a match and compute edge vs book.
    """
    model = bundle['model']
    home_stats = bundle['home_stats']
    away_stats = bundle['away_stats']
    ref_stats = bundle['ref_stats']
    league_avg = bundle['league_avg']
    
    # Get team stats
    home_rate = home_stats.loc[home, 'home_cards_avg'] if home in home_stats.index else league_avg
    away_rate = away_stats.loc[away, 'away_cards_avg'] if away in away_stats.index else league_avg
    
    # Get referee strictness
    if referee and referee in ref_stats.index:
        ref_strict = ref_stats.loc[referee, 'strictness']
    elif referee:
        ref_strict = REFEREE_FALLBACK.get(referee, 1.0)
    else:
        ref_strict = 1.0
    
    # Binary features
    pair = (home, away)
    rev_pair = (away, home)
    is_derby = 1 if any(pair == d or rev_pair == d for d in MAJOR_DERBIES) else 0
    is_london = 1 if home in LONDON_TEAMS and away in LONDON_TEAMS else 0
    is_top6 = 1 if home in TOP_6 and away in TOP_6 else 0
    
    # Build prediction input
    X_pred = pd.DataFrame({
        'const': [1],
        'home_cards_rate': [home_rate],
        'away_cards_rate': [away_rate],
        'ref_strictness': [ref_strict],
        'is_major_derby': [is_derby],
        'is_london_derby': [is_london],
        'is_top6_clash': [is_top6],
    })
    
    # Predict expected cards (λ)
    lambda_val = model.predict(X_pred)[0]
    
    # Compute probabilities
    probs = {
        'o25': 1 - poisson.cdf(2, lambda_val),
        'o35': 1 - poisson.cdf(3, lambda_val),
        'o45': 1 - poisson.cdf(4, lambda_val),
        'o55': 1 - poisson.cdf(5, lambda_val),
    }
    
    # Fair odds
    fair = {k: 1/v if v > 0.01 else 99.0 for k, v in probs.items()}
    
    # Edge calculation
    edge_o35 = None
    edge_o45 = None
    if book_odds:
        if 'o35_cards' in book_odds:
            book_implied = 1 / book_odds['o35_cards']
            edge_o35 = probs['o35'] - book_implied
        if 'o45_cards' in book_odds:
            book_implied = 1 / book_odds['o45_cards']
            edge_o45 = probs['o45'] - book_implied
    
    return {
        'lambda': lambda_val,
        'probs': probs,
        'fair': fair,
        'edge_o35': edge_o35,
        'edge_o45': edge_o45,
        'features': {
            'home_rate': home_rate,
            'away_rate': away_rate,
            'ref_strict': ref_strict,
            'is_derby': is_derby,
            'is_london': is_london,
            'is_top6': is_top6,
        }
    }


def run_system():
    """Main entry point."""
    print("\n" + "="*80)
    print("🎯 FOOTBALL EDGE SYSTEM V2 - Enhanced Multi-Variable Model")
    print("="*80)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print(f"\n📂 Loaded {len(df)} matches ({df['date'].min().date()} to {df['date'].max().date()})")
    
    # Train model
    bundle = train_model(df, use_time_weights=True)
    
    # Get upcoming fixtures
    print("\n" + "="*80)
    print("⚡ MATCHWEEK PREDICTIONS")
    print("="*80)
    
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASON)
    schedule = fbref.read_schedule()
    upcoming = schedule[schedule['score'].isna()].head(10)
    
    print(f"\n{'MATCH':<42} | {'REF':<16} | {'λ':>5} | {'@O3.5':>6} | {'@O4.5':>6} | {'EDGE':>7} | SIGNAL")
    print("-" * 105)
    
    all_preds = []
    
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
        
        # Generate signal
        λ = pred['lambda']
        e35 = pred['edge_o35']
        e45 = pred['edge_o45']
        
        signal = ""
        if e35 and e35 > 0.08:
            signal = f"💰 O3.5 +{e35*100:.0f}%"
        elif e45 and e45 > 0.08:
            signal = f"💰 O4.5 +{e45*100:.0f}%"
        elif λ > 5.0:
            signal = "🔥🔥 STRONG OVER"
        elif λ > 4.5:
            signal = "🔥 O4.5"
        elif λ > 4.0:
            signal = "✅ O3.5"
        elif λ < 3.0:
            signal = "⚠️ UNDER"
        else:
            signal = "-"
        
        # Match tags
        tag = ""
        if pred['features']['is_derby']:
            tag = "🔥"
        elif pred['features']['is_london']:
            tag = "⚔️"
        elif pred['features']['is_top6']:
            tag = "⭐"
        
        ref_display = f"{ref_tag}{ref}" if ref else "<NA>"
        edge_display = f"{e35*100:+.0f}%" if e35 else "-"
        
        print(f"{tag}{home:<20} vs {away:<18} | {ref_display:<16} | {λ:>5.2f} | {pred['fair']['o35']:>6.2f} | {pred['fair']['o45']:>6.2f} | {edge_display:>7} | {signal}")
        
        all_preds.append({
            'match': match_key,
            'lambda': λ,
            'edge_o35': e35,
            'edge_o45': e45,
            'signal': signal,
        })
    
    # Summary
    print("\n" + "="*80)
    print("📊 BETTING SUMMARY")
    print("="*80)
    
    value_bets = [p for p in all_preds if p['edge_o35'] and p['edge_o35'] > 0.05]
    strong_overs = [p for p in all_preds if p['lambda'] > 4.5]
    unders = [p for p in all_preds if p['lambda'] < 3.2]
    
    print(f"\n   💰 VALUE BETS (>5% edge on O3.5):")
    if value_bets:
        for b in sorted(value_bets, key=lambda x: -x['edge_o35']):
            print(f"      • {b['match']}: {b['edge_o35']*100:+.1f}% edge (λ={b['lambda']:.2f})")
    else:
        print("      None identified")
    
    print(f"\n   🔥 STRONG OVERS (λ > 4.5):")
    if strong_overs:
        for s in sorted(strong_overs, key=lambda x: -x['lambda']):
            print(f"      • {s['match']}: λ={s['lambda']:.2f}")
    else:
        print("      None identified")
    
    print(f"\n   ⚠️ UNDER CANDIDATES (λ < 3.2):")
    if unders:
        for u in sorted(unders, key=lambda x: x['lambda']):
            print(f"      • {u['match']}: λ={u['lambda']:.2f}")
    else:
        print("      None identified")
    
    print("\n" + "="*80)
    print("💡 NOTE: Add real bookmaker odds to MANUAL_ODDS for accurate edge calculation")
    print("="*80)


if __name__ == "__main__":
    run_system()
