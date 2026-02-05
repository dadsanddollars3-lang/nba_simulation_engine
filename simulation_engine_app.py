"""
NBA SIMULATION ENGINE - USER-FRIENDLY INTERFACE

This is a separate app from the betting app.
It helps you discover patterns in historical data.

NO CODING REQUIRED - Just click buttons and follow instructions.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import psycopg
from psycopg.types.json import Json
import time

st.set_page_config(page_title="NBA Simulation Engine", layout="wide", page_icon="🔬")

# =======================
# Database Connection (Same as betting app)
# =======================

def S(key: str, default=None):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets.get(key)
    except Exception:
        pass
    return os.environ.get(key, default)

DATABASE_URL_RAW = S("DATABASE_URL")

def get_db_conn():
    """Simple database connection"""
    if not DATABASE_URL_RAW:
        raise RuntimeError("Missing DATABASE_URL in secrets")
    return psycopg.connect(DATABASE_URL_RAW, connect_timeout=12)

# =======================
# GitHub Data URLs
# =======================

GITHUB_BASE = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets"

DATA_SOURCES = {
    "games": f"{GITHUB_BASE}/game.csv",
    "team_stats": f"{GITHUB_BASE}/team_boxscore.csv",
    "player_stats": f"{GITHUB_BASE}/player_boxscore.csv",
    # Play-by-play is too large for direct download - we'll handle separately
}

# =======================
# Step 1: Database Setup
# =======================

def setup_simulation_database():
    """Create tables for simulation engine"""
    
    ddl = """
    -- Historical games (from GitHub)
    create table if not exists sim_historical_games (
        game_id text primary key,
        season text not null,
        game_date date not null,
        home_team_id integer,
        away_team_id integer,
        home_team_abbr text,
        away_team_abbr text,
        home_score integer,
        away_score integer,
        point_diff integer,
        total_points integer,
        season_type text,
        loaded_at timestamptz default now()
    );
    create index if not exists idx_sim_games_season on sim_historical_games(season);
    create index if not exists idx_sim_games_date on sim_historical_games(game_date);
    create index if not exists idx_sim_games_teams on sim_historical_games(home_team_abbr, away_team_abbr);
    
    -- Team stats (from GitHub)
    create table if not exists sim_team_stats (
        stat_id bigserial primary key,
        game_id text not null,
        team_id integer,
        team_abbr text,
        is_home boolean,
        pace numeric,
        possessions numeric,
        off_rating numeric,
        def_rating numeric,
        efg_pct numeric,
        tov_pct numeric,
        orb_pct numeric,
        ft_rate numeric,
        loaded_at timestamptz default now()
    );
    create index if not exists idx_sim_team_game on sim_team_stats(game_id);
    create index if not exists idx_sim_team_abbr on sim_team_stats(team_abbr);
    
    -- Player stats (from GitHub)
    create table if not exists sim_player_stats (
        stat_id bigserial primary key,
        game_id text not null,
        player_id integer,
        player_name text,
        team_id integer,
        team_abbr text,
        minutes numeric,
        points integer,
        rebounds integer,
        assists integer,
        plus_minus integer,
        usage_pct numeric,
        true_shooting numeric,
        loaded_at timestamptz default now()
    );
    create index if not exists idx_sim_player_game on sim_player_stats(game_id);
    create index if not exists idx_sim_player_name on sim_player_stats(player_name);
    
    -- Simulation results
    create table if not exists sim_results (
        sim_id bigserial primary key,
        sim_name text not null,
        sim_type text not null,
        run_date timestamptz default now(),
        n_simulations integer,
        parameters jsonb,
        results jsonb,
        notes text
    );
    
    -- Discovered correlations
    create table if not exists sim_correlations (
        corr_id bigserial primary key,
        correlation_type text not null,
        factor_a text not null,
        factor_b text not null,
        correlation_coef numeric,
        sample_size integer,
        confidence_level numeric,
        discovered_at timestamptz default now(),
        notes text
    );
    """
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

# =======================
# Step 2: Download Data from GitHub
# =======================

def download_github_data(data_type: str, sample_size: int = None):
    """
    Download data from GitHub
    
    Args:
        data_type: 'games', 'team_stats', or 'player_stats'
        sample_size: If provided, only load this many rows (for testing)
    """
    
    url = DATA_SOURCES.get(data_type)
    if not url:
        raise ValueError(f"Unknown data type: {data_type}")
    
    st.info(f"📥 Downloading {data_type} from GitHub...")
    
    try:
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Read CSV
        df = pd.read_csv(url)
        
        if sample_size:
            df = df.head(sample_size)
        
        st.success(f"✅ Downloaded {len(df):,} rows of {data_type}")
        return df
        
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        return None

def load_games_to_database(df: pd.DataFrame):
    """Load game data into database"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    rows_inserted = 0
    batch_size = 1000
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for idx, row in df.iterrows():
                try:
                    # Calculate derived fields
                    point_diff = row.get('home_score', 0) - row.get('away_score', 0)
                    total_points = row.get('home_score', 0) + row.get('away_score', 0)
                    
                    cur.execute(
                        """
                        insert into sim_historical_games(
                            game_id, season, game_date, home_team_id, away_team_id,
                            home_team_abbr, away_team_abbr, home_score, away_score,
                            point_diff, total_points, season_type
                        )
                        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        on conflict (game_id) do nothing
                        """,
                        (
                            str(row.get('game_id')),
                            str(row.get('season', '')),
                            row.get('game_date'),
                            int(row.get('home_team_id', 0)) if pd.notna(row.get('home_team_id')) else None,
                            int(row.get('away_team_id', 0)) if pd.notna(row.get('away_team_id')) else None,
                            str(row.get('home_team_abbreviation', '')),
                            str(row.get('away_team_abbreviation', '')),
                            int(row.get('home_score', 0)),
                            int(row.get('away_score', 0)),
                            point_diff,
                            total_points,
                            str(row.get('season_type', 'Regular'))
                        )
                    )
                    
                    rows_inserted += 1
                    
                    # Update progress
                    if rows_inserted % batch_size == 0:
                        conn.commit()
                        progress = min(rows_inserted / len(df), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Loaded {rows_inserted:,} / {len(df):,} games...")
                
                except Exception as e:
                    st.warning(f"Skipped row {idx}: {e}")
                    continue
            
            conn.commit()
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Loaded {rows_inserted:,} games successfully!")
    
    return rows_inserted

def load_team_stats_to_database(df: pd.DataFrame):
    """Load team stats into database"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    rows_inserted = 0
    batch_size = 1000
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for idx, row in df.iterrows():
                try:
                    cur.execute(
                        """
                        insert into sim_team_stats(
                            game_id, team_id, team_abbr, is_home, pace, possessions,
                            off_rating, def_rating, efg_pct, tov_pct, orb_pct, ft_rate
                        )
                        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            str(row.get('game_id')),
                            int(row.get('team_id', 0)) if pd.notna(row.get('team_id')) else None,
                            str(row.get('team_abbreviation', '')),
                            bool(row.get('is_home', False)),
                            float(row.get('pace', 0)) if pd.notna(row.get('pace')) else None,
                            float(row.get('possessions', 0)) if pd.notna(row.get('possessions')) else None,
                            float(row.get('off_rating', 0)) if pd.notna(row.get('off_rating')) else None,
                            float(row.get('def_rating', 0)) if pd.notna(row.get('def_rating')) else None,
                            float(row.get('efg_pct', 0)) if pd.notna(row.get('efg_pct')) else None,
                            float(row.get('tov_pct', 0)) if pd.notna(row.get('tov_pct')) else None,
                            float(row.get('orb_pct', 0)) if pd.notna(row.get('orb_pct')) else None,
                            float(row.get('ft_rate', 0)) if pd.notna(row.get('ft_rate')) else None,
                        )
                    )
                    
                    rows_inserted += 1
                    
                    if rows_inserted % batch_size == 0:
                        conn.commit()
                        progress = min(rows_inserted / len(df), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Loaded {rows_inserted:,} / {len(df):,} team stats...")
                
                except Exception as e:
                    st.warning(f"Skipped row {idx}: {e}")
                    continue
            
            conn.commit()
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Loaded {rows_inserted:,} team stats successfully!")
    
    return rows_inserted

# =======================
# Step 3: Data Validation
# =======================

def validate_loaded_data():
    """Check what data we have loaded"""
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Count games
            cur.execute("SELECT COUNT(*), MIN(season), MAX(season) FROM sim_historical_games")
            game_count, min_season, max_season = cur.fetchone()
            
            # Count team stats
            cur.execute("SELECT COUNT(*) FROM sim_team_stats")
            team_stat_count = cur.fetchone()[0]
            
            # Count player stats
            cur.execute("SELECT COUNT(*) FROM sim_player_stats")
            player_stat_count = cur.fetchone()[0]
    
    return {
        "games": game_count or 0,
        "min_season": min_season,
        "max_season": max_season,
        "team_stats": team_stat_count or 0,
        "player_stats": player_stat_count or 0,
    }

# =======================
# UI: Main Interface
# =======================

st.title("🔬 NBA Simulation Engine")
st.caption("Discover hidden patterns in NBA data - No coding required!")

# Show current status
st.sidebar.header("📊 Data Status")

try:
    stats = validate_loaded_data()
    st.sidebar.metric("Historical Games", f"{stats['games']:,}")
    if stats['min_season']:
        st.sidebar.caption(f"Seasons: {stats['min_season']} to {stats['max_season']}")
    st.sidebar.metric("Team Stats", f"{stats['team_stats']:,}")
    st.sidebar.metric("Player Stats", f"{stats['player_stats']:,}")
except:
    st.sidebar.warning("Database not set up yet")

# =======================
# Step-by-Step Tabs
# =======================

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Setup", 
    "2️⃣ Load Data", 
    "3️⃣ Run Simulations",
    "4️⃣ View Results"
])

# TAB 1: SETUP
with tab1:
    st.header("Step 1: Database Setup")
    st.markdown("""
    **What this does:** Creates the tables needed to store NBA data for simulations.
    
    **When to do this:** Only once, the first time you use this tool.
    
    **How long it takes:** Less than 1 minute.
    """)
    
    if st.button("🔧 Set Up Database", type="primary", use_container_width=True):
        with st.spinner("Setting up database tables..."):
            try:
                setup_simulation_database()
                st.success("✅ Database is ready!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Setup failed: {e}")
                st.info("💡 Make sure DATABASE_URL is set in your Streamlit secrets")

# TAB 2: LOAD DATA
with tab2:
    st.header("Step 2: Load Historical Data")
    st.markdown("""
    **What this does:** Downloads NBA game data from GitHub and loads it into your database.
    
    **How long it takes:**
    - Test mode (1,000 games): ~2-3 minutes
    - Full load (48,000+ games): ~20-30 minutes
    
    **Recommendation:** Start with test mode to make sure everything works!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Test Mode")
        st.caption("Load 1,000 games to test")
        
        if st.button("Load Test Data", use_container_width=True):
            with st.spinner("Downloading test data..."):
                # Download games
                games_df = download_github_data("games", sample_size=1000)
                
                if games_df is not None:
                    # Load to database
                    with st.spinner("Loading games to database..."):
                        n_games = load_games_to_database(games_df)
                    
                    st.success(f"✅ Test data loaded: {n_games} games")
    
    with col2:
        st.subheader("🚀 Full Load")
        st.caption("Load all historical data (10 seasons)")
        
        if st.button("Load All Data", use_container_width=True):
            st.warning("⚠️ This will take 20-30 minutes. Are you sure?")
            
            if st.button("Yes, I'm sure - Load Everything"):
                # Download games
                with st.spinner("Downloading all games (this may take a few minutes)..."):
                    games_df = download_github_data("games")
                
                if games_df is not None:
                    # Load to database
                    with st.spinner("Loading to database..."):
                        n_games = load_games_to_database(games_df)
                    
                    st.success(f"✅ Full data loaded: {n_games:,} games")
                    st.balloons()
                
                # Download team stats
                with st.spinner("Downloading team stats..."):
                    team_df = download_github_data("team_stats")
                
                if team_df is not None:
                    with st.spinner("Loading team stats..."):
                        n_team = load_team_stats_to_database(team_df)
                    
                    st.success(f"✅ Team stats loaded: {n_team:,} rows")

# TAB 3: RUN SIMULATIONS
with tab3:
    st.header("Step 3: Run Simulations")
    st.markdown("""
    **Coming next:** This is where you'll discover patterns and correlations.
    
    For now, make sure Steps 1 and 2 are complete!
    """)
    
    st.info("🚧 Simulation features coming in the next update!")
    
    # Show data summary
    try:
        stats = validate_loaded_data()
        if stats['games'] > 0:
            st.success(f"✅ Ready to simulate with {stats['games']:,} historical games!")
        else:
            st.warning("⚠️ No data loaded yet. Complete Step 2 first.")
    except:
        st.error("❌ Database not set up. Complete Step 1 first.")

# TAB 4: RESULTS
with tab4:
    st.header("Step 4: View Results")
    st.markdown("""
    **Coming next:** View discovered correlations and patterns.
    
    This will show you things like:
    - Back-to-back fatigue effects
    - Lineup-specific win rates
    - Pace variance patterns
    - Player prop correlations
    """)
    
    st.info("🚧 Results viewer coming in the next update!")

# =======================
# Help Section
# =======================

with st.expander("❓ Help & Troubleshooting"):
    st.markdown("""
    ## Common Issues
    
    **"Missing DATABASE_URL in secrets"**
    - Go to your Streamlit app settings
    - Click "Secrets"
    - Add: `DATABASE_URL = "your_database_url_here"`
    
    **"Download failed"**
    - Check your internet connection
    - GitHub might be temporarily down
    - Try again in a few minutes
    
    **"Database connection failed"**
    - Make sure your database is running
    - Check that DATABASE_URL is correct
    - Verify your database allows connections
    
    ## Need More Help?
    
    Contact support or check the documentation.
    """)

# =======================
# Footer
# =======================

st.markdown("---")
st.caption("🔬 Simulation Engine v1.0 | Built for NBA edge research")
