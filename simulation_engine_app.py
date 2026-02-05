"""
NBA SIMULATION ENGINE - UPDATED FOR REAL DATA STRUCTURE

This uses the official load_nba_data function from the GitHub repo.
Data types available:
- nbastats: Play-by-play from stats.nba.com
- shotdetail: Shot charts
- pbpstats: Possession-level stats
- datanba: data.nba.com format
- cdnnba: cdn.nba.com format  
- nbastatsv3: New NBA API format
- matchups: Matchup data
"""

import os
import tarfile
from pathlib import Path
from itertools import product
from urllib.request import urlopen
from typing import Union, Sequence, Optional, List
from io import BytesIO, TextIOWrapper
import csv

import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import psycopg
from psycopg.types.json import Json
import time

st.set_page_config(page_title="NBA Simulation Engine", layout="wide", page_icon="🔬")

# =======================
# Database Connection
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
# Official NBA Data Loader Function
# =======================

def load_nba_data(seasons: Union[Sequence, int] = range(2020, 2025),
                  data: Union[Sequence, str] = "nbastats",
                  seasontype: str = 'rg',
                  league: str = 'nba',
                  in_memory: bool = True,
                  use_pandas: bool = True) -> Optional[Union[List, pd.DataFrame]]:
    """
    Loading NBA play-by-play dataset from github repository
    
    Args:
        seasons: Year(s) of start of season (e.g., 2023 for 2023-24 season)
        data: Data type - 'nbastats', 'shotdetail', 'pbpstats', etc.
        seasontype: 'rg' (regular season) or 'po' (playoffs)
        league: 'nba' or 'wnba'
        in_memory: Load directly into memory (True) vs save to disk (False)
        use_pandas: Return pandas DataFrame (True) vs list (False)
    
    Returns:
        pd.DataFrame if use_pandas=True, List if use_pandas=False
    """
    
    if isinstance(seasons, int):
        seasons = (seasons,)
    if isinstance(data, str):
        data = (data,)

    if (len(data) > 1) & in_memory:
        raise ValueError("in_memory=True only works with single data type")

    # Build file names
    if seasontype == 'rg':
        need_data = tuple(["_".join([d, str(season)]) for (d, season) in product(data, seasons)])
    elif seasontype == 'po':
        need_data = tuple(["_".join([d, seasontype, str(season)]) 
                          for (d, seasontype, season) in product(data, (seasontype,), seasons)])
    else:
        need_data_rg = tuple(["_".join([d, str(season)]) for (d, season) in product(data, seasons)])
        need_data_po = tuple(["_".join([d, seasontype, str(season)]) 
                             for (d, seasontype, season) in product(data, ('po',), seasons)])
        need_data = need_data_rg + need_data_po
    
    if league.lower() == 'wnba':
        need_data = ['wnba_' + x for x in need_data]

    # Get file URLs from GitHub
    with urlopen("https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt") as f:
        v = f.read().decode('utf-8').strip()

    name_v = [string.split("=")[0] for string in v.split("\n")]
    element_v = [string.split("=")[1] for string in v.split("\n")]

    need_name = [name for name in name_v if name in need_data]
    need_element = [element for (name, element) in zip(name_v, element_v) if name in need_data]

    if in_memory:
        if use_pandas:
            table = pd.DataFrame()
        else:
            table = []
    
    for i in range(len(need_name)):
        with urlopen(need_element[i]) as response:
            if response.status != 200:
                raise Exception(f"Failed to download file: {response.status}")
            file_content = response.read()
            
            if in_memory:
                with tarfile.open(fileobj=BytesIO(file_content), mode='r:xz') as tar:
                    csv_file_name = "".join([need_name[i], ".csv"])
                    csv_file = tar.extractfile(csv_file_name)
                    
                    if use_pandas:
                        df_chunk = pd.read_csv(csv_file)
                        table = pd.concat([table, df_chunk], axis=0, ignore_index=True)
                    else:
                        csv_reader = csv.reader(TextIOWrapper(csv_file, encoding="utf-8"))
                        for row in csv_reader:
                            table.append(row)
    
    if in_memory:
        return table
    else:
        return None

# =======================
# Database Setup
# =======================

def setup_simulation_database():
    """Create tables for simulation engine"""
    
    ddl = """
    -- Play-by-play events (from nbastats)
    create table if not exists sim_play_by_play (
        pbp_id bigserial primary key,
        game_id text not null,
        eventnum integer,
        eventmsgtype integer,
        eventmsgactiontype integer,
        period integer,
        pctimestring text,
        homedescription text,
        visitordescription text,
        neutraldescription text,
        score text,
        scoremargin text,
        player1_id bigint,
        player1_name text,
        player1_team_id bigint,
        player1_team_abbreviation text,
        player2_id bigint,
        player2_name text,
        player2_team_id bigint,
        loaded_at timestamptz default now()
    );
    create index if not exists idx_sim_pbp_game on sim_play_by_play(game_id);
    create index if not exists idx_sim_pbp_period on sim_play_by_play(game_id, period);
    
    -- Shot details
    create table if not exists sim_shot_details (
        shot_id bigserial primary key,
        game_id text not null,
        game_event_id integer,
        player_id bigint,
        player_name text,
        team_id bigint,
        team_name text,
        period integer,
        minutes_remaining integer,
        seconds_remaining integer,
        event_type text,
        action_type text,
        shot_type text,
        shot_zone_basic text,
        shot_zone_area text,
        shot_zone_range text,
        shot_distance integer,
        loc_x integer,
        loc_y integer,
        shot_attempted_flag integer,
        shot_made_flag integer,
        game_date date,
        htm text,
        vtm text,
        loaded_at timestamptz default now()
    );
    create index if not exists idx_sim_shots_game on sim_shot_details(game_id);
    create index if not exists idx_sim_shots_player on sim_shot_details(player_name);
    
    -- Game summaries (derived from PBP)
    create table if not exists sim_game_summary (
        game_id text primary key,
        game_date date,
        season text,
        home_team text,
        away_team text,
        home_score integer,
        away_score integer,
        point_diff integer,
        total_points integer,
        loaded_at timestamptz default now()
    );
    
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
# Data Loading Functions
# =======================

def load_pbp_to_database(df: pd.DataFrame, progress_callback=None):
    """Load play-by-play data into database"""
    
    rows_inserted = 0
    batch_size = 1000
    
    # First, extract game summaries
    game_summaries = extract_game_summaries_from_pbp(df)
    
    # Load game summaries (each in its own transaction)
    for _, game in game_summaries.iterrows():
        try:
            with get_db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        insert into sim_game_summary(
                            game_id, season, home_team, away_team,
                            home_score, away_score, point_diff, total_points
                        )
                        values (%s,%s,%s,%s,%s,%s,%s,%s)
                        on conflict (game_id) do nothing
                        """,
                        (
                            str(game['game_id']),
                            str(game['season']),
                            str(game['home_team']),
                            str(game['away_team']),
                            int(game['home_score']),
                            int(game['away_score']),
                            int(game['point_diff']),
                            int(game['total_points'])
                        )
                    )
                conn.commit()
        except Exception as e:
            # Skip failed game summaries silently
            pass
    
    # Load play-by-play events
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for idx, row in df.iterrows():
                try:
                    cur.execute(
                        """
                        insert into sim_play_by_play(
                            game_id, eventnum, eventmsgtype, eventmsgactiontype,
                            period, pctimestring, homedescription, visitordescription,
                            neutraldescription, score, scoremargin,
                            player1_id, player1_name, player1_team_id, player1_team_abbreviation,
                            player2_id, player2_name, player2_team_id
                        )
                        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            str(row.get('GAME_ID', '')),
                            int(row.get('EVENTNUM', 0)) if pd.notna(row.get('EVENTNUM')) else None,
                            int(row.get('EVENTMSGTYPE', 0)) if pd.notna(row.get('EVENTMSGTYPE')) else None,
                            int(row.get('EVENTMSGACTIONTYPE', 0)) if pd.notna(row.get('EVENTMSGACTIONTYPE')) else None,
                            int(row.get('PERIOD', 0)) if pd.notna(row.get('PERIOD')) else None,
                            str(row.get('PCTIMESTRING', '')) if pd.notna(row.get('PCTIMESTRING')) else None,
                            str(row.get('HOMEDESCRIPTION', '')) if pd.notna(row.get('HOMEDESCRIPTION')) else None,
                            str(row.get('VISITORDESCRIPTION', '')) if pd.notna(row.get('VISITORDESCRIPTION')) else None,
                            str(row.get('NEUTRALDESCRIPTION', '')) if pd.notna(row.get('NEUTRALDESCRIPTION')) else None,
                            str(row.get('SCORE', '')) if pd.notna(row.get('SCORE')) else None,
                            str(row.get('SCOREMARGIN', '')) if pd.notna(row.get('SCOREMARGIN')) else None,
                            int(row.get('PLAYER1_ID', 0)) if pd.notna(row.get('PLAYER1_ID')) else None,
                            str(row.get('PLAYER1_NAME', '')) if pd.notna(row.get('PLAYER1_NAME')) else None,
                            int(row.get('PLAYER1_TEAM_ID', 0)) if pd.notna(row.get('PLAYER1_TEAM_ID')) else None,
                            str(row.get('PLAYER1_TEAM_ABBREVIATION', '')) if pd.notna(row.get('PLAYER1_TEAM_ABBREVIATION')) else None,
                            int(row.get('PLAYER2_ID', 0)) if pd.notna(row.get('PLAYER2_ID')) else None,
                            str(row.get('PLAYER2_NAME', '')) if pd.notna(row.get('PLAYER2_NAME')) else None,
                            int(row.get('PLAYER2_TEAM_ID', 0)) if pd.notna(row.get('PLAYER2_TEAM_ID')) else None,
                        )
                    )
                    
                    rows_inserted += 1
                    
                    if rows_inserted % batch_size == 0:
                        conn.commit()
                        if progress_callback:
                            progress_callback(rows_inserted, len(df))
                
                except Exception as e:
                    continue
            
            conn.commit()
    
    return rows_inserted

def extract_game_summaries_from_pbp(df: pd.DataFrame) -> pd.DataFrame:
    """Extract game-level summaries from play-by-play data"""
    
    # Get unique games
    games = df['GAME_ID'].unique()
    
    summaries = []
    for game_id in games:
        try:
            game_df = df[df['GAME_ID'] == game_id].sort_values(['PERIOD', 'EVENTNUM'])
            
            # Get last event (should have final score)
            last_event = game_df.iloc[-1]
            
            # Try to parse score from SCORE field
            score_str = str(last_event.get('SCORE', ''))
            
            if score_str and '-' in score_str:
                parts = score_str.strip().split('-')
                if len(parts) == 2:
                    try:
                        score1 = int(parts[0].strip())
                        score2 = int(parts[1].strip())
                        
                        # Get team abbreviations
                        home_team = 'UNK'
                        away_team = 'UNK'
                        
                        # Try to get home team from first event with home description
                        home_events = game_df[game_df['HOMEDESCRIPTION'].notna()]
                        if not home_events.empty:
                            home_team = str(home_events.iloc[0].get('PLAYER1_TEAM_ABBREVIATION', 'UNK'))
                        
                        # Try to get away team from first event with visitor description  
                        away_events = game_df[game_df['VISITORDESCRIPTION'].notna()]
                        if not away_events.empty:
                            away_team = str(away_events.iloc[0].get('PLAYER1_TEAM_ABBREVIATION', 'UNK'))
                        
                        # Parse season from GAME_ID (format: 00XSSSSSSSS where SS is season year)
                        season = '2023-24'  # Default
                        try:
                            game_id_str = str(game_id)
                            if len(game_id_str) >= 5:
                                season_code = game_id_str[3:5]
                                season_year = 2000 + int(season_code)
                                season = f"{season_year}-{str(season_year + 1)[-2:]}"
                        except:
                            pass
                        
                        summaries.append({
                            'game_id': str(game_id),
                            'season': season,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': score1,
                            'away_score': score2,
                            'point_diff': score1 - score2,
                            'total_points': score1 + score2
                        })
                    except (ValueError, TypeError):
                        pass
        except Exception:
            continue
    
    return pd.DataFrame(summaries)

# =======================
# Data Validation
# =======================

def validate_loaded_data():
    """Check what data we have loaded"""
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Count PBP events
            cur.execute("SELECT COUNT(*) FROM sim_play_by_play")
            pbp_count = cur.fetchone()[0] or 0
            
            # Count games
            cur.execute("SELECT COUNT(*) FROM sim_game_summary")
            game_count = cur.fetchone()[0] or 0
            
            # Count shots
            cur.execute("SELECT COUNT(*) FROM sim_shot_details")
            shot_count = cur.fetchone()[0] or 0
    
    return {
        "pbp_events": pbp_count,
        "games": game_count,
        "shots": shot_count,
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
    st.sidebar.metric("Play-by-Play Events", f"{stats['pbp_events']:,}")
    st.sidebar.metric("Games", f"{stats['games']:,}")
    st.sidebar.metric("Shot Details", f"{stats['shots']:,}")
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
    **What this does:** Creates the tables needed to store NBA play-by-play data.
    
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
    **Available Data Types:**
    - **nbastats**: Play-by-play from stats.nba.com (recommended to start)
    - **shotdetail**: Shot charts with coordinates
    - **pbpstats**: Possession-level statistics
    
    **How long it takes:**
    - Test mode (1 season): ~5-10 minutes
    - Full load (4 seasons): ~30-60 minutes
    
    **Storage impact:**
    - 1 season PBP: ~500MB
    - 4 seasons PBP: ~2GB
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Test Mode")
        st.caption("Load 1 season (2023-24) to test")
        
        data_type = st.selectbox("Data type", ["nbastats", "shotdetail", "pbpstats"], key="test_data")
        
        if st.button("Load Test Data (1 Season)", use_container_width=True):
            with st.spinner(f"Downloading {data_type} for 2023-24 season..."):
                try:
                    # Load data
                    df = load_nba_data(
                        seasons=2023,
                        data=data_type,
                        seasontype='rg',
                        in_memory=True,
                        use_pandas=True
                    )
                    
                    if df is not None and not df.empty:
                        st.success(f"✅ Downloaded {len(df):,} rows")
                        
                        # Show preview
                        st.dataframe(df.head(10))
                        
                        # Load to database
                        with st.spinner("Loading to database..."):
                            if data_type == "nbastats":
                                n_rows = load_pbp_to_database(df)
                                st.success(f"✅ Loaded {n_rows:,} play-by-play events")
                            else:
                                st.info(f"{data_type} loading function coming soon!")
                    else:
                        st.error("No data returned")
                        
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
                    st.code(str(e))
    
    with col2:
        st.subheader("🚀 Full Load")
        st.caption("Load multiple seasons")
        
        start_season = st.number_input("Start season", min_value=2014, max_value=2024, value=2020)
        end_season = st.number_input("End season", min_value=2014, max_value=2024, value=2023)
        full_data_type = st.selectbox("Data type", ["nbastats", "shotdetail", "pbpstats"], key="full_data")
        
        if st.button("Load Multiple Seasons", use_container_width=True):
            st.warning(f"⚠️ This will load {end_season - start_season + 1} seasons. This may take 30-60 minutes.")
            
            if st.button("Yes, I'm sure - Start Loading"):
                seasons_to_load = range(start_season, end_season + 1)
                
                for season in seasons_to_load:
                    with st.spinner(f"Loading season {season}-{season+1}..."):
                        try:
                            df = load_nba_data(
                                seasons=season,
                                data=full_data_type,
                                seasontype='rg',
                                in_memory=True,
                                use_pandas=True
                            )
                            
                            if df is not None and not df.empty:
                                if full_data_type == "nbastats":
                                    n_rows = load_pbp_to_database(df)
                                    st.success(f"✅ Season {season}: {n_rows:,} events")
                        except Exception as e:
                            st.error(f"❌ Season {season} failed: {e}")

# TAB 3: RUN SIMULATIONS
with tab3:
    st.header("Step 3: Run Simulations")
    st.info("🚧 Simulation features coming in the next update!")
    
    try:
        stats = validate_loaded_data()
        if stats['pbp_events'] > 0:
            st.success(f"✅ Ready to simulate with {stats['pbp_events']:,} play-by-play events!")
        else:
            st.warning("⚠️ No data loaded yet. Complete Step 2 first.")
    except:
        st.error("❌ Database not set up. Complete Step 1 first.")

# TAB 4: RESULTS
with tab4:
    st.header("Step 4: View Results")
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
    
    **"Download failed" or "404 error"**
    - The GitHub repo uses compressed archives
    - Make sure you're using the official load_nba_data function
    - Check that the season exists (data goes back to 1996)
    
    **"Database connection failed"**
    - Make sure your database is running
    - Check that DATABASE_URL is correct
    - Verify your database allows connections
    
    ## Data Types Explained
    
    - **nbastats**: Official NBA play-by-play (recommended)
    - **shotdetail**: Every shot with X/Y coordinates
    - **pbpstats**: Possession-level aggregations
    - **datanba**: Alternative PBP format
    - **cdnnba**: CDN format (newer)
    - **nbastatsv3**: Latest API version
    """)

st.markdown("---")
st.caption("🔬 Simulation Engine v2.0 | Using official NBA data loader")
