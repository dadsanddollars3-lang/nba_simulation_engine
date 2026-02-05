"""
NBA SIMULATION ENGINE - UPDATED FOR REAL DATA STRUCTURE

This uses the official load_nba_data function from the GitHub repo.
Data types available:
- nbastats: Play-by-play from stats.nba.com
- shotdetail: Shot charts
- pbpstats: Possession-level stats
- datanba: data.nba.com format"""
NBA SIMULATION ENGINE - COMPLETE SYSTEM
All 12 data types supported with proper reset and analytics
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import psycopg
from psycopg.types.json import Json

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
    if not DATABASE_URL_RAW:
        raise RuntimeError("Missing DATABASE_URL in secrets")
    return psycopg.connect(DATABASE_URL_RAW, connect_timeout=12)

# =======================
# Database Reset (WORKING)
# =======================

def reset_database():
    """DROP ALL TABLES - Complete reset"""
    drop_sql = """
    DROP TABLE IF EXISTS sim_datanba CASCADE;
    DROP TABLE IF EXISTS sim_datanba_po CASCADE;
    DROP TABLE IF EXISTS sim_matchups CASCADE;
    DROP TABLE IF EXISTS sim_matchups_po CASCADE;
    DROP TABLE IF EXISTS sim_nbastats CASCADE;
    DROP TABLE IF EXISTS sim_nbastats_po CASCADE;
    DROP TABLE IF EXISTS sim_nbastatsv3 CASCADE;
    DROP TABLE IF EXISTS sim_nbastatsv3_po CASCADE;
    DROP TABLE IF EXISTS sim_pbpstats CASCADE;
    DROP TABLE IF EXISTS sim_pbpstats_po CASCADE;
    DROP TABLE IF EXISTS sim_shotdetail CASCADE;
    DROP TABLE IF EXISTS sim_shotdetail_po CASCADE;
    DROP TABLE IF EXISTS sim_results CASCADE;
    DROP TABLE IF EXISTS sim_correlations CASCADE;
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(drop_sql)
        conn.commit()

# =======================
# Database Setup
# =======================

def setup_simulation_database():
    """Create all tables"""
    
    ddl = """
    -- DATANBA (data.nba format)
    create table if not exists sim_datanba (
        id bigserial primary key,
        evt integer,
        wallclk text,
        cl text,
        de text,
        locX integer,
        locY integer,
        opt1 integer,
        opt2 text,
        opt3 text,
        opt4 text,
        mtype integer,
        etype integer,
        opid text,
        tid integer,
        pid integer,
        hs integer,
        vs integer,
        epid text,
        oftid integer,
        ord integer,
        pts integer,
        PERIOD integer,
        GAME_ID text
    );
    create index if not exists idx_datanba_game on sim_datanba(GAME_ID);
    
    create table if not exists sim_datanba_po (
        id bigserial primary key,
        evt integer,
        wallclk text,
        cl text,
        de text,
        locX integer,
        locY integer,
        opt1 integer,
        opt2 text,
        opt3 text,
        opt4 text,
        mtype integer,
        etype integer,
        opid text,
        tid integer,
        pid integer,
        hs integer,
        vs integer,
        epid text,
        oftid integer,
        ord integer,
        pts integer,
        PERIOD integer,
        GAME_ID text
    );
    create index if not exists idx_datanba_po_game on sim_datanba_po(GAME_ID);
    
    -- MATCHUPS
    create table if not exists sim_matchups (
        id bigserial primary key,
        game_id text,
        away_team_id bigint,
        home_team_id bigint,
        team_id bigint,
        team_name text,
        team_city text,
        team_tricode text,
        team_slug text,
        person_id bigint,
        first_name text,
        family_name text,
        name_i text,
        player_slug text,
        position text,
        comment text,
        jersey_num text,
        matchups_person_id bigint,
        matchups_first_name text,
        matchups_family_name text,
        matchups_name_i text,
        matchups_player_slug text,
        matchups_jersey_num text,
        matchup_minutes numeric,
        matchup_minutes_sort numeric,
        partial_possessions numeric,
        percentage_defender_total_time numeric,
        percentage_offensive_total_time numeric,
        percentage_total_time_both_on numeric,
        switches_on integer,
        player_points integer,
        team_points integer,
        matchup_assists integer,
        matchup_potential_assists integer,
        matchup_turnovers integer,
        matchup_blocks integer,
        matchup_field_goals_made integer,
        matchup_field_goals_attempted integer,
        matchup_field_goals_percentage numeric,
        matchup_three_pointers_made integer,
        matchup_three_pointers_attempted integer,
        matchup_three_pointers_percentage numeric,
        help_blocks integer,
        help_field_goals_made integer,
        help_field_goals_attempted integer,
        help_field_goals_percentage numeric,
        matchup_free_throws_made integer,
        matchup_free_throws_attempted integer,
        shooting_fouls integer
    );
    create index if not exists idx_matchups_game on sim_matchups(game_id);
    create index if not exists idx_matchups_player on sim_matchups(person_id);
    
    create table if not exists sim_matchups_po (
        id bigserial primary key,
        game_id text,
        away_team_id bigint,
        home_team_id bigint,
        team_id bigint,
        team_name text,
        team_city text,
        team_tricode text,
        team_slug text,
        person_id bigint,
        first_name text,
        family_name text,
        name_i text,
        player_slug text,
        position text,
        comment text,
        jersey_num text,
        matchups_person_id bigint,
        matchups_first_name text,
        matchups_family_name text,
        matchups_name_i text,
        matchups_player_slug text,
        matchups_jersey_num text,
        matchup_minutes numeric,
        matchup_minutes_sort numeric,
        partial_possessions numeric,
        percentage_defender_total_time numeric,
        percentage_offensive_total_time numeric,
        percentage_total_time_both_on numeric,
        switches_on integer,
        player_points integer,
        team_points integer,
        matchup_assists integer,
        matchup_potential_assists integer,
        matchup_turnovers integer,
        matchup_blocks integer,
        matchup_field_goals_made integer,
        matchup_field_goals_attempted integer,
        matchup_field_goals_percentage numeric,
        matchup_three_pointers_made integer,
        matchup_three_pointers_attempted integer,
        matchup_three_pointers_percentage numeric,
        help_blocks integer,
        help_field_goals_made integer,
        help_field_goals_attempted integer,
        help_field_goals_percentage numeric,
        matchup_free_throws_made integer,
        matchup_free_throws_attempted integer,
        shooting_fouls integer
    );
    create index if not exists idx_matchups_po_game on sim_matchups_po(game_id);
    create index if not exists idx_matchups_po_player on sim_matchups_po(person_id);
    
    -- NBASTATS (stats.nba.com format)
    create table if not exists sim_nbastats (
        id bigserial primary key,
        GAME_ID text,
        EVENTNUM integer,
        EVENTMSGTYPE integer,
        EVENTMSGACTIONTYPE integer,
        PERIOD integer,
        WCTIMESTRING text,
        PCTIMESTRING text,
        HOMEDESCRIPTION text,
        NEUTRALDESCRIPTION text,
        VISITORDESCRIPTION text,
        SCORE text,
        SCOREMARGIN text,
        PERSON1TYPE integer,
        PLAYER1_ID bigint,
        PLAYER1_NAME text,
        PLAYER1_TEAM_ID bigint,
        PLAYER1_TEAM_CITY text,
        PLAYER1_TEAM_NICKNAME text,
        PLAYER1_TEAM_ABBREVIATION text,
        PERSON2TYPE integer,
        PLAYER2_ID bigint,
        PLAYER2_NAME text,
        PLAYER2_TEAM_ID bigint,
        PLAYER2_TEAM_CITY text,
        PLAYER2_TEAM_NICKNAME text,
        PLAYER2_TEAM_ABBREVIATION text,
        PERSON3TYPE integer,
        PLAYER3_ID bigint,
        PLAYER3_NAME text,
        PLAYER3_TEAM_ID bigint,
        PLAYER3_TEAM_CITY text,
        PLAYER3_TEAM_NICKNAME text,
        PLAYER3_TEAM_ABBREVIATION text,
        VIDEO_AVAILABLE_FLAG integer
    );
    create index if not exists idx_nbastats_game on sim_nbastats(GAME_ID);
    
    create table if not exists sim_nbastats_po (
        id bigserial primary key,
        GAME_ID text,
        EVENTNUM integer,
        EVENTMSGTYPE integer,
        EVENTMSGACTIONTYPE integer,
        PERIOD integer,
        WCTIMESTRING text,
        PCTIMESTRING text,
        HOMEDESCRIPTION text,
        NEUTRALDESCRIPTION text,
        VISITORDESCRIPTION text,
        SCORE text,
        SCOREMARGIN text,
        PERSON1TYPE integer,
        PLAYER1_ID bigint,
        PLAYER1_NAME text,
        PLAYER1_TEAM_ID bigint,
        PLAYER1_TEAM_CITY text,
        PLAYER1_TEAM_NICKNAME text,
        PLAYER1_TEAM_ABBREVIATION text,
        PERSON2TYPE integer,
        PLAYER2_ID bigint,
        PLAYER2_NAME text,
        PLAYER2_TEAM_ID bigint,
        PLAYER2_TEAM_CITY text,
        PLAYER2_TEAM_NICKNAME text,
        PLAYER2_TEAM_ABBREVIATION text,
        PERSON3TYPE integer,
        PLAYER3_ID bigint,
        PLAYER3_NAME text,
        PLAYER3_TEAM_ID bigint,
        PLAYER3_TEAM_CITY text,
        PLAYER3_TEAM_NICKNAME text,
        PLAYER3_TEAM_ABBREVIATION text,
        VIDEO_AVAILABLE_FLAG integer
    );
    create index if not exists idx_nbastats_po_game on sim_nbastats_po(GAME_ID);
    
    -- NBASTATSV3 (version 3 format)
    create table if not exists sim_nbastatsv3 (
        id bigserial primary key,
        actionNumber integer,
        clock text,
        period integer,
        teamId bigint,
        teamTricode text,
        personId bigint,
        playerName text,
        playerNameI text,
        xLegacy integer,
        yLegacy integer,
        shotDistance integer,
        shotResult text,
        isFieldGoal integer,
        scoreHome integer,
        scoreAway integer,
        pointsTotal integer,
        location text,
        description text,
        actionType text,
        subType text,
        videoAvailable integer,
        shotValue integer,
        actionId integer,
        gameId text
    );
    create index if not exists idx_nbastatsv3_game on sim_nbastatsv3(gameId);
    
    create table if not exists sim_nbastatsv3_po (
        id bigserial primary key,
        actionNumber integer,
        clock text,
        period integer,
        teamId bigint,
        teamTricode text,
        personId bigint,
        playerName text,
        playerNameI text,
        xLegacy integer,
        yLegacy integer,
        shotDistance integer,
        shotResult text,
        isFieldGoal integer,
        scoreHome integer,
        scoreAway integer,
        pointsTotal integer,
        location text,
        description text,
        actionType text,
        subType text,
        videoAvailable integer,
        shotValue integer,
        actionId integer,
        gameId text
    );
    create index if not exists idx_nbastatsv3_po_game on sim_nbastatsv3_po(gameId);
    
    -- PBPSTATS (pbpstats.com format)
    create table if not exists sim_pbpstats (
        id bigserial primary key,
        ENDTIME text,
        EVENTS text,
        FG2A integer,
        FG2M integer,
        FG3A integer,
        FG3M integer,
        GAMEDATE text,
        GAMEID text,
        NONSHOOTINGFOULSTHATRESULTEDINFTS integer,
        OFFENSIVEREBOUNDS integer,
        OPPONENT text,
        PERIOD integer,
        SHOOTINGFOULSDRAWN integer,
        STARTSCOREDIFFERENTIAL integer,
        STARTTIME text,
        STARTTYPE text,
        TURNOVERS integer,
        DESCRIPTION text,
        URL text
    );
    create index if not exists idx_pbpstats_game on sim_pbpstats(GAMEID);
    
    create table if not exists sim_pbpstats_po (
        id bigserial primary key,
        ENDTIME text,
        EVENTS text,
        FG2A integer,
        FG2M integer,
        FG3A integer,
        FG3M integer,
        GAMEDATE text,
        GAMEID text,
        NONSHOOTINGFOULSTHATRESULTEDINFTS integer,
        OFFENSIVEREBOUNDS integer,
        OPPONENT text,
        PERIOD integer,
        SHOOTINGFOULSDRAWN integer,
        STARTSCOREDIFFERENTIAL integer,
        STARTTIME text,
        STARTTYPE text,
        TURNOVERS integer,
        DESCRIPTION text,
        URL text
    );
    create index if not exists idx_pbpstats_po_game on sim_pbpstats_po(GAMEID);
    
    -- SHOTDETAIL
    create table if not exists sim_shotdetail (
        id bigserial primary key,
        GRID_TYPE text,
        GAME_ID text,
        GAME_EVENT_ID integer,
        PLAYER_ID bigint,
        PLAYER_NAME text,
        TEAM_ID bigint,
        TEAM_NAME text,
        PERIOD integer,
        MINUTES_REMAINING integer,
        SECONDS_REMAINING integer,
        EVENT_TYPE text,
        ACTION_TYPE text,
        SHOT_TYPE text,
        SHOT_ZONE_BASIC text,
        SHOT_ZONE_AREA text,
        SHOT_ZONE_RANGE text,
        SHOT_DISTANCE integer,
        LOC_X integer,
        LOC_Y integer,
        SHOT_ATTEMPTED_FLAG integer,
        SHOT_MADE_FLAG integer,
        GAME_DATE text,
        HTM text,
        VTM text
    );
    create index if not exists idx_shotdetail_game on sim_shotdetail(GAME_ID);
    
    create table if not exists sim_shotdetail_po (
        id bigserial primary key,
        GRID_TYPE text,
        GAME_ID text,
        GAME_EVENT_ID integer,
        PLAYER_ID bigint,
        PLAYER_NAME text,
        TEAM_ID bigint,
        TEAM_NAME text,
        PERIOD integer,
        MINUTES_REMAINING integer,
        SECONDS_REMAINING integer,
        EVENT_TYPE text,
        ACTION_TYPE text,
        SHOT_TYPE text,
        SHOT_ZONE_BASIC text,
        SHOT_ZONE_AREA text,
        SHOT_ZONE_RANGE text,
        SHOT_DISTANCE integer,
        LOC_X integer,
        LOC_Y integer,
        SHOT_ATTEMPTED_FLAG integer,
        SHOT_MADE_FLAG integer,
        GAME_DATE text,
        HTM text,
        VTM text
    );
    create index if not exists idx_shotdetail_po_game on sim_shotdetail_po(GAME_ID);
    
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
# Generic Batch Loader
# =======================

def load_data_batch(df: pd.DataFrame, table_name: str, progress_callback=None):
    """Generic batch loader for any table"""
    
    rows_inserted = 0
    batch_size = 1000
    batch_rows = []
    
    # Get column names from dataframe
    columns = df.columns.tolist()
    placeholders = ','.join(['%s'] * len(columns))
    column_names = ','.join(columns)
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for idx, row in df.iterrows():
                try:
                    # Convert row to tuple
                    row_data = tuple(None if pd.isna(val) else val for val in row)
                    batch_rows.append(row_data)
                    
                    # Insert batch when full
                    if len(batch_rows) >= batch_size:
                        cur.executemany(
                            f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})",
                            batch_rows
                        )
                        conn.commit()
                        rows_inserted += len(batch_rows)
                        batch_rows = []
                        
                        if progress_callback:
                            progress_callback(rows_inserted, len(df))
                
                except Exception as e:
                    continue
            
            # Insert remaining rows
            if batch_rows:
                cur.executemany(
                    f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})",
                    batch_rows
                )
                conn.commit()
                rows_inserted += len(batch_rows)
    
    return rows_inserted

# =======================
# Data Quality Stats
# =======================

def get_data_stats():
    """Get stats for all tables"""
    
    tables = [
        'sim_datanba', 'sim_datanba_po',
        'sim_matchups', 'sim_matchups_po',
        'sim_nbastats', 'sim_nbastats_po',
        'sim_nbastatsv3', 'sim_nbastatsv3_po',
        'sim_pbpstats', 'sim_pbpstats_po',
        'sim_shotdetail', 'sim_shotdetail_po'
    ]
    
    stats = {}
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for table in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    stats[table] = count or 0
                except:
                    stats[table] = 0
    
    return stats

# =======================
# UI
# =======================

st.title("🔬 NBA Simulation Engine - Complete System")
st.caption("All 12 data types supported | Proper reset | Analytics dashboard")

# Sidebar stats
st.sidebar.header("📊 Database Status")
try:
    stats = get_data_stats()
    total_rows = sum(stats.values())
    st.sidebar.metric("Total Rows", f"{total_rows:,}")
    
    with st.sidebar.expander("Details by Type"):
        for table, count in stats.items():
            if count > 0:
                st.metric(table.replace('sim_', ''), f"{count:,}")
except:
    st.sidebar.warning("Database not set up")

# =======================
# Tabs
# =======================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 Setup & Reset",
    "📤 Upload Data",
    "📊 Data Quality",
    "🎯 Simulations"
])

# TAB 1: SETUP & RESET
with tab1:
    st.header("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Initial Setup")
        if st.button("Create All Tables", use_container_width=True, type="primary"):
            with st.spinner("Creating tables..."):
                try:
                    setup_simulation_database()
                    st.success("✅ All tables created!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
    
    with col2:
        st.subheader("🗑️ Complete Reset")
        st.warning("⚠️ This will DELETE ALL DATA")
        if st.button("RESET DATABASE", use_container_width=True):
            if st.button("⚠️ Confirm - Delete Everything"):
                with st.spinner("Resetting database..."):
                    try:
                        reset_database()
                        setup_simulation_database()
                        st.success("✅ Database reset complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Reset failed: {e}")

# TAB 2: UPLOAD DATA
with tab2:
    st.header("Upload Data Files")
    
    st.info("""
    **Supported file types:**
    - datanba_YYYY.csv / datanba_po_YYYY.csv
    - matchups_YYYY.csv / matchups_po_YYYY.csv
    - nbastats_YYYY.csv / nbastats_po_YYYY.csv
    - nbastatsv3_YYYY.csv / nbastatsv3_po_YYYY.csv
    - pbpstats_YYYY.csv / pbpstats_po_YYYY.csv
    - shotdetail_YYYY.csv / shotdetail_po_YYYY.csv
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.success(f"✅ File selected: {uploaded_file.name}")
        
        # Auto-detect table
        filename_lower = uploaded_file.name.lower()
        
        if 'datanba_po' in filename_lower:
            table_name = 'sim_datanba_po'
        elif 'datanba' in filename_lower:
            table_name = 'sim_datanba'
        elif 'matchups_po' in filename_lower:
            table_name = 'sim_matchups_po'
        elif 'matchups' in filename_lower:
            table_name = 'sim_matchups'
        elif 'nbastatsv3_po' in filename_lower:
            table_name = 'sim_nbastatsv3_po'
        elif 'nbastatsv3' in filename_lower:
            table_name = 'sim_nbastatsv3'
        elif 'nbastats_po' in filename_lower:
            table_name = 'sim_nbastats_po'
        elif 'nbastats' in filename_lower:
            table_name = 'sim_nbastats'
        elif 'pbpstats_po' in filename_lower:
            table_name = 'sim_pbpstats_po'
        elif 'pbpstats' in filename_lower:
            table_name = 'sim_pbpstats'
        elif 'shotdetail_po' in filename_lower:
            table_name = 'sim_shotdetail_po'
        elif 'shotdetail' in filename_lower:
            table_name = 'sim_shotdetail'
        else:
            table_name = None
        
        if table_name:
            st.info(f"📊 Detected table: **{table_name}**")
        else:
            st.error("❌ Could not detect data type from filename")
        
        if table_name and st.button("Load This File", use_container_width=True, type="primary"):
            status_container = st.empty()
            progress_container = st.empty()
            
            try:
                status_container.info(f"📖 Reading {uploaded_file.name}...")
                df = pd.read_csv(uploaded_file)
                
                status_container.success(f"✅ Loaded {len(df):,} rows")
                
                status_container.info(f"💾 Loading to {table_name}...")
                progress_bar = progress_container.progress(0)
                
                def update_progress(current, total):
                    progress_bar.progress(min(current / total, 1.0))
                
                n_rows = load_data_batch(df, table_name, progress_callback=update_progress)
                
                progress_bar.progress(1.0)
                status_container.success(f"✅ Loaded {n_rows:,} rows to {table_name}!")
                st.balloons()
                
            except Exception as e:
                status_container.error(f"❌ Failed: {e}")
                st.code(str(e))

# TAB 3: DATA QUALITY
with tab3:
    st.header("Data Quality Dashboard")
    
    if st.button("🔄 Refresh Stats"):
        st.rerun()
    
    try:
        stats = get_data_stats()
        
        # Regular season vs Playoffs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regular Season Data")
            st.metric("datanba", f"{stats['sim_datanba']:,}")
            st.metric("matchups", f"{stats['sim_matchups']:,}")
            st.metric("nbastats", f"{stats['sim_nbastats']:,}")
            st.metric("nbastatsv3", f"{stats['sim_nbastatsv3']:,}")
            st.metric("pbpstats", f"{stats['sim_pbpstats']:,}")
            st.metric("shotdetail", f"{stats['sim_shotdetail']:,}")
        
        with col2:
            st.subheader("Playoff Data")
            st.metric("datanba_po", f"{stats['sim_datanba_po']:,}")
            st.metric("matchups_po", f"{stats['sim_matchups_po']:,}")
            st.metric("nbastats_po", f"{stats['sim_nbastats_po']:,}")
            st.metric("nbastatsv3_po", f"{stats['sim_nbastatsv3_po']:,}")
            st.metric("pbpstats_po", f"{stats['sim_pbpstats_po']:,}")
            st.metric("shotdetail_po", f"{stats['sim_shotdetail_po']:,}")
        
        # Total
        total = sum(stats.values())
        st.metric("**TOTAL ROWS**", f"{total:,}")
        
    except Exception as e:
        st.error(f"Error loading stats: {e}")

# TAB 4: SIMULATIONS
with tab4:
    st.header("Simulations & Analysis")
    st.info("🚧 Coming next: Correlation discovery and simulation tools")

st.markdown("---")
st.caption("🔬 Complete NBA Simulation Engine | All data types supported")
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
    """Load play-by-play data into database using batch inserts"""
    
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
    
    # Load play-by-play events in BATCHES (much faster)
    batch_rows = []
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for idx, row in df.iterrows():
                try:
                    batch_rows.append((
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
                    ))
                    
                    # When batch is full, insert all at once
                    if len(batch_rows) >= batch_size:
                        cur.executemany(
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
                            batch_rows
                        )
                        conn.commit()
                        rows_inserted += len(batch_rows)
                        batch_rows = []
                        
                        if progress_callback:
                            progress_callback(rows_inserted, len(df))
                
                except Exception as e:
                    continue
            
            # Insert remaining rows
            if batch_rows:
                cur.executemany(
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
                    batch_rows
                )
                conn.commit()
                rows_inserted += len(batch_rows)
    
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
        st.subheader("📤 Upload Your Data")
        st.caption("Upload NBA data CSV files (one year at a time)")
        
        st.info("""
        **Supported files:**
        - nbastats_YYYY.csv (play-by-play)
        - datanba_YYYY.csv (alternative format)
        - shotdetail_YYYY.csv (shot charts)
        
        **Start with ONE year to test!**
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            st.success(f"✅ File selected: {uploaded_file.name}")
            
            if st.button("Load This File", use_container_width=True, type="primary"):
                status_container = st.empty()
                progress_container = st.empty()
                preview_container = st.empty()
                
                try:
                    # Read CSV
                    status_container.info(f"📖 Reading {uploaded_file.name}...")
                    df = pd.read_csv(uploaded_file)
                    
                    status_container.success(f"✅ Loaded {len(df):,} rows from {uploaded_file.name}")
                    
                    # Show preview
                    with preview_container.expander("📊 Preview Data"):
                        st.dataframe(df.head(20))
                        st.caption(f"Total rows: {len(df):,}")
                        st.caption(f"Columns: {', '.join(df.columns.tolist()[:10])}")
                        
                        if 'GAME_ID' in df.columns:
                            unique_games = df['GAME_ID'].nunique()
                            st.caption(f"Unique games: {unique_games:,}")
                            sample_ids = df['GAME_ID'].unique()[:3]
                            st.caption(f"Sample Game IDs: {sample_ids}")
                    
                    # Determine data type from filename
                    filename_lower = uploaded_file.name.lower()
                    if 'nbastats' in filename_lower or 'datanba' in filename_lower:
                        data_type = "play-by-play"
                    elif 'shot' in filename_lower:
                        data_type = "shot details"
                    else:
                        data_type = "unknown"
                    
                    # Load to database
                    status_container.info(f"💾 Loading {len(df):,} rows to database...")
                    progress_bar = progress_container.progress(0)
                    
                    if data_type == "play-by-play":
                        def update_progress(current, total):
                            progress_bar.progress(min(current / total, 1.0))
                        
                        n_rows = load_pbp_to_database(df, progress_callback=update_progress)
                        progress_bar.progress(1.0)
                        status_container.success(f"✅ Loaded {n_rows:,} play-by-play events!")
                        st.balloons()
                    else:
                        status_container.warning(f"⚠️ {data_type} loading not yet implemented")
                
                except Exception as e:
                    status_container.error(f"❌ Failed: {e}")
                    st.code(str(e))
    
    with col2:
        st.subheader("💾 Batch Upload")
        st.caption("Upload multiple years at once")
        
        st.info("""
        **Coming soon:** 
        Upload multiple CSV files and load them sequentially.
        
        **For now:** Upload one file at a time on the left.
        """)
        
        st.caption("Recommended: Start with 1 year, verify it works, then add more.")

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
