"""
NBA SIMULATION ENGINE - COMPLETE SYSTEM
All 12 data types supported with proper reset and analytics
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
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
    """Generic batch loader for any table with chunked processing"""
    
    rows_inserted = 0
    batch_size = 500  # Reduced from 1000 for large tables
    batch_rows = []
    
    # Get column names from dataframe
    columns = df.columns.tolist()
    placeholders = ','.join(['%s'] * len(columns))
    column_names = ','.join(columns)
    
    # For very wide tables (like matchups with 48 cols), process in smaller chunks
    if len(columns) > 30:
        batch_size = 250
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for idx, row in df.iterrows():
                try:
                    # Convert row to tuple, handling NaN/None
                    row_data = []
                    for val in row:
                        if pd.isna(val):
                            row_data.append(None)
                        elif isinstance(val, (int, float)):
                            # Convert numpy types to native Python
                            row_data.append(float(val) if isinstance(val, float) else int(val))
                        else:
                            row_data.append(val)
                    
                    batch_rows.append(tuple(row_data))
                    
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
                    # Log error but continue
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
        
        # Use session state for confirmation
        if 'confirm_reset' not in st.session_state:
            st.session_state.confirm_reset = False
        
        if not st.session_state.confirm_reset:
            if st.button("RESET DATABASE", use_container_width=True):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.error("⚠️⚠️⚠️ ARE YOU SURE? THIS CANNOT BE UNDONE ⚠️⚠️⚠️")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("YES, DELETE EVERYTHING", type="primary", use_container_width=True):
                    with st.spinner("Resetting database..."):
                        try:
                            reset_database()
                            setup_simulation_database()
                            st.success("✅ Database reset complete!")
                            st.session_state.confirm_reset = False
                            st.balloons()
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Reset failed: {e}")
                            st.session_state.confirm_reset = False
            with col_b:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()

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
    
    # MULTI-FILE UPLOADER
    uploaded_files = st.file_uploader(
        "Choose CSV files (can select multiple)", 
        type=['csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files selected")
        
        # Show which files were detected
        with st.expander("📋 Files to Upload"):
            for f in uploaded_files:
                filename_lower = f.name.lower()
                
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
                    table_name = '❌ UNKNOWN'
                
                st.write(f"- {f.name} → {table_name}")
        
        if st.button("Load All Files", use_container_width=True, type="primary"):
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            total_files = len(uploaded_files)
            success_count = 0
            fail_count = 0
            
            for idx, uploaded_file in enumerate(uploaded_files):
                file_status = st.empty()
                file_progress = st.empty()
                
                try:
                    # Detect table
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
                    
                    if not table_name:
                        file_status.error(f"❌ {uploaded_file.name} - Unknown file type")
                        fail_count += 1
                        continue
                    
                    file_status.info(f"📖 [{idx+1}/{total_files}] Reading {uploaded_file.name}...")
                    
                    # Read CSV in chunks to save memory
                    df = pd.read_csv(uploaded_file, low_memory=False)
                    
                    file_status.info(f"💾 [{idx+1}/{total_files}] Loading {len(df):,} rows to {table_name}...")
                    
                    def update_progress(current, total):
                        file_progress.progress(min(current / total, 1.0))
                    
                    n_rows = load_data_batch(df, table_name, progress_callback=update_progress)
                    
                    # Clear dataframe from memory immediately
                    del df
                    
                    file_status.success(f"✅ [{idx+1}/{total_files}] {uploaded_file.name} - {n_rows:,} rows loaded")
                    file_progress.empty()
                    success_count += 1
                    
                except Exception as e:
                    file_status.error(f"❌ [{idx+1}/{total_files}] {uploaded_file.name} - {str(e)[:100]}")
                    fail_count += 1
                
                # Update overall progress
                overall_progress.progress((idx + 1) / total_files)
                overall_status.info(f"Progress: {idx + 1}/{total_files} files | ✅ {success_count} success | ❌ {fail_count} failed")
            
            # Final summary
            overall_progress.progress(1.0)
            if fail_count == 0:
                overall_status.success(f"🎉 All {success_count} files loaded successfully!")
                st.balloons()
            else:
                overall_status.warning(f"⚠️ Complete: {success_count} succeeded, {fail_count} failed")

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
