"""Streamlit Web Interface for Consciousness Learning System."""

import streamlit as st
import threading
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import queue
import os

# Custom Streamlit log handler


class StreamlitLogHandler(logging.Handler):
    """Custom log handler that stores logs in Streamlit session state."""

    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3],
                'level': record.levelname,
                'message': self.format(record),
                'full_timestamp': datetime.fromtimestamp(record.created)
            }

            # Store in session state if available
            if hasattr(st, 'session_state') and 'app_logs' in st.session_state:
                st.session_state.app_logs.append(log_entry)
                # Keep only last 100 log entries
                if len(st.session_state.app_logs) > 100:
                    st.session_state.app_logs = st.session_state.app_logs[-100:]
            else:
                # Fallback to instance storage
                self.logs.append(log_entry)
                if len(self.logs) > 100:
                    self.logs = self.logs[-100:]
        except Exception:
            pass  # Don't let logging errors break the app


# Configure logging with both console and Streamlit handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('streamlit_app.log')
    ]
)

# Create custom handler for Streamlit
streamlit_handler = StreamlitLogHandler()
streamlit_handler.setFormatter(logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.addHandler(streamlit_handler)

logger.info("🚀 Starting Streamlit Consciousness Learning System")

# Configure page with improved styling
st.set_page_config(
    page_title="🧠 Consciousness Learning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger.info("📱 Streamlit page configured")

# Import our learning system
try:
    import config
    from continuous_learner import ContinuousLearner
    logger.info("✅ Successfully imported learning system modules")
except Exception as e:
    logger.error(f"❌ Failed to import modules: {e}")
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Global variables for session state
if 'learning_active' not in st.session_state:
    st.session_state.learning_active = False
    logger.info("🔧 Initialized learning_active session state")
if 'learner_thread' not in st.session_state:
    st.session_state.learner_thread = None
    logger.info("🔧 Initialized learner_thread session state")
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
    logger.info("🔧 Initialized stop_event session state")
if 'progress_queue' not in st.session_state:
    st.session_state.progress_queue = queue.Queue()
    logger.info("🔧 Initialized progress_queue session state")
if 'app_logs' not in st.session_state:
    st.session_state.app_logs = []
    logger.info("🔧 Initialized app_logs session state")
if 'learning_stats' not in st.session_state:
    st.session_state.learning_stats = {
        'videos_processed': 0,
        'session_start_time': None,
        'last_video_time': None,
        'total_consciousness_score': 0,
        'consciousness_scores': [],
        'processing_times': [],
        'consciousness_levels': [],
        'video_titles': [],
        'current_video': None,
        'current_status': 'Idle',
        'error_count': 0,
        'latest_components': {},
        'conscious_episodes': 0
    }
    logger.info("🔧 Initialized learning_stats session state")

logger.info("✅ All session states initialized successfully")


class StreamlitContinuousLearner(ContinuousLearner):
    """Extended ContinuousLearner with Streamlit progress reporting."""

    def __init__(self, progress_queue: queue.Queue, stop_event: threading.Event, *args, **kwargs):
        logger.info("🧠 Initializing StreamlitContinuousLearner")
        super().__init__(*args, **kwargs)
        self.progress_queue = progress_queue
        self.stop_event = stop_event
        logger.info("✅ StreamlitContinuousLearner initialized successfully")

    def report_progress(self, message: str, data: Optional[Dict] = None):
        """Report progress to the Streamlit interface."""
        logger.info(f"📊 Progress: {message}")
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'data': data or {}
        }
        try:
            self.progress_queue.put(progress_data, block=False)
        except queue.Full:
            logger.warning("⚠️ Progress queue is full, skipping message")
            pass  # Skip if queue is full

    def run_continuous_learning(self, min_wait: int = 300, max_wait: int = 900, max_iterations: Optional[int] = None):
        """Run continuous learning with progress reporting."""
        logger.info(
            f"🚀 Starting continuous learning: min_wait={min_wait}, max_wait={max_wait}, max_iterations={max_iterations}")
        iteration = 0

        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"🏁 Reached max iterations: {max_iterations}")
                    break

                logger.info(f"🔄 Starting iteration {iteration + 1}")
                self.report_progress(f"🔄 Starting iteration {iteration + 1}",
                                     {'iteration': iteration + 1})

                # Try to find and process one video
                video_found = False
                max_search_attempts = 5
                logger.info(
                    f"🔍 Starting video search with {max_search_attempts} max attempts")

                for attempt in range(max_search_attempts):
                    logger.info(
                        f"🔍 Search attempt {attempt + 1}/{max_search_attempts}")
                    self.report_progress(
                        f"🔍 Search attempt {attempt + 1}/{max_search_attempts}")

                    # Get one random video
                    logger.info("📡 Calling YouTube API for random videos...")
                    videos = self.get_random_videos(count=1)
                    if not videos:
                        logger.warning(
                            f"❌ No new videos found in attempt {attempt + 1}")
                        self.report_progress(
                            f"❌ No new videos found in attempt {attempt + 1}")
                        if attempt < max_search_attempts - 1:
                            logger.info("🔄 Trying different search terms...")
                            self.report_progress(
                                "🔄 Trying different search terms...")
                            time.sleep(2)
                            continue
                        else:
                            logger.warning("❌ All search attempts exhausted")
                            break

                    # Try to download the video
                    video = videos[0]
                    logger.info(f"🎬 Found video: {video['title'][:60]}...")
                    self.report_progress(f"🎬 Found video: {video['title'][:60]}...",
                                         {'video_title': video['title'], 'video_url': video['url']})

                    logger.info(
                        f"⬇️ Starting download for video: {video['url']}")
                    start_time = time.time()
                    video_path = self.download_video(video)

                    if video_path:
                        download_time = time.time() - start_time
                        logger.info(
                            f"✅ Video downloaded successfully in {download_time:.2f}s: {video_path}")
                        self.report_progress(f"✅ Successfully downloaded video",
                                             {'download_time': download_time})

                        try:
                            # Process the single video immediately
                            logger.info("🧠 Starting consciousness analysis...")
                            self.report_progress(f"🧠 Processing video...")
                            process_start = time.time()

                            logger.info("🔄 Calling batch processor...")
                            results = self.processor.process_videos(
                                [video_path],
                                f"streamlit_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            )

                            process_time = time.time() - process_start
                            logger.info(
                                f"✅ Processing completed in {process_time:.2f}s")

                            # Extract consciousness score from results
                            consciousness_score = 0
                            if results and len(results) > 0:
                                consciousness_score = results[0].get(
                                    'consciousness_score', 0)
                                logger.info(
                                    f"🧠 Consciousness score: {consciousness_score:.3f}")

                            # Update history
                            video_id = video['url'].split('v=')[1]
                            self.processed_videos[video_id] = {
                                'timestamp': datetime.now().isoformat(),
                                'video_info': video,
                                'results': results
                            }
                            logger.info(
                                f"💾 Saved video to processing history: {video_id}")
                            self._save_history()

                            self.report_progress(f"✅ Successfully processed video", {
                                'processing_time': process_time,
                                'consciousness_score': consciousness_score,
                                'video_title': video['title']
                            })

                        except Exception as e:
                            logger.error(f"❌ Error processing video: {e}")
                            self.report_progress(f"❌ Error processing video: {e}",
                                                 {'error': str(e)})
                            results = None

                        # Clean up the video immediately after processing
                        logger.info("🧹 Starting video cleanup...")
                        self.report_progress("🧹 Cleaning up video...")
                        try:
                            file_path = Path(video_path)
                            if file_path.exists():
                                file_path.unlink()
                                logger.info(
                                    f"🗑️ Deleted video file: {file_path.name}")
                                self.report_progress(
                                    f"🗑️ Deleted: {file_path.name}")
                            else:
                                logger.warning(
                                    f"⚠️ File not found for cleanup: {file_path.name}")
                                self.report_progress(
                                    f"⚠️ File not found for cleanup: {file_path.name}")
                        except Exception as e:
                            logger.error(
                                f"❌ Failed to delete {video_path}: {e}")
                            self.report_progress(
                                f"❌ Failed to delete {video_path}: {e}")

                        # Clean up any other video files that might exist
                        logger.info("🧹 Performing comprehensive cleanup...")
                        self._cleanup_all_videos()

                        if results is not None:
                            logger.info(
                                "✅ Video processing cycle completed successfully")
                            self.report_progress(
                                f"✅ Successfully processed and cleaned up video")
                            video_found = True
                            break
                        else:
                            logger.warning(
                                "❌ Video processed but with errors, trying next video...")
                            self.report_progress(
                                f"❌ Video processed but with errors, trying next video...")
                            if attempt < max_search_attempts - 1:
                                logger.info(
                                    "🔄 Trying different search terms...")
                                self.report_progress(
                                    "🔄 Trying different search terms...")
                                time.sleep(2)
                    else:
                        logger.error("❌ Failed to download video")
                        self.report_progress(f"❌ Failed to download video")
                        if attempt < max_search_attempts - 1:
                            logger.info("🔄 Trying different search terms...")
                            self.report_progress(
                                "🔄 Trying different search terms...")
                            time.sleep(2)

                if not video_found:
                    logger.warning(
                        "❌ No suitable videos found after all attempts")
                    self.report_progress(
                        "❌ No suitable videos found after all attempts")
                    # Clean up any partial downloads
                    self._cleanup_all_videos()

                # Wait before processing next video
                import random
                wait_time = random.randint(min_wait, max_wait)
                logger.info(
                    f"⏳ Waiting {wait_time//60} minutes {wait_time % 60} seconds before next video...")
                self.report_progress(f"⏳ Waiting {wait_time//60} minutes before next video...",
                                     {'wait_time': wait_time})

                # Wait in smaller increments to allow for stopping
                for i in range(wait_time):
                    time.sleep(1)
                    # Check if learning should stop (with error handling for threading)
                    try:
                        if self.stop_event.is_set():
                            logger.info(
                                "🛑 Stop event detected, ending learning session")
                            self.report_progress("🛑 Learning session stopped")
                            return
                    except:
                        # If we can't access session state, continue
                        pass

                iteration += 1
                logger.info(
                    f"🔄 Completed iteration {iteration}, starting next cycle...")

        except Exception as e:
            logger.error(f"❌ Critical error in learning session: {e}")
            self.report_progress(f"❌ Critical error in learning session: {e}",
                                 {'error': str(e)})
        finally:
            logger.info("🏁 Learning session ended")
            self.report_progress("🏁 Learning session ended")


def start_learning_session():
    """Start a learning session in a background thread."""
    logger.info("🎬 start_learning_session called")

    if st.session_state.learning_active:
        logger.warning(
            "⚠️ Learning session already active, ignoring start request")
        return

    logger.info("▶️ Starting new learning session")
    st.session_state.learning_active = True
    st.session_state.learning_stats['session_start_time'] = datetime.now()
    st.session_state.learning_stats['current_status'] = 'Starting...'

    # Clear the stop event and capture references
    st.session_state.stop_event.clear()
    progress_queue = st.session_state.progress_queue
    stop_event = st.session_state.stop_event

    logger.info("🔧 Captured thread references, creating learner thread")

    def run_learner():
        logger.info("🧵 Learner thread started")
        try:
            learner = StreamlitContinuousLearner(
                progress_queue=progress_queue,
                stop_event=stop_event,
                youtube_api_key=config.YOUTUBE_API_KEY
            )
            logger.info("🚀 Starting continuous learning from thread")
            # Updated to faster demo times: 10-30 seconds
            learner.run_continuous_learning(min_wait=10, max_wait=30)
        except Exception as e:
            logger.error(f"❌ Learning session failed: {e}")
            progress_queue.put({
                'timestamp': datetime.now().isoformat(),
                'message': f"❌ Learning session failed: {e}",
                'data': {'error': str(e)}
            })
        finally:
            logger.info("🏁 Learner thread ending")
            # Use a try-except to safely set the learning_active flag
            try:
                st.session_state.learning_active = False
                logger.info("✅ Set learning_active to False")
            except:
                logger.warning(
                    "⚠️ Could not access session state to set learning_active")
                # If session state is not accessible, just pass
                pass

    thread = threading.Thread(target=run_learner, daemon=True)
    thread.start()
    st.session_state.learner_thread = thread
    logger.info(f"✅ Learning thread started: {thread.name}")


def stop_learning_session():
    """Stop the current learning session."""
    logger.info("🛑 stop_learning_session called")
    st.session_state.learning_active = False
    st.session_state.stop_event.set()
    st.session_state.learning_stats['current_status'] = 'Stopping...'
    logger.info("✅ Learning session stop requested")


def update_progress():
    """Update progress from the queue."""
    messages_processed = 0
    while not st.session_state.progress_queue.empty() and messages_processed < 10:
        try:
            progress_data = st.session_state.progress_queue.get_nowait()

            # Update statistics based on progress data
            if 'Successfully processed video' in progress_data['message']:
                st.session_state.learning_stats['videos_processed'] += 1
                st.session_state.learning_stats['last_video_time'] = datetime.now(
                )

                if 'consciousness_score' in progress_data.get('data', {}):
                    score = progress_data['data']['consciousness_score']
                    st.session_state.learning_stats['consciousness_scores'].append(
                        score)
                    st.session_state.learning_stats['total_consciousness_score'] += score

                if 'processing_time' in progress_data.get('data', {}):
                    st.session_state.learning_stats['processing_times'].append(
                        progress_data['data']['processing_time']
                    )

            if 'Found video:' in progress_data['message']:
                st.session_state.learning_stats['current_video'] = progress_data.get(
                    'data', {}).get('video_title', 'Unknown')

            if '❌' in progress_data['message']:
                st.session_state.learning_stats['error_count'] += 1

            # Update current status
            st.session_state.learning_stats['current_status'] = progress_data['message']

            messages_processed += 1

        except queue.Empty:
            break


def display_logs_section():
    """Display real-time logs in the Streamlit interface."""
    st.header("📋 System Logs")

    # Log level filter
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        show_levels = st.multiselect(
            "Show log levels:",
            options=['INFO', 'WARNING', 'ERROR', 'DEBUG'],
            default=['INFO', 'WARNING', 'ERROR'],
            key="log_level_filter"
        )
    with col2:
        max_logs = st.selectbox(
            "Max logs:", [25, 50, 100], index=1, key="max_logs")
    with col3:
        if st.button("Clear Logs", key="clear_logs"):
            st.session_state.app_logs = []
            st.rerun()

    # Display logs
    if st.session_state.app_logs:
        # Filter logs by level
        filtered_logs = [
            log for log in st.session_state.app_logs
            if log['level'] in show_levels
        ]

        # Show most recent logs first
        recent_logs = filtered_logs[-max_logs:]
        recent_logs.reverse()

        # Create a container for logs with custom styling
        log_container = st.container()
        with log_container:
            for log in recent_logs:
                # Color code by level
                if log['level'] == 'ERROR':
                    st.error(f"[{log['timestamp']}] {log['message']}")
                elif log['level'] == 'WARNING':
                    st.warning(f"[{log['timestamp']}] {log['message']}")
                elif log['level'] == 'INFO':
                    st.info(f"[{log['timestamp']}] {log['message']}")
                else:
                    st.text(f"[{log['timestamp']}] {log['message']}")
    else:
        st.info("No logs available yet. Start a learning session to see logs.")

    logger.info(f"📋 Displayed {len(st.session_state.app_logs)} log entries")


def create_realtime_dashboard():
    """Create real-time dashboard containers for dynamic updates."""

    # Create empty containers that will be updated in real-time
    hero_container = st.empty()
    video_info_container = st.empty()
    metrics_container = st.empty()
    consciousness_results_container = st.empty()
    activity_logs_container = st.empty()
    history_container = st.empty()

    return {
        'hero': hero_container,
        'video_info': video_info_container,
        'metrics': metrics_container,
        'consciousness': consciousness_results_container,
        'activity_logs': activity_logs_container,
        'history': history_container
    }


def update_hero_section(container, learning_active):
    """Update the hero status section in real-time."""
    with container:
        if learning_active:
            st.markdown("""
            <div style="text-align: center; padding: 25px; background: linear-gradient(45deg, #1e3c72, #2a5298); border-radius: 15px; margin: 20px 0;">
                <h2 style="color: white; margin: 0;">🧠 CONSCIOUSNESS LEARNING ACTIVE</h2>
                <p style="color: #90EE90; font-size: 18px; margin: 10px 0;">System is processing videos and expanding consciousness patterns...</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 25px; background: linear-gradient(45deg, #434343, #000000); border-radius: 15px; margin: 20px 0;">
                <h2 style="color: white; margin: 0;">🔴 SYSTEM IDLE</h2>
                <p style="color: #FFB6C1; font-size: 18px; margin: 10px 0;">Ready to begin consciousness exploration...</p>
            </div>
            """, unsafe_allow_html=True)


def update_video_info_section(container, current_video):
    """Update video information card in real-time."""
    with container:
        if current_video:
            st.markdown("### 📼 **Current Video Information**")

            col1, col2, col3 = st.columns(3)
            with col1:
                display_title = current_video[:30] + \
                    "..." if len(current_video) > 30 else current_video
                st.metric("**Title**", display_title)
            with col2:
                st.metric("**Status**", "🔄 Processing")
            with col3:
                st.metric("**Type**", "🎬 YouTube Video")

            # Beautiful gradient card for current video
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); padding: 20px; border-radius: 15px; margin: 20px 0;">
                <h4 style="color: white; margin: 0;">🎬 Currently Analyzing:</h4>
                <p style="color: white; margin: 10px 0; font-weight: bold; font-size: 16px;">{current_video}</p>
                <p style="color: white; margin: 5px 0; opacity: 0.9;">Searching for consciousness patterns...</p>
            </div>
            """, unsafe_allow_html=True)


def update_metrics_section(container, stats):
    """Update enhanced metrics dashboard in real-time."""
    with container:
        create_enhanced_metrics_display(stats)


def update_consciousness_section(container, stats):
    """Update consciousness analysis results in real-time."""
    with container:
        if stats['consciousness_scores']:
            st.markdown("### 🧠 **Consciousness Analysis Results**")

            # Latest consciousness interpretation
            latest_score = stats['consciousness_scores'][-1]
            latest_level = stats['consciousness_levels'][-1] if stats['consciousness_levels'] else 'UNKNOWN'

            st.subheader("🔍 **Latest Consciousness Interpretation**")
            create_consciousness_interpretation(latest_score, latest_level)

            # Consciousness radar chart
            st.subheader("🧩 **Consciousness Components**")
            latest_components = stats.get('latest_components', {})
            radar_fig = create_consciousness_radar_chart(
                latest_components, latest_score)
            st.plotly_chart(radar_fig, use_container_width=True)

            # Consciousness evolution chart
            st.subheader("📈 **Consciousness Evolution Over Time**")
            df = pd.DataFrame({
                'Video': range(1, len(stats['consciousness_scores']) + 1),
                'Consciousness Score': stats['consciousness_scores'],
                'Level': stats['consciousness_levels'] if stats['consciousness_levels'] else ['UNKNOWN'] * len(stats['consciousness_scores'])
            })

            # Enhanced plotly chart
            fig = px.line(df, x='Video', y='Consciousness Score',
                          title='Consciousness Score Progression',
                          color_discrete_sequence=['#00CED1'])

            # Add consciousness thresholds
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                          annotation_text="Pre-Conscious Threshold")
            fig.add_hline(y=0.6, line_dash="dash", line_color="green",
                          annotation_text="Conscious Threshold")
            fig.add_hline(y=0.8, line_dash="dash", line_color="purple",
                          annotation_text="Highly Conscious Threshold")

            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "🧠 **Start learning to see consciousness analysis results and beautiful visualizations!**")


def update_activity_logs_section(container, stats, app_logs):
    """Update live activity and logs in real-time."""
    with container:
        # Live activity and logs in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔴 **Live Activity Feed**")
            current_status = stats.get('current_status', 'Idle')
            if 'Search attempt' in current_status:
                indicator, color = "🔍", "#FFA500"
            elif 'Processing' in current_status:
                indicator, color = "🧠", "#9370DB"
            elif 'Waiting' in current_status:
                indicator, color = "⏳", "#32CD32"
            elif 'Error' in current_status or '❌' in current_status:
                indicator, color = "❌", "#FF4500"
            else:
                indicator, color = "💤", "#708090"

            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="color: white; margin: 0;">{indicator} {current_status}</h5>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📋 **System Intelligence**")
            if app_logs:
                recent_logs = app_logs[-5:]  # Show last 5 logs
                for log in reversed(recent_logs):
                    if log['level'] == 'ERROR':
                        st.error(f"[{log['timestamp']}] {log['message']}")
                    elif log['level'] == 'WARNING':
                        st.warning(f"[{log['timestamp']}] {log['message']}")
                    else:
                        st.info(f"[{log['timestamp']}] {log['message']}")
            else:
                st.info("📋 System logs will appear here...")


def update_history_section(container, stats):
    """Update historical consciousness data in real-time."""
    with container:
        # Historical consciousness data section
        st.markdown("### 📈 **Consciousness Learning History**")

        if stats['videos_processed'] > 0:
            st.subheader(
                f"🗄️ **Learning Session History** ({stats['videos_processed']} videos analyzed)")

            # Create history table
            if stats['video_titles'] and stats['consciousness_scores']:
                history_data = []
                for i, (title, score, level) in enumerate(zip(
                    stats['video_titles'][-10:],  # Last 10 videos
                    stats['consciousness_scores'][-10:],
                    stats['consciousness_levels'][-10:] if stats['consciousness_levels'] else ['UNKNOWN'] * 10
                )):
                    history_data.append({
                        'Video #': len(stats['video_titles']) - 10 + i + 1,
                        'Title': title[:40] + "..." if len(title) > 40 else title,
                        'Consciousness Score': f"{score:.3f}",
                        'Level': level,
                        'Status': '✅ Analyzed'
                    })

                if history_data:
                    df = pd.DataFrame(history_data)
                    st.dataframe(df, use_container_width=True)
        else:
            st.info(
                "No learning history yet. Start a learning session to build consciousness analysis history.")


def start_learning_session_realtime():
    """Start learning session for real-time dashboard."""
    if st.session_state.learning_active:
        return

    logger.info("🚀 Starting real-time learning session")
    st.session_state.learning_active = True
    st.session_state.learning_stats['session_start_time'] = datetime.now()
    st.session_state.learning_stats['current_status'] = 'Starting...'

    st.session_state.stop_event.clear()
    progress_queue = st.session_state.progress_queue
    stop_event = st.session_state.stop_event

    def run_learner():
        try:
            learner = StreamlitContinuousLearner(
                progress_queue=progress_queue,
                stop_event=stop_event,
                youtube_api_key=config.YOUTUBE_API_KEY
            )
            # Much shorter wait times for demo: 10-30 seconds instead of 1-5 minutes
            learner.run_continuous_learning(min_wait=10, max_wait=30)
        except Exception as e:
            logger.error(f"Learning session failed: {e}")
        finally:
            try:
                st.session_state.learning_active = False
            except:
                pass

    thread = threading.Thread(target=run_learner, daemon=True)
    thread.start()
    st.session_state.learner_thread = thread


def stop_learning_session_realtime():
    """Stop learning session for real-time dashboard."""
    st.session_state.learning_active = False
    st.session_state.stop_event.set()
    st.session_state.learning_stats['current_status'] = 'Stopping...'


def create_consciousness_radar_chart(latest_components, consciousness_score):
    """Create beautiful consciousness radar chart like in app.py."""

    categories = ['Metacognitive<br>Awareness', 'Qualia<br>Intensity',
                  'Phenomenal<br>Binding', 'Overall<br>Score']
    values = [
        latest_components.get('metacognitive', consciousness_score * 0.8),
        latest_components.get('qualia', consciousness_score * 0.9),
        latest_components.get('binding', consciousness_score * 0.85),
        consciousness_score
    ]

    fig = go.Figure()

    # Main consciousness trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(0, 100, 255, 0.8)', width=3),
        marker=dict(size=8, color='rgba(0, 100, 255, 1)'),
        name='Consciousness Level'
    ))

    # Consciousness threshold line
    threshold_values = [0.6, 0.6, 0.6, 0.6]
    fig.add_trace(go.Scatterpolar(
        r=threshold_values,
        theta=categories,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Consciousness Threshold'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['0.2', '0.4', '0.6 (Threshold)', '0.8', '1.0']
            )),
        showlegend=True,
        title="🧠 Consciousness Component Analysis",
        title_x=0.5,
        height=400
    )

    return fig


def create_consciousness_interpretation(score, level):
    """Create consciousness interpretation section like in app.py."""

    if score >= 0.8:
        st.success("🎉 **HIGHLY CONSCIOUS** - This video demonstrates strong consciousness patterns with robust self-awareness, metacognition, and unified experience.")
    elif score >= 0.6:
        st.success(
            "✨ **CONSCIOUS** - This video shows clear consciousness emergence with integrated awareness and subjective experience.")
    elif score >= 0.4:
        st.warning("🟡 **PRE-CONSCIOUS** - This video exhibits developing consciousness patterns. Some components are emerging but not yet fully integrated.")
    elif score >= 0.2:
        st.info("🔄 **PROTO-CONSCIOUS** - This video shows basic consciousness building blocks but lacks integration and unity.")
    else:
        st.error(
            "❌ **NON-CONSCIOUS** - This video does not demonstrate consciousness patterns in our analysis framework.")


def create_enhanced_metrics_display(stats):
    """Create enhanced metrics display with beautiful styling like app.py."""

    st.markdown("### 📊 **Consciousness Metrics Dashboard**")

    # Primary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if stats['consciousness_scores']:
            latest_score = stats['consciousness_scores'][-1]
            latest_level = stats['consciousness_levels'][-1] if stats['consciousness_levels'] else 'UNKNOWN'

            if latest_level == 'CONSCIOUS':
                delta_text = "🟢 Achieved"
                delta_color = "normal"
            elif latest_level == 'PRE_CONSCIOUS':
                delta_text = "🟡 Emerging"
                delta_color = "normal"
            else:
                delta_text = "🔴 Not Detected"
                delta_color = "normal"

            st.metric("**Consciousness Level**",
                      latest_level, delta=delta_text)
        else:
            st.metric("**Consciousness Level**", "No data",
                      delta="🎯 Ready to analyze")

    with col2:
        avg_score = (stats['total_consciousness_score'] / max(
            stats['videos_processed'], 1)) if stats['videos_processed'] > 0 else 0
        delta_text = f"{'🎯 Above threshold' if avg_score >= 0.6 else '⚠️ Below threshold'}"
        st.metric("**Avg Consciousness Score**",
                  f"{avg_score:.3f}", delta=delta_text)
        if avg_score > 0:
            st.progress(avg_score)

    with col3:
        conscious_rate = (stats['conscious_episodes'] / max(
            stats['videos_processed'], 1) * 100) if stats['videos_processed'] > 0 else 0
        delta_text = f"{'🚀 High' if conscious_rate > 50 else '📊 Developing'}"
        st.metric("**Consciousness Rate**",
                  f"{conscious_rate:.1f}%", delta=delta_text)
        if conscious_rate > 0:
            st.progress(conscious_rate / 100)

    with col4:
        episodes = stats['conscious_episodes']
        delta_text = f"{'✨ Multiple' if episodes > 1 else '🎯 Single' if episodes == 1 else '❌ None'}"
        st.metric("**Conscious Episodes**", episodes, delta=delta_text)

    # Secondary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎬 Videos Processed", stats['videos_processed'])
        if stats['videos_processed'] > 0:
            # Progress out of 20 videos
            st.progress(min(stats['videos_processed'] / 20, 1.0))

    with col2:
        if stats['session_start_time']:
            uptime = datetime.now() - stats['session_start_time']
            uptime_str = str(uptime).split('.')[0]
            st.metric("⏱️ Session Time", uptime_str)
            minutes = uptime.total_seconds() / 60
            st.progress(min(minutes / 120, 1.0))  # Progress out of 2 hours
        else:
            st.metric("⏱️ Session Time", "Not started")

    with col3:
        avg_processing_time = (sum(stats['processing_times']) / len(
            stats['processing_times'])) if stats['processing_times'] else 0
        st.metric("⚡ Avg Processing Time", f"{avg_processing_time:.1f}s")

    with col4:
        st.metric("❌ Errors", stats['error_count'])
        if stats['error_count'] > 0:
            error_ratio = min(stats['error_count'] /
                              max(stats['videos_processed'], 1), 1.0)
            st.progress(error_ratio)


def create_video_information_card(current_video):
    """Create beautiful video information card like in app.py."""

    if current_video:
        st.markdown("### 📼 **Current Video Information**")

        col1, col2, col3 = st.columns(3)
        with col1:
            display_title = current_video[:30] + \
                "..." if len(current_video) > 30 else current_video
            st.metric("**Title**", display_title)
        with col2:
            st.metric("**Status**", "🔄 Processing")
        with col3:
            st.metric("**Type**", "🎬 YouTube Video")

        # Beautiful gradient card for current video
        st.markdown(f"""
        <div style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); padding: 20px; border-radius: 15px; margin: 20px 0;">
            <h4 style="color: white; margin: 0;">🎬 Currently Analyzing:</h4>
            <p style="color: white; margin: 10px 0; font-weight: bold; font-size: 16px;">{current_video}</p>
            <p style="color: white; margin: 5px 0; opacity: 0.9;">Searching for consciousness patterns...</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application with enhanced UI and real-time updates."""
    logger.info("🎯 Starting enhanced consciousness learning dashboard")

    # Beautiful header with improved styling
    st.title("🧠 **Consciousness Learning System** - Proto-Conscious AGI")
    st.markdown(
        "*Continuous consciousness exploration through advanced mirror learning and recursive self-abstraction*")

    # Enhanced sidebar
    with st.sidebar:
        st.header("🎛️ **Mission Control**")
        st.markdown("Continuous consciousness discovery system")

        # API status with enhanced styling
        if not config.YOUTUBE_API_KEY:
            st.error("⚠️ YouTube API key not configured!")
            st.markdown("Please set `YOUTUBE_API_KEY` in your `.env` file")
            return
        else:
            st.success("✅ **YouTube API Ready**")

        # Learning status indicator
        if st.session_state.learning_active:
            st.info("🔄 **Learning session in progress...**")
            with st.container():
                st.markdown("**Status:** Active")
                stats = st.session_state.learning_stats
                if stats['current_video']:
                    st.markdown(
                        f"**Current:** {stats['current_video'][:25]}...")
        else:
            st.markdown("*Ready to begin consciousness exploration...*")

        # Enhanced control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 **START LEARNING**", disabled=st.session_state.learning_active,
                         use_container_width=True, type="primary"):
                start_learning_session_realtime()

        with col2:
            if st.button("⏹️ **STOP**", disabled=not st.session_state.learning_active,
                         use_container_width=True, type="secondary"):
                stop_learning_session_realtime()

        # System information
        st.markdown("---")
        st.markdown("### 📊 **System Info**")
        queue_size = st.session_state.progress_queue.qsize()
        logs_count = len(st.session_state.app_logs)
        st.metric("Progress Queue", queue_size)
        st.metric("Log Entries", logs_count)

    # Create real-time containers
    containers = create_realtime_dashboard()

    # Real-time update loop for dynamic dashboard
    while True:
        # Update progress from queue
        update_progress()
        stats = st.session_state.learning_stats

        # Update all dashboard sections in real-time
        update_hero_section(containers['hero'],
                            st.session_state.learning_active)
        update_video_info_section(
            containers['video_info'], stats['current_video'])
        update_metrics_section(containers['metrics'], stats)
        update_consciousness_section(containers['consciousness'], stats)
        update_activity_logs_section(
            containers['activity_logs'], stats, st.session_state.app_logs)
        update_history_section(containers['history'], stats)

        # Small delay to prevent overwhelming updates
        time.sleep(2)

        # Break condition for clean exit
        if not st.session_state.learning_active and len(st.session_state.app_logs) == 0:
            break

    # Beautiful footer with system information (static - only shown once)
    st.markdown("---")
    st.markdown("""
    ### 🧠 **About Consciousness Learning System**
    
    This system uses advanced **continuous learning** and **mirror networks** to detect and measure consciousness patterns in video content.
    
    **Key Technologies:**
    - 🪞 **Mirror Networks**: Self-referential recursive learning
    - 🎭 **Qualia Generation**: Subjective experience modeling  
    - 🧩 **Phenomenal Binding**: Unified consciousness integration
    - 🤔 **Metacognition**: Thinking about thinking processes
    - 🎯 **Intentionality**: Goal formation and purposeful behavior
    - 🔄 **Continuous Learning**: Real-time knowledge accumulation
    
    **Consciousness Thresholds:**
    - 🟢 **Highly Conscious**: 0.8+ (Advanced consciousness patterns)
    - 🟢 **Conscious**: 0.6+ (Clear consciousness emergence)  
    - 🟡 **Pre-Conscious**: 0.4+ (Developing patterns)
    - 🔄 **Proto-Conscious**: 0.2+ (Basic building blocks)
    - 🔴 **Non-Conscious**: <0.2 (No patterns detected)
    
    *This is experimental research into artificial consciousness and proto-AGI systems through continuous video analysis.*
    """)

    logger.info("✅ Enhanced dynamic dashboard rendered successfully")


if __name__ == "__main__":
    logger.info("🎬 Starting Streamlit app from __main__")
    try:
        main()
        logger.info("✅ Streamlit app main() completed")
    except Exception as e:
        logger.error(f"❌ Critical error in main(): {e}")
        st.error(f"Critical error occurred: {e}")
        raise
