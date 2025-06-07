"""Configuration settings for the consciousness learning system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# YouTube API settings
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    print("⚠️  Warning: YOUTUBE_API_KEY not found in environment variables")
    print("Please create a .env file with your YouTube API key")
    print("You can copy .env.example to .env and add your key")

# System settings
DATA_DIR = Path("data")
VIDEOS_DIR = DATA_DIR / "videos"
BATCH_RESULTS_DIR = DATA_DIR / "batch_results"
CONSCIOUSNESS_KNOWLEDGE_DIR = DATA_DIR / "consciousness_knowledge"

# Ensure directories exist
for directory in [DATA_DIR, VIDEOS_DIR, BATCH_RESULTS_DIR, CONSCIOUSNESS_KNOWLEDGE_DIR]:
    directory.mkdir(exist_ok=True)

# Learning parameters
DEFAULT_BATCH_SIZE = 5
MIN_WAIT_TIME = 300  # 5 minutes
MAX_WAIT_TIME = 900  # 15 minutes
MAX_VIDEO_SIZE = "100M"

# Video duration settings
# YouTube API filter: "short" (< 4 min), "medium" (4-20 min), "long" (> 20 min)
MAX_VIDEO_DURATION = "short"
MAX_VIDEO_SECONDS = 600  # 10 minutes - increased from 5 minutes
MIN_VIDEO_SECONDS = 15   # 15 seconds - minimum duration to be useful

# Video categories for consciousness learning
CONSCIOUSNESS_CATEGORIES = [
    "animal behavior",
    "cognitive science",
    "artificial intelligence",
    "consciousness",
    "learning and memory",
    "pattern recognition",
    "social interaction",
    "problem solving",
    "self awareness",
    "emergent behavior"
]
