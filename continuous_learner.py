"""Continuous learning system that processes random YouTube videos."""

# IMMEDIATE CLEANUP - Run before ANY imports
import config
from batch_processor import BatchProcessor
from consciousness_runner import EnhancedConsciousnessSystem, SystemConfiguration
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import yt_dlp
from typing import Dict, List, Optional
from datetime import datetime
import json
import random
import time
from pathlib import Path
import os


def cleanup_videos_immediately():
    """Clean up videos folder before any other code runs."""
    import os
    from pathlib import Path

    videos_dir = Path("data/videos")
    videos_dir.mkdir(exist_ok=True)

    cleaned_count = 0
    for file_path in videos_dir.iterdir():
        if file_path.is_file():
            try:
                file_path.unlink()
                cleaned_count += 1
                print(f"🗑️ Deleted: {file_path.name}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path.name}: {e}")

    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} file(s) immediately")
    return cleaned_count


# RUN CLEANUP IMMEDIATELY
cleanup_videos_immediately()

# Now safe to import everything else


class ContinuousLearner:
    """Continuously fetches and processes random YouTube videos for consciousness analysis."""

    def __init__(self, youtube_api_key: Optional[str] = None, system_config: Optional[SystemConfiguration] = None):
        """Initialize the continuous learner.
        
        Args:
            youtube_api_key: YouTube Data API key (optional, will use env var if not provided)
            system_config: Optional system configuration
        """
        self.api_key = youtube_api_key or config.YOUTUBE_API_KEY
        if not self.api_key:
            raise ValueError(
                "YouTube API key not provided. Either pass it to the constructor or "
                "set the YOUTUBE_API_KEY environment variable."
            )

        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.config = system_config or SystemConfiguration()
        self.processor = BatchProcessor(self.config)
        self.videos_dir = config.VIDEOS_DIR
        self.history_file = self.videos_dir / "processed_videos.json"
        self.processed_videos = self._load_history()

    def _cleanup_all_videos(self):
        """Clean up all video files in the videos directory."""
        cleaned_count = 0

        # Files to keep (whitelist)
        keep_files = {'video_metadata.json',
                      'processed_videos.json', '.gitkeep'}

        for file_path in self.videos_dir.iterdir():
            if file_path.is_file() and file_path.name not in keep_files:
                try:
                    file_path.unlink()
                    cleaned_count += 1
                    print(f"🗑️ Deleted: {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to delete {file_path.name}: {e}")

        if cleaned_count > 0:
            print(
                f"🧹 Cleaned up {cleaned_count} video file(s) from previous runs")
        else:
            print("🧹 No video files to clean up")

    def _load_history(self) -> Dict[str, Dict]:
        """Load history of processed videos."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_history(self):
        """Save history of processed videos."""
        with open(self.history_file, 'w') as f:
            json.dump(self.processed_videos, f, indent=2)

    def get_random_videos(self, count: int = config.DEFAULT_BATCH_SIZE) -> List[Dict]:
        """Get random YouTube videos using random search queries.
        
        Args:
            count: Number of videos to fetch
            
        Returns:
            List of video information dictionaries
        """
        videos = []
        search_query = random.choice(config.CONSCIOUSNESS_CATEGORIES)

        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=search_query,
                part='id,snippet',
                maxResults=count * 2,  # Request more to filter
                type='video',
                videoDuration=config.MAX_VIDEO_DURATION,
                safeSearch='strict'
            ).execute()

            # Filter and format videos
            for item in search_response.get('items', []):
                if len(videos) >= count:
                    break

                video_id = item['id']['videoId']

                # Skip if already processed
                if video_id in self.processed_videos:
                    continue

                videos.append({
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'category': search_query
                })

        except HttpError as e:
            print(f"❌ Error fetching videos: {e}")

        return videos

    def download_video(self, video_info: Dict) -> Optional[str]:
        """Download a single video using yt-dlp.
        
        Args:
            video_info: Video information dictionary
            
        Returns:
            Path to downloaded video if successful, None otherwise
        """
        print(f"🎬 Attempting to download: {video_info['title']}")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(self.videos_dir / '%(id)s.%(ext)s'),
            'max_filesize': config.MAX_VIDEO_SIZE,
            'quiet': False,  # Enable logging to see what's happening
            'no_warnings': False,
            'merge_output_format': 'mp4',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }]
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_info['url'], download=True)
                if info is None:  # Video download failed
                    print(f"❌ Video download failed: {video_info['title']}")
                    return None

                # Look for the downloaded file
                video_id = info['id']
                possible_extensions = ['.mp4', '.webm', '.mkv', '.m4v']

                for ext in possible_extensions:
                    video_path = self.videos_dir / f"{video_id}{ext}"
                    if video_path.exists():
                        print(
                            f"✅ Successfully downloaded: {video_info['title']}")
                        return str(video_path)

                # Also check for files with format codes (like .f605.mp4)
                for file_path in self.videos_dir.iterdir():
                    if file_path.is_file() and video_id in file_path.name:
                        print(
                            f"✅ Successfully downloaded: {video_info['title']}")
                        return str(file_path)

                print(
                    f"❌ Download completed but file not found: {video_info['title']}")
                return None
        except Exception as e:
            print(f"❌ Failed to download video '{video_info['title']}': {e}")
            return None

    def run_continuous_learning(self,
                                min_wait: int = config.MIN_WAIT_TIME,
                                max_wait: int = config.MAX_WAIT_TIME,
                                max_iterations: Optional[int] = None):
        """Run continuous learning process - one video at a time.
        
        Args:
            min_wait: Minimum wait time between videos in seconds
            max_wait: Maximum wait time between videos in seconds
            max_iterations: Maximum number of iterations (None for infinite)
        """
        iteration = 0

        while True:
            if max_iterations and iteration >= max_iterations:
                break

            print(f"\n🔄 Starting iteration {iteration + 1}")
            print(f"📚 Total videos processed: {len(self.processed_videos)}")

            # Try to find and process one video
            video_found = False
            max_search_attempts = 5

            for attempt in range(max_search_attempts):
                print(f"🔍 Search attempt {attempt + 1}/{max_search_attempts}")

                # Get one random video
                videos = self.get_random_videos(count=1)
                if not videos:
                    print(f"❌ No new videos found in attempt {attempt + 1}")
                    if attempt < max_search_attempts - 1:
                        print("🔄 Trying different search terms...")
                        time.sleep(2)
                        continue
                    else:
                        break

                # Try to download the video
                video = videos[0]
                print(f"🎬 Found video: {video['title'][:60]}...")
                video_path = self.download_video(video)

                # Check if video file actually exists, even if download_video returned None
                potential_video_files = list(self.videos_dir.glob(
                    "*.mp4")) + list(self.videos_dir.glob("*.webm")) + list(self.videos_dir.glob("*.mkv"))

                if video_path or potential_video_files:
                    # Use the returned path or find the most recent video file
                    if not video_path and potential_video_files:
                        video_path = str(
                            max(potential_video_files, key=lambda p: p.stat().st_mtime))
                        print(
                            f"📁 Found downloaded video file: {Path(video_path).name}")

                    print(
                        f"✅ Video available for processing: {Path(video_path).name}")

                    try:
                        # Process the single video immediately
                        print(f"🧠 Processing video...")
                        results = self.processor.process_videos(
                            [video_path],
                            f"single_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )

                        # Update history
                        video_id = video['url'].split('v=')[1]
                        self.processed_videos[video_id] = {
                            'timestamp': datetime.now().isoformat(),
                            'video_info': video,
                            'results': results
                        }
                        self._save_history()
                        print(f"✅ Successfully processed video")

                    except Exception as e:
                        print(f"❌ Error processing video: {e}")
                        results = None

                    # Always clean up the video after processing (or failed processing)
                    print("🧹 Cleaning up video...")
                    try:
                        file_path = Path(video_path)
                        if file_path.exists():
                            file_path.unlink()
                            print(f"🗑️ Deleted: {file_path.name}")
                        else:
                            print(
                                f"⚠️ File not found for cleanup: {file_path.name}")
                    except Exception as e:
                        print(f"❌ Failed to delete {video_path}: {e}")

                    # Clean up any other video files that might exist
                    self._cleanup_all_videos()

                    if results is not None:
                        print(f"✅ Successfully processed and cleaned up video")
                        video_found = True
                        break
                    else:
                        print(
                            f"⚠️ Video processed but with errors, will try next video...")
                        if attempt < max_search_attempts - 1:
                            print("🔄 Trying different search terms...")
                            time.sleep(2)
                else:
                    print(f"❌ No video file found (download failed or filtered out)")
                    if attempt < max_search_attempts - 1:
                        print("🔄 Trying different search terms...")
                        time.sleep(2)

            if not video_found:
                print("❌ No suitable videos found after all attempts")
                # Clean up any partial downloads
                self._cleanup_all_videos()

            # Wait before processing next video
            wait_time = random.randint(min_wait, max_wait)
            print(f"\n⏳ Waiting {wait_time//60} minutes before next video...")
            time.sleep(wait_time)

            iteration += 1


def main():
    """Command line interface for continuous learning."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run continuous learning on YouTube videos - one video at a time")
    parser.add_argument('--api_key', type=str,
                        help="YouTube Data API key (optional if set in env)")
    parser.add_argument('--min_wait', type=int, default=config.MIN_WAIT_TIME,
                        help="Minimum wait time between videos (seconds)")
    parser.add_argument('--max_wait', type=int, default=config.MAX_WAIT_TIME,
                        help="Maximum wait time between videos (seconds)")
    parser.add_argument('--max_iterations', type=int,
                        help="Maximum number of videos to process (optional)")
    args = parser.parse_args()

    try:
        learner = ContinuousLearner(args.api_key)
        print(f"🎬 No duration filtering - processing all video lengths")
        print(f"🔄 Processing videos one at a time continuously...")
        learner.run_continuous_learning(
            min_wait=args.min_wait,
            max_wait=args.max_wait,
            max_iterations=args.max_iterations
        )
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo set the API key, either:")
        print("1. Pass it as an argument: --api_key YOUR_API_KEY")
        print("2. Set it as an environment variable:")
        print("   export YOUTUBE_API_KEY=YOUR_API_KEY")


if __name__ == "__main__":
    main()
