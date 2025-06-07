"""Download a curated set of test videos for consciousness analysis."""
import os
from pathlib import Path
import yt_dlp
from typing import List, Dict
import json

# Test videos representing different aspects of consciousness
TEST_VIDEOS = [
    {
        "url": "https://www.youtube.com/watch?v=VNqNnUJVcVs",
        "description": "Mirror test with animals",
        "duration": "2:00"
    },
    {
        "url": "https://www.youtube.com/watch?v=ktkjUjcZid0",
        "description": "Human facial expressions",
        "duration": "1:30"
    },
    {
        "url": "https://www.youtube.com/watch?v=YNLC0wJSHxI",
        "description": "AI learning patterns",
        "duration": "2:00"
    },
    {
        "url": "https://www.youtube.com/watch?v=hCRUq903hoA",
        "description": "Pattern recognition",
        "duration": "1:45"
    },
    {
        "url": "https://www.youtube.com/watch?v=apzXGEbZht0",
        "description": "Social interaction",
        "duration": "2:00"
    },
    {
        "url": "https://www.youtube.com/watch?v=7k4HYCNEYkk",
        "description": "Learning behaviors",
        "duration": "1:30"
    },
    {
        "url": "https://www.youtube.com/watch?v=lFEgohhfxOA",
        "description": "Problem solving",
        "duration": "2:00"
    },
    {
        "url": "https://www.youtube.com/watch?v=_JmA2ClUvUY",
        "description": "Memory patterns",
        "duration": "1:45"
    },
    {
        "url": "https://www.youtube.com/watch?v=dKjCWfuvYxQ",
        "description": "Emergent behavior",
        "duration": "2:00"
    },
    {
        "url": "https://www.youtube.com/watch?v=gaFKqOBTj9w",
        "description": "Self organization",
        "duration": "1:30"
    }
]


def download_video(video_info: Dict, output_dir: Path) -> bool:
    """Download a single video using yt-dlp."""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'max_filesize': '100M',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'fragment_retries': 10,
        'ignoreerrors': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"📥 Downloading: {video_info['description']}")
            ydl.download([video_info['url']])
        return True
    except Exception as e:
        print(f"❌ Failed to download {video_info['description']}: {e}")
        return False


def main():
    """Download all test videos."""
    # Create videos directory if it doesn't exist
    videos_dir = Path("data/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing videos except metadata
    for file in videos_dir.glob("*.mp4"):
        file.unlink()

    print(f"🎬 Downloading {len(TEST_VIDEOS)} test videos...")

    # Track successful downloads
    successful = 0
    for video in TEST_VIDEOS:
        if download_video(video, videos_dir):
            successful += 1

    print(
        f"\n✅ Successfully downloaded {successful}/{len(TEST_VIDEOS)} videos")
    print(f"📁 Videos saved to: {videos_dir}")

    # Save video metadata
    metadata_file = videos_dir / "video_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(TEST_VIDEOS, f, indent=2)
    print(f"📝 Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    main()
