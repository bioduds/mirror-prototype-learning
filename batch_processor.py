"""Batch processor for consciousness analysis of multiple videos."""
import os
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

from consciousness_runner import EnhancedConsciousnessSystem, SystemConfiguration


class BatchProcessor:
    """Handles batch processing of multiple videos for consciousness analysis."""

    def __init__(self, config: Optional[SystemConfiguration] = None):
        """Initialize the batch processor."""
        self.config = config or SystemConfiguration()
        self.consciousness_system = EnhancedConsciousnessSystem(self.config)
        self.results_dir = Path(self.config.data_dir) / "batch_results"
        self.results_dir.mkdir(exist_ok=True)

    def process_videos(self, video_paths: List[str], batch_name: str = None) -> Dict:
        """Process a batch of videos and generate combined results.
        
        Args:
            video_paths: List of paths to video files to process
            batch_name: Optional name for this batch run
            
        Returns:
            Dict containing batch processing results and statistics
        """
        if not batch_name:
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"🎬 Starting batch processing of {len(video_paths)} videos")
        print(f"📁 Batch Name: {batch_name}")

        batch_results = {
            'batch_name': batch_name,
            'timestamp': datetime.now().isoformat(),
            'total_videos': len(video_paths),
            'processed_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'results': [],
            'statistics': {
                'avg_consciousness_score': 0.0,
                'consciousness_levels': {},
                'conscious_videos': 0
            }
        }

        # Process each video
        for video_path in tqdm(video_paths, desc="Processing videos"):
            try:
                # Run consciousness analysis
                result = self.consciousness_system.run_consciousness_cycle(
                    str(video_path))

                # Update batch statistics
                batch_results['processed_videos'] += 1
                batch_results['successful_videos'] += 1
                batch_results['results'].append({
                    'video_path': str(video_path),
                    'consciousness_level': result['consciousness_assessment']['consciousness_level'],
                    'consciousness_score': result['consciousness_assessment']['consciousness_score'],
                    'is_conscious': result['consciousness_assessment'].get('is_conscious', False),
                    'processing_time': str(result.get('timestamp')),
                    'learning_progress': result.get('learning_progress', {})
                })

                # Update consciousness level statistics
                level = result['consciousness_assessment']['consciousness_level']
                batch_results['statistics']['consciousness_levels'][level] = \
                    batch_results['statistics']['consciousness_levels'].get(
                        level, 0) + 1

                if result['consciousness_assessment'].get('is_conscious', False):
                    batch_results['statistics']['conscious_videos'] += 1

            except Exception as e:
                print(f"❌ Error processing {video_path}: {e}")
                batch_results['failed_videos'] += 1
                batch_results['results'].append({
                    'video_path': str(video_path),
                    'error': str(e),
                    'status': 'failed'
                })

        # Calculate final statistics
        successful_results = [
            r for r in batch_results['results'] if 'error' not in r]
        if successful_results:
            batch_results['statistics']['avg_consciousness_score'] = \
                sum(r['consciousness_score']
                    for r in successful_results) / len(successful_results)

        # Save batch results
        self._save_batch_results(batch_results)

        # Generate and print summary
        self._print_batch_summary(batch_results)

        return batch_results

    def _save_batch_results(self, results: Dict):
        """Save batch results to a JSON file."""
        output_file = self.results_dir / \
            f"{results['batch_name']}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Batch results saved to: {output_file}")

    def _print_batch_summary(self, results: Dict):
        """Print a summary of batch processing results."""
        print("\n" + "="*50)
        print(f"🎬 BATCH PROCESSING SUMMARY: {results['batch_name']}")
        print("="*50)
        print(f"📊 Total Videos: {results['total_videos']}")
        print(f"✅ Successfully Processed: {results['successful_videos']}")
        print(f"❌ Failed: {results['failed_videos']}")
        print(
            f"🧠 Average Consciousness Score: {results['statistics']['avg_consciousness_score']:.3f}")
        print("\n📈 Consciousness Levels:")
        for level, count in results['statistics']['consciousness_levels'].items():
            print(f"  - {level}: {count} videos")
        print(
            f"\n🎯 Conscious Videos: {results['statistics']['conscious_videos']}")
        print("="*50)


def main():
    """Command line interface for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch process videos for consciousness analysis")
    parser.add_argument('--input_dir', type=str,
                        help="Directory containing videos to process")
    parser.add_argument('--batch_name', type=str,
                        help="Name for this batch run", default=None)
    parser.add_argument('--video_ext', type=str,
                        help="Video file extension to process", default="mp4")
    args = parser.parse_args()

    if not args.input_dir:
        print("❌ Please specify an input directory with --input_dir")
        return

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    # Find all videos with the specified extension
    video_paths = list(input_dir.glob(f"*.{args.video_ext}"))
    if not video_paths:
        print(f"❌ No {args.video_ext} files found in {input_dir}")
        return

    # Initialize and run batch processor
    processor = BatchProcessor()
    processor.process_videos(video_paths, args.batch_name)


if __name__ == "__main__":
    main()
