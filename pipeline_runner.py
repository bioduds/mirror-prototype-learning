#!/usr/bin/env python3
"""
Mirror Prototype Learning - Complete Pipeline Runner
Integrates all stages of the mirror learning architecture.
"""

import os
import sys
import time
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime


class MirrorPipelineRunner:
    """Complete pipeline runner for mirror prototype learning."""

    def __init__(self, video_dir: str = "data/videos", output_dir: str = "vectors"):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Pipeline stages in order
        self.pipeline_stages = [
            ("mirror.py", "Perception & Feature Extraction"),
            ("encoder.py", "MirrorNet Autoencoder"),
            ("attention.py", "Temporal Attention"),
            ("self.py", "Self-Reference Learning"),
            ("fusion.py", "Consciousness Fusion"),
            ("extractor.py", "CLIP Feature Extraction"),
            ("clustering.py", "Pattern Analysis")
        ]

        # Expected outputs for each stage
        self.expected_outputs = {
            "mirror.py": ["pca_features.npy", "pca_coords.npy"],
            "encoder.py": ["mirrornet_latents.npy", "mirrornet_reconstructed.npy"],
            "attention.py": ["mirror_attention_output.npy"],
            "self.py": ["self_reference_vector.npy"],
            "fusion.py": ["fused_consciousness_vectors.npy"],
            "extractor.py": ["clip_features.npy"],
            "clustering.py": ["clustering_results.npy"]
        }

        self.log_file = "pipeline_log.txt"

    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)

        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        self.log("🔍 Checking prerequisites...")

        # Check video directory
        if not self.video_dir.exists():
            self.log(f"❌ Video directory not found: {self.video_dir}")
            return False

        # Check for video files
        video_files = list(self.video_dir.glob("*.mp4"))
        if not video_files:
            self.log(f"❌ No video files found in {self.video_dir}")
            return False

        self.log(f"✅ Found {len(video_files)} video file(s)")

        # Check Python scripts exist
        missing_scripts = []
        for script, _ in self.pipeline_stages:
            if not Path(script).exists():
                missing_scripts.append(script)

        if missing_scripts:
            self.log(f"❌ Missing scripts: {missing_scripts}")
            return False

        self.log("✅ All pipeline scripts found")
        return True

    def run_script(self, script_name: str, description: str) -> bool:
        """Run a single pipeline script."""
        self.log(f"🚀 Running {script_name}: {description}")

        try:
            start_time = time.time()

            # Run the script
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                self.log(
                    f"✅ {script_name} completed successfully in {elapsed:.1f}s")

                # Check expected outputs
                missing_outputs = []
                for expected_file in self.expected_outputs.get(script_name, []):
                    if not Path(expected_file).exists():
                        missing_outputs.append(expected_file)

                if missing_outputs:
                    self.log(
                        f"⚠️ Warning: Missing expected outputs: {missing_outputs}")
                else:
                    self.log(f"✅ All expected outputs generated")

                return True
            else:
                self.log(
                    f"❌ {script_name} failed with return code {result.returncode}")
                self.log(f"Error output: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"❌ {script_name} timed out after 5 minutes")
            return False
        except Exception as e:
            self.log(f"❌ Error running {script_name}: {str(e)}")
            return False

    def run_pipeline(self, start_from: Optional[str] = None) -> Dict:
        """Run the complete pipeline."""
        self.log("🧠 Starting Mirror Prototype Learning Pipeline")

        # Check prerequisites
        if not self.check_prerequisites():
            return {"success": False, "error": "Prerequisites not met"}

        # Create unique run ID for this execution
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log(f"🆔 Pipeline Run ID: {run_id}")

        results = {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "stages": {},
            "success": True
        }

        # Determine starting point
        start_idx = 0
        if start_from:
            for i, (script, _) in enumerate(self.pipeline_stages):
                if script == start_from:
                    start_idx = i
                    break
            self.log(f"📍 Starting from stage {start_idx + 1}: {start_from}")

        # Run pipeline stages
        for i, (script, description) in enumerate(self.pipeline_stages[start_idx:], start_idx):
            stage_start = time.time()

            self.log(f"📊 Stage {i + 1}/{len(self.pipeline_stages)}: {script}")

            success = self.run_script(script, description)
            stage_time = time.time() - stage_start

            results["stages"][script] = {
                "success": success,
                "duration": stage_time,
                "description": description
            }

            if not success:
                self.log(f"❌ Pipeline failed at stage {i + 1}: {script}")
                results["success"] = False
                results["failed_stage"] = script
                break

            self.log(f"✅ Stage {i + 1} completed successfully")

        results["end_time"] = datetime.now().isoformat()
        results["total_duration"] = sum(
            stage["duration"] for stage in results["stages"].values())

        # Save results
        self.save_run_results(results)

        if results["success"]:
            self.log("🎉 Pipeline completed successfully!")
            self.generate_summary()
        else:
            self.log("💥 Pipeline failed - check logs for details")

        return results

    def save_run_results(self, results: Dict):
        """Save pipeline run results."""
        results_file = f"pipeline_results_{results['run_id']}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        self.log(f"💾 Results saved to {results_file}")

    def generate_summary(self):
        """Generate a summary of pipeline outputs."""
        self.log("📋 Generating pipeline summary...")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "outputs": {}
        }

        # Check all output files
        for script, expected_files in self.expected_outputs.items():
            summary["outputs"][script] = {}
            for file_name in expected_files:
                file_path = Path(file_name)
                if file_path.exists():
                    try:
                        # Try to load and get shape info for .npy files
                        if file_name.endswith('.npy'):
                            data = np.load(file_path)
                            summary["outputs"][script][file_name] = {
                                "exists": True,
                                "shape": list(data.shape),
                                "dtype": str(data.dtype),
                                "size_mb": file_path.stat().st_size / (1024 * 1024)
                            }
                        else:
                            summary["outputs"][script][file_name] = {
                                "exists": True,
                                "size_mb": file_path.stat().st_size / (1024 * 1024)
                            }
                    except Exception as e:
                        summary["outputs"][script][file_name] = {
                            "exists": True,
                            "error": str(e),
                            "size_mb": file_path.stat().st_size / (1024 * 1024)
                        }
                else:
                    summary["outputs"][script][file_name] = {"exists": False}

        # Save summary
        with open("pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.log("✅ Summary saved to pipeline_summary.json")

        # Print key metrics
        self.log("\n📊 PIPELINE SUMMARY:")
        for script, files in summary["outputs"].items():
            self.log(f"  {script}:")
            for file_name, info in files.items():
                if info["exists"]:
                    if "error" in info:
                        self.log(
                            f"    ⚠️ {file_name}: Error - {info['error']} ({info.get('size_mb', 0):.1f}MB)")
                    elif "shape" in info:
                        self.log(
                            f"    ✅ {file_name}: {info['shape']} ({info.get('size_mb', 0):.1f}MB)")
                    else:
                        self.log(
                            f"    ✅ {file_name}: {info.get('size_mb', 0):.1f}MB")
                else:
                    self.log(f"    ❌ {file_name}: Missing")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mirror Prototype Learning Pipeline")
    parser.add_argument("--video-dir", default="data/videos",
                        help="Video directory path")
    parser.add_argument(
        "--start-from", help="Start from specific script (e.g., encoder.py)")
    parser.add_argument("--stage", help="Run only a specific stage")

    args = parser.parse_args()

    runner = MirrorPipelineRunner(video_dir=args.video_dir)

    if args.stage:
        # Run single stage
        script_name = args.stage if args.stage.endswith(
            '.py') else f"{args.stage}.py"
        description = next(
            (desc for script, desc in runner.pipeline_stages if script == script_name), "Single Stage")
        success = runner.run_script(script_name, description)
        if success:
            print(f"✅ {script_name} completed successfully")
        else:
            print(f"❌ {script_name} failed")
            sys.exit(1)
    else:
        # Run full pipeline
        results = runner.run_pipeline(start_from=args.start_from)
        if not results["success"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
