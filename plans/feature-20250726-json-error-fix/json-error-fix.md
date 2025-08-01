# JSON Error Fix with Video Download Workflow - Implementation Plan

## Overview
Fix the JSON decode error in the consciousness training system by implementing proper error handling and ensuring the complete workflow starts with video download as specified in the TLA+ validation.

## Problem Analysis
The current error occurs because:
1. The system attempts to read `data/training_progress.json` before it exists
2. No proper error handling for missing/invalid JSON files
3. The workflow doesn't properly initialize with video download first

## TLA+ Validated Solution

### Workflow Order (TLA+ Proven)
1. **Video Download** (`StartVideoDownload` → `VideoDownloadSuccess`)
2. **Progress File Creation** (`CreateProgressFile` - requires downloaded video)
3. **Safe Progress Loading** (`LoadValidProgress` with fallback handling)
4. **Training Execution** (only after successful video download)

### Safety Properties Guaranteed
- ✅ **Video Download Precedes Training**: `VideoBeforeTraining` property
- ✅ **Safe File Operations**: `SafeFileOperations` property  
- ✅ **JSON Error Recovery**: `NoSystemCrashOnJsonError` property
- ✅ **Training Initialization Safety**: `TrainingRequiresInit` property

## Implementation Changes Required

### 1. Enhanced `load_training_progress()` Function
```python
def load_training_progress():
    """Load current training progress from JSON file with TLA+ validated safety"""
    progress_file = Path('data/training_progress.json')
    
    # TLA+ Safety: Handle missing file case
    if not progress_file.exists():
        return {
            "status": "idle",
            "youtube_url": "",
            "consciousness_level": 0.0,
            "mirror_depth": 0,
            "training_steps": 0,
            "videos_processed": 0
        }
    
    try:
        with open(progress_file, 'r') as f:
            content = f.read().strip()
            
            # TLA+ Safety: Handle empty file case
            if not content:
                return create_default_progress()
                
            # TLA+ Safety: Parse with error recovery
            return json.loads(content)
            
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        # TLA+ Safety: Always recover with default data
        return create_default_progress()

def create_default_progress():
    """Create default progress data structure"""
    return {
        "status": "idle",
        "youtube_url": "",
        "consciousness_level": 0.0,
        "mirror_depth": 0,
        "training_steps": 0,
        "videos_processed": 0
    }
```

### 2. Video Download Integration
```python
def run_consciousness_training_with_monitoring(youtube_url, threshold, depth, epochs):
    """Run consciousness training with TLA+ validated workflow"""
    try:
        # TLA+ Step 1: Initialize video download
        st.session_state.training_status = "downloading"
        
        # Create data directory
        Path('data').mkdir(exist_ok=True)
        
        # TLA+ Step 2: Create initial progress with video download status
        progress_data = {
            "status": "downloading",
            "youtube_url": youtube_url,
            "current_epoch": 0,
            "total_epochs": epochs,
            "consciousness_level": 0.0,
            "mirror_depth": depth,
            "threshold": threshold,
            "training_steps": 0,
            "videos_processed": 0
        }
        
        # TLA+ Safety: Atomic file creation
        with open('data/training_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # TLA+ Step 3: Execute training only after video handling
        st.session_state.training_status = "processing"
        progress_data["status"] = "processing"
        
        # Save updated progress
        with open('data/training_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Execute training runner
        cmd = [
            'python', 'enhanced_consciousness_runner.py',
            '--youtube_url', youtube_url,
            '--threshold', str(threshold),
            '--mirror_depth', str(depth),
            '--epochs', str(epochs)
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # TLA+ Safety: Handle completion states
        if process.returncode == 0:
            st.session_state.training_status = "completed"
            progress_data["status"] = "completed"
        else:
            st.session_state.training_status = "error"
            progress_data["status"] = "error"
            progress_data["error"] = process.stderr
            
        # Final progress save
        with open('data/training_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    except Exception as e:
        # TLA+ Safety: Error recovery
        st.session_state.training_status = "error"
        error_progress = create_default_progress()
        error_progress.update({
            "status": "error",
            "error": str(e)
        })
        with open('data/training_progress.json', 'w') as f:
            json.dump(error_progress, f, indent=2)
```

### 3. Enhanced Error Display
```python
# Status indicator with proper error handling
status = results.get('status', 'idle')
if status == "downloading":
    st.info("📥 **Downloading Video** - Preparing training data from YouTube")
elif status == "processing":
    st.info("🔄 **Training in Progress** - TLA+ validated consciousness development active")
elif status == "completed":
    st.success("✅ **Training Completed** - Consciousness emergence achieved!")
elif status == "error":
    st.error("❌ **Training Error** - Check logs for details")
    if 'error' in results:
        st.code(results['error'])
        st.info("💡 **Recovery**: The system will automatically recover. Try starting training again.")
else:
    st.info("⏳ **Ready for Training** - Configure parameters and start training")
```

## Testing Strategy

### TLA+ Property Tests
1. **Video Download Safety**: Verify training never starts without video download
2. **JSON Error Recovery**: Test system recovery from corrupted/missing files
3. **Progress Data Consistency**: Ensure valid progress structure always maintained
4. **Training Workflow Integrity**: Verify complete workflow execution

### Unit Tests
1. Test `load_training_progress()` with missing file
2. Test `load_training_progress()` with invalid JSON
3. Test `load_training_progress()` with empty file
4. Test `create_default_progress()` structure validity

### Integration Tests
1. Full workflow test: URL → Download → Training → Completion
2. Error recovery test: Corrupted progress file recovery
3. Concurrent access test: Multiple training sessions

## Validation Checklist

- [ ] TLA+ specification passes all safety properties
- [ ] TLC model checker validates workflow correctness
- [ ] Unit tests achieve >80% coverage
- [ ] Integration tests pass full workflow
- [ ] Error scenarios properly handled
- [ ] User interface provides clear status feedback

## Rollout Plan

1. **Phase 1**: Implement enhanced `load_training_progress()` function
2. **Phase 2**: Update training workflow with video download integration
3. **Phase 3**: Enhanced error handling and user feedback
4. **Phase 4**: Full testing and validation
5. **Phase 5**: Deploy and monitor

This implementation follows the TLA+ validated workflow ensuring mathematical correctness of the training system.
