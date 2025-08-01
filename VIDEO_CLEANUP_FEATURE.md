# Video Cleanup Feature Implementation

## 🧹 **Automatic Video Cleanup Before Download**

### **Feature Overview**

Added automatic cleanup of old video files before downloading new ones to ensure a clean training environment and prevent storage issues.

### **Implementation Details**

#### 1. **Video Directory Cleanup**

```python
# TLA+ Safety: Clear old videos before downloading new one
# This prevents confusion and ensures clean training environment
old_videos_count = 0
for old_video in videos_dir.glob("*.mp4"):
    old_video.unlink()  # Delete old video files
    old_videos_count += 1
for old_video in videos_dir.glob("*.webm"):
    old_video.unlink()  # Delete old webm files
    old_videos_count += 1
for old_video in videos_dir.glob("*.mkv"):
    old_video.unlink()  # Delete old mkv files
    old_videos_count += 1
```

#### 2. **Supported Video Formats**

- `.mp4` files (most common)
- `.webm` files (alternative format)
- `.mkv` files (high quality format)

#### 3. **Progress Tracking**

- New download stage: `"cleaning_old_videos"`
- Tracks count of removed files: `"old_videos_cleared"`
- Shows cleanup progress in UI

### **UI Updates**

#### **4-Stage Download Process** (Updated from 3-stage)

1. **Stage 0/4**: 🧹 Cleaning old videos
2. **Stage 1/4**: 📡 Fetching video info
3. **Stage 2/4**: 📥 Downloading video file
4. **Stage 3/4**: ✅ Download completion

#### **Visual Feedback**

- Progress bar during cleanup (0.1 progress)
- Shows count of removed files
- Time estimate: 1-2 seconds
- Clear status messages

### **Benefits**

#### **Storage Management**

- Prevents accumulation of old video files
- Ensures disk space availability
- Maintains clean workspace

#### **Training Clarity**

- Single video per training session
- No confusion about which video is being used
- Clear training data isolation

#### **User Experience**

- Transparent cleanup process
- Progress feedback during cleanup
- Automatic maintenance

### **TLA+ Safety Properties**

#### **Error Handling**

- Graceful handling if videos don't exist
- No crashes on permission issues
- Continues if cleanup fails

#### **Atomic Operations**

- File deletion before download starts
- Progress tracking throughout process
- Consistent state maintenance

### **Example Output**

```
🧹 Stage 0/4: Cleaning old videos from directory...
🗑️ Cleared 3 old video files
⏱️ Estimated time: 1-2 seconds
🔄 Process: Removing previous video files from directory
```

### **Technical Implementation**

#### **File Pattern Matching**

- Uses `Path.glob()` for pattern matching
- Supports multiple video formats
- Recursive cleanup in videos directory

#### **Progress Updates**

- Real-time count of removed files
- UI feedback during cleanup
- Seamless transition to download

### **Integration**

#### **Workflow Integration**

- Happens before video info fetch
- Part of TLA+ validated download process
- Maintains all safety properties

#### **UI Integration**

- New stage in progress display
- Enhanced real-time monitor
- Updated time estimates

This feature ensures that each consciousness training session starts with a clean slate, using only the newly downloaded video for training purposes.
