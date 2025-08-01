---- MODULE JsonProgressHandler ----
EXTENDS Naturals, Sequences, TLC, FiniteSets

\* Constants representing file states, data types, and video states
CONSTANTS FileExists, FileNotExists, ValidJson, InvalidJson, EmptyContent,
          VideoDownloaded, VideoNotDownloaded, DownloadInProgress, DownloadFailed

\* Variables representing system state
VARIABLES 
    youtube_url,        \* Input YouTube URL
    video_state,        \* State of video download
    file_state,         \* Current state of progress file
    file_content,       \* Content of the progress file  
    progress_data,      \* Current progress data structure
    training_active,    \* Whether training is currently active
    error_occurred,     \* Whether an error has occurred
    system_status       \* Overall system status

\* Type definitions
VideoStates == {VideoDownloaded, VideoNotDownloaded, DownloadInProgress, DownloadFailed}
FileStates == {FileExists, FileNotExists}
ContentTypes == {ValidJson, InvalidJson, EmptyContent}
SystemStatuses == {"idle", "downloading", "processing", "completed", "error"}

\* Valid YouTube URL patterns (simplified)
ValidYouTubeUrls == {"youtube.com/watch", "youtu.be/", "youtube.com/shorts"}

\* Valid progress data structure
ValidProgressData == [
    status: SystemStatuses,
    youtube_url: STRING,
    consciousness_level: Nat,
    mirror_depth: 0..4,
    training_steps: Nat,
    videos_processed: Nat
]

\* Default progress data when file is missing or invalid
DefaultProgressData == [
    status |-> "idle",
    youtube_url |-> "",
    consciousness_level |-> 0,
    mirror_depth |-> 0, 
    training_steps |-> 0,
    videos_processed |-> 0
]

\* Type invariant
TypeInv == 
    /\ youtube_url \in STRING
    /\ video_state \in VideoStates
    /\ file_state \in FileStates
    /\ file_content \in ContentTypes  
    /\ progress_data \in ValidProgressData
    /\ training_active \in BOOLEAN
    /\ error_occurred \in BOOLEAN
    /\ system_status \in SystemStatuses

\* Initial state - no video downloaded, no file exists yet
Init == 
    /\ youtube_url = ""
    /\ video_state = VideoNotDownloaded
    /\ file_state = FileNotExists
    /\ file_content = EmptyContent
    /\ progress_data = DefaultProgressData
    /\ training_active = FALSE
    /\ error_occurred = FALSE
    /\ system_status = "idle"

\* Action: Start video download
StartVideoDownload ==
    /\ youtube_url /= ""
    /\ video_state = VideoNotDownloaded
    /\ system_status = "idle"
    /\ video_state' = DownloadInProgress
    /\ system_status' = "downloading"
    /\ UNCHANGED <<youtube_url, file_state, file_content, progress_data, training_active, error_occurred>>

\* Action: Video download succeeds
VideoDownloadSuccess ==
    /\ video_state = DownloadInProgress
    /\ video_state' = VideoDownloaded
    /\ system_status' = "idle"  \* Ready for training
    /\ UNCHANGED <<youtube_url, file_state, file_content, progress_data, training_active, error_occurred>>

\* Action: Video download fails
VideoDownloadFailure ==
    /\ video_state = DownloadInProgress
    /\ video_state' = DownloadFailed
    /\ system_status' = "error"
    /\ error_occurred' = TRUE
    /\ UNCHANGED <<youtube_url, file_state, file_content, progress_data, training_active>>

\* Action: Create progress file when training starts (requires downloaded video)
CreateProgressFile ==
    /\ training_active = FALSE
    /\ video_state = VideoDownloaded
    /\ file_state = FileNotExists
    /\ file_state' = FileExists
    /\ file_content' = ValidJson
    /\ progress_data' \in {pd \in ValidProgressData : 
        pd.status = "processing"}
    /\ training_active' = TRUE
    /\ system_status' = "processing"
    /\ UNCHANGED <<youtube_url, video_state, error_occurred>>

\* Action: Successfully load valid JSON file
LoadValidProgress ==
    /\ file_state = FileExists
    /\ file_content = ValidJson
    /\ \E new_data \in ValidProgressData :
        progress_data' = new_data
    /\ error_occurred' = FALSE
    /\ UNCHANGED <<youtube_url, video_state, file_state, file_content, training_active, system_status>>

\* Action: Handle missing file by using default data
HandleMissingFile ==
    /\ file_state = FileNotExists
    /\ progress_data' = DefaultProgressData
    /\ error_occurred' = FALSE
    /\ UNCHANGED <<youtube_url, video_state, file_state, file_content, training_active, system_status>>

\* Action: Handle invalid JSON by using default data and recovering
HandleInvalidJson ==
    /\ file_state = FileExists
    /\ file_content = InvalidJson
    /\ progress_data' = DefaultProgressData
    /\ file_content' = ValidJson  \* Recovery: overwrite with valid JSON
    /\ error_occurred' = TRUE     \* Mark that error occurred
    /\ UNCHANGED <<youtube_url, video_state, file_state, training_active, system_status>>

\* Action: File becomes corrupted (models real-world failure)
FileCorruption ==
    /\ file_state = FileExists
    /\ file_content = ValidJson
    /\ file_content' = InvalidJson
    /\ error_occurred' = TRUE
    /\ UNCHANGED <<youtube_url, video_state, file_state, progress_data, training_active, system_status>>

\* Action: Complete training
CompleteTraining ==
    /\ training_active = TRUE
    /\ system_status = "processing"
    /\ video_state = VideoDownloaded
    /\ training_active' = FALSE
    /\ system_status' = "completed"
    /\ UNCHANGED <<youtube_url, video_state, file_state, file_content, progress_data, error_occurred>>

\* Next state relation
Next ==
    \/ StartVideoDownload
    \/ VideoDownloadSuccess
    \/ VideoDownloadFailure
    \/ CreateProgressFile
    \/ LoadValidProgress  
    \/ HandleMissingFile
    \/ HandleInvalidJson
    \/ FileCorruption
    \/ CompleteTraining

\* Specification
Spec == Init /\ [][Next]_<<youtube_url, video_state, file_state, file_content, progress_data, training_active, error_occurred, system_status>>

\* Safety Properties

\* SAFETY 1: System always has valid progress data
ProgressDataAlwaysValid == 
    []( progress_data \in ValidProgressData )

\* SAFETY 2: No JSON decode errors crash the system  
NoSystemCrashOnJsonError ==
    []( (file_content = InvalidJson) => (progress_data = DefaultProgressData) )

\* SAFETY 3: File operations are safe
SafeFileOperations ==
    []( (file_state = FileNotExists) => (progress_data = DefaultProgressData) )

\* SAFETY 4: Training only starts with downloaded video
TrainingRequiresVideo ==
    []( training_active => (video_state = VideoDownloaded) )

\* SAFETY 5: Training requires proper initialization  
TrainingRequiresInit ==
    []( training_active => (file_state = FileExists /\ video_state = VideoDownloaded) )

\* SAFETY 6: Error recovery maintains system stability
ErrorRecoveryStable ==
    []( error_occurred => <>(file_content = ValidJson) )

\* SAFETY 7: Video download precedes training
VideoBeforeTraining ==
    []( (system_status = "processing") => (video_state = VideoDownloaded) )

\* Liveness Properties

\* LIVENESS 1: Video download eventually completes or fails
VideoDownloadProgresses ==
    []( (video_state = DownloadInProgress) => <>(video_state \in {VideoDownloaded, DownloadFailed}) )

\* LIVENESS 2: Progress data eventually becomes available
ProgressEventuallyAvailable ==
    <>(progress_data \in ValidProgressData)

\* LIVENESS 3: Errors are eventually resolved
ErrorsEventuallyResolved ==
    [](error_occurred => <>(~error_occurred \/ file_content = ValidJson))

\* LIVENESS 4: Training can eventually complete
TrainingCanComplete ==
    [](training_active => <>(system_status = "completed"))

\* Combined safety property
Safety == 
    /\ ProgressDataAlwaysValid
    /\ NoSystemCrashOnJsonError  
    /\ SafeFileOperations
    /\ TrainingRequiresVideo
    /\ TrainingRequiresInit
    /\ VideoBeforeTraining

\* Combined liveness property  
Liveness ==
    /\ VideoDownloadProgresses
    /\ ProgressEventuallyAvailable
    /\ ErrorsEventuallyResolved
    /\ TrainingCanComplete

====
