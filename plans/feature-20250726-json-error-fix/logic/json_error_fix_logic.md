# Predicate Logic Circuit for Complete Training Workflow with Video Download

## Domain Objects
- `youtube_url(U)` - U is a valid YouTube URL
- `video_file(V)` - V is a downloaded video file
- `file_path(P)` - P is a valid file path
- `file_exists(P)` - file P exists on filesystem  
- `file_readable(P)` - file P can be read
- `json_valid(C)` - content C is valid JSON
- `json_empty(C)` - content C is empty string
- `progress_data(D)` - D is valid progress data structure
- `error_state(E)` - E represents an error condition
- `download_successful(U, V)` - video V successfully downloaded from URL U
- `video_accessible(V)` - video file V is accessible for processing

## State Variables
- `training_active(T)` - training session T is active
- `video_downloaded(U, V)` - video V downloaded from URL U
- `download_progress(U, P)` - download progress P for URL U
- `file_created(P, T)` - file P created for training T
- `file_content(P, C)` - file P contains content C
- `system_state(S)` - current system state S
- `training_step(T, S)` - training T is at step S

## Core Predicates

### Video Download Predicates
```prolog
% Valid YouTube URL format
valid_youtube_url(U) :- 
    youtube_url(U),
    (contains(U, "youtube.com/watch") ; contains(U, "youtu.be/") ; contains(U, "youtube.com/shorts")).

% Video download process
download_video(U, V) :-
    valid_youtube_url(U),
    video_file(V),
    download_successful(U, V),
    video_accessible(V).

% Download failure handling
download_failed(U) :-
    valid_youtube_url(U),
    not exists(V, download_successful(U, V)).
```

### File Safety Predicates
```prolog
% File must exist before reading
safe_read(P) :- file_exists(P), file_readable(P).

% Content must be valid JSON or empty
safe_json_parse(C) :- json_valid(C).
safe_json_parse(C) :- json_empty(C).

% Default data when file missing or invalid
default_progress_data(D) :- 
    D = {status: "idle", consciousness_level: 0.0, mirror_depth: 0, training_steps: 0, videos_processed: 0}.
```

### Training Lifecycle Predicates
```prolog
% Training initialization requires video download first
training_init(T, U) :- 
    valid_youtube_url(U),
    exists(V, download_video(U, V)),
    directory_exists("data"),
    file_path(P),
    P = "data/training_progress.json".

% Progress file creation is atomic and includes video info
create_progress_file(T, U, P, D) :-
    training_init(T, U),
    file_path(P),
    progress_data(D),
    D.youtube_url = U,
    D.status = "processing",
    file_created(P, T),
    file_content(P, json_serialize(D)).

% Training step progression requires video
training_step_advance(T, U, S_old, S_new) :-
    training_active(T),
    exists(V, video_downloaded(U, V)),
    training_step(T, S_old),
    S_new = S_old + 1,
    training_step(T, S_new).

% Safe progress loading with fallback
load_progress_safe(P, D) :-
    file_exists(P),
    file_content(P, C),
    safe_json_parse(C),
    progress_data(D),
    json_deserialize(C, D).

load_progress_safe(P, D) :-
    not file_exists(P),
    default_progress_data(D).

load_progress_safe(P, D) :-
    file_exists(P),
    file_content(P, C),
    not safe_json_parse(C),
    default_progress_data(D).
```

### Error Handling Predicates
```prolog
% JSON decode error conditions
json_decode_error(P) :-
    file_exists(P),
    file_content(P, C),
    not json_valid(C),
    not json_empty(C).

% Recovery from JSON errors
recover_from_json_error(P, D) :-
    json_decode_error(P),
    default_progress_data(D),
    file_content(P, json_serialize(D)).
```

### Invariants
```prolog
% System must always have valid progress data
invariant_valid_progress :- 
    forall(T, training_active(T) -> exists(D, progress_data(D))).

% File operations must be safe
invariant_safe_operations :-
    forall(P, (file_read_attempt(P) -> safe_read(P))).

% No training without proper video download and initialization  
invariant_training_safety :-
    forall(T, training_active(T) -> 
        exists(U, exists(V, valid_youtube_url(U) /\ video_downloaded(U, V) /\ training_init(T, U)))).

% Video download must precede training
invariant_download_before_training :-
    forall(T, forall(U, training_active(T) /\ youtube_url(U) -> 
        exists(V, video_downloaded(U, V)))).

% Progress data must include video source
invariant_progress_includes_video :-
    forall(T, forall(D, training_active(T) /\ progress_data(D) ->
        exists(U, valid_youtube_url(U) /\ D.youtube_url = U))).
```

## Safety Properties

1. **Video Download Safety**: Never start training without successful video download
2. **File Existence Safety**: Never attempt to read non-existent files
3. **JSON Parse Safety**: Always handle invalid JSON gracefully
4. **Data Consistency**: Progress data structure always valid
5. **Error Recovery**: System recovers from any JSON parsing error
6. **Training Integrity**: Training only proceeds with valid video and initialization

## Liveness Properties

1. **Video Download Progress**: Video download eventually completes or fails definitively
2. **Progress Availability**: Progress data eventually becomes available
3. **Error Resolution**: JSON errors eventually resolved
4. **Training Continuation**: Training continues after error recovery
