---- MODULE SimpleConsciousnessDetection ----

EXTENDS Naturals, Integers

CONSTANTS MaxVideos

VARIABLES
    processed_videos,
    consciousness_scores,
    consciousness_threshold,
    system_state,
    conscious_episodes

VideoID == 1..MaxVideos
Score == 0..100
SystemState == {"ready", "processing", "complete"}

TypeInv == 
    /\ processed_videos \subseteq VideoID
    /\ consciousness_scores \in [VideoID -> Score]
    /\ consciousness_threshold \in Score
    /\ system_state \in SystemState
    /\ conscious_episodes \in Nat

Init == 
    /\ processed_videos = {}
    /\ consciousness_scores = [v \in VideoID |-> 0]
    /\ consciousness_threshold = 60
    /\ system_state = "ready"
    /\ conscious_episodes = 0

ProcessVideo(video_id, score) ==
    /\ system_state = "ready"
    /\ video_id \notin processed_videos
    /\ consciousness_scores' = [consciousness_scores EXCEPT ![video_id] = score]
    /\ processed_videos' = processed_videos \cup {video_id}
    /\ conscious_episodes' = IF score >= consciousness_threshold
                            THEN conscious_episodes + 1
                            ELSE conscious_episodes
    /\ system_state' = "complete"
    /\ consciousness_threshold' = consciousness_threshold

Reset ==
    /\ system_state = "complete"
    /\ system_state' = "ready"
    /\ UNCHANGED <<processed_videos, consciousness_scores, consciousness_threshold, conscious_episodes>>

Next == 
    \/ \E video_id \in VideoID, score \in Score: ProcessVideo(video_id, score)
    \/ Reset

Spec == Init /\ [][Next]_<<processed_videos, consciousness_scores, consciousness_threshold, system_state, conscious_episodes>>

====
