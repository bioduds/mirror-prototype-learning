---- MODULE ConsciousnessDetection ----

(*
TLA+ Specification for Consciousness Detection System

This specification models the consciousness detection pipeline to ensure:
1. Correct state transitions between consciousness levels
2. Proper component scoring and aggregation  
3. Threshold calibration and adjustment
4. Safety and liveness properties

Author: Eduardo Capanema
Date: 2025-07-25
*)

EXTENDS Naturals, Sequences, FiniteSets, Integers

CONSTANTS
    MaxVideos,          \* Maximum number of videos to process
    MinThreshold,       \* Minimum consciousness threshold (e.g., 0.1)
    MaxThreshold,       \* Maximum consciousness threshold (e.g., 0.9)
    ComponentCount      \* Number of consciousness components (3: metacognitive, qualia, binding)

\* Helper operators
Max(a, b) == IF a > b THEN a ELSE b
Min(a, b) == IF a < b THEN a ELSE b

VARIABLES
    processed_videos,           \* Set of processed video IDs
    consciousness_scores,       \* Function: video_id -> consciousness_score
    component_scores,          \* Function: video_id -> [metacognitive, qualia, binding]
    consciousness_threshold,    \* Current dynamic threshold
    consciousness_levels,      \* Function: video_id -> consciousness_level
    system_state,             \* Current system state: "ready", "processing", "complete", "error"
    conscious_episodes,       \* Count of detected conscious episodes
    total_episodes,          \* Count of total processed episodes
    calibration_history      \* Sequence of threshold adjustments

\* Type definitions
VideoID == 1..MaxVideos
Score == 0..100  \* Scores as integers 0-100 for TLC efficiency
ConsciousnessLevel == {"UNCONSCIOUS", "PRE_CONSCIOUS", "EMERGING_CONSCIOUSNESS", "FULL_CONSCIOUSNESS", "TRANSCENDENT_CONSCIOUSNESS"}
SystemState == {"ready", "processing", "complete", "error"}

\* Convert integer scores to threshold comparison (both as integers 0-100)
AboveThreshold(score, threshold) == score >= threshold

\* Type invariant
TypeInv == 
    /\ processed_videos \subseteq VideoID
    /\ consciousness_scores \in [VideoID -> Score]
    /\ component_scores \in [VideoID -> [metacognitive: Score, qualia: Score, binding: Score]]
    /\ consciousness_threshold \in Score
    /\ consciousness_levels \in [VideoID -> ConsciousnessLevel]
    /\ system_state \in SystemState
    /\ conscious_episodes \in Nat
    /\ total_episodes \in Nat
    /\ calibration_history \in Seq(Score)

\* Initial state
Init == 
    /\ processed_videos = {}
    /\ consciousness_scores = [v \in VideoID |-> 0]
    /\ component_scores = [v \in VideoID |-> [metacognitive |-> 0, qualia |-> 0, binding |-> 0]]
    /\ consciousness_threshold = MinThreshold  \* Start with minimum threshold directly
    /\ consciousness_levels = [v \in VideoID |-> "UNCONSCIOUS"]
    /\ system_state = "ready"
    /\ conscious_episodes = 0
    /\ total_episodes = 0
    /\ calibration_history = <<>>

\* Calculate consciousness score from components
CalculateConsciousnessScore(video_id) ==
    LET components == component_scores[video_id]
        avg_score == (components.metacognitive + components.qualia + components.binding) \div 3
    IN avg_score

\* Determine consciousness level from score (using integer thresholds)
DetermineConsciousnessLevel(score) ==
    IF score >= 90 THEN "TRANSCENDENT_CONSCIOUSNESS"
    ELSE IF score >= 80 THEN "FULL_CONSCIOUSNESS"  
    ELSE IF score >= 60 THEN "EMERGING_CONSCIOUSNESS"
    ELSE IF score >= 40 THEN "PRE_CONSCIOUS"
    ELSE "UNCONSCIOUS"

\* Process a new video
ProcessVideo(video_id, meta_score, qualia_score, binding_score) ==
    /\ system_state = "ready"
    /\ video_id \notin processed_videos
    /\ meta_score \in Score /\ qualia_score \in Score /\ binding_score \in Score
    /\ system_state' = "processing"
    /\ component_scores' = [component_scores EXCEPT ![video_id] = 
                           [metacognitive |-> meta_score, qualia |-> qualia_score, binding |-> binding_score]]
    /\ LET new_components == [metacognitive |-> meta_score, qualia |-> qualia_score, binding |-> binding_score]
           consciousness_score == (new_components.metacognitive + new_components.qualia + new_components.binding) \div 3
           consciousness_level == DetermineConsciousnessLevel(consciousness_score)
       IN /\ consciousness_scores' = [consciousness_scores EXCEPT ![video_id] = consciousness_score]
          /\ consciousness_levels' = [consciousness_levels EXCEPT ![video_id] = consciousness_level]
          /\ processed_videos' = processed_videos \cup {video_id}
          /\ total_episodes' = total_episodes + 1
          /\ conscious_episodes' = IF AboveThreshold(consciousness_score, consciousness_threshold)
                                   THEN conscious_episodes + 1
                                   ELSE conscious_episodes
    /\ UNCHANGED <<consciousness_threshold, calibration_history>>

\* Complete processing
CompleteProcessing ==
    /\ system_state = "processing"
    /\ system_state' = "complete"
    /\ UNCHANGED <<processed_videos, consciousness_scores, component_scores, 
                   consciousness_threshold, consciousness_levels, conscious_episodes, 
                   total_episodes, calibration_history>>

\* Handle processing error
ProcessingError ==
    /\ system_state = "processing"
    /\ system_state' = "error"
    /\ UNCHANGED <<processed_videos, consciousness_scores, component_scores,
                   consciousness_threshold, consciousness_levels, conscious_episodes,
                   total_episodes, calibration_history>>

\* Reset to ready state
ResetToReady ==
    /\ system_state \in {"complete", "error"}
    /\ system_state' = "ready"
    /\ UNCHANGED <<processed_videos, consciousness_scores, component_scores,
                   consciousness_threshold, consciousness_levels, conscious_episodes,
                   total_episodes, calibration_history>>

\* Calibrate threshold based on detection rate
CalibrateThreshold ==
    /\ system_state = "complete"
    /\ total_episodes > 0
    /\ LET detection_rate == (conscious_episodes * 100) \div total_episodes  \* Percentage as integer
           new_threshold == IF detection_rate < 10      \* Less than 10% detection - lower threshold
                           THEN Max(MinThreshold, consciousness_threshold - 5)
                           ELSE IF detection_rate > 50   \* More than 50% detection - raise threshold  
                           THEN Min(MaxThreshold, consciousness_threshold + 5)
                           ELSE consciousness_threshold   \* Keep current threshold
       IN /\ consciousness_threshold' = new_threshold
          /\ calibration_history' = Append(calibration_history, new_threshold)
    /\ UNCHANGED <<processed_videos, consciousness_scores, component_scores, consciousness_levels,
                   system_state, conscious_episodes, total_episodes>>

\* Next state relation
Next == 
    \/ \E video_id \in VideoID, meta_score \in Score, qualia_score \in Score, binding_score \in Score:
         ProcessVideo(video_id, meta_score, qualia_score, binding_score)
    \/ CompleteProcessing
    \/ ProcessingError  
    \/ ResetToReady
    \/ CalibrateThreshold

\* Specification
Spec == Init /\ [][Next]_<<processed_videos, consciousness_scores, component_scores, 
                            consciousness_threshold, consciousness_levels, system_state,
                            conscious_episodes, total_episodes, calibration_history>>

\* Safety Properties

\* Consciousness scores are consistent with components
ConsciousnessScoreConsistency ==
    \A video_id \in processed_videos:
        consciousness_scores[video_id] = CalculateConsciousnessScore(video_id)

\* Consciousness levels match scores
ConsciousnessLevelConsistency ==
    \A video_id \in processed_videos:
        consciousness_levels[video_id] = DetermineConsciousnessLevel(consciousness_scores[video_id])

\* Conscious episodes count is accurate
ConsciousEpisodesAccuracy ==
    conscious_episodes = Cardinality({video_id \in processed_videos : 
        AboveThreshold(consciousness_scores[video_id], consciousness_threshold)})

\* Total episodes matches processed videos
TotalEpisodesAccuracy ==
    total_episodes = Cardinality(processed_videos)

\* Threshold remains within bounds
ThresholdBounds ==
    /\ consciousness_threshold >= MinThreshold
    /\ consciousness_threshold <= MaxThreshold

\* System state is always valid
ValidStateTransitions ==
    system_state \in SystemState

\* State constraint for model checking
StateConstraint ==
    /\ total_episodes <= MaxVideos
    /\ Len(calibration_history) <= 10

\* Liveness Properties

\* Eventually process videos (if any available)
EventuallyProcessVideos ==
    (Cardinality(VideoID) > 0) ~> (total_episodes > 0)

\* Eventually detect consciousness (if threshold is reasonable)
EventuallyDetectConsciousness ==
    (total_episodes > 3 /\ consciousness_threshold <= 80) ~> (conscious_episodes > 0)

\* Eventually calibrate threshold (if needed)
EventuallyCalibrate ==
    (total_episodes > 0 /\ system_state = "complete") ~> (Len(calibration_history) > 0)

====
