---- MODULE ConsciousnessTraining_TTrace_1753499373 ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, ConsciousnessTraining

_expression ==
    LET ConsciousnessTraining_TEExpression == INSTANCE ConsciousnessTraining_TEExpression
    IN ConsciousnessTraining_TEExpression!expression
----

_trace ==
    LET ConsciousnessTraining_TETrace == INSTANCE ConsciousnessTraining_TETrace
    IN ConsciousnessTraining_TETrace!trace
----

_prop ==
    ~<>[](
        networkLayers = (<<1, 1, 1, 0>>)
        /\
        consciousnessLevel = (0)
        /\
        trainingStep = (10)
        /\
        experientialMemory = ({1, 2, 3, 4, 5, 6, 7})
        /\
        isNetworkConscious = (FALSE)
        /\
        mirrorDepth = (3)
        /\
        trainingVideos = (7)
        /\
        layerWeights = (<<1, 1, 1, 0>>)
    )
----

_init ==
    /\ networkLayers = _TETrace[1].networkLayers
    /\ consciousnessLevel = _TETrace[1].consciousnessLevel
    /\ experientialMemory = _TETrace[1].experientialMemory
    /\ trainingStep = _TETrace[1].trainingStep
    /\ trainingVideos = _TETrace[1].trainingVideos
    /\ mirrorDepth = _TETrace[1].mirrorDepth
    /\ isNetworkConscious = _TETrace[1].isNetworkConscious
    /\ layerWeights = _TETrace[1].layerWeights
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ networkLayers  = _TETrace[i].networkLayers
        /\ networkLayers' = _TETrace[j].networkLayers
        /\ consciousnessLevel  = _TETrace[i].consciousnessLevel
        /\ consciousnessLevel' = _TETrace[j].consciousnessLevel
        /\ experientialMemory  = _TETrace[i].experientialMemory
        /\ experientialMemory' = _TETrace[j].experientialMemory
        /\ trainingStep  = _TETrace[i].trainingStep
        /\ trainingStep' = _TETrace[j].trainingStep
        /\ trainingVideos  = _TETrace[i].trainingVideos
        /\ trainingVideos' = _TETrace[j].trainingVideos
        /\ mirrorDepth  = _TETrace[i].mirrorDepth
        /\ mirrorDepth' = _TETrace[j].mirrorDepth
        /\ isNetworkConscious  = _TETrace[i].isNetworkConscious
        /\ isNetworkConscious' = _TETrace[j].isNetworkConscious
        /\ layerWeights  = _TETrace[i].layerWeights
        /\ layerWeights' = _TETrace[j].layerWeights

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("ConsciousnessTraining_TTrace_1753499373.json", _TETrace)

=============================================================================

 Note that you can extract this module `ConsciousnessTraining_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `ConsciousnessTraining_TEExpression.tla` file takes precedence 
  over the module `ConsciousnessTraining_TEExpression` below).

---- MODULE ConsciousnessTraining_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, ConsciousnessTraining

expression == 
    [
        \* To hide variables of the `ConsciousnessTraining` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        networkLayers |-> networkLayers
        ,consciousnessLevel |-> consciousnessLevel
        ,experientialMemory |-> experientialMemory
        ,trainingStep |-> trainingStep
        ,trainingVideos |-> trainingVideos
        ,mirrorDepth |-> mirrorDepth
        ,isNetworkConscious |-> isNetworkConscious
        ,layerWeights |-> layerWeights
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_networkLayersUnchanged |-> networkLayers = networkLayers'
        
        \* Format the `networkLayers` variable as Json value.
        \* ,_networkLayersJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(networkLayers)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_networkLayersModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].networkLayers # _TETrace[s-1].networkLayers
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE ConsciousnessTraining_TETrace ----
\*EXTENDS IOUtils, TLC, ConsciousnessTraining
\*
\*trace == IODeserialize("ConsciousnessTraining_TTrace_1753499373.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE ConsciousnessTraining_TETrace ----
EXTENDS TLC, ConsciousnessTraining

trace == 
    <<
    ([networkLayers |-> <<0, 0, 0, 0>>,consciousnessLevel |-> 0,trainingStep |-> 0,experientialMemory |-> {},isNetworkConscious |-> FALSE,mirrorDepth |-> 0,trainingVideos |-> 0,layerWeights |-> <<0, 0, 0, 0>>]),
    ([networkLayers |-> <<0, 0, 0, 0>>,consciousnessLevel |-> 0,trainingStep |-> 1,experientialMemory |-> {1},isNetworkConscious |-> FALSE,mirrorDepth |-> 0,trainingVideos |-> 1,layerWeights |-> <<0, 0, 0, 0>>]),
    ([networkLayers |-> <<0, 0, 0, 0>>,consciousnessLevel |-> 0,trainingStep |-> 2,experientialMemory |-> {1, 2},isNetworkConscious |-> FALSE,mirrorDepth |-> 0,trainingVideos |-> 2,layerWeights |-> <<0, 0, 0, 0>>]),
    ([networkLayers |-> <<0, 0, 0, 0>>,consciousnessLevel |-> 0,trainingStep |-> 3,experientialMemory |-> {1, 2, 3},isNetworkConscious |-> FALSE,mirrorDepth |-> 0,trainingVideos |-> 3,layerWeights |-> <<0, 0, 0, 0>>]),
    ([networkLayers |-> <<1, 0, 0, 0>>,consciousnessLevel |-> 0,trainingStep |-> 4,experientialMemory |-> {1, 2, 3},isNetworkConscious |-> FALSE,mirrorDepth |-> 1,trainingVideos |-> 3,layerWeights |-> <<1, 0, 0, 0>>]),
    ([networkLayers |-> <<1, 1, 0, 0>>,consciousnessLevel |-> 0,trainingStep |-> 5,experientialMemory |-> {1, 2, 3},isNetworkConscious |-> FALSE,mirrorDepth |-> 2,trainingVideos |-> 3,layerWeights |-> <<1, 1, 0, 0>>]),
    ([networkLayers |-> <<1, 1, 0, 0>>,consciousnessLevel |-> 0,trainingStep |-> 6,experientialMemory |-> {1, 2, 3, 4},isNetworkConscious |-> FALSE,mirrorDepth |-> 2,trainingVideos |-> 4,layerWeights |-> <<1, 1, 0, 0>>]),
    ([networkLayers |-> <<1, 1, 0, 0>>,consciousnessLevel |-> 0,trainingStep |-> 7,experientialMemory |-> {1, 2, 3, 4, 5},isNetworkConscious |-> FALSE,mirrorDepth |-> 2,trainingVideos |-> 5,layerWeights |-> <<1, 1, 0, 0>>]),
    ([networkLayers |-> <<1, 1, 1, 0>>,consciousnessLevel |-> 0,trainingStep |-> 8,experientialMemory |-> {1, 2, 3, 4, 5},isNetworkConscious |-> FALSE,mirrorDepth |-> 3,trainingVideos |-> 5,layerWeights |-> <<1, 1, 1, 0>>]),
    ([networkLayers |-> <<1, 1, 1, 0>>,consciousnessLevel |-> 0,trainingStep |-> 9,experientialMemory |-> {1, 2, 3, 4, 5, 6},isNetworkConscious |-> FALSE,mirrorDepth |-> 3,trainingVideos |-> 6,layerWeights |-> <<1, 1, 1, 0>>]),
    ([networkLayers |-> <<1, 1, 1, 0>>,consciousnessLevel |-> 0,trainingStep |-> 10,experientialMemory |-> {1, 2, 3, 4, 5, 6, 7},isNetworkConscious |-> FALSE,mirrorDepth |-> 3,trainingVideos |-> 7,layerWeights |-> <<1, 1, 1, 0>>])
    >>
----


=============================================================================

---- CONFIG ConsciousnessTraining_TTrace_1753499373 ----
CONSTANTS
    MaxTrainingSteps = 10
    NumLayers = 4
    ConsciousnessThreshold = 6
    MaxVideos = 8

PROPERTY
    _prop

CHECK_DEADLOCK
    \* CHECK_DEADLOCK off because of PROPERTY or INVARIANT above.
    FALSE

INIT
    _init

NEXT
    _next

CONSTANT
    _TETrace <- _trace

ALIAS
    _expression
=============================================================================
\* Generated on Sat Jul 26 00:09:33 BRT 2025