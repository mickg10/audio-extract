--------------------------- MODULE OracleBindingGate ---------------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Methods, Resolutions, PrimaryResolution, SensitivityResolutions,
          SelectedMethod

ASSUME /\ SelectedMethod \in Methods
       /\ PrimaryResolution \in Resolutions
       /\ SensitivityResolutions \subseteq Resolutions \ {PrimaryResolution}
       /\ Resolutions = {PrimaryResolution} \cup SensitivityResolutions

Cells == Methods \X Resolutions
Decisions == {"UNLOADED", "INVALID_EVIDENCE", "ACTIONABLE",
              "RESOLUTION_SENSITIVE", "NO_ACTIONABLE_GAP"}

VARIABLES phase, valid, passes, decision
vars == <<phase, valid, passes, decision>>

TypeOK ==
    /\ phase \in {"Choose", "Decided"}
    /\ valid \in [Cells -> BOOLEAN]
    /\ passes \in [Cells -> BOOLEAN]
    /\ decision \in Decisions

AllValid == \A cell \in Cells : valid[cell]
SelectedPrimaryPass == passes[<<SelectedMethod, PrimaryResolution>>]
SelectedSensitivityPass ==
    \E resolution \in SensitivityResolutions :
        passes[<<SelectedMethod, resolution>>]

DecisionOf ==
    IF ~AllValid THEN "INVALID_EVIDENCE"
    ELSE IF SelectedPrimaryPass THEN "ACTIONABLE"
    ELSE IF SelectedSensitivityPass THEN "RESOLUTION_SENSITIVE"
    ELSE "NO_ACTIONABLE_GAP"

Init ==
    /\ phase = "Choose"
    /\ valid \in [Cells -> BOOLEAN]
    /\ passes \in [Cells -> BOOLEAN]
    /\ decision = "UNLOADED"

Decide ==
    /\ phase = "Choose"
    /\ phase' = "Decided"
    /\ decision' = DecisionOf
    /\ UNCHANGED <<valid, passes>>

Done ==
    /\ phase = "Decided"
    /\ UNCHANGED vars

Next == Decide \/ Done
Spec == Init /\ [][Next]_vars

ActionableSound ==
    decision = "ACTIONABLE" =>
        /\ AllValid
        /\ SelectedPrimaryPass

SensitivitySound ==
    decision = "RESOLUTION_SENSITIVE" =>
        /\ AllValid
        /\ ~SelectedPrimaryPass
        /\ SelectedSensitivityPass

InvalidSound ==
    decision = "INVALID_EVIDENCE" => ~AllValid

NoGapSound ==
    decision = "NO_ACTIONABLE_GAP" =>
        /\ AllValid
        /\ ~SelectedPrimaryPass
        /\ ~SelectedSensitivityPass

NoCrossMethodPromotion ==
    decision = "ACTIONABLE" => passes[<<SelectedMethod, PrimaryResolution>>]

NoSensitivityPromotion ==
    decision = "ACTIONABLE" => ~(
        ~passes[<<SelectedMethod, PrimaryResolution>>]
        /\ SelectedSensitivityPass
    )

TerminalDecision ==
    phase = "Decided" => decision \in Decisions \ {"UNLOADED"}
=============================================================================
