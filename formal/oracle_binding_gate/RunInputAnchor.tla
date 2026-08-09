----------------------------- MODULE RunInputAnchor -----------------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Digests
ASSUME Digests # {}

NoDigest == "NONE"
Phases == {"Prepare", "Prepared", "Running", "Reported", "Verified", "Rejected"}

VARIABLES phase, currentDigest, preparedDigest, reportDigest,
          commitMatches, cleanCodeState, worksMatch, runConfigMatches,
          policyMatches, sourceGroupsValid, globalBasisMatches,
          truthManifestMatches, auditsMatch

vars == <<phase, currentDigest, preparedDigest, reportDigest,
          commitMatches, cleanCodeState, worksMatch, runConfigMatches,
          policyMatches, sourceGroupsValid, globalBasisMatches,
          truthManifestMatches, auditsMatch>>

TypeOK ==
    /\ phase \in Phases
    /\ currentDigest \in Digests
    /\ preparedDigest \in Digests \cup {NoDigest}
    /\ reportDigest \in Digests \cup {NoDigest}
    /\ commitMatches \in BOOLEAN
    /\ cleanCodeState \in BOOLEAN
    /\ worksMatch \in BOOLEAN
    /\ runConfigMatches \in BOOLEAN
    /\ policyMatches \in BOOLEAN
    /\ sourceGroupsValid \in BOOLEAN
    /\ globalBasisMatches \in BOOLEAN
    /\ truthManifestMatches \in BOOLEAN
    /\ auditsMatch \in BOOLEAN

AllFacts ==
    /\ commitMatches
    /\ cleanCodeState
    /\ worksMatch
    /\ runConfigMatches
    /\ policyMatches
    /\ sourceGroupsValid
    /\ globalBasisMatches
    /\ truthManifestMatches
    /\ auditsMatch

Init ==
    /\ phase = "Prepare"
    /\ currentDigest \in Digests
    /\ preparedDigest = NoDigest
    /\ reportDigest = NoDigest
    /\ <<commitMatches, cleanCodeState, worksMatch, runConfigMatches,
         policyMatches, sourceGroupsValid, globalBasisMatches,
         truthManifestMatches, auditsMatch>> \in BOOLEAN \X BOOLEAN \X BOOLEAN
         \X BOOLEAN \X BOOLEAN \X BOOLEAN \X BOOLEAN \X BOOLEAN \X BOOLEAN

Prepare ==
    /\ phase = "Prepare"
    /\ phase' = "Prepared"
    /\ preparedDigest' = currentDigest
    /\ UNCHANGED <<currentDigest, reportDigest,
                    commitMatches, cleanCodeState, worksMatch, runConfigMatches,
                    policyMatches, sourceGroupsValid, globalBasisMatches,
                    truthManifestMatches, auditsMatch>>

MutateBeforeRun ==
    /\ phase \in {"Prepare", "Prepared"}
    /\ currentDigest' \in Digests
    /\ UNCHANGED <<phase, preparedDigest, reportDigest,
                    commitMatches, cleanCodeState, worksMatch, runConfigMatches,
                    policyMatches, sourceGroupsValid, globalBasisMatches,
                    truthManifestMatches, auditsMatch>>

StartRun ==
    /\ phase = "Prepared"
    /\ currentDigest = preparedDigest
    /\ phase' = "Running"
    /\ UNCHANGED <<currentDigest, preparedDigest, reportDigest,
                    commitMatches, cleanCodeState, worksMatch, runConfigMatches,
                    policyMatches, sourceGroupsValid, globalBasisMatches,
                    truthManifestMatches, auditsMatch>>

FinishRun ==
    /\ phase = "Running"
    /\ phase' = "Reported"
    /\ reportDigest' = preparedDigest
    /\ UNCHANGED <<currentDigest, preparedDigest,
                    commitMatches, cleanCodeState, worksMatch, runConfigMatches,
                    policyMatches, sourceGroupsValid, globalBasisMatches,
                    truthManifestMatches, auditsMatch>>

MutateAfterReport ==
    /\ phase = "Reported"
    /\ currentDigest' \in Digests
    /\ UNCHANGED <<phase, preparedDigest, reportDigest,
                    commitMatches, cleanCodeState, worksMatch, runConfigMatches,
                    policyMatches, sourceGroupsValid, globalBasisMatches,
                    truthManifestMatches, auditsMatch>>

Verify ==
    /\ phase = "Reported"
    /\ phase' = IF /\ currentDigest = preparedDigest
                    /\ reportDigest = preparedDigest
                    /\ AllFacts
                 THEN "Verified"
                 ELSE "Rejected"
    /\ UNCHANGED <<currentDigest, preparedDigest, reportDigest,
                    commitMatches, cleanCodeState, worksMatch, runConfigMatches,
                    policyMatches, sourceGroupsValid, globalBasisMatches,
                    truthManifestMatches, auditsMatch>>

Done ==
    /\ phase \in {"Verified", "Rejected"}
    /\ UNCHANGED vars

Next == Prepare \/ MutateBeforeRun \/ StartRun \/ FinishRun \/ MutateAfterReport
        \/ Verify \/ Done

Spec == Init /\ [][Next]_vars

VerifiedSound ==
    phase = "Verified" =>
        /\ preparedDigest # NoDigest
        /\ reportDigest = preparedDigest
        /\ currentDigest = preparedDigest
        /\ AllFacts

NoPosthocPairing ==
    phase = "Verified" => reportDigest = currentDigest

NoVerificationWithoutPreparation ==
    phase = "Verified" => preparedDigest # NoDigest

CleanCodeRequired ==
    phase = "Verified" => cleanCodeState

CompleteProvenanceRequired ==
    phase = "Verified" =>
        /\ sourceGroupsValid
        /\ globalBasisMatches
        /\ truthManifestMatches
        /\ auditsMatch
=============================================================================
