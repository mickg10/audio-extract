-------------------- MODULE PreregisteredOracleBinding --------------------
EXTENDS TLC

VARIABLES policyPinned,
          exactResolutions,
          voicedNonempty,
          noVocalNonempty,
          manifestsPhysicallyDistinct,
          witnessPreexists,
          reportBindsWitness,
          runInputPreexists,
          claimPreexists,
          externalAnchorValid,
          digestBindingsValid,
          outputAbsentThroughPreflight,
          preflightValid,
          reportBindsRunInputClaim,
          hallReferenceComplete,
          hallMetricEvidenceValid,
          allEvidenceValid,
          primaryPass,
          sensitivityPass,
          decision

vars ==
    <<policyPinned, exactResolutions, voicedNonempty, noVocalNonempty,
      manifestsPhysicallyDistinct, witnessPreexists, reportBindsWitness,
      runInputPreexists, claimPreexists, externalAnchorValid,
      digestBindingsValid, outputAbsentThroughPreflight, preflightValid,
      reportBindsRunInputClaim,
      hallReferenceComplete, hallMetricEvidenceValid,
      allEvidenceValid, primaryPass, sensitivityPass, decision>>

Decisions ==
    {"UNDECIDED", "INVALID_EVIDENCE", "ACTIONABLE",
     "RESOLUTION_SENSITIVE", "NO_ACTIONABLE_GAP"}

TypeOK ==
    /\ policyPinned \in BOOLEAN
    /\ exactResolutions \in BOOLEAN
    /\ voicedNonempty \in BOOLEAN
    /\ noVocalNonempty \in BOOLEAN
    /\ manifestsPhysicallyDistinct \in BOOLEAN
    /\ witnessPreexists \in BOOLEAN
    /\ reportBindsWitness \in BOOLEAN
    /\ runInputPreexists \in BOOLEAN
    /\ claimPreexists \in BOOLEAN
    /\ externalAnchorValid \in BOOLEAN
    /\ digestBindingsValid \in BOOLEAN
    /\ outputAbsentThroughPreflight \in BOOLEAN
    /\ preflightValid \in BOOLEAN
    /\ reportBindsRunInputClaim \in BOOLEAN
    /\ hallReferenceComplete \in BOOLEAN
    /\ hallMetricEvidenceValid \in BOOLEAN
    /\ allEvidenceValid \in BOOLEAN
    /\ primaryPass \in BOOLEAN
    /\ sensitivityPass \in BOOLEAN
    /\ decision \in Decisions

RunContractValid ==
    /\ runInputPreexists
    /\ claimPreexists
    /\ externalAnchorValid
    /\ digestBindingsValid
    /\ outputAbsentThroughPreflight
    /\ preflightValid
    /\ reportBindsRunInputClaim
    /\ hallReferenceComplete
    /\ hallMetricEvidenceValid

PreregistrationValid ==
    /\ policyPinned
    /\ exactResolutions
    /\ voicedNonempty
    /\ noVocalNonempty
    /\ manifestsPhysicallyDistinct
    /\ witnessPreexists
    /\ reportBindsWitness
    /\ RunContractValid

Complete == PreregistrationValid /\ allEvidenceValid

DecisionOf ==
    IF ~Complete THEN "INVALID_EVIDENCE"
    ELSE IF primaryPass THEN "ACTIONABLE"
    ELSE IF sensitivityPass THEN "RESOLUTION_SENSITIVE"
    ELSE "NO_ACTIONABLE_GAP"

Init ==
    /\ policyPinned \in BOOLEAN
    /\ exactResolutions \in BOOLEAN
    /\ voicedNonempty \in BOOLEAN
    /\ noVocalNonempty \in BOOLEAN
    /\ manifestsPhysicallyDistinct \in BOOLEAN
    /\ witnessPreexists \in BOOLEAN
    /\ reportBindsWitness \in BOOLEAN
    /\ runInputPreexists \in BOOLEAN
    /\ claimPreexists \in BOOLEAN
    /\ externalAnchorValid \in BOOLEAN
    /\ digestBindingsValid \in BOOLEAN
    /\ outputAbsentThroughPreflight \in BOOLEAN
    /\ preflightValid \in BOOLEAN
    /\ reportBindsRunInputClaim \in BOOLEAN
    /\ hallReferenceComplete \in BOOLEAN
    /\ hallMetricEvidenceValid \in BOOLEAN
    /\ allEvidenceValid \in BOOLEAN
    /\ primaryPass \in BOOLEAN
    /\ sensitivityPass \in BOOLEAN
    /\ decision = "UNDECIDED"

Decide ==
    /\ decision = "UNDECIDED"
    /\ decision' = DecisionOf
    /\ UNCHANGED
        <<policyPinned, exactResolutions, voicedNonempty, noVocalNonempty,
          manifestsPhysicallyDistinct, witnessPreexists, reportBindsWitness,
          runInputPreexists, claimPreexists, externalAnchorValid,
          digestBindingsValid, outputAbsentThroughPreflight, preflightValid,
          reportBindsRunInputClaim,
          hallReferenceComplete, hallMetricEvidenceValid,
          allEvidenceValid, primaryPass, sensitivityPass>>

Done ==
    /\ decision # "UNDECIDED"
    /\ UNCHANGED vars

Next == Decide \/ Done
Spec == Init /\ [][Next]_vars

ActionableSound ==
    decision = "ACTIONABLE" =>
        /\ PreregistrationValid
        /\ allEvidenceValid
        /\ primaryPass

NoRoundedResolutionPromotion ==
    ~exactResolutions => decision # "ACTIONABLE"

NoAliasedManifestPromotion ==
    ~manifestsPhysicallyDistinct => decision # "ACTIONABLE"

NoPosthocWitnessPromotion ==
    (~witnessPreexists \/ ~reportBindsWitness) => decision # "ACTIONABLE"

NoUnclaimedV3Promotion ==
    (~runInputPreexists \/ ~claimPreexists \/ ~externalAnchorValid) =>
        decision # "ACTIONABLE"

NoInvalidV3BindingPromotion ==
    (~digestBindingsValid \/ ~preflightValid \/ ~reportBindsRunInputClaim) =>
        decision # "ACTIONABLE"

NoPostOutputClaimPromotion ==
    ~outputAbsentThroughPreflight => decision # "ACTIONABLE"

NoMissingHallPromotion ==
    (~hallReferenceComplete \/ ~hallMetricEvidenceValid) =>
        decision # "ACTIONABLE"

SensitivityDoesNotPromote ==
    (~primaryPass /\ sensitivityPass) => decision # "ACTIONABLE"

InvalidSound ==
    decision = "INVALID_EVIDENCE" => ~Complete

=============================================================================
