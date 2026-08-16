#!/usr/bin/env python3
"""Verifier for the canonical immutable voice-gate calibration artifact.

REJECTS (nonzero exit + reason) on:
  - changed canonical bytes (recompute canonical_sha256 and compare)
  - any missing required hash (models, configs, code, f32 parents, regen)
  - a lossy_origin file used as an analysis parent
  - acoustic threshold != -10.0 (frozen)
  - candidate/control set mismatch (a scored candidate without a matched-envelope control+seeds)
  - missing/empty scramble seeds
  - any unavailable detector
  - verdict recomputation mismatch (re-derive each verdict from stored scores via the binding gate rule)
Also enforces internal consistency of the detector<->removal-DAG independence bookkeeping and
prints a prominent NON-INDEPENDENCE warning (the median-ensemble acoustic detector shares model
family with the removal ensemble; acoustic-PASS is provisional). Prints PASS + summary if all hold.

Usage: python3 verify_voice_gate_artifact.py [path-to-voice_gate_calibration_v1.json]
"""
import json, sys, hashlib, re

DEFAULT = "/home/mickg/ab_pairs/voice_gate_calibration_v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FROZEN_ACOUSTIC_THRESHOLD_DB = -10.0

fails = []
warns = []
def REJECT(msg): fails.append(msg)
def WARN(msg): warns.append(msg)
def need_hash(v, where):
    if not (isinstance(v, str) and HEX64.match(v)):
        REJECT(f"missing/invalid required hash: {where} = {v!r}")


# ---- binding gate rule (identical to builder) ----
def acoustic_pass(worst_db, thr):
    return worst_db is not None and worst_db <= thr
def acoustic_available(worst_db):
    return worst_db is not None
def corrected_asr_hard_fail(cu, hardfail_thr):
    return cu is not None and cu >= hardfail_thr
def asr_available(cu):
    return cu is not None
def provenance_valid(f32_sha, lossy):
    return bool(f32_sha) and not lossy
def gate_evaluate(worst_db, cu, f32_sha, lossy, acoustic_thr, hardfail_thr):
    ap = acoustic_pass(worst_db, acoustic_thr); aa = acoustic_available(worst_db)
    hf = corrected_asr_hard_fail(cu, hardfail_thr); sa = asr_available(cu)
    pv = provenance_valid(f32_sha, lossy)
    passed = ap and (not hf) and aa and sa and pv
    if not aa: reason = "fail_closed:acoustic_unavailable"
    elif not ap: reason = "fail:acoustic(worst_db>-10)"
    elif hf: reason = "fail:corrected_asr(control_unique>=floor+margin)"
    elif not pv: reason = "fail_closed:provenance(lossy_origin_or_missing_f32)"
    elif not sa: reason = "fail_closed:asr_corrected_unmeasured_on_this_pcm"
    else: reason = "pass"
    return dict(pass_=bool(passed), reason=reason,
                components=dict(acoustic_pass=ap, acoustic_available=aa,
                                corrected_ASR_hard_fail=hf, asr_available=sa, provenance_valid=pv))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    obj = json.load(open(path))

    # ============ CHECK 1: canonical bytes ============
    if "canonical_sha256" not in obj:
        REJECT("no canonical_sha256 field present")
        stored = None
    else:
        stored = obj["canonical_sha256"]
    payload = {k: v for k, v in obj.items() if k != "canonical_sha256"}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    recomputed = hashlib.sha256(canon).hexdigest()
    if stored != recomputed:
        REJECT(f"canonical_sha256 mismatch: stored={stored} recomputed={recomputed} (artifact bytes changed)")

    # ============ CHECK 2/... structural + thresholds ============
    bgr = obj.get("binding_gate_rule", {})
    if bgr.get("acoustic_threshold_db") != FROZEN_ACOUSTIC_THRESHOLD_DB:
        REJECT(f"binding_gate_rule.acoustic_threshold_db != {FROZEN_ACOUSTIC_THRESHOLD_DB}")
    ad = obj.get("acoustic_detector", {})
    if ad.get("threshold_db") != FROZEN_ACOUSTIC_THRESHOLD_DB:
        REJECT(f"acoustic_detector.threshold_db != {FROZEN_ACOUSTIC_THRESHOLD_DB}")

    # ---- unavailable detector ----
    if ad.get("availability") is not True:
        REJECT("acoustic detector marked unavailable")
    asrd = obj.get("asr_detector", {})
    cal = asrd.get("calibration", {})
    if cal.get("availability") is not True:
        REJECT("ASR detector/calibration marked unavailable")

    # ---- required model/config hashes ----
    for name, h in obj.get("model_hashes", {}).items():
        need_hash(h, f"model_hashes[{name}]")
    for mk, mv in ad.get("models", {}).items():
        need_hash(mv.get("sha256"), f"acoustic_detector.models.{mk}.sha256")
        need_hash(mv.get("config_sha256"), f"acoustic_detector.models.{mk}.config_sha256")
    detector_shas = set()
    for mv in ad.get("models", {}).values():
        detector_shas.add(mv.get("sha256"))
    # ---- required code hashes ----
    for ck, cv in obj.get("code_hashes", {}).items():
        need_hash(cv, f"code_hashes[{ck}]")
    # ---- regen hashes ----
    pr = obj.get("provenance_regeneration", {})
    need_hash(pr.get("B_melband_ft_pcm_sha256"), "provenance_regeneration.B_melband_ft_pcm_sha256")
    need_hash(pr.get("M_maxagg_pcm_sha256"), "provenance_regeneration.M_maxagg_pcm_sha256")
    need_hash(pr.get("A_champion_repro_pcm_sha256"), "provenance_regeneration.A_champion_repro_pcm_sha256")

    # ---- collect the set of KNOWN lossy listening-copy hashes (must never be an analysis parent) ----
    lossy_hashes = set()
    for c in obj.get("candidates", {}).values():
        llc = (c.get("provenance") or {}).get("lossy_listening_copy")
        if llc and llc.get("pcm_sha256"):
            lossy_hashes.add(llc["pcm_sha256"])
    allh = (obj.get("provenance_regeneration", {}) or {}).get("all_pcm_hashes", {})
    for tag, e in allh.items():
        if isinstance(e, dict) and e.get("lossy_origin") and e.get("pcm_sha256"):
            lossy_hashes.add(e["pcm_sha256"])

    # ============ ASR control/seed presence ============
    cm = cal.get("corrected_metric", {})
    seeds = cm.get("phase_scramble_seeds")
    seed_ov = cm.get("phase_scramble_seed_overlaps_pct")
    if not seeds:
        REJECT("missing/empty phase-scramble seeds (corrected_metric.phase_scramble_seeds)")
    if not seed_ov or not isinstance(seed_ov, dict) or len(seed_ov) == 0:
        REJECT("missing/empty phase-scramble seed overlaps")
    else:
        for k, v in seed_ov.items():
            if not isinstance(v, (int, float)):
                REJECT(f"non-numeric phase-scramble seed overlap {k}={v!r}")
    controls = cal.get("transcripts_and_controls", {})
    zero_voice = [k for k, v in controls.items() if v.get("kind") == "zero_voice_control"]
    if not zero_voice:
        REJECT("no matched-envelope zero_voice_control present in ASR calibration")
    # floor/threshold consistency
    floor = cm.get("floor_pct"); margin = cm.get("margin_pct"); hardfail_thr = cm.get("hardfail_threshold_pct")
    if floor is None or margin is None or hardfail_thr is None:
        REJECT("ASR corrected_metric floor/margin/hardfail_threshold missing")
        hardfail_thr = 10 ** 9
    elif round(floor + margin, 6) != round(hardfail_thr, 6):
        REJECT(f"ASR hardfail_threshold_pct({hardfail_thr}) != floor({floor})+margin({margin})")

    # ============ per-candidate checks + verdict recomputation ============
    cands = obj.get("candidates", {})
    if not cands:
        REJECT("no candidates present")
    scored = []
    verdict_rows = []
    for name, c in cands.items():
        prov = c.get("provenance", {})
        ac = c.get("acoustic", {})
        asb = c.get("asr", {})
        lossy = bool(prov.get("lossy_origin"))
        f32 = prov.get("analysis_parent_f32_pcm_sha256")
        worst = ac.get("worst_db")
        cu = asb.get("corrected_control_unique_pct")

        # required hashes per candidate
        if lossy:
            llc = prov.get("lossy_listening_copy") or {}
            need_hash(llc.get("pcm_sha256"), f"candidates[{name}].lossy_listening_copy.pcm_sha256")
            # a lossy candidate MUST NOT claim a lossless f32 parent
            if prov.get("analysis_parent_lossless") or f32:
                REJECT(f"lossy_origin candidate {name} claims an f32 analysis parent (lossy used as parent)")
            if prov.get("provenance_valid"):
                REJECT(f"lossy_origin candidate {name} has provenance_valid=true (lossy used as valid parent)")
        else:
            need_hash(f32, f"candidates[{name}].analysis_parent_f32_pcm_sha256")
            if f32 in lossy_hashes:
                REJECT(f"candidate {name} uses a KNOWN lossy hash {f32} as its f32 analysis parent")
            if not prov.get("analysis_parent_lossless"):
                REJECT(f"non-lossy candidate {name} not marked analysis_parent_lossless")

        # acoustic availability must be present (detector produced a score)
        if ac.get("available") is not True:
            REJECT(f"candidate {name} acoustic detector unavailable (no worst_db)")

        # independence bookkeeping consistency (recompute shared hashes)
        rd = c.get("removal_dag", {})
        if rd.get("dag_known"):
            dag_shas = set(rd.get("dag_model_sha256_set", []))
            shared_recomputed = sorted(detector_shas & dag_shas)
            if sorted(rd.get("acoustic_detector_shared_hashes", [])) != shared_recomputed:
                REJECT(f"candidate {name} removal_dag.acoustic_detector_shared_hashes inconsistent with recompute")
            indep_expected = (len(shared_recomputed) == 0)
            if rd.get("acoustic_detector_independent") != indep_expected:
                REJECT(f"candidate {name} acoustic_detector_independent flag inconsistent")
            if not indep_expected and ac.get("acoustic_pass"):
                WARN(f"{name}: acoustic-PASS but detector NON-INDEPENDENT of removal DAG (shared {shared_recomputed}) -> circular/provisional")
        else:
            if rd.get("acoustic_detector_independent") is not None:
                WARN(f"{name}: removal DAG unknown but independence flag not null")

        # recompute stored per-signal booleans
        if ac.get("acoustic_pass") != acoustic_pass(worst, FROZEN_ACOUSTIC_THRESHOLD_DB):
            REJECT(f"candidate {name} stored acoustic_pass != recompute")
        if asb.get("corrected_ASR_hard_fail") != corrected_asr_hard_fail(cu, hardfail_thr):
            REJECT(f"candidate {name} stored corrected_ASR_hard_fail != recompute")
        if asb.get("corrected_available") != asr_available(cu):
            REJECT(f"candidate {name} stored corrected_available != recompute")

        # recompute overall verdict
        rv = gate_evaluate(worst, cu, f32, lossy, FROZEN_ACOUSTIC_THRESHOLD_DB, hardfail_thr)
        sv = c.get("gate_verdict", {})
        if sv.get("pass_") != rv["pass_"]:
            REJECT(f"candidate {name} gate_verdict.pass_ mismatch: stored={sv.get('pass_')} recompute={rv['pass_']}")
        if sv.get("reason") != rv["reason"]:
            REJECT(f"candidate {name} gate_verdict.reason mismatch: stored={sv.get('reason')} recompute={rv['reason']}")
        if sv.get("components") != rv["components"]:
            REJECT(f"candidate {name} gate_verdict.components mismatch: stored={sv.get('components')} recompute={rv['components']}")

        if asr_available(cu):
            scored.append(name)
        verdict_rows.append((name, worst, ac.get("acoustic_pass"), cu,
                             asb.get("corrected_ASR_hard_fail"), prov.get("provenance_valid"),
                             rv["pass_"], rv["reason"]))

    # ============ candidate/control set: every SCORED candidate needs matched-envelope controls+seeds ============
    if scored and (not seeds or not zero_voice):
        REJECT(f"scored candidates {scored} exist but matched-envelope control/seed set is missing")

    # ---- anchor consistency: floor == V1_inverted corrected control-unique ----
    v1 = cands.get("V1_inverted", {})
    v1cu = (v1.get("asr") or {}).get("corrected_control_unique_pct")
    if v1cu is not None and floor is not None and round(v1cu, 6) != round(floor, 6):
        REJECT(f"ASR floor_pct({floor}) != V1_inverted corrected control-unique({v1cu})")

    # ---- detector independence top-level assertions ----
    if ad.get("independent_of_removal_dag") is not False:
        WARN("acoustic_detector.independent_of_removal_dag is not False — expected False (median-ensemble is circular)")
    if asrd.get("independent_of_removal_dag") is not True:
        REJECT("asr_detector.independent_of_removal_dag is not True")

    # ================= REPORT =================
    if fails:
        print("VERIFY: REJECT")
        for f in fails:
            print("  [FAIL]", f)
        if warns:
            print("  --- warnings ---")
            for w in warns:
                print("  [WARN]", w)
        sys.exit(1)

    print("VERIFY: PASS")
    print(f"  artifact: {path}")
    print(f"  schema {obj.get('schema')} {obj.get('schema_version')} artifact_version {obj.get('artifact_version')} track {obj.get('track')}")
    print(f"  canonical_sha256 OK: {recomputed}")
    print(f"  acoustic threshold FROZEN at {FROZEN_ACOUSTIC_THRESHOLD_DB} dB; ASR hard-fail threshold {hardfail_thr}% (floor {floor} + margin {margin})")
    print(f"  candidates={len(cands)}  scored(asr)={len(scored)}  zero_voice_controls={len(zero_voice)}  phase_scramble_seeds={seeds}")
    print(f"  all model/config/code/f32/regen hashes present; no lossy file used as an analysis parent")
    print(f"  A_champion reproduces iter1 (determinism): {pr.get('A_champion_reproduces_iter1')}")
    print("  per-candidate verdict (recomputed == stored):")
    print("    %-14s %9s %5s %7s %7s %6s  %s" % ("candidate","worst_db","acPS","cu%","asrHF","prov","verdict|reason"))
    for (name, worst, acps, cu, hf, pv, p, r) in verdict_rows:
        print("    %-14s %9s %5s %7s %7s %6s  %s|%s" % (
            name, worst, acps, cu, hf, pv, ("PASS" if p else "FAIL"), r))
    if warns:
        print("  --- NON-INDEPENDENCE / advisory warnings (do not invalidate the artifact) ---")
        for w in warns:
            print("  [WARN]", w)
    print("  NOTE: no candidate is a certified PASS; the acoustic-PASS deliverable (C_cascade/Z, Z2) fail-closes on")
    print("        corrected-ASR-unmeasured (== needs_human_ab). Acoustic detector is NON-INDEPENDENT of the removal")
    print("        ensemble (circular) — acoustic-PASS is PROVISIONAL pending independent-family/human confirmation.")
    sys.exit(0)


if __name__ == "__main__":
    main()
