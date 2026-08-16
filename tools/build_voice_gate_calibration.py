#!/usr/bin/env python3
"""Build the canonical immutable voice-gate calibration artifact for verdi_slow_voices.

Assembles gpt56's REQUIRED artifact from ALREADY-MEASURED data + the f32-provenance regeneration.
Recomputes every candidate's overall verdict from stored scores via the BINDING GATE RULE and
stores it. Emits voice_gate_calibration_v1.json with a canonical_sha256 over the canonical
serialization of everything except that field.

Run: python3 build_artifact.py
"""
import json, hashlib, os, time

AB = "/home/mickg/ab_pairs"
CAND = "/home/mickg/gate_candidates/verdi_slow_voices"
REPO = "/home/mickg/audio-extract"
OUT = f"{AB}/voice_gate_calibration_v1.json"

# ---- FROZEN gate parameters ----
ACOUSTIC_THRESHOLD_DB = -10.0
ASR_FLOOR_PCT = 26.5           # V1_inverted (human voice-free anchor) corrected control-unique = 9/34
ASR_MARGIN_PCT = 5.9           # -> hard-fail threshold 32.4% = theoretical control-unique ceiling (34 - |control union 23|)
N_REF = 34
CONTROL_UNION_SIZE = 23        # union_priming_recoverable_refwords over the 8 zero-voice controls
ASR_HARDFAIL_THRESHOLD_PCT = round(ASR_FLOOR_PCT + ASR_MARGIN_PCT, 1)  # 32.4

# ---- verified model/config sha256 (sha256sum on research6) ----
MODEL_HASHES = {
    "MDX23C-8KFFT-InstVoc_HQ.ckpt": "49d51472769e34a2501cd1da782346a3212555c3a5619fc2c53507445528d816",
    "vocals_mel_band_roformer.ckpt": "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e",
    "vocals_mel_band_roformer.yaml": "b958b29c8f7195f0d86bee6759a33980db675c4ecaf2fcaa80fa125828e6cd38",
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt": "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
    "model_bs_roformer_ep_317_sdr_12.9755.yaml": "2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f",
    "mdx_model_data.json": "1aca8f9bcc57233bc714029663a9ec2345d9c7721f91e5e08f46392a879c6a9a",
    "melband_ft_step600.ckpt": "1c12a6b764b9c329890c0db8b16d5dfb91e67f16d0130a2892ad7cbab9edf302",
}
SHA_MDX = MODEL_HASHES["MDX23C-8KFFT-InstVoc_HQ.ckpt"]
SHA_MEL = MODEL_HASHES["vocals_mel_band_roformer.ckpt"]
SHA_BS = MODEL_HASHES["model_bs_roformer_ep_317_sdr_12.9755.ckpt"]
SHA_MELFT = MODEL_HASHES["melband_ft_step600.ckpt"]
DETECTOR_MODEL_SHAS = sorted([SHA_MDX, SHA_MEL, SHA_BS])

SOFTWARE = {
    "detector_env": {"python": "3.12.13", "torch": "2.13.0+cu130", "numpy": "2.4.6",
                     "soundfile": "0.14.0", "audio_separator": "0.44.5",
                     "venv": "/home/mickg/audio-extract/.venv (uv run --no-sync via run.sh)"},
    "asr_env": {"python": "3.12.13", "faster_whisper": "1.2.1", "ctranslate2": "4.8.1",
                "onnxruntime": "1.28.0", "numpy": "2.5.2",
                "whisper_model": "large-v3", "device": "cuda", "compute_type": "float16",
                "venv": "/home/mickg/asr_venv"},
}

CODE_FILES = {
    "voice_gate2.py": f"{REPO}/audio_extract/voice_gate2.py",
    "voice_gate.py": f"{REPO}/audio_extract/voice_gate.py",
    "render_ab.py": f"{REPO}/audio_extract/render_ab.py",
    "render_aggressive.py": f"{REPO}/audio_extract/render_aggressive.py",
    "asr_gate.py": "/home/mickg/asr_gate.py",
    "asr_controls.py": "/home/mickg/asr_controls.py",
    "asr_seeds.py": "/home/mickg/asr_seeds.py",
    "dsp_controls.py": "/home/mickg/dsp_controls.py",
    "dsp_seeds.py": "/home/mickg/dsp_seeds.py",
    "finalize.py": "/home/mickg/finalize.py",
    "regen_bm.py": "/home/mickg/gate_candidates/regen_bm.py",
}


def sha256_file(p):
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    return json.load(open(p))


# =========================================================================================
#  BINDING GATE RULE  (implemented verbatim; used identically by the verifier)
# =========================================================================================
def acoustic_pass(worst_db):
    return worst_db is not None and worst_db <= ACOUSTIC_THRESHOLD_DB

def acoustic_available(worst_db):
    return worst_db is not None

def corrected_asr_hard_fail(cu_pct):
    return cu_pct is not None and cu_pct >= ASR_HARDFAIL_THRESHOLD_PCT

def asr_available(cu_pct):
    return cu_pct is not None

def provenance_valid(f32_sha, lossy_origin):
    return bool(f32_sha) and not lossy_origin

def gate_evaluate(worst_db, cu_pct, f32_sha, lossy_origin):
    ap = acoustic_pass(worst_db); aa = acoustic_available(worst_db)
    hf = corrected_asr_hard_fail(cu_pct); sa = asr_available(cu_pct)
    pv = provenance_valid(f32_sha, lossy_origin)
    passed = ap and (not hf) and aa and sa and pv
    # primary reason (acoustic is the PRIMARY hard veto)
    if not aa:
        reason = "fail_closed:acoustic_unavailable"
    elif not ap:
        reason = "fail:acoustic(worst_db>-10)"
    elif hf:
        reason = "fail:corrected_asr(control_unique>=floor+margin)"
    elif not pv:
        reason = "fail_closed:provenance(lossy_origin_or_missing_f32)"
    elif not sa:
        reason = "fail_closed:asr_corrected_unmeasured_on_this_pcm"
    else:
        reason = "pass"
    return dict(pass_=bool(passed), reason=reason,
                components=dict(acoustic_pass=ap, acoustic_available=aa,
                                corrected_ASR_hard_fail=hf, asr_available=sa,
                                provenance_valid=pv))


def main():
    ev = load(f"{AB}/voice_gate2_verdi.json")["candidates"]
    man = {n: load(f"{CAND}/{n}.manifest.json") for n in
           ["iter0", "iter1", "iter2", "iter2_Z2_comp"]}
    asr = load(f"{AB}/asr_report.json")
    prim = load(f"{AB}/asr_priming_controls.json")
    regen = load(f"{CAND}/provenance_regen.json")
    det = regen["detector_perwindow"]
    H = regen["hashes"]
    S = regen["stages"]

    def hval(tag):
        e = H.get(tag, {})
        return e.get("pcm_sha256")

    code_hashes = {k: sha256_file(v) for k, v in CODE_FILES.items()}

    # ---- ASR corrected metric (from priming controls) ----
    corr = prim["corrected"]
    ms = corr["multiseed_floor"]
    R = prim["results"]
    ref_words = [w["word"] for w in asr["_reference"]["verdi_slow_voices"]["lyric_words"]]

    def control_unique_pct(name, n_recovered, overlap_pct):
        """Exactly computable only where the recovered ref-word set is known."""
        if name == "V1_inverted":
            return ms["V1_unique_overlap_corrected_pct"]           # 26.5 (9/34)
        if n_recovered == 0:
            return 0.0
        if overlap_pct is not None and abs(overlap_pct - 100.0) < 1e-6 and n_recovered == N_REF:
            return round(100.0 * (N_REF - CONTROL_UNION_SIZE) / N_REF, 1)   # 32.4 (all-34 minus union)
        return None   # recovered_ref_idx not stored -> not exactly computable

    def raw_asr(name):
        e = asr.get(f"verdi_slow_voices__{name}")
        if not e:
            return None
        return dict(raw_overlap_pct=e["overlap_pct"], n_recovered=e["n_recovered"],
                    n_ref_scored=e["n_ref_scored"], no_speech_prob=e["no_speech_prob"],
                    avg_logprob=e["avg_logprob"], wf_corr_vs_orig=e.get("wf_corr_vs_orig"),
                    transcript=e["transcript"],
                    old_15pct_fail=e["asr_fail"])

    # ---------------------------------------------------------------------------
    #  CANDIDATE SPECS
    #  (canonical lossless f32 candidates + lossy-only listening-copy variants)
    #  det_tag -> per-window detector recompute key; asr_name -> asr_report key
    # ---------------------------------------------------------------------------
    DAG_BASE = [{"model": "MDX23C-8KFFT-InstVoc_HQ.ckpt", "sha256": SHA_MDX},
                {"model": "vocals_mel_band_roformer.ckpt", "sha256": SHA_MEL},
                {"model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "sha256": SHA_BS}]
    DAG_FT = [{"model": "MDX23C-8KFFT-InstVoc_HQ.ckpt", "sha256": SHA_MDX},
              {"model": "melband_ft_step600.ckpt", "sha256": SHA_MELFT},
              {"model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "sha256": SHA_BS}]

    SPECS = [
        # name, iteration, parent_name, det_tag, asr_name, f32_source_tag, lossy_origin, recipe, dag_models, dag_known, aliases, terminal
        dict(name="O_original", iteration=0, parent=None, det_tag="iter0", asr_name="O_original",
             f32_tag="iter0", lossy=False, recipe="raw program mixture (nothing removed)",
             dag=[], dag_known=True, aliases=["iter0"], terminal=None,
             m4a_tag="O_original_m4a"),
        dict(name="A_champion", iteration=1, parent="O_original", det_tag="iter1", asr_name="A_champion",
             f32_tag="iter1", lossy=False, recipe="mix - median3(MDX23C,MelBand_base,BS)",
             dag=DAG_BASE, dag_known=True, aliases=["iter1"], terminal=None,
             m4a_tag="A_champion_m4a"),
        dict(name="C_cascade", iteration=2, parent="A_champion", det_tag="iter2", asr_name="C_cascade",
             f32_tag="iter2", lossy=False,
             recipe="cascade 2nd pass: A1 - median3(MDX23C,MelBand_base,BS)(A1)  [== Z deliverable]",
             dag=DAG_BASE, dag_known=True, aliases=["iter2", "Z", "Z_gate2_iter2"],
             terminal="needs_human_ab", m4a_tag="C_cascade_m4a"),
        dict(name="Z2", iteration=2, parent="C_cascade", det_tag="iter2_Z2_comp", asr_name=None,
             f32_tag="iter2_Z2_comp", lossy=False,
             recipe="spectral_envelope_lift(Z->mix over mix no-voice frames; nfft4096 hop1024; gain clip[1,2.5] smoothed)  [DSP only]",
             dag=DAG_BASE, dag_known=True, aliases=["iter2_Z2_comp", "Z2_gate2_iter2_comp"],
             terminal=None, m4a_tag="Z2_gate2_iter2_comp_m4a"),
        dict(name="B_melband_ft", iteration=None, parent="O_original", det_tag="B_melband_ft", asr_name="B_melband_ft",
             f32_tag=None, f32_from_stage="B_melband_ft", lossy=False,
             recipe="mix - median3(MDX23C,MelBand_ft@step600,BS)",
             dag=DAG_FT, dag_known=True, aliases=[], terminal=None, m4a_tag="B_melband_ft_m4a"),
        dict(name="M_maxagg", iteration=None, parent="O_original", det_tag="M_maxagg", asr_name="M_maxagg",
             f32_tag=None, f32_from_stage="M_maxagg", lossy=False,
             recipe="mix - per_sample_per_channel_argmax_abs(MDX23C,MelBand_base,BS)",
             dag=DAG_BASE, dag_known=True, aliases=[], terminal=None, m4a_tag="M_maxagg_m4a"),
        # ---- lossy-only variants (listening copies; NOT valid analysis parents) ----
        dict(name="V1_inverted", iteration=None, parent=None, det_tag=None, asr_name="V1_inverted",
             f32_tag=None, lossy=True, recipe="UNVERIFIED (generation code not located); m4a listening copy only",
             dag=None, dag_known=False, aliases=[], terminal=None, m4a_tag="V1_inverted_m4a",
             human_anchor="voice-free"),
        dict(name="V2_mdx23c", iteration=None, parent=None, det_tag=None, asr_name="V2_mdx23c",
             f32_tag=None, lossy=True, recipe="UNVERIFIED (name implies MDX23C-only); m4a listening copy only",
             dag=None, dag_known=False, aliases=[], terminal=None, m4a_tag="V2_mdx23c_m4a"),
        dict(name="V3_roformer", iteration=None, parent=None, det_tag=None, asr_name="V3_roformer",
             f32_tag=None, lossy=True, recipe="UNVERIFIED (name implies a roformer variant); m4a listening copy only",
             dag=None, dag_known=False, aliases=[], terminal=None, m4a_tag="V3_roformer_m4a"),
        dict(name="V4_deechorev", iteration=None, parent=None, det_tag=None, asr_name="V4_deechorev",
             f32_tag=None, lossy=True, recipe="UNVERIFIED (name implies de-echo/reverb variant); m4a listening copy only",
             dag=None, dag_known=False, aliases=[], terminal=None, m4a_tag="V4_deechorev_m4a"),
        dict(name="V5_ensmax", iteration=None, parent=None, det_tag=None, asr_name="V5_ensmax",
             f32_tag=None, lossy=True, recipe="UNVERIFIED (name implies ensemble-max variant); m4a listening copy only",
             dag=None, dag_known=False, aliases=[], terminal=None, m4a_tag="V5_ensmax_m4a"),
    ]

    def f32_hash_for(sp):
        if sp.get("f32_tag"):
            return hval(sp["f32_tag"])
        if sp.get("f32_from_stage"):
            return S[sp["f32_from_stage"]]["pcm_sha256"]
        return None   # lossy-only

    candidates = {}
    for sp in SPECS:
        name = sp["name"]
        evc = ev.get(name, {})
        # ---- acoustic (prefer FRESH lossless f32 per-window recompute; else frozen m4a value) ----
        dtag = sp["det_tag"]
        if dtag and dtag in det:
            d = det[dtag]
            worst = d["worst_db"]
            acoustic = dict(
                worst_db=worst, threshold_db=ACOUSTIC_THRESHOLD_DB,
                acoustic_pass=acoustic_pass(worst), available=acoustic_available(worst),
                measured_on="lossless_f32_recomputed",
                n_windows=d["n_windows"], n_kept=d["n_kept"], n_novoice=d["n_novoice"],
                window_starts_samples=d["window_starts_samples"],
                per_window_rel_db=d["per_window_rel_db"],
                frozen_table_worst_db=evc.get("acoustic_worst_db"))
        else:
            worst = evc.get("acoustic_worst_db")
            acoustic = dict(
                worst_db=worst, threshold_db=ACOUSTIC_THRESHOLD_DB,
                acoustic_pass=acoustic_pass(worst), available=acoustic_available(worst),
                measured_on="m4a_decode(lossy)_frozen_table",
                n_windows=None, n_kept=None, n_novoice=None,
                window_starts_samples=None, per_window_rel_db=None,
                frozen_table_worst_db=worst)

        # ---- removal DAG + detector-independence diagnostic ----
        dag = sp["dag"]
        dag_shas = sorted({m["sha256"] for m in dag}) if dag else []
        shared = sorted(set(DETECTOR_MODEL_SHAS) & set(dag_shas))
        if not sp["dag_known"]:
            indep = None
            indep_note = "UNPROVEN: removal DAG unknown (lossy-only variant; generation code not located)"
        elif not dag:
            indep = True
            indep_note = "trivially independent: raw mixture, nothing removed"
        else:
            indep = (len(shared) == 0)
            indep_note = ("NON-INDEPENDENT: detector shares model(s) with removal DAG (circular)"
                          if shared else "independent: no shared model sha with removal DAG")
        removal_dag = dict(recipe=sp["recipe"], models=dag, dag_known=sp["dag_known"],
                           dag_model_sha256_set=dag_shas, parent=sp["parent"],
                           iteration=sp["iteration"], aliases=sp["aliases"],
                           acoustic_detector_shared_hashes=shared,
                           acoustic_detector_independent=indep,
                           independence_note=indep_note)

        # ---- ASR ----
        rawa = raw_asr(sp["asr_name"]) if sp["asr_name"] else None
        n_rec = rawa["n_recovered"] if rawa else None
        ov = rawa["raw_overlap_pct"] if rawa else None
        cu = control_unique_pct(name, n_rec if n_rec is not None else -1, ov)
        # canonical f32 candidates that were NOT ASR'd on the exact PCM: cu stays null unless it is
        # a trivially-derivable value (0% or 100%->32.4%). C_cascade/Z2 f32 were not ASR'd:
        cu_note = None
        if name in ("C_cascade", "Z2") and sp["f32_tag"] is not None:
            # raw ASR (if any) came from a lossy m4a SIBLING, not this exact f32 PCM
            if name == "C_cascade":
                cu = None
                cu_note = ("not measured on exact f32 PCM; lossy m4a sibling C_cascade measured "
                           "raw_overlap=0.0% -> control_unique 0% (diagnostic only)")
            else:
                cu = None
                cu_note = "not measured on exact f32 PCM (no matching ASR run for this signal)"
        asr_block = dict(
            raw=rawa, raw_asr_rule_status="SUPERSEDED (raw_overlap>=15%)",
            raw_asr_source=("m4a listening copy" if rawa else None),
            corrected_metric="control_unique_overlap",
            corrected_control_unique_pct=cu,
            corrected_available=asr_available(cu),
            corrected_note=cu_note,
            asr_floor_pct=ASR_FLOOR_PCT, asr_margin_pct=ASR_MARGIN_PCT,
            asr_hardfail_threshold_pct=ASR_HARDFAIL_THRESHOLD_PCT,
            corrected_ASR_hard_fail=corrected_asr_hard_fail(cu))

        # ---- provenance ----
        f32sha = f32_hash_for(sp)
        m4a = H.get(sp["m4a_tag"], {}) if sp.get("m4a_tag") else {}
        prov = dict(
            analysis_parent_f32_pcm_sha256=f32sha,
            analysis_parent_lossless=(not sp["lossy"] and bool(f32sha)),
            lossy_origin=bool(sp["lossy"]),
            f32_source=(S[sp["f32_from_stage"]]["f32_path"] if sp.get("f32_from_stage")
                        else (H.get(sp["f32_tag"], {}).get("path") if sp.get("f32_tag") else None)),
            lossy_listening_copy=dict(path=m4a.get("path"), pcm_sha256=m4a.get("pcm_sha256"),
                                      lossy_origin=True) if m4a else None,
            provenance_valid=provenance_valid(f32sha, sp["lossy"]),
            note=("lossy-origin listening copy — NOT a valid analysis parent" if sp["lossy"]
                  else "lossless f32 analysis parent"))

        # ---- release checks (section 7) ----
        release = dict(theft_db=evc.get("theft_db"), hall_db=evc.get("hall_db"),
                       stereo_db=evc.get("stereo_db"), terminal=sp["terminal"],
                       release_pass_rule="theft_db<-6 AND hall_db>-6 AND stereo_db>-6")

        # ---- BINDING GATE VERDICT (recomputed from stored scores) ----
        verdict = gate_evaluate(acoustic["worst_db"], cu, f32sha, sp["lossy"])

        cand = dict(
            identity=dict(role=name, iteration=sp["iteration"], parent=sp["parent"],
                          aliases=sp["aliases"], recipe=sp["recipe"]),
            provenance=prov,
            removal_dag=removal_dag,
            acoustic=acoustic,
            asr=asr_block,
            release_checks=release,
            gate_verdict=verdict,
        )
        if sp.get("human_anchor"):
            cand["human_anchor"] = dict(label=sp["human_anchor"], calibration_only=True)
        candidates[name] = cand

    # ---- per-iteration table (section 7, from f32 manifests) ----
    per_iter = []
    for n, mm in (("iter0", man["iter0"]), ("iter1", man["iter1"]),
                  ("iter2", man["iter2"]), ("iter2_Z2_comp", man["iter2_Z2_comp"])):
        per_iter.append(dict(iteration=mm["iteration"], node=n, recipe=mm["recipe"],
                             parent=mm["parent"], pcm_sha256=mm["pcm_sha256"],
                             acoustic_worst_db=mm["acoustic_worst_db"],
                             acoustic_pass=mm["acoustic_pass"], theft_db=mm["theft_db"],
                             hall_db=mm["hall_db"], stereo_db=mm["stereo_db"],
                             terminal=("needs_human_ab" if n == "iter2" else None)))

    # ---- ASR calibration section (5) ----
    controls = {}
    for k, v in R.items():
        if v.get("kind") in ("zero_voice_control", "vocalband_test") or k.endswith("_validation"):
            controls[k] = {kk: v.get(kk) for kk in
                           ("kind", "path", "overlap_pct", "n_recovered", "n_ref_scored",
                            "recovered_ref_idx", "no_speech_prob", "transcript") if kk in v}

    asr_calibration = dict(
        reference=dict(n_ref_scored=N_REF, mode=asr["_reference"]["verdi_slow_voices"]["reference_mode"],
                       lyric_words=asr["_reference"]["verdi_slow_voices"]["lyric_words"],
                       lyric_words_list=ref_words),
        asr_params={k: prim["_meta"][k] for k in
                    ("model", "device", "compute_type", "language_forced", "tol_s", "minlen",
                     "vad_filter", "condition_on_previous_text", "beam_size", "temperature")},
        corrected_metric=dict(
            name="control_unique_overlap",
            definition="fraction of reference lyric words the candidate recovered that NO matched-envelope zero-voice control recovered",
            floor_pct=ASR_FLOOR_PCT,
            floor_source="V1_inverted human-labeled VOICE-FREE anchor: control-unique 9/34 = 26.5% (highest control-unique attributable to pure envelope+timing priming)",
            margin_pct=ASR_MARGIN_PCT,
            hardfail_threshold_pct=ASR_HARDFAIL_THRESHOLD_PCT,
            hardfail_threshold_rationale="26.5 (voice-free floor) + 5.9 = 32.4% = theoretical control-unique ceiling (34 ref words - 23 recovered by control union); ADD-ONLY veto, intentionally conservative — acoustic remains the sole active veto on this material",
            control_union_size=CONTROL_UNION_SIZE,
            phase_scramble_seeds=[1, 2, 3, 4],
            phase_scramble_seed_overlaps_pct=ms["phase_scramble_seed_overlaps"],
            phase_scramble_floor_max_pct=ms["phase_scramble_floor_max_pct"],
            phase_scramble_floor_mean_pct=ms["phase_scramble_floor_mean_pct"],
            n_zero_voice_controls=ms["n_zero_voice_controls"],
            union_priming_recoverable_refwords=ms["union_priming_recoverable_refwords"],
            V1_overlap_pct=ms["V1_overlap_pct"],
            V1_unique_vs_union_floor_n=ms["V1_unique_vs_union_floor_n"],
            V1_unique_overlap_corrected_pct=ms["V1_unique_overlap_corrected_pct"],
            v1_unique_ref_idx=ms["v1_unique_ref_idx"],
            v1_unique_words=ms["v1_unique_words"],
            mechanism="Whisper hallucinates lyrics from the orchestral envelope+timing (priming); raw overlap cannot distinguish voice from voice-free."),
        transcripts_and_controls=controls,
        historical_diagnostics=dict(
            asr_report_json="/home/mickg/ab_pairs/asr_report.json (per-variant raw overlap; OLD 15% rule; SUPERSEDED but preserved)",
            asr_priming_controls_json="/home/mickg/ab_pairs/asr_priming_controls.json (priming controls; corrected metric; preserved)"),
        availability=True,
    )

    # ---- assemble payload (everything except canonical_sha256) ----
    payload = dict(
        schema="voice_gate_calibration",
        schema_version="1.0.0",
        artifact_version="v1",
        track="verdi_slow_voices",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        created_on="research6 (100.73.131.92)",
        purpose="Immutable, hash-verified voice-gate calibration record for verdi_slow_voices. "
                "Records provenance, detector/ASR calibration, per-candidate scores, and the "
                "binding-gate verdict recomputed from stored scores.",
        canonical_serialization_spec="canonical_sha256 = sha256(json.dumps(payload, sort_keys=True, "
                                     "separators=(',',':'), ensure_ascii=False).encode()) over ALL fields "
                                     "EXCEPT canonical_sha256 itself.",
        binding_gate_rule=dict(
            formula="pass <=> (acoustic_score_db <= -10.0) AND (corrected_ASR_hard_fail == false) "
                    "AND (acoustic_available AND asr_available AND provenance_valid)",
            acoustic_is_primary_hard_veto=True,
            corrected_asr_role="ADD-ONLY: can only add a fail; never overrides an acoustic fail; never grants a pass",
            fail_closed="any detector unavailable/invalid OR provenance invalid => fail",
            acoustic_threshold_db=ACOUSTIC_THRESHOLD_DB,
            definitions=dict(
                acoustic_score_db="worst-window rel_db from the FROZEN median-ensemble detector",
                acoustic_pass="acoustic_score_db <= -10.0",
                acoustic_available="acoustic_worst_db is not null",
                corrected_ASR_hard_fail=f"corrected_control_unique_pct is not null AND >= {ASR_HARDFAIL_THRESHOLD_PCT}",
                asr_available="corrected_control_unique_pct is not null (a corrected-ASR determination exists for this candidate's analyzed PCM)",
                provenance_valid="candidate has a lossless f32 pcm_sha256 AND lossy_origin == false"),
        ),
        superseded_rules=dict(
            asr_raw_overlap_15pct=dict(
                old_rule="asr_fail = raw reference-overlap% >= 15%",
                status="SUPERSEDED",
                reason="raw overlap is confounded by envelope+timing PRIMING: the human-labeled VOICE-FREE "
                       "anchor V1_inverted yields 94.1% raw overlap and phase-scramble voice-free controls "
                       "yield 26.5-55.9%. Replaced by the corrected control-unique overlap vs matched-envelope controls.")),
        detector_circularity_caveat=dict(
            concern="The FROZEN acoustic detector uses median(MDX23C, MelBand_base, BS) — the SAME model family "
                    "used to REMOVE vocals in the cascade candidates. Residual voice that survived removal lies "
                    "partly in this ensemble's null space by construction, so a cascade acoustic-PASS partly "
                    "reflects optimizing against this exact metric rather than proven absence of human-audible voice.",
            evidence="Per-candidate acoustic_detector_shared_hashes shows the detector shares ALL 3 model shas with "
                     "the removal DAG of iter1/iter2(Z)/Z2/M and 2 of 3 with B. A peer research thread reported "
                     "human-audible residual voice at verdi ~20-28s in cascade/inverted variants this detector rates as passing.",
            implication="acoustic-PASS from this detector is PROVISIONAL and MUST be confirmed by human A/B and/or an "
                        "INDEPENDENT-family detector (e.g. HTDemucs + pYIN + VAD). This is why the passing deliverable "
                        "carries terminal=needs_human_ab and why the binding rule fail-closes iter2/Z/Z2 here.",
            recommended_v2="seal a v2 with an independent-family detector as PRIMARY acoustic veto; demote these "
                           "median-ensemble scores to a diagnostic section.",
            disputed_human_label="a peer thread asserts V1_inverted has human-audible voice at ~20-28s, contradicting "
                                 "this artifact's task-given human label (V1_inverted = voice-free). UNRESOLVED — surfaced for user adjudication."),
        acoustic_detector=dict(
            name="median_ensemble_voice_gate2",
            family="median_ensemble_shared_with_remover",
            independent_of_removal_dag=False,
            independence_assertion_basis="EXPLICIT: the three detector model shas are compared against every "
                                         "candidate's removal-DAG model shas (see candidates[*].removal_dag).",
            algorithm="vmed = per-sample median of the 3 separator vocal estimates of the CANDIDATE; per 1.0s window "
                      "/ 0.5s hop compute program energy we=mean(ORIGINAL-mix window^2); silence-gate keep windows "
                      "we>max(we)*1e-3; rel_db=10log10(mean(vmed_win^2)/we); worst_db=max(rel_db over kept); PASS iff worst_db<=-10.0",
            sample_rate=44100, channels=2,
            window_s=1.0, hop_s=0.5, window_samples=44100, hop_samples=22050,
            silence_gate_rule="keep window iff program_energy > max(program_energy)*1e-3",
            normalization_rule="vocal energy normalized by the ORIGINAL PROGRAM MIX window energy (NOT the candidate)",
            aggregation="worst-window = max(rel_db) over kept windows",
            threshold_db=ACOUSTIC_THRESHOLD_DB, threshold_status="FROZEN/provisional",
            threshold_calibration=dict(
                anchors=dict(V1_inverted_voice_free_PASS=ev.get("V1_inverted", {}).get("acoustic_worst_db"),
                             A_champion_voice_FAIL=ev.get("A_champion", {}).get("acoustic_worst_db"),
                             B_melband_ft_voice_FAIL=ev.get("B_melband_ft", {}).get("acoustic_worst_db")),
                rationale="threshold -10 dB sits between the voice-free anchor (V1_inverted ~ -11.6) and the "
                          "voice anchors (A/B ~ -7.0), and above the corpus orchestra-only floor (-11..-19 dB).",
                note="anchor worst_db values are from lossy m4a decode; used as-frozen for calibration."),
            models=dict(
                MDX23C=dict(file="MDX23C-8KFFT-InstVoc_HQ.ckpt", sha256=SHA_MDX,
                            config="mdx_model_data.json", config_sha256=MODEL_HASHES["mdx_model_data.json"]),
                MelBand_base=dict(file="vocals_mel_band_roformer.ckpt", sha256=SHA_MEL,
                                  config="vocals_mel_band_roformer.yaml",
                                  config_sha256=MODEL_HASHES["vocals_mel_band_roformer.yaml"]),
                BS_roformer=dict(file="model_bs_roformer_ep_317_sdr_12.9755.ckpt", sha256=SHA_BS,
                                 config="model_bs_roformer_ep_317_sdr_12.9755.yaml",
                                 config_sha256=MODEL_HASHES["model_bs_roformer_ep_317_sdr_12.9755.yaml"])),
            model_dir="/home/mickg/models",
            software=SOFTWARE["detector_env"],
            code=dict(voice_gate2=code_hashes["voice_gate2.py"], voice_gate=code_hashes["voice_gate.py"]),
            availability=True,
        ),
        asr_detector=dict(
            engine="faster_whisper", family="whisper_ASR",
            independent_of_removal_dag=True,
            independence_assertion_basis="EXPLICIT: whisper large-v3 is a different architecture, never used in "
                                         "vocal removal; no shared weights/config/adapter with any removal model; "
                                         "no ASR output was used to construct or tune any candidate.",
            software=SOFTWARE["asr_env"],
            code={k: code_hashes[k] for k in
                  ("asr_gate.py", "asr_controls.py", "asr_seeds.py", "dsp_controls.py", "dsp_seeds.py")},
            calibration=asr_calibration,
        ),
        human_anchors=dict(
            V1_inverted=dict(label="voice-free", expected_acoustic="PASS", calibration_only=True,
                             acoustic_worst_db=ev.get("V1_inverted", {}).get("acoustic_worst_db"),
                             note="lossy-origin listening copy -> NOT a valid analysis parent; used ONLY to calibrate "
                                  "the acoustic threshold and the ASR floor. As a gate candidate it fail-closes on provenance."),
            A_champion=dict(label="voice", expected_acoustic="FAIL", calibration_only=True,
                            acoustic_worst_db=ev.get("A_champion", {}).get("acoustic_worst_db")),
            B_melband_ft=dict(label="voice", expected_acoustic="FAIL", calibration_only=True,
                              acoustic_worst_db=ev.get("B_melband_ft", {}).get("acoustic_worst_db")),
            calibration_result="FROZEN acoustic threshold -10 dB correctly classifies all three anchors "
                               "(V1_inverted PASS; A_champion & B_melband_ft FAIL).",
            disputed="peer thread asserts V1_inverted has audible voice ~20-28s (contradicts voice-free label); unresolved.",
        ),
        candidates=candidates,
        per_iteration_table=per_iter,
        cascade_worst_db_curve=[man["iter0"]["acoustic_worst_db"], man["iter1"]["acoustic_worst_db"],
                                man["iter2"]["acoustic_worst_db"]],
        provenance_regeneration=dict(
            regen_script="regen_bm.py", regen_script_sha256=code_hashes["regen_bm.py"],
            A_champion_reproduces_iter1=S["A_champion_repro"]["reproduces_iter1"],
            A_champion_repro_pcm_sha256=S["A_champion_repro"]["pcm_sha256"],
            B_melband_ft_pcm_sha256=S["B_melband_ft"]["pcm_sha256"],
            B_melband_ft_f32_path=S["B_melband_ft"]["f32_path"],
            M_maxagg_pcm_sha256=S["M_maxagg"]["pcm_sha256"],
            M_maxagg_f32_path=S["M_maxagg"]["f32_path"],
            ffmpeg_version=regen.get("ffmpeg_version"),
            determinism_note="separators reproduce iter1 (A_champion) byte-for-byte -> B/M f32 are faithful "
                             "lossless renders; fresh detector worst_db reproduces the frozen table for iter0/iter1.",
            all_pcm_hashes=H,
        ),
        model_hashes=MODEL_HASHES,
        code_hashes=code_hashes,
        data_separation=dict(
            calibration="verdi human anchors (V1_inverted/A_champion/B_melband_ft) + corpus orchestra-only controls",
            note="v1 does NOT establish disjoint candidate-selection / threshold-calibration / final-eval groups; "
                 "a final untouched holdout (e.g. Cantolopera) is recommended for v2.",
        ),
        notes=[
            "The old 15% raw-ASR-overlap rule is SUPERSEDED by the corrected control-unique metric.",
            "No candidate receives a certified PASS: acoustically-failing candidates fail on acoustic; lossy-origin "
            "variants (V1-V5, C_cascade m4a) fail-closed on provenance; the acoustically-passing lossless deliverable "
            "(C_cascade/Z, Z2) fail-closes on corrected-ASR-unmeasured, consistent with terminal=needs_human_ab.",
            "Lossy m4a listening copies give materially different acoustic worst_db (up to ~4 dB) vs their lossless f32 "
            "parents (e.g. O_original m4a -1.69 vs iter0 f32 -5.71) — only lossless f32 parents are valid for gating.",
            "The acoustic detector is NOT independent of the removal ensemble (see detector_circularity_caveat).",
        ],
    )

    # ---- canonical hash ----
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode()
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    out = dict(payload)
    out["canonical_sha256"] = canonical_sha256

    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("WROTE", OUT)
    print("canonical_sha256 =", canonical_sha256)
    print("n_candidates =", len(candidates))
    print("\nPER-CANDIDATE VERDICTS:")
    print("%-14s %9s %5s %8s %9s %6s  %s" % ("candidate", "worst_db", "acPS", "cu_pct", "asrHF", "prov", "verdict/reason"))
    for name, c in candidates.items():
        a = c["acoustic"]; s = c["asr"]; v = c["gate_verdict"]
        print("%-14s %9s %5s %8s %9s %6s  %s|%s" % (
            name, a["worst_db"], a["acoustic_pass"], s["corrected_control_unique_pct"],
            s["corrected_ASR_hard_fail"], c["provenance"]["provenance_valid"],
            ("PASS" if v["pass_"] else "FAIL"), v["reason"]))


if __name__ == "__main__":
    main()
