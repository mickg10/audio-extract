#!/usr/bin/env python3
"""Step 5 WORK-LEVEL scorer (research6). Scores 4 recipes over every (work, condition)
case, building ONE TrackLayout per WORK (its conditions pooled as challenge cases +
one concatenated multi-passage no-vocal control), so a later frozen-selector run can
consume the whole work at once. Persists step5_results.json AND keeps each work's
manifest.sqlite for the LOWO select phase. Disk-careful: challenge audio is symlinked,
the concatenated control is deleted after scoring, temp stems cleaned per case."""
import os, sys, json, time, glob, hashlib, traceback, shutil, gc, sqlite3
import numpy as np
import soundfile as sf

REPO = os.path.expanduser("~/audio-extract")
sys.path.insert(0, REPO)
from audio_extract import dsp
from audio_extract.separate import Separator
from audio_extract.storage import TrackLayout
from audio_extract import cli_autonomous as cla

SR = 44100
MODEL_DIR = "/home/mickg/models"
TP = os.path.expanduser("~/truth_pairs")
SC = os.path.expanduser("~/step5_cases")
WORK = os.path.expanduser("~/step5_work")
LIB = os.path.join(WORK, "lib")
STEM_TMP = os.path.join(WORK, "_stem_tmp")

MODEL_FILES = {"MDX23C": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
               "MelBand": "vocals_mel_band_roformer.ckpt",
               "BS": "model_bs_roformer_ep_317_sdr_12.9755.ckpt"}

# WORK -> [(condition, case_dir)] ; grouping unit is the WORK (conditions never split)
WORKS = {
    "donizetti":          [("dry",    os.path.join(TP, "bologna_donizetti"))],
    "puccini":            [("dry",    os.path.join(TP, "bologna_puccini"))],
    "aalto_mozart":       [("dry",    os.path.join(TP, "aalto_mozart_dry")),
                           ("hall",   os.path.join(TP, "aalto_mozart_hall"))],
    "spheres_mozart":     [("dry",    os.path.join(SC, "spheres_mozart_dry")),
                           ("hall",   os.path.join(SC, "spheres_mozart_hall")),
                           ("stereo", os.path.join(SC, "spheres_mozart_stereo"))],
    "spheres_tchaikovsky":[("dry",    os.path.join(SC, "spheres_tchaikovsky_dry")),
                           ("hall",   os.path.join(SC, "spheres_tchaikovsky_hall")),
                           ("stereo", os.path.join(SC, "spheres_tchaikovsky_stereo"))],
    "verdi":              [("dry",    os.path.join(TP, "bologna_verdi"))],   # FROZEN test-v1
}

_sep_cache, _stem_cache, _timings = {}, {}, []
_cur = {"work": None, "cond": None}


def _mixhash(mix):
    a = np.ascontiguousarray(dsp.as2d(mix), dtype=np.float64)
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


def estimator_factory(model_name):
    fname = MODEL_FILES[model_name]

    def _est(mix):
        h = _mixhash(mix); ck = (model_name, h)
        if ck in _stem_cache:
            return _stem_cache[ck]
        sep = _sep_cache.get(model_name)
        if sep is None:
            sep = Separator(fname, overlap=8, model_dir=MODEL_DIR, output_dir=STEM_TMP)
            _sep_cache[model_name] = sep
            print(f"      [load] {model_name}", flush=True)
        tmpwav = os.path.join(STEM_TMP, f"in_{model_name}_{h}.wav")
        sf.write(tmpwav, dsp.as2d(mix).astype("float32"), SR, subtype="FLOAT")
        t0 = time.time()
        out = sep.separate_file(tmpwav)
        _timings.append({"model": model_name, "work": _cur["work"], "cond": _cur["cond"],
                         "sep_time_s": round(time.time() - t0, 2),
                         "n": int(dsp.as2d(mix).shape[0]),
                         "ch": int(dsp.as2d(mix).shape[1]),
                         "stem_ch": {k: list(np.asarray(v).shape) for k, v in out.stems.items()}})
        _stem_cache[ck] = out.stems
        for p in glob.glob(os.path.join(STEM_TMP, "*.wav")):
            try: os.remove(p)
            except OSError: pass
        return out.stems
    return _est


def _hex(work, cond):
    return hashlib.sha256(f"{work}::{cond}".encode()).hexdigest()[:32]


def build_work_layout(work, conditions):
    """One layout: concat orchestras -> canonical + one no_vocal_control passage per
    condition; one challenge dir + challenge_case row per condition."""
    from audio_extract.manifest_v2 import ManifestV2
    lay = TrackLayout(LIB, work); lay.ensure()
    orch_parts, passages, cond_meta = [], [], {}
    off = 0
    for cond, cdir in conditions:
        orch, _ = sf.read(os.path.join(cdir, "orchestra_only.wav"), dtype="float64", always_2d=True)
        n = orch.shape[0]
        orch_parts.append(orch)
        passages.append({"passage_id": f"ctl_{work}_{cond}", "start_sample": off,
                         "end_sample": off + n, "tags": ["no_vocal_control"],
                         "features": {"vocal_energy_ratio": 0.0}})
        cond_meta[cond] = {"cid": f"sha256:{_hex(work, cond)}",
                           "meta": json.load(open(os.path.join(cdir, "meta.json"))),
                           "case_dir": cdir, "frames": int(n)}
        off += n
    canon_arr = np.concatenate(orch_parts, axis=0)
    canon = lay.source_dir / "canonical.f32.wav"
    sf.write(str(canon), canon_arr.astype("float32"), SR, subtype="FLOAT")
    (lay.passages_dir / "passages.v1.json").write_text(json.dumps(
        {"schema": "passages/v1", "passages": passages}, indent=2))
    with ManifestV2(lay.manifest_sqlite) as m2:
        for cond, cdir in conditions:
            hx = _hex(work, cond); cid = f"sha256:{hx}"
            ch = lay.root / "challenges" / f"sha256_{hx}"; ch.mkdir(parents=True, exist_ok=True)
            for link, src in [("mixture.f32.wav", os.path.join(cdir, "mix_with_voice.wav")),
                              ("target.f32.wav", os.path.join(cdir, "orchestra_only.wav"))]:
                p = ch / link
                if p.exists() or p.is_symlink(): p.unlink()
                os.symlink(src, p)
            m2.add_challenge_case(challenge_id=cid, challenge_type="track_remix",
                                  mixture_node_id=cid, target_node_id=cid + "#target",
                                  recipe={"schema": "audio-extract/step5/work-condition",
                                          "work": work, "condition": cond},
                                  klass={"work": work, "condition": cond})
    return lay, cond_meta


def build_recipe_grid():
    specs = []
    def add(name, spec):
        spec = dict(spec); spec["overlap"] = 8
        specs.append({"name": name, "spec": spec, "recipe_id": cla.recipe_spec_id(spec)})
    add("residual:MDX23C", {"kind": "residual", "models": ["MDX23C"]})
    add("residual:MelBand", {"kind": "residual", "models": ["MelBand"]})
    add("residual:BS", {"kind": "residual", "models": ["BS"]})
    add("ens_median:MDX23C/MelBand/BS",
        {"kind": "ensemble_residual", "models": ["MDX23C", "MelBand", "BS"], "algo": "median"})
    return specs


def main():
    if os.path.exists(LIB): shutil.rmtree(LIB)
    os.makedirs(LIB, exist_ok=True); os.makedirs(STEM_TMP, exist_ok=True)
    t0 = time.time()
    grid = build_recipe_grid()
    specs = [dict(g["spec"], recipe_id=g["recipe_id"]) for g in grid]
    print("=== RECIPES ===", flush=True)
    for g in grid: print(f"  {g['name']:32s} {g['recipe_id']}", flush=True)
    results = {"sr": SR, "grid": grid, "works": {}}
    for work, conditions in WORKS.items():
        print(f"\n=== WORK {work}  conditions={[c for c,_ in conditions]} ===", flush=True)
        _cur["work"] = work
        lay, cond_meta = build_work_layout(work, conditions)
        for cond, cdir in conditions:
            _cur["cond"] = cond
        tw = time.time()
        try:
            out = cla.run_challenges(lay, SR, recipe_specs=specs,
                                     estimator_factory=estimator_factory, persist_outputs=False)
        except Exception:
            print("  RUN FAILED\n", traceback.format_exc(), flush=True)
            results["works"][work] = {"error": traceback.format_exc()}
            _stem_cache.clear(); gc.collect(); continue
        # pull residual_construction + theft_rows from the sqlite theft_assay rows
        con = sqlite3.connect(str(lay.manifest_sqlite)); con.row_factory = sqlite3.Row
        theft_extra = {}
        for r in con.execute("SELECT * FROM challenge_result WHERE challenge_id='theft_assay'"):
            res = json.loads(r["result_json"]); rid = r["candidate_recipe_id"]
            rc = res.get("residual_construction", [])
            theft_extra[rid] = {
                "residual_construction": rc,
                "min_rc_si": min([c["si_sdr_db"] for c in rc], default=None),
                "mean_rc_band": float(np.mean([c["band_envelope_err_db"] for c in rc])) if rc else None,
                "theft_rows": res.get("theft_rows", []),
                "max_theft_broadband": max([t["theft_broadband"] for t in res.get("theft_rows", [])], default=None),
            }
        con.close()
        matrix = {}
        for g in grid:
            rid = g["recipe_id"]; cell = out["matrix"].get(rid, {})
            matrix[rid] = {"name": g["name"], "cases": cell.get("cases", {}),
                           "theft_mean": cell.get("theft_mean"), **theft_extra.get(rid, {})}
        results["works"][work] = {
            "conditions": {c: {"cid": cond_meta[c]["cid"], "frames": cond_meta[c]["frames"],
                               "meta": cond_meta[c]["meta"]} for c, _ in conditions},
            "wall_s": round(time.time() - tw, 1), "matrix": matrix}
        # console summary
        for g in grid:
            rid = g["recipe_id"]; mm = matrix[rid]
            for cond, _ in conditions:
                cid = cond_meta[cond]["cid"]; pc = mm["cases"].get(cid, {})
                print(f"  {g['name']:28s} [{cond:6s}] SISDR={pc.get('si_sdr_db')} "
                      f"hole={round(pc.get('event_hole_depth_db') or 0,1)} "
                      f"vintf={pc.get('vocal_interference_ratio')} band={pc.get('band_envelope_err_db')} "
                      f"stW={pc.get('stereo_width_err')} theft={mm.get('theft_mean')} "
                      f"min_rc_si={round(mm.get('min_rc_si') or 0,1)}", flush=True)
        # free disk: drop the big concatenated control, keep manifest.sqlite for select phase
        try: (lay.source_dir / "canonical.f32.wav").unlink()
        except OSError: pass
        _stem_cache.clear(); gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    results["timings"] = _timings
    results["total_wall_s"] = round(time.time() - t0, 1)
    outp = os.path.join(WORK, "step5_results.json")
    json.dump(results, open(outp, "w"), indent=1, default=str)
    print(f"\nWROTE {outp} (total {results['total_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
