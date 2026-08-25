#!/usr/bin/env python3
"""Upload pipeline orchestrator (Mac side, stdlib only).

Usage: upload_pipeline.py <local_src_path> <track_base>
Relays step-by-step progress into the track's workspace thread by POSTing
action_reply records to the local ab.py server, then runs the research6
auto_track driver, pulls the variant m4as + detector timelines back, and
posts the final recommendation. Runs detached; log at /tmp/pipeline_<track>.log.
"""
import json
import os
import subprocess
import sys
import urllib.request

R6 = "mickg@100.73.131.92"
R6PY = "/home/mickg/audio-extract/.venv/bin/python"
API = "http://localhost:8766/api/vote"
ROOT = os.path.expanduser("~/audio-extract")


def post(track, text, user="pipeline"):
    rec = {"track": "var:" + track, "vote": "action_reply", "user": user, "text": text}
    try:
        req = urllib.request.Request(API, data=json.dumps(rec).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        print("post failed:", e, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    src, track = sys.argv[1], sys.argv[2]
    fname = os.path.basename(src)
    try:
        # 1. probe
        pr = sh(["ffprobe", "-v", "quiet", "-show_entries",
                 "format=duration,bit_rate:stream=channels,sample_rate",
                 "-of", "json", src])
        info = json.loads(pr.stdout or "{}")
        dur = float(info.get("format", {}).get("duration", 0))
        st = (info.get("streams") or [{}])[0]
        post(track, f"📥 **Upload received: `{fname}`** — {dur:.1f}s, "
                    f"{st.get('channels','?')}ch @ {st.get('sample_rate','?')} Hz. "
                    f"Saved to the masters directory. Starting the pipeline...")
        if dur < 3:
            post(track, "❌ file too short / not decodable — pipeline stopped.")
            return
        # 2. ship to GPU box
        r = sh(["scp", "-q", src, f"{R6}:/home/mickg/lib_in/{fname}"])
        if r.returncode != 0:
            post(track, "❌ could not copy to the GPU box: " + r.stderr[:200]); return
        post(track, "🚚 copied to the render box; analyzing + rendering (a few minutes)...")
        # 3. stream the driver
        proc = subprocess.Popen(
            ["ssh", R6, R6PY, "-u", "/home/mickg/tobacco_variants/auto_track.py",
             f"/home/mickg/lib_in/{fname}", track],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        done_meta = None
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            if line.startswith("STEP: DONE "):
                try:
                    done_meta = json.loads(line[len("STEP: DONE "):])
                except Exception:
                    done_meta = {}
                continue
            if line.startswith("STEP: "):
                post(track, "⚙️ " + line[6:])
        proc.wait()
        if proc.returncode != 0 or done_meta is None:
            post(track, f"❌ render driver exited with code {proc.returncode} — see /tmp/pipeline_{track}.log on the Mac.")
            return
        # 4. pull m4as
        r = sh(["bash", "-c",
                f"scp -q '{R6}:/home/mickg/tobacco_variants/{track}/{track}__*.m4a' "
                f"{ROOT}/variants/"])
        if r.returncode != 0:
            post(track, "❌ could not pull rendered variants: " + r.stderr[:200]); return
        letters = sorted(x.split("__")[1][0] for x in os.listdir(ROOT + "/variants")
                         if x.startswith(track + "__"))
        # 5. pull timelines
        sh(["scp", "-q", f"{R6}:/mnt/bigdisk/mickg/vad_scan/vad_timelines.json",
            ROOT + "/variants/.vad_timelines.json"])
        post(track, f"🃏 **Card is live** with variants {'/'.join(letters)} and detector bars — "
                    f"seamless switching, downloads, the works.")
        # 6. recommendation
        cleanest = (done_meta or {}).get("cleanest", "?")
        post(track, f"✅ **Pipeline done.** Detector's cleanest variant: **{cleanest.split('_')[0]}** "
                    f"({cleanest}). That is the machine's opinion only — ears decide. "
                    f"Ask here (🔧) for deeper analysis, spectrograms, or a re-render with different settings.")
        # 7. mark upload done
        up = ROOT + "/variants/.uploads.json"
        try:
            d = json.load(open(up))
            if track in d:
                d[track]["status"] = "done"
                json.dump(d, open(up, "w"), indent=1)
        except Exception:
            pass
    except Exception as e:
        post(track, "❌ pipeline crashed: " + repr(e)[:300])
        raise


if __name__ == "__main__":
    main()
