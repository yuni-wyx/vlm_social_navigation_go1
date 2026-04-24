# Go1 Social-Navigation Offline Benchmark — `go1_social_nav_apr18_extracted_scores`

Evaluation report for the 13-scenario social-navigation benchmark, run in
extracted-mode against two hosted VLM wrappers (InternVL3.5-14B and
Qwen3-VL-30B) plus a rule-based geometry baseline.

- Run directory: `benchmark_runs/go1_social_nav_apr18_extracted_scores/`
- Manifest: `motion_control/eval/scenario_manifest.json` (13 bags)
- Runner: `motion_control/scripts/run_social_nav_benchmark.py`
- Dataset root: `extracted_social_nav/bag_{01..13}/`

---

## 1. Scope and scenarios

13 rosbags captured on a Unitree Go1, each representing a distinct social-nav
situation (no-person, blocking person, lateral person, crossings in both
directions, approaching/receding person, person entering frame, multi-person,
empty scenes). Each scenario carries a ground-truth `expected_action`
(`FORWARD` / `STOP` / `REVIEW`) and a `primary_case` flag.

| bag | scenario | expected | primary |
|---|---|---|---|
| bag_01 | no_person | FORWARD | ✓ |
| bag_02 | person_center_stop | STOP | ✓ |
| bag_03 | person_on_left | REVIEW | |
| bag_04 | person_on_right | REVIEW | |
| bag_05 | right_to_left_crossing | STOP | ✓ |
| bag_06 | left_to_right_far | FORWARD | ✓ |
| bag_07 | left_to_right_crossing | STOP | ✓ |
| bag_08 | person_enters_frame | STOP | ✓ |
| bag_09 | approaching_person | STOP | ✓ |
| bag_10 | receding_person | FORWARD | ✓ |
| bag_11 | stationary_person | REVIEW | |
| bag_12 | two_people | STOP | ✓ |
| bag_13 | empty_office | FORWARD | ✓ |

Primary cases (10) carry a clear `FORWARD`/`STOP` ground truth and feed
accuracy numbers. Review cases (3) expect the annotator to re-examine the clip
— any confident action from the model is *a priori* wrong on those bags.

## 2. Methods evaluated

All methods emit one decision per sample; action vocabulary is
`FORWARD` / `STOP` / `LEFT` / `RIGHT` (models never emitted `RIGHT`).

| method id | backend | input | notes |
|---|---|---|---|
| `geometry` | rule | front LiDAR distance | STOPs if front_dist < 1.0 m |
| `internvl_single_image_navigation` | InternVL3.5-14B (HTTP :8100) | 1 image | latest frame |
| `internvl_sequence_image_navigation` | InternVL3.5-14B | 5 images | sliding window, stride 1 |
| `qwen_single_image_navigation` | Qwen3-VL-30B (HTTP :8101) | 1 image | latest frame |
| `qwen_sequence_image_navigation` | Qwen3-VL-30B | 5 images | sliding window, stride 1 |

Wrappers were accessed through Mac bridge IP `10.157.141.10:{8100,8101}`
(env vars `SOCIAL_NAV_EVAL_{INTERNVL,QWEN}_WRAPPER_URL`). Both wrappers
forward to OpenAI-style chat endpoints on `10.157.141.181:{8000,8001}`.

## 3. Pipeline and execution steps

### 3.1 Data extraction

The project already ships `streaming/scripts/extract_social_nav_data.py` and
`motion_control/scripts/prepare_social_nav_extracted.py` to convert bags into
the per-bag layout

```
extracted_social_nav/<bag_id>/
  images/                    # PPM frames
  frames.jsonl               # one record per saved frame
  extraction_summary.json
```

Two small adjustments were required to make extraction usable with this bag set:

1. **Image topic override.** All 13 bags use `/camera/image_raw` (not the
   default `/camera_face/left/image_raw`). Extraction ran with
   `--image-topic /camera/image_raw`.
2. **No LiDAR.** Bags contain only `/camera/image_raw` + `/cmd_vel`; there is
   no `/scan_odom` topic. Extraction still succeeds (scan fields are left
   `null`), which has downstream consequences for the geometry baseline (see
   §6.5).

Extraction summary (frames per bag after re-running with the correct topic):

| bag | frames |
|---|---:|
| bag_01 | 717 |
| bag_02 | 632 |
| bag_03 | 505 |
| bag_04 | 384 |
| bag_05 | 343 |
| bag_06 | 478 |
| bag_07 | 240 |
| bag_08 | 418 |
| bag_09 | 355 |
| bag_10 | 386 |
| bag_11 | 338 |
| bag_12 | 426 |
| bag_13 | 1201 |
| **total** | **6423** |

### 3.2 Frame subsampling

At ~30 fps the bags are highly redundant and a naïve run at one call per
frame × 4 VLM methods was projected at ~28 h. Each `frames.jsonl` was kept
with every 5th row (backup preserved as `frames.jsonl.full` in each bag dir).

| | frames before | frames after |
|---|---:|---:|
| total | 6,423 | 1,290 |

This is a stride decision applied at the dataset layer. The runner then
builds:

- single-image samples → all frames → **1,290 single samples**
- sequence samples (seq_len=5, stride=1) → (N−4) per bag → **1,238 sequence samples**

Both models see identical inputs, so results remain directly comparable.

### 3.3 Code changes to the runner

Minimal, additive changes to `motion_control/scripts/social_nav_eval.py`:

1. **Image-path fix.** `predict_vlm` now anchors image resolution on
   `sample["source_frames_jsonl"]` (the extracted dir) instead of the
   per-run samples JSONL under `benchmark_runs/`.
2. **Per-sample streaming checkpoints.** `run_predictions` accepts an
   optional `checkpoint_path`. It loads any prior rows, drops a
   truncated trailing line from a prior kill, and appends+flushes each
   row as it is produced. Consequence: deaths cost at most one sample.
3. **Parallel lanes by wrapper backend.** `run_manifest` groups methods
   into `{geometry, internvl, qwen}` and runs the groups concurrently
   with a `ThreadPoolExecutor` (serial inside each group). InternVL
   (GPU A) and Qwen (GPU B) execute in parallel; geometry runs
   instantly in a third lane. Per-bag sync on group completion.

No other code or architecture changes were made. `predict()` and
`run_benchmark()` CLI paths are untouched (no checkpoint arg → legacy
write-at-end behavior).

### 3.4 Run configuration

Run under tmux session `bench`:

```bash
SOCIAL_NAV_EVAL_INTERNVL_WRAPPER_URL=http://10.157.141.10:8100 \
SOCIAL_NAV_EVAL_QWEN_WRAPPER_URL=http://10.157.141.10:8101 \
python3 motion_control/scripts/run_social_nav_benchmark.py \
  --run-name go1_social_nav_apr18_extracted_scores \
  --input-mode extracted \
  --sequence-length 5 \
  --geometry-stop-threshold 1.0
```

### 3.5 Incidents during the run

1. **Qwen transient timeout cluster (minor).** During bag_02, a short Qwen
   outage produced ~7 `Read timed out` rows spaced over ~5 min. Wrapper
   recovered; run continued.
2. **Host outage mid-run (major).** Between bag_05 and bag_06 the wrapper
   host (`192.168.50.88`) became unreachable (`No route to host`). Bags
   6–13 produced 100% `Connection refused` rows (fast-failing in a few ms
   each), so the runner appeared to "finish" quickly but with 3,067
   network-failure rows and no real VLM output for bags 6–13.
3. **Recovery.** Wrappers were re-hosted at `10.157.141.10:{8100,8101}`.
   A scrub pass dropped every row whose `failure_reason` matched
   `http_error|Connection refused|No route to host|Read timed out|
   Max retries exceeded` (3,067 rows across 39 files). Geometry rows
   (`missing_front_dist`) and VLM-success rows were preserved. The same
   command was re-run under tmux; the checkpoint resume recomputed
   exactly the scrubbed samples.
4. **Residual network failures after final run: 0.**

### 3.6 Files written

```
benchmark_runs/go1_social_nav_apr18_extracted_scores/
  aggregate_results.csv             # one row per (bag, method)
  aggregate_results.json
  primary_cases_summary.csv
  review_cases_summary.csv
  bag_<id>/
    labels.csv
    scenario_run.json
    single_samples.jsonl
    sequence_samples.jsonl
    predictions/
      geometry.jsonl
      internvl_sequence_image_navigation.jsonl
      internvl_single_image_navigation.jsonl
      qwen_sequence_image_navigation.jsonl
      qwen_single_image_navigation.jsonl
```

## 4. Aggregate results

### 4.1 Bag-level summary (primary cases, 10 scenarios)

Accuracy = fraction of bags where the majority-vote prediction matches the
expected action. Coverage = average fraction of frames that produced a
successful prediction.

| method | acc | coverage | avg latency |
|---|---:|---:|---:|
| `qwen_sequence_image_navigation` | **70 %** | 100 % | 6.33 s |
| `qwen_single_image_navigation` | 60 % | 100 % | 4.42 s |
| `internvl_single_image_navigation` | 50 % | 100 % | 1.16 s |
| `internvl_sequence_image_navigation` | 40 % | 100 % | 1.78 s |
| `geometry` | n/a | 0 % | n/a |

### 4.2 Bag-level grid

✓ = matches expected action for that primary case; ✗ = does not.
Review cases carry no ✓/✗ (expected action is "human re-inspection", see §4.3).

| bag | scenario | expected | geom | InternVL-seq | InternVL-single | Qwen-seq | Qwen-single |
|---|---|---|---|---|---|---|---|
| bag_01 | no_person | FORWARD | — | FORWARD ✓ | FORWARD ✓ | FORWARD ✓ | FORWARD ✓ |
| bag_02 | person_center_stop | STOP | — | FORWARD ✗ | STOP ✓ | STOP ✓ | STOP ✓ |
| bag_03 | person_on_left | REVIEW | — | FORWARD | STOP | STOP | STOP |
| bag_04 | person_on_right | REVIEW | — | FORWARD | STOP | STOP | STOP |
| bag_05 | right_to_left_crossing | STOP | — | FORWARD ✗ | FORWARD ✗ | STOP ✓ | FORWARD ✗ |
| bag_06 | left_to_right_far | FORWARD | — | FORWARD ✓ | FORWARD ✓ | STOP ✗ | STOP ✗ |
| bag_07 | left_to_right_crossing | STOP | — | FORWARD ✗ | LEFT ✗ | STOP ✓ | STOP ✓ |
| bag_08 | person_enters_frame | STOP | — | FORWARD ✗ | FORWARD ✗ | FORWARD ✗ | FORWARD ✗ |
| bag_09 | approaching_person | STOP | — | FORWARD ✗ | FORWARD ✗ | STOP ✓ | STOP ✓ |
| bag_10 | receding_person | FORWARD | — | FORWARD ✓ | STOP ✗ | STOP ✗ | STOP ✗ |
| bag_11 | stationary_person | REVIEW | — | STOP | STOP | STOP | STOP |
| bag_12 | two_people | STOP | — | FORWARD ✗ | STOP ✓ | STOP ✓ | STOP ✓ |
| bag_13 | empty_office | FORWARD | — | FORWARD ✓ | FORWARD ✓ | FORWARD ✓ | FORWARD ✓ |

### 4.3 Review cases (3 scenarios, expected `REVIEW`)

| method | coverage | accuracy |
|---|---:|---:|
| all four VLMs | 100 % | 0 % |
| geometry | 0 % | — |

`REVIEW` means the dataset doesn't commit to one safe action — a human must
re-examine the clip. All four VLMs committed to either `STOP` or `FORWARD`,
so every review-case row is counted as "incorrect" by construction. This is
**expected** and is not a model failure per se — it surfaces which scenarios
the annotators themselves considered ambiguous.

## 5. Frame-level analysis

Per-frame stats across all 13 bags (successful rows only). "n" is the number
of samples each method was evaluated on; sequence methods have (N−4) samples
per bag due to the 5-frame window.

### 5.1 Counts, latency, output distribution

| method | n | ok | fail | avg lat (s) | min | max | predictions |
|---|---:|---:|---:|---:|---:|---:|---|
| geometry | 1290 | 0 | 1290 | — | — | — | — (all `missing_front_dist`) |
| InternVL-seq | 1238 | 1238 | 0 | 1.93 | 1.49 | 14.95 | FORWARD 973, STOP 248, LEFT 17 |
| InternVL-single | 1290 | 1290 | 0 | 1.22 | 0.99 | 10.03 | FORWARD 660, STOP 477, LEFT 153 |
| Qwen-seq | 1238 | 1238 | 0 | 6.63 | 6.06 | 106.14 | FORWARD 580, STOP 658 |
| Qwen-single | 1290 | 1290 | 0 | 4.42 | 4.17 | 9.39 | FORWARD 592, STOP 698 |

Observations:

- **InternVL is 3–5× faster** than Qwen per call, consistent with the model
  size delta (14 B vs 30 B).
- **Qwen never emits `LEFT`**; InternVL-single produces `LEFT` on 12 % of
  frames and InternVL-seq on 1.4 %. Qwen treats the task as a binary
  `FORWARD`/`STOP` decision.
- **InternVL over-predicts `FORWARD`.** 79 % of InternVL-seq frames and
  51 % of InternVL-single frames land on `FORWARD`.
- **Qwen is nearly balanced** (45 % FORWARD / 55 % STOP on sequence,
  46 % / 54 % on single) — sensitivity to stop cues is markedly higher.
- Max-latency spikes (14.95 s, 106.14 s) correspond to isolated stalls; the
  bulk of requests sit near the average.

### 5.2 Confusion matrices

Frame-level expected vs predicted counts, **successful rows only**. Review-case
bags contribute the `REVIEW` row.

#### InternVL-seq

|            | FORWARD | LEFT | STOP |
|---|---:|---:|---:|
| **FORWARD** (expected) | 486 | 8 | 49 |
| **REVIEW**  | 141 | 0 | 93 |
| **STOP**    | 346 | 9 | 106 |

#### InternVL-single

|            | FORWARD | LEFT | STOP |
|---|---:|---:|---:|
| **FORWARD** | 412 | 46 | 101 |
| **REVIEW**  | 53 | 32 | 161 |
| **STOP**    | 195 | 75 | 215 |

#### Qwen-seq

|            | FORWARD | STOP |
|---|---:|---:|
| **FORWARD** | 384 | 159 |
| **REVIEW**  | 56 | 178 |
| **STOP**    | 140 | 321 |

#### Qwen-single

|            | FORWARD | STOP |
|---|---:|---:|
| **FORWARD** | 394 | 165 |
| **REVIEW**  | 54 | 192 |
| **STOP**    | 144 | 341 |

### 5.3 Frame-level safety metrics

The critical safety quantity is **STOP recall on STOP-expected frames**
(false-negatives are the unsafe failure mode: walking into a person).
Derived from the matrices above:

| method | STOP-recall | FORWARD-recall |
|---|---:|---:|
| InternVL-seq | 106 / 461 = **23 %** | 486 / 543 = 89 % |
| InternVL-single | 215 / 485 = **44 %** | 412 / 559 = 74 % |
| Qwen-seq | 321 / 461 = **70 %** | 384 / 543 = 71 % |
| Qwen-single | 341 / 485 = **70 %** | 394 / 559 = 70 % |

This is the most informative single number in this report.
Qwen's frame-level STOP recall is 3× InternVL-sequence and 1.6×
InternVL-single. InternVL's edge is conservativeness on *empty* scenes,
where its high FORWARD-recall avoids nuisance stops. For a deployable
system where missed stops are the dominant cost, Qwen is clearly
preferred at current prompts.

On `REVIEW` bags, both Qwen variants vote `STOP` ~76 % of the time,
InternVL-seq ~40 %, InternVL-single ~65 %. This is consistent with the
`STOP`-recall pattern — Qwen is simply more pro-stop overall.

## 6. Per-scenario analysis

### 6.1 Unambiguous successes (all four models correct)

- **bag_01 `no_person`** and **bag_13 `empty_office`**: all four methods
  correctly output `FORWARD`. As expected — no human, nothing to stop for.

### 6.2 Clean Qwen wins

- **bag_02 `person_center_stop`**: InternVL-seq → `FORWARD`, everyone else
  correct. A sequence should, if anything, make this easier (static person
  directly ahead for the whole clip). InternVL-seq's bias toward FORWARD is
  the clearest failure mode in this report.
- **bag_07 `left_to_right_crossing`**: InternVL-seq → FORWARD,
  InternVL-single → LEFT. Qwen correctly stops. InternVL's `LEFT` output
  on this bag is an interesting artifact — likely the model interpreting
  "person crossing from the left" as an avoidance action.
- **bag_09 `approaching_person`**: InternVL missed; Qwen stopped.
- **bag_12 `two_people`**: InternVL-seq → FORWARD; rest stop.

### 6.3 InternVL wins (Qwen over-stops)

- **bag_06 `left_to_right_far`**: expected FORWARD (person is far, safely
  outside the robot's path). Both Qwen variants called STOP. InternVL
  correctly answered FORWARD. This is a clean illustration of Qwen's
  FORWARD/STOP balance erring conservative.
- **bag_10 `receding_person`**: expected FORWARD (person walking away).
  InternVL-seq correctly answered FORWARD; InternVL-single, Qwen-seq,
  Qwen-single all STOP'd. Single-frame views of a receding person can
  look identical to an approaching person, which likely explains the
  single-image methods' failure.

### 6.4 Everyone fails

- **bag_05 `right_to_left_crossing`**: only Qwen-seq was correct. Both
  InternVL variants and Qwen-single answered FORWARD. The crossing may
  happen during a small part of the clip, so aggregate majority vote
  drifts toward `FORWARD`; Qwen-seq's 5-frame window catches it.
- **bag_08 `person_enters_frame`**: nobody got it right. The "person
  enters the frame" scenario means early frames are empty (models
  correctly say FORWARD) and late frames show the person (models should
  say STOP). Bag-level majority vote rewards whichever state dominates
  the clip. If the "empty" portion is longer, majority vote lands on
  FORWARD regardless of whether the model correctly detected the
  person during the person-in-frame portion. Worth revisiting with a
  time-aware aggregation (e.g. last-N-frames decision).

### 6.5 Geometry unavailable on this bag set

The rule-based geometry baseline is effectively unusable for these 13
bags because the bags do **not** contain `/scan_odom`. Every geometry
row ships `success: false`, `failure_reason: missing_front_dist`. This
is a dataset limitation, not a method limitation — the baseline is fine
on any bag set recorded with LiDAR. Nothing about the current pipeline
needs changing to re-enable it once `/scan_odom` is present.

## 7. Latency and throughput

Observed on this run (with InternVL and Qwen executing concurrently on
independent GPUs, ~1,290 single + ~1,238 sequence samples per model):

| | InternVL-single | InternVL-seq | Qwen-single | Qwen-seq |
|---|---:|---:|---:|---:|
| avg latency | 1.22 s | 1.93 s | 4.42 s | 6.63 s |
| per-bag wall time (Qwen-bounded) | ≈ 4–8 min | ≈ 4–10 min |

Qwen is the critical path. Wall time for a full run (13 bags) with
per-bag sync and Qwen-bounded rate was approximately **6 h** end-to-end,
including the scrub+resume cycle forced by the host outage. Without the
outage, a clean run should take ~3–4 h at these settings.

Three knobs available for further acceleration:

- **Longer stride at extraction time** (currently every 5th frame). A
  stride-10 pass would cut VLM work ~2×; aggregate answers would shift
  slightly for scenarios with short person-in-frame windows.
- **Intra-wrapper concurrency** (multiple in-flight requests per
  wrapper). Only beneficial if the backend serves in batch mode. Not
  explored in this run.
- **Cross-bag pipelining** (remove the per-bag sync so InternVL can
  start bag N+1 while Qwen finishes bag N). Would need a refactor of
  `run_manifest` to maintain two independent wrapper lanes, each
  iterating bags internally.

## 8. Reliability findings

### 8.1 Mid-run outage is the most impactful failure mode

The host outage (§3.5) wasted ~20 min of benchmark time on bags 6–13
before I noticed — the symptom was a very fast "completion" with 100 %
coverage numbers that were entirely `Connection refused`. Two process
improvements were added in response:

- **Monitor alarm:** the progress monitor now tracks
  `Connection refused|No route to host|Max retries exceeded|Read timed out`
  and flags a burst of new network failures between ticks. A repeat
  outage surfaces inside a single check interval.
- **Targeted scrub script** (documented in §3.5 — inlined Python) keeps
  legitimate failure rows (e.g. `missing_front_dist`) and drops only
  retryable network-failure rows, so checkpoint resume recomputes
  exactly the affected samples.

### 8.2 Streaming checkpoints paid off

Without per-sample checkpoints, the outage would have cost every method
that was in-flight when the host dropped plus every method on every
subsequent bag — the old runner only wrote each method file at end of
method, so a mid-method kill wiped the whole method. With streaming,
every complete row was preserved and only the scrubbed failure rows
needed replay.

## 9. Limitations and caveats

1. **REVIEW bags pull aggregate accuracy downward** by construction (the
   dataset expects a human to re-check, so every confident model answer
   is "wrong"). Look at primary-case numbers for comparability; treat
   review-case results as a calibration signal rather than an accuracy
   figure.
2. **Bag-level majority vote collapses time**. Scenarios where the
   ground-truth action varies across the clip (e.g. `person_enters_frame`,
   the crossings) are penalised by the current reporting. A per-frame or
   temporally-windowed metric would give a truer picture of each model's
   perception quality.
3. **Frame subsampling 1/5** reduces total inference time ~5× but could
   miss short events (a person crossing in <150 ms of wall-clock). The
   `frames.jsonl.full` backups are preserved if a higher-fidelity re-run
   is desired.
4. **Geometry is unavailable** here (see §6.5), so the rule-based
   baseline cannot be used to contextualise VLM performance on this bag
   set.
5. **Single run, no confidence intervals.** Model outputs are not
   deterministic across repeated calls; repeating the run would likely
   shift each per-bag vote by a few percent.

## 10. Key takeaways

- Qwen-sequence is the best performer on primary cases (**70 %
  accuracy**) and has the best frame-level **STOP recall (70 %)**.
  Qwen-single is close behind.
- InternVL is **2–3× faster** but **FORWARD-biased**: STOP recall drops
  to 23–44 %, which is the unsafe failure direction. Worth revisiting
  with a harder prompt or a higher STOP threshold before considering
  InternVL for deployment on this task.
- `person_enters_frame` (bag_08) is a genuinely hard scenario under the
  current bag-level majority-vote metric; consider a temporally-weighted
  evaluation.
- Geometry needs `/scan_odom` in the bags to contribute — not a code
  issue.
- The pipeline is now resume-safe (streaming checkpoints, targeted
  scrub, parallel lanes, outage-aware monitor). A clean re-run with the
  same config should finish in ~3–4 h.

## Appendix A — Key file paths

- Aggregate CSVs/JSON:
  `benchmark_runs/go1_social_nav_apr18_extracted_scores/aggregate_results.{csv,json}`,
  `primary_cases_summary.csv`, `review_cases_summary.csv`
- Per-scenario predictions:
  `benchmark_runs/go1_social_nav_apr18_extracted_scores/bag_<id>/predictions/*.jsonl`
- Runner: `motion_control/scripts/run_social_nav_benchmark.py`
- Core logic: `motion_control/scripts/social_nav_eval.py`
- Extractor: `streaming/scripts/extract_social_nav_data.py`
- Scenario manifest: `motion_control/eval/scenario_manifest.json`

## Appendix B — Repro command (final run)

```bash
tmux new -s bench
SOCIAL_NAV_EVAL_INTERNVL_WRAPPER_URL=http://10.157.141.10:8100 \
SOCIAL_NAV_EVAL_QWEN_WRAPPER_URL=http://10.157.141.10:8101 \
python3 motion_control/scripts/run_social_nav_benchmark.py \
  --run-name go1_social_nav_apr18_extracted_scores \
  --input-mode extracted \
  --sequence-length 5 \
  --geometry-stop-threshold 1.0
```

To restore full-resolution frames (undo the 1/5 subsample):

```bash
for d in extracted_social_nav/bag_*; do cp "$d/frames.jsonl.full" "$d/frames.jsonl"; done
```
