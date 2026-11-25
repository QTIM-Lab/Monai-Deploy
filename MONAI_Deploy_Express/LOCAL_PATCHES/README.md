# Fixed MONAI Deploy services (until upstream releases them)

Express's stock gateway and workflow manager lose results when runs overlap, which forced
one-series-at-a-time sending (~20 s per series: days per model on the dashboard's Orthanc). Two fixes, each on
its own branch in [`../../upstream/`](../../upstream/README.md): `build.sh` builds both images from those branches at
pinned commits; `switch.sh stock|pr|fixed` picks what Express runs (edits `.env`). Hands-on comparison: [TESTING.md](TESTING.md).

| Service | Bug (stock image) | Fix (branch → image) | Upstream |
|---|---|---|---|
| Informatics Gateway 0.5.1-3 | Exports are de-duplicated by task name only; a second run's `export-*` is refused ("already queued"), never acked, and redelivered ~30 min later. | PR #544 (bluna301) as is: `pull/544/head` = upstream `main` + the fix (key = workflow instance + task) → `monai-deploy-informatics-gateway:pr544-af5e15c` | [IG PR #544](https://github.com/Project-MONAI/monai-deploy-informatics-gateway/pull/544), open since 2026-03 |
| Workflow Manager 0.2.1-2 | `ConditionalParameterParser` is a singleton holding the current workflow instance in fields; concurrent router evaluations clobber it (`ArgumentNullException: WorkflowInstance`), the condition counts as false, and the run ends "Succeeded" with no MAP run. | ours: `fix-parser-race` = upstream `develop` + one commit (lock around its public entry points + a unit test) → `monai-deploy-workflow-manager:fix-parser-race-4e76602` | not reported yet (2026-11-20); PR from the QTIM-Lab fork |

Both fix branches sit on upstream's current dev line, not on the exact release revisions: WM `develop` equals the
0.2.1-2 source except a CI file; IG `main` adds a database-config refactor (#527) and dependency updates, which run
fine with our `configs/informatics-gateway.json` (tested 2026-11-20: healthy, AEs loaded, 10/10 concurrent runs).

## Evidence (this host, disc_detection MAP, practice Orthanc)

| Images | Concurrent runs | Lost | Wall time |
|---|---|---|---|
| stock IG + stock WM | 2 | 1 (export stuck, ~30 min) | — |
| fixed IG + stock WM | 10 | 2 (router) | 24 s |
| fixed IG + fixed WM | 10 × 3, then 21, then 10 + 10 on the branch builds | **0 / 71** | 21–27 s per wave |

WM unit test (`…WhenEvaluatedConcurrently_EachInstanceResolvesItsOwnPayload`, in the commit): fails on `develop`
5/5, passes with the fix 5/5. Orthanc kept the same 21 SRs after ~80 re-runs (results stay idempotent).

## Use

```bash
LOCAL_PATCHES/build.sh            # once per host, ~5 min: builds the pr and fixed images
LOCAL_PATCHES/switch.sh fixed     # or stock / pr; no argument shows the current mode
```

When upstream releases a fix: point `.env` at their image (`switch.sh` constants) and drop that build.
