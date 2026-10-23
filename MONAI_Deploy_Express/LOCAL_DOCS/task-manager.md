# Task Manager (TM)

Container `mdl-tm` (0.2.1-2), config `configs/task-manager.json`, working dirs `.md/mdtm/`. Runs
`docker` tasks by talking to the host's Docker socket.

- **`concurrency: 1` limits how many tasks it *launches* at once, not how many run.** The Docker
  plugin returns as soon as a container starts, so every study's MAP runs in parallel. Nothing in this
  version runs jobs strictly one after another.
- **It never removes task containers.** One exited container per run accumulates. Clean up with
  `docker container prune -f --filter label=org.monai.deploy.app-sdk` (MAP images carry that label;
  running containers are never touched).
- Always sets `Runtime=nvidia`; the MAP sees every GPU. It cannot set `--ulimit stack`.
- Input is staged read-only as `<study>/<series>/*.dcm` plus a `.dcm.json` sidecar per file. The SDK
  `DICOMDataLoaderOperator` reads that layout as-is.
- The MAP runs as its packaged user (`holoscan`, 1000:1000); task dirs on this filesystem come out 777,
  so it can write its output.

```bash
docker logs mdl-tm --since 10m | grep -E "TaskDispatch|containerId"
docker ps -a --filter label=org.monai.deploy.app-sdk     # MAP containers, running and exited
```
