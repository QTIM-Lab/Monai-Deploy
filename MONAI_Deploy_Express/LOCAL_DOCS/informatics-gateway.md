# Informatics Gateway (IG)

Container `mdl-ig` (0.5.1-3). DICOM in on 104, REST on 5000. Config `configs/informatics-gateway.json`;
AEs, sources and destinations live in MongoDB (database `InformaticsGateway`) and are (re)created at every
start by the `configure-ig` service running `configs/config-ig.sh`. `.md/mdig/` holds incoming payload files.

| Thing | Value | Set in |
|---|---|---|
| Listening AE | `MONAI-DEPLOY`, grouping by study (`0020,000D`), 5 s timeout | `config-ig.sh` |
| Loop guard | AE `ignoredSopClasses`: SR 88.11/22/33/34/35, SEG 66.4 | `config-ig.sh` |
| Accepted sender | source `ORTHANC` = AE `ORTHANC` @ 172.29.0.100 | `config-ig.sh` |
| Export target | destination `ORTHANC` = 172.29.0.100:4242 | `config-ig.sh` |
| Export AE (calling) | `MONAISCU` | `informatics-gateway.json` `dicom.scu` |
| Disk watermark | 99 % (upstream 95) | `informatics-gateway.json` |

```bash
curl -s http://127.0.0.1:5000/config/ae | python3 -m json.tool
curl -s http://127.0.0.1:5000/config/source
curl -s http://127.0.0.1:5000/config/destination
# live change, e.g. re-create the AE (config-ig.sh does the same at start)
curl -X DELETE http://127.0.0.1:5000/config/ae/MONAI-DEPLOY
curl -X POST http://127.0.0.1:5000/config/ae -H 'Content-Type: application/json' -d '{"name":"MONAI-DEPLOY","aeTitle":"MONAI-DEPLOY","ignoredSopClasses":[...]}'
```

Behaviour worth knowing:

- Each group (study) that arrives becomes one payload, which becomes one workflow run.
- Ignored SOP classes are accepted (C-STORE success) and discarded; log line:
  `Instance ignored due to matching SOP Class UID`.
- Above the watermark it **rejects every association** (`NoReasonGiven`). `/scratch90` is ~97 % full.
- Received files go to MinIO, not to disk; see [storage.md](storage.md).
