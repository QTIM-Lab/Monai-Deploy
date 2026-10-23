# Orthanc (practice instance)

Container `mdl-orthanc`, config `configs/orthanc.json`, data `.md/orthanc/`. Stands in for the real
Orthanc on another server. It knows nothing about MONAI: `MONAI-DEPLOY` is just a DICOM modality.

```json
"DicomModalities": { "MONAI-DEPLOY": ["MONAI-DEPLOY", "informatics-gateway", 104] }
```

## The auto-route Lua (`configs/orthanc-autoroute.lua`)

Mounted by `docker-compose.override.yml`, loaded by `"LuaScripts"` in `orthanc.json`.

- Fires on `OnStableSeries`: `StableAge` seconds (15) after a series stops receiving instances.
- Sends that series to `MONAI-DEPLOY`, unless its first instance's `RemoteAET` is `MONAISCU`.
- Per series, not per study: results land in the source study, so a study trigger would re-fire on
  every result.
- No modality or SOP class logic on purpose. Which AI runs is the workflow router's job; what MONAI
  accepts is the IG AE's.

`MONAISCU` is the IG's **export** AE (`dicom.scu.aeTitle` in `configs/informatics-gateway.json`). It is
not `MONAI-DEPLOY`, the AE the IG listens on. If the two drift, the IG's `ignoredSopClasses` still stops
a loop.

Changes take effect on `docker compose restart orthanc`. Check it: `docker logs mdl-orthanc 2>&1 | grep autoroute`.

## Moving to the real Orthanc

On that server: add modality `MONAI-DEPLOY` → this host:104, and an equivalent sender rule (Lua,
Python plugin, whatever its admin prefers) that skips `MONAISCU`. Here: expose IG port 104 beyond
`127.0.0.1` (firewalled to that host), re-register the IG source/destination
([informatics-gateway.md](informatics-gateway.md)), drop the `orthanc` service.
