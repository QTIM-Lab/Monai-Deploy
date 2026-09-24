"""Shadow Mode — AI inference running alongside the imaging workflow, not in it.

Phase 2 scope: execute an existing MAP against a local DICOM file. Orthanc
discovery, eligibility/routing, provenance and idempotency are deliberately
absent — they belong in layers above `execution`, not inside it.
"""
