"""Fetch a patient's or a study's DICOM files from Orthanc into a temporary folder.

    from orthanc import fetch
    with fetch(patient="1001") as folder:      # or study="<StudyInstanceUID>"
        ...                                    # folder holds the DICOM files; removed afterwards

Data comes from the PACS, never from a path on this machine: load your own data into
Orthanc first (e.g. MONAI_Deploy_Express/LOCAL_SCRIPTS/upload.sh /path/to/dicom).
Orthanc URL: $ORTHANC_URL, default the practice instance. Standard library only.
"""

import io
import json
import os
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

ORTHANC_URL = os.environ.get("ORTHANC_URL", "http://127.0.0.1:18042")


def _call(path: str, body: dict = None) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(ORTHANC_URL + path, data=data)) as response:
        return response.read()


@contextmanager
def fetch(patient: str = None, study: str = None):
    """Yield a temporary folder with every DICOM file of one patient (PatientID) or one study."""
    level, key, value = ("Patient", "PatientID", patient) if patient else ("Study", "StudyInstanceUID", study)
    if not value:
        raise ValueError("give patient= or study=")
    ids = json.loads(_call("/tools/find", {"Level": level, "Query": {key: value}}))
    if not ids:
        raise LookupError(f"no {level.lower()} {value} in Orthanc at {ORTHANC_URL}")
    with tempfile.TemporaryDirectory(prefix="orthanc-") as folder:
        for orthanc_id in ids:
            zipfile.ZipFile(io.BytesIO(_call(f"/{level.lower()}s/{orthanc_id}/archive"))).extractall(folder)
        yield Path(folder)
