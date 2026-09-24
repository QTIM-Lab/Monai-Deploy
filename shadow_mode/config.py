"""Registry of the MAPs Shadow Mode knows how to run.

A MAP is treated as an opaque container: a name, an image, and the resources it
needs. Nothing ophthalmology-specific belongs here. Deciding *which* MAP should
run against a given instance is the eligibility/routing layer's job; this module
only records *how* to run one once that decision is made.
"""

from dataclasses import dataclass
from typing import Tuple

# Standard MAP filesystem contract, identical across every image built by
# `monai-deploy package`.
MAP_INPUT_DIR = "/var/holoscan/input"
MAP_OUTPUT_DIR = "/var/holoscan/output"

# Holoscan warns below 32 MB and can segfault without it.
MAP_STACK_BYTES = 33554432


@dataclass(frozen=True)
class MapSpec:
    """Everything needed to invoke one MAP image."""

    name: str
    image: str
    gpu: bool = True
    timeout_s: int = 600

    # Which file types this MAP's loader understands. The executor stages
    # exactly one matching file per run.
    input_suffixes: Tuple[str, ...] = (".dcm",)

    # The UID/GID the image was packaged with. MAP dependencies install into
    # that user's site-packages (/home/holoscan/.local/lib/python3.12/...).
    # `monai-deploy run` substitutes the *host* UID/GID by default, Python then
    # drops that directory from sys.path, and every import inside the MAP fails
    # with a ModuleNotFoundError that looks like a missing dependency but is
    # really a mismatched identity. The docker backend sidesteps this by not
    # passing --user at all, letting the image use its own intended user.
    package_uid: int = 1000
    package_gid: int = 1000


MAPS = {
    "laterality": MapSpec(
        name="laterality",
        image="airead_laterality_app-x64-workstation-dgpu-linux-amd64:1.0",
        input_suffixes=(".dcm",),
    ),
    # Registered so the runner itself can be smoke-tested without involving the
    # laterality model or a GPU. These are the tutorial MAPs, not part of the
    # Shadow Mode pipeline.
    "mednist_prebuilt": MapSpec(
        name="mednist_prebuilt",
        image="mednist_prebuilt_app-x64-workstation-dgpu-linux-amd64:1.0",
        input_suffixes=(".jpeg", ".jpg"),
    ),
    "simple_imaging": MapSpec(
        name="simple_imaging",
        image="simple_imaging_app-x64-workstation-dgpu-linux-amd64:1.0",
        gpu=False,
        input_suffixes=(".png",),
    ),
}


def get_map(name: str) -> MapSpec:
    """Look up a MAP by name, failing with the list of known names."""
    try:
        return MAPS[name]
    except KeyError:
        known = ", ".join(sorted(MAPS)) or "<none registered>"
        raise KeyError(f"Unknown MAP {name!r}. Registered MAPs: {known}") from None
