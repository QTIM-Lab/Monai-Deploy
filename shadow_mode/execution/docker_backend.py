"""Run a MAP by invoking `docker run` directly.

Preferred backend for Shadow Mode. It needs no Holoscan CLI in the service
environment, and — because it passes no `--user` — the container runs as the
image's own intended user, which is by construction the identity the MAP was
packaged with. That makes the UID/GID mismatch failure mode structurally
impossible rather than merely configured-around.

The cost is that the MAP filesystem contract (/var/holoscan/...) is spelled out
here instead of being supplied by the CLI.
"""

from pathlib import Path
from typing import List, Optional

from shadow_mode.config import MAP_INPUT_DIR, MAP_OUTPUT_DIR, MAP_STACK_BYTES, MapSpec
from shadow_mode.execution.executor import MapExecutor


class DockerMapExecutor(MapExecutor):
    """Executes MAPs via the Docker CLI."""

    backend = "docker"

    def __init__(self, docker_bin: str = "docker", gpus: Optional[str] = "all"):
        """
        Args:
            docker_bin: docker executable, overridable for podman-style shims.
            gpus: value for --gpus on GPU-requesting MAPs. None disables the flag.
        """
        self.docker_bin = docker_bin
        self.gpus = gpus

    def _build_command(self, spec: MapSpec, stage_dir: Path, output_dir: Path, run_name: str) -> List[str]:
        command = [
            self.docker_bin,
            "run",
            "--rm",
            "--name",
            run_name,
            # Holoscan warns below 32 MB and can segfault without it.
            f"--ulimit=stack={MAP_STACK_BYTES}",
        ]
        if spec.gpu and self.gpus:
            command += ["--gpus", self.gpus]
        command += [
            "-v",
            f"{stage_dir}:{MAP_INPUT_DIR}",
            "-v",
            f"{output_dir}:{MAP_OUTPUT_DIR}",
            spec.image,
        ]
        return command
