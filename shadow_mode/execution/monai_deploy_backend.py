"""Run a MAP through the `monai-deploy run` CLI.

Kept alongside the docker backend because it is the vendor-supported path and
the one the app READMEs document, so it is the right thing to reach for when
deciding whether a failure is ours or the SDK's.

It requires holoscan-cli in whatever environment Shadow Mode runs from, and it
requires --uid/--gid to be stated explicitly: the CLI otherwise substitutes the
host's UID/GID, Python drops the packaging user's --user site-packages from
sys.path, and every import inside the MAP fails with ModuleNotFoundError. The
values come from MapSpec.package_uid/gid rather than os.getuid().
"""

from pathlib import Path
from typing import List, Optional

from shadow_mode.config import MapSpec
from shadow_mode.execution.executor import MapExecutor


class MonaiDeployMapExecutor(MapExecutor):
    """Executes MAPs via the monai-deploy CLI."""

    backend = "monai-deploy"

    def __init__(
        self,
        monai_deploy_bin: str = "monai-deploy",
        gpus: Optional[str] = "all",
        log_level: str = "INFO",
    ):
        """
        Args:
            monai_deploy_bin: CLI executable. Give an absolute path when Shadow
                Mode runs outside the pyenv that holds holoscan-cli.
            gpus: value for --gpus on GPU-requesting MAPs. None disables the flag.
            log_level: passed through as -l.
        """
        self.monai_deploy_bin = monai_deploy_bin
        self.gpus = gpus
        self.log_level = log_level

    def _build_command(self, spec: MapSpec, stage_dir: Path, output_dir: Path, run_name: str) -> List[str]:
        command = [
            self.monai_deploy_bin,
            "run",
            "-l",
            self.log_level,
            "--rm",
            "--name",
            run_name,
            # Must be the packaging identity, not ours. See module docstring.
            "--uid",
            str(spec.package_uid),
            "--gid",
            str(spec.package_gid),
        ]
        if spec.gpu and self.gpus:
            command += ["--gpus", self.gpus]
        command += [
            "-i",
            str(stage_dir),
            "-o",
            str(output_dir),
            spec.image,
        ]
        return command
