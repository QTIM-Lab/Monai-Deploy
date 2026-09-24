"""MAP execution backends.

Callers should depend on `MapExecutor` and `get_executor`, never on a concrete
backend, so that swapping in a Workflow Manager adapter later touches only this
package.
"""

from shadow_mode.execution.docker_backend import DockerMapExecutor
from shadow_mode.execution.executor import MapExecutionError, MapExecutor, RunResult
from shadow_mode.execution.monai_deploy_backend import MonaiDeployMapExecutor

BACKENDS = {
    "docker": DockerMapExecutor,
    "monai-deploy": MonaiDeployMapExecutor,
}

DEFAULT_BACKEND = "docker"


def get_executor(backend: str = DEFAULT_BACKEND, **kwargs) -> MapExecutor:
    """Build an executor by backend name."""
    try:
        return BACKENDS[backend](**kwargs)
    except KeyError:
        known = ", ".join(sorted(BACKENDS))
        raise KeyError(f"Unknown backend {backend!r}. Available: {known}") from None


__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "DockerMapExecutor",
    "MapExecutionError",
    "MapExecutor",
    "MonaiDeployMapExecutor",
    "RunResult",
    "get_executor",
]
