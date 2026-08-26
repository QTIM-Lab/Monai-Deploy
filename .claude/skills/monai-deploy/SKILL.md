---
name: monai-deploy
description: Use when working on MONAI Deploy App SDK apps in this repo — writing Operators/Applications, packaging MAPs with `monai-deploy package`, running MAPs via `monai-deploy run` or `docker run`, debugging MAP import errors or Holoscan/CUDA version conflicts, or planning MONAI Bundle-to-MAP model handoff. Triggers on "MONAI Deploy", "MAP", "holoscan", "monai-deploy package/run", "app.yaml", "Operator", "MedNIST app", "simple imaging app".
---

# MONAI Deploy in this repo

A learning ladder through the official MONAI Deploy App SDK tutorials, aimed at
eventually running real model inference (AIREADI laterality classifier) and
defining a model-developer → deployment handoff contract.

**No Kubernetes.** Not at this stage, for any reason.

## Environment — three distinct Pythons

Never conflate these. Most confusion in this project comes from mixing them up.

| Environment | Python | Purpose |
|---|---|---|
| `~/.pyenv/versions/3.10.0/envs/monai_deploy` | 3.10.0 | Host dev + MAP packaging |
| Inside a built MAP image | 3.12 | MAP runtime |
| Bare `python` shim | 3.9.13 | **Wrong env** — pyenv global, unrelated |

Activate the right one:

```bash
pyenv activate monai_deploy
# or absolute:
/persist/python_virtual_environments/bearceb/.pyenv/versions/3.10.0/envs/monai_deploy/bin/python
```

## Pinned versions — do not change casually

Host `monai_deploy` env:

```
monai-deploy-app-sdk  4.0.0
holoscan-cli          4.2.0
holoscan-cu13         4.2.0
```

These were arrived at by debugging. Installing monai-deploy-app-sdk 4.0.0 pulls
newer Holoscan by default; Holoscan CLI 4.2 then refuses to package:

```
Invalid SDK version specified (4.5.0): valid values are: 4.2.0
```

`holoscan-cu12` 4.5.0 was removed. Do not reintroduce it. MONAI Deploy App SDK
4.0.0 expects the **cu13** Holoscan variant.

If a version change looks genuinely necessary, explain why and get agreement
before touching anything.

## CUDA: host is 12, MAP is 13

Host exposes `libcudart.so.12`; driver is 580.173.02 (supports up to CUDA 13);
2× RTX A4000. Holoscan cu13 therefore cannot run from the host venv — running
`python app.py` directly on the host is not a supported path here.

**This is not a blocker.** The MAP ships its own CUDA 13 userspace and reaches
the GPU via the NVIDIA container stack. Prefer containerized execution.

**Do not upgrade host CUDA** — this is a shared server and others depend on
CUDA 12.

Important distinction: this conflict is **Holoscan-specific**. Plain PyTorch
training does not import Holoscan, and cu12 torch wheels bundle their own CUDA
userspace, so host-side GPU *training* is viable in the pyenv even though
host-side *Holoscan* is not.

## The UID/GID invariant — read this before debugging any MAP import error

**Package-time UID/GID must equal runtime UID/GID.**

MAP dependencies install into the packaging user's user-site directory:
`/home/holoscan/.local/lib/python3.12/site-packages`. `monai-deploy run`
defaults to substituting the *host* UID/GID (here `214196116:214196116`).
Python then drops that path from `sys.path` and every MAP import fails:

```
ModuleNotFoundError: No module named 'monai'
```

The module is present. The identity is wrong. Verify with:

```bash
# works — image's own user
docker run --rm --entrypoint python3 <image> -c "import monai; print(monai.__path__)"

# fails — host UID substituted
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python3 <image> \
  -c "import sys; print('\n'.join(sys.path)); import monai"
```

Bind mounts are irrelevant to this. Only identity matters.

For a future shared deployment, do **not** bake in a personal UID — define a
fixed service identity (e.g. a `monai-deploy` service user) used consistently at
package and run time.

## Packaging a MAP

```bash
monai-deploy package <app_dir> \
  -c <app_dir>/app.yaml \
  -t <name>:1.0 \
  --platform x86_64 \
  -l DEBUG
```

Produces `<name>-x64-workstation-dgpu-linux-amd64:1.0`.

For apps with a model, add `--models <path>`, which places the model under
`/opt/holoscan/models` in the image.

The app's `requirements.txt` is baked into the MAP. In this repo it pins the
MONAI Deploy/Holoscan versions explicitly rather than inheriting them — keep
those pins consistent with the host env.

`app.yaml` declares title/version, input/output formats, and resource requests
(cpu/gpu/memory/gpuMemory).

## Running a MAP — three known-good invocations

```bash
# 1. monai-deploy run, with the packaging identity supplied explicitly
monai-deploy run --uid 1000 --gid 1000 \
  -i <app_dir>/test_input_folder \
  -o <app_dir>/output_path \
  <name>-x64-workstation-dgpu-linux-amd64:1.0

# 2. plain docker, letting the image use its own intended user (holoscan 1000:1000)
docker run --rm --gpus all --ulimit stack=33554432 \
  -v "$(pwd)/<app_dir>/test_input_folder:/var/holoscan/input" \
  -v "$(pwd)/<app_dir>/output_path:/var/holoscan/output" \
  <name>-x64-workstation-dgpu-linux-amd64:1.0

# 3. package with your own identity so both sides agree
monai-deploy package ... --uid "$(id -u)" --gid "$(id -g)"
```

Standard MAP paths inside the container: `/var/holoscan/input`,
`/var/holoscan/output`, `/opt/holoscan/models`.

Note: option 2 works because Docker uses the image's *intended* user, **not**
because of root.

## Inspecting a built MAP

```bash
docker run --rm --entrypoint python3 <image> -m pip list
docker run --rm --entrypoint bash <image> -c "ls /opt/holoscan/models"
```

Do not assume a dependency is missing just because it isn't in
`requirements.txt` — check the image.

## Application structure

`Application.compose()` instantiates operators and wires them with `add_flow`:

```python
class App(Application):
    name = "simple_imaging_app"

    def compose(self):
        app_context = Application.init_app_context(self.argv)
        sample_data_path = Path(app_context.input_path)
        output_data_path = Path(app_context.output_path)

        # CountCondition(self, 1) makes the pipeline run exactly once
        sobel_op = SobelOperator(self, CountCondition(self, 1),
                                 input_path=sample_data_path, name="sobel_op")
        median_op = MedianOperator(self, name="median_op")
        gaussian_op = GaussianOperator(self, output_folder=output_data_path,
                                       name="gaussian_op")
        self.add_flow(sobel_op, median_op, {("out1", "in1")})
        self.add_flow(median_op, gaussian_op, {("out1", "in1")})
```

The Application object is the first positional arg to each Operator; everything
else is kwargs. Decorators like `@resource(cpu=1)` are not available in SDK 4.0.

`__main__.py` just does `from app import App; App().run()` so the directory is
runnable as a module and packageable.

## Repo layout

- `Simple_Image_Processing/` — Tutorial 1, Sobel → Median → Gaussian. **Working**:
  packaged and executed successfully.
- `MedNIST_Classifier_App/` — Tutorial 2, train-it-yourself. In progress.
- `MedNIST_Classifier_App_Pre-Built_Model/` — Tutorial 2 prebuilt variant. Stub.

## Working style

The user is an experienced engineer (Python, Docker, Linux, DICOM, MONAI
Bundles) — skip fundamentals. They arrive with issues already diagnosed and
expect that work respected. Keep focus on learning MONAI Deploy and reaching
real model inference; don't over-engineer the tutorials.
