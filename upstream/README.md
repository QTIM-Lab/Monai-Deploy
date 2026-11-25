# upstream — clones of MONAI Deploy services we fix

Git-ignored except this file: each folder is a clone of an upstream Project-MONAI repo (plus the QTIM-Lab fork),
where fixes are made and proposed upstream. Never `git add` them to this repo. Nothing here is needed to run
Express: `MONAI_Deploy_Express/LOCAL_PATCHES/build.sh` builds the fixed images at pinned commits, from GitHub (or
from these clones when present).

| Folder | Branch | What |
|---|---|---|
| `monai-deploy-workflow-manager` | `develop` | upstream's dev line (tracks `upstream/develop`) |
| | `fix-parser-race` | **our PR**: `develop` + one commit (lock around the singleton `ConditionalParameterParser` + unit test) |
| `monai-deploy-informatics-gateway` | `main` | upstream's main (tracks `upstream/main`) |
| | `pr-544` | **their PR** #544 as on GitHub, for testing and review |

One fix branch per service, used both for the upstream PR and for the image Express runs. When upstream
releases the fix, Express goes back to their image and the branch is done.

Remotes: `upstream` = Project-MONAI (fetch from here); `origin` = the QTIM-Lab fork (push here):
`git remote add origin git@github.com:QTIM-Lab/monai-deploy-<name>.git`.

## Routine

- Upstream moved on: `git fetch upstream && git rebase upstream/develop` on `fix-parser-race`,
  `git push --force-with-lease`.
- Changed our fix: commit on `fix-parser-race`, update `WM_COMMIT` in `LOCAL_PATCHES/build.sh`, rebuild,
  `LOCAL_PATCHES/switch.sh fixed`.
- PR #544 updated: `git fetch upstream pull/544/head:pr-544 --force`, update `IG_COMMIT` in `build.sh`, rebuild.
- Upstream released a fix: point `MONAI_Deploy_Express/.env` at their image (`switch.sh` constants); drop that build.
- Bug context and evidence: `MONAI_Deploy_Express/LOCAL_PATCHES/README.md` and `TESTING.md`.
