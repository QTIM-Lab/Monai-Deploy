# Upstream: taking MONAI's updates

This folder is a `git subtree` of `Project-MONAI/monai-deploy` → `deploy/monai-deploy-express`, added at
upstream `0c1f13a`; our edits are separate commits on top. Service fixes arrive as new image tags in `.env`;
this is only for Express's own files.

```bash
git remote add monai-deploy https://github.com/Project-MONAI/monai-deploy.git   # once per clone
git config remote.monai-deploy.fetch '+refs/heads/main:refs/remotes/monai-deploy/main'
git log --grep="git-subtree-dir: MONAI_Deploy_Express" -1                       # upstream version we're on
git fetch monai-deploy                                                           # anything new?
# take it (git >= spring 2026: just the split line without the worktree)
git worktree add --detach /tmp/monai-upstream monai-deploy/main
git -C /tmp/monai-upstream subtree split --prefix=deploy/monai-deploy-express HEAD -b express-upstream
git worktree remove /tmp/monai-upstream
git subtree merge --prefix=MONAI_Deploy_Express express-upstream && git branch -D express-upstream
```

The worktree works around a git 2.43 subtree bug (split checks the prefix in the working tree; fixed upstream in
"subtree: validate --prefix against commit in split"). The merge keeps our edits and stops only where MONAI
changed the same lines.
