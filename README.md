# Reproduction for kubeflow/trainer#4240

Standalone branch, not part of Trainer. It runs the `local-container-mnist` training
setup (two containers, `torchrun --rdzv-backend=static`, gloo on CPU, the same
commands the Kubeflow SDK container backend uses) many times on GitHub-hosted
runners, on the runtime image before and after kubeflow/trainer#3780:

- `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`
- `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime`

A run counts as `HANG` when a container is still running 420s after rank 0
printed `Training is finished`. For `SLOW` and `HANG` runs the job saves `ps`
and `py-spy dump --native` output from both containers, plus their logs, as artifacts.
