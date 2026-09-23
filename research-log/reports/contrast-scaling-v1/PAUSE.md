# Training intentionally paused

The user requested a pause and mid-run GPU validation before deciding whether to change the training method. **Do not automatically resume training.**

The primary run stopped after **1692 complete family updates**, with no completed updates discarded. Saved checkpoint: `checkpoints/contrast-scaling-v1/primary/partial-1692`. Weights identity, all375optimizer states at step1692, CPU/CUDA RNG state and contiguous history have been verified. The production GPU next-update check passed at the unchanged1e-7 tolerance: probability difference0 and next-update parameter difference0. Both verification branches were discarded; the saved checkpoint was unchanged and training was not resumed.

The running process had no cooperative pause flag. A temporary guard at its status-report write path triggered the existing exception checkpoint handler at the explicit completed-update boundary. This was first verified with a real tiny-model optimizer/resume test. Checkpoint writes remained available. The guard was removed after both processes exited. Raw interruption reports/logs are preserved in `pause-audit/`; public pipeline status now records an intentional pause, not a completed experiment. No runtime source, packages, training data or original default were changed.

The5,000-family endpoint and repeat-data control are unfinished. The automatic continuation is paused. Full GPU validation is now complete for all seven checkpoints: 800 contrast judgments plus 300 original-task judgments per checkpoint. See [GPU results](mid-gpu-validation/RESULTS.md). The recommendation is to keep the current recipe paused while the next experiment is decided. Final-test and Jev outcomes remain unopened.
