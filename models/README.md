# Model checkpoints

Checkpoints are **not committed** to Git (~99 MB each). Place the trained model here as:

```
models/avocado_ripeness_resnet50.pth
```

Ways to get it:

1. **Download** it from the project's GitHub Releases page and save it here.
2. **Train** your own: see "Training" in the main README.
3. **Convert** a checkpoint produced by the original research notebook:

   ```bash
   python -m avocado_ai.ml.convert_checkpoint old_checkpoint.pth models/avocado_ripeness_resnet50.pth
   ```

Checkpoints in the current format are loaded with `torch.load(weights_only=True)`,
so they cannot run arbitrary code when loaded.
