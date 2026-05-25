# Results Folder

This folder is used for all generated outputs from the modelling pipeline.

Before running the pipeline, this folder may be empty except for `.gitkeep` files and this README.

## Generated Outputs

When `run_all.py` or individual model scripts are executed, the following outputs are generated here:

- model summary CSV files
- model prediction CSV files
- removed-days CSV files
- execution logs
- M7 gate-history CSV files

## Logs

Execution logs are stored in:

```text
Results/logs/
```

Each model script receives a separate log file when the full pipeline is executed through:

```bash
python3 run_all.py
```

## Gate History Outputs

M7 gate-history files are stored in:

```text
Results/Gate_P/
Results/Gate_D/
```

- `Gate_P`: gate histories for the price variant
- `Gate_D`: gate histories for the delta variant

## Git Tracking

Generated result files are excluded from Git by `.gitignore`.

This keeps the repository lightweight and ensures that outputs can be regenerated from the code and fixed data snapshot.