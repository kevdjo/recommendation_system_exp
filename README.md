# recommendation_system_exp

Databricks batch workload deployed via Databricks Asset Bundles (DAB).

- **Project type**: `modeling_batch`
- **Compute type**: `serverless`
- **Target timezone**: `Asia/Jakarta`

## Quickstart

```bash
# 1. First-time setup
make init

# 2. Confirm local Databricks Connect auth
make connect-test

# 3. Run test suite & static checks
make test
make lint

# 4. Build wheel
make build

# 5. Validate bundle
make validate

# 6. Deploy to personal dev sandbox
make deploy-dev
# 7. Run jobs in dev
make run-train-dev

```

## Structure

```
src/recommendation_system_exp/
├── common/         # SparkSession, task values, MLflow helpers
├── config/         # config.yaml + Pydantic schema
├── training/       # Training tasks (extract -> transform)
└── inference/      # Inference task (init)

resources/
├── jobs.yml        # DAB job definitions
└── experiments.yml # Per-target MLflow experiments
```
