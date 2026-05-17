# Run all experiments included in the final report. Run this script from the root directory of the project. Command:
# ./scripts/reproduability.sh

# === DP EXPERIMENTS ===
./scripts/sweep_dp.sh 125m
./scripts/sweep_dp.sh 350m
./scripts/sweep_dp.sh 760m
./scripts/sweep_dp.sh 1.5b
./scripts/sweep_dp.sh 3b
./scripts/sweep_dp.sh 8b

# === DATA LOADER WORKERS EXPERIMENTS ===
./scripts/sweep_workers.sh 125m