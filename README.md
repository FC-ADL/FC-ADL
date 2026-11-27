# FC-ADL

Code and results for the paper *FC-ADL: Efficient Microservice Anomaly Detection and Localisation Through Functional Connectivity*.

FC-ADL infers time-varying functional connectivity (FC) between microservices from their CPU usage, detects faults as a change in the structure of the FC graphs, and ranks candidate root causes by how much their dependencies change.

```python
import fcadl

changepoint, runtime = fcadl.detect(df)      # df: one column of CPU usage per microservice
ranking, runtime = fcadl.localise(df)        # top-5 candidate root causes
```

## Layout

```
fcadl/          the method (tapered correlation, DeltaCon, LMDS, HDBSCAN, localisation)
baselines/      baseline detection (BARO, N-Sigma, SPOT, BIRCH) and RCA (BARO, N-Sigma, CIRCA, ε-Diagnosis, PC+PR, LiNGAM+PR)
experiments/    data loading and the experiment runners
figures/        scripts for Figs. 1-7 and the appendix table
results/        the results reported in the paper
```

## Setup

Python 3.10:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps sfr-pyrca==1.0.1
```

`sfr-pyrca` (CIRCA and ε-Diagnosis) is installed without its dependencies, as it otherwise pulls in `javabridge`, which needs Java and is not used here.

## Data

Download the data from https://www.kaggle.com/datasets/fcadldata/fc-adl-dataset into `data/`. It contains the social network and hotel reservation recordings, and, for Figs. 6 and 7, `alibaba/MSCPU-48h.csv`: CPU usage of 28,315 microservices over 48 hours at 60 s, taken from the Alibaba cluster-trace-microservices-v2022.

Each recording is a folder `data/<service>_<fault>_sin_<run>/` with `cpu_usage.csv` and `fault_inject.txt` (the fault onset as a sample index).

## Reproducing the paper

The figures and the table can be produced from the included results without the data:

```bash
./reproduce.sh figures
```

Outputs are written to `output/`. To rerun the experiments first (this needs the data):

```bash
./reproduce.sh all
```

or run a single experiment, e.g. `python -m experiments.run_detection fcadl baro`. The detection and RCA runners replace the rows of the methods they run. The scalability runners keep completed measurements so that a long run can be resumed; delete `results/scalability.json` and `results/single_window.json` to start them again. Runtimes depend on the machine; the paper's were measured on an Apple M5 Max.
