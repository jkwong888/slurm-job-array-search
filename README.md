# California Housing training sweep

This example downloads the California Housing regression dataset, runs one
PyTorch hyperparameter trial per Slurm array task, and summarizes the models by
validation RMSE. The dataset, search configuration, and results must be on storage visible to the
Slurm compute nodes.

## 1. Prepare the dataset

Use a Python environment with the packages in `requirements.txt`:

```bash
python prepare_data.py --output-path /data/datasets/california_housing
```

This downloads the source data once and writes an 80/10/10 split as
`train.parquet`, `validate.parquet`, and `test.parquet` in that directory. The
default split seed is 42. Existing split files are not overwritten.

## 2. Submit the Slurm job array

Put `search.json` on shared storage.

```bash
mkdir -p /data/configs
cp search.json /data/configs/search.json
```

Make sure the training image has been
built and pushed with the current scripts. The submit script needs `jq` and
`sbatch`; each compute node needs access to the image through Pyxis.

```bash
export DATASET_PATH=/data/datasets/california_housing
export SWEEP_DIR=/data/sweep
./submit_sweep.sh --search-config /data/configs/search.json
```

This generates an array of jobs with length equal to the number of trials in `search.json` and uses `sbatch` 
to schedule them on the slurm cluster.  As they execute, they write their results to a job directory in`/data/sweep/${SLURM_JOB_ID}/<trial number>`

If you want to run this containerized, build the container image and set `TRAIN_IMAGE` before submission.  Take a look at [sweep.sbatch](./sweep.sbatch) to see how it submits the job to slurm using the container image.  It's also possible to submit the job without containers and just run the code directly provided that every worker node can access these scripts from the same relative paths.

## 3. Summarize completed trials

After every array task has finished, run:

```bash
python summarize.py \
  --search-config /data/configs/search.json \
  --output-dir /data/sweep/<JOB_ID>
```

The summarizer verifies each expected trial, writes `trials.jsonl` with metrics
and model paths, and writes `best.json` with the best validation result and a
`model_paths` mapping for every trial. Model paths are relative to the sweep
directory (`/data/sweep/54` in this example).  

Here is a sample output:

```bash
2026-10-02T18:07:40+0000 INFO summarize: summarizing 6 trials from /data/test/60
2026-10-02T18:07:40+0000 INFO summarize: trial=0 model_path=0/model.pt val_rmse=0.6515
2026-10-02T18:07:40+0000 INFO summarize: trial=1 model_path=1/model.pt val_rmse=0.6732
2026-10-02T18:07:40+0000 INFO summarize: trial=2 model_path=2/model.pt val_rmse=0.5878
2026-10-02T18:07:40+0000 INFO summarize: trial=3 model_path=3/model.pt val_rmse=0.6233
2026-10-02T18:07:40+0000 INFO summarize: trial=4 model_path=4/model.pt val_rmse=0.5702
2026-10-02T18:07:40+0000 INFO summarize: trial=5 model_path=5/model.pt val_rmse=0.5783
2026-10-02T18:07:40+0000 INFO summarize: best trial=4 val_rmse=0.5702 val_mae=0.3994 checkpoint=4/model.pt
```