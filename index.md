# Introduction
Welcome to the AutoDQM ML user tutorial! The [AutoDQM_ML](https://github.com/AutoDQM/AutoDQM_ML) repository is a toolkit for developing machine learning algorithms to detect anomalies in offline DQM histograms.

With this tool, you can do the following:
1. Grab data from DQM histograms of interest on `/eos` and write to a single `pandas dataframe`.
2. Train machine learning algorithms that can be used for anomaly detection.
3. Compare the performance of these ML algorithms and statistical tests.

# Setup
**1. Clone repository**
```
git clone https://github.com/AutoDQM/AutoDQM_ML.git
cd AutoDQM_ML
```
**2. Install dependencies**

Dependencies are listed in ```environment.yml``` and installed using `conda`. `conda` is a package and environment management system, see [documentation](https://docs.conda.io/en/latest/) for more details.

If you do not already have `conda` set up on your system, you can install (for linux) with:
```
curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
```
You can then set `conda` to be available upon login with
```
~/miniconda3/bin/conda init # adds conda setup to your ~/.bashrc, so relogin after executing this line
```

Once `conda` is installed and set up, install dependencies with
```
conda env create -f environment.yml -p <path to install conda env>
```

Note: if you are running on `lxplus`, you may run into permissions errors, which may be fixed with:
```
chmod 755 -R /afs/cern.ch/user/s/<your_user_name>/.conda
```
and then rerunning the command to create the `conda` env. The resulting `conda env` can also be several GB in size, so it may also be advisable to specify the installation location in your work area if running on `lxplus`, i.e. running the `conda env create` command with `-p /afs/cern.ch/work/...`.

**3. Install autodqm-ml**

**Users** can install with:
```
python setup.py install
```
**Developers** are suggested to install with:
```
pip install -e .
```
to avoid rerunning the whole installation every time there is a change.

Once your setup is installed, you can activate your python environment with
```
conda activate autodqm-ml
```

**Note**: `CMSSW` environments can interfere with `conda` environments. Recommended to unset your CMSSW environment (if any) by running
```
eval `scram unsetenv -sh`
```
before attempting installation and each time before activating the `conda` environment.

# Using the tool

## 1. Fetching Data
The first step in developing anomaly detection algorithms is fetching the data with which we want to train and assess our algorithms.

The [`scripts/fetch_data.py`](https://github.com/AutoDQM/AutoDQM_ML/blob/main/scripts/fetch_data.py) script can be used to do this. It is essentially a wrapper to the `autodqm_ml.data_prep.data_fetcher.DataFetcher` class. 

There are 3 main command line arguments to pay attention to when using this script.

First, the `--tag` argument should be given a `str` input which will be included in your output file names.

Second, the `--contents` option should be given a `str` input which is a path to a `json` file indicating which histograms you would like to fetch data for. Here is a "contents" `json` file for grabbing a small set of histograms from the CSC and EMTF subsystems:
```json
{
    "CSC": [
        "/Run summary/CSCOfflineMonitor/Occupancy/hORecHits",
        "/Run summary/CSCOfflineMonitor/Occupancy/hOSegments",
        "/Run summary/CSCOfflineMonitor/Segments/hSTimeCombinedSerial",
        "/Run summary/CSCOfflineMonitor/Segments/hSTimeVsTOF",
        "/Run summary/CSCOfflineMonitor/Segments/hSTimeVsZ"
    ],
    "L1T" : [
        "/Run summary/L1TStage2EMTF/emtfTrackBX",
        "/Run summary/L1TStage2EMTF/emtfTrackEta",
        "/Run summary/L1TStage2EMTF/emtfTrackOccupancy",
        "/Run summary/L1TStage2EMTF/emtfTrackPhi",
        "/Run summary/L1TStage2EMTF/emtfTrackQualityVsMode"
    ]
}
``` 
Each key in the "contents" `json` should correspond to a subsystem directory and the corresponding value should be a list of histograms, given as the relative path within the `root` file starting from the subsystem directory.

Third, the `--datasets` option should be given a `str` input which is a path to a `json` file indicating which primary datasets, years, productions, eras and/or years you would like to grab histograms for. As an example, consider the following "datasets" `json`:
```json
{
    "primary_datasets" : ["SingleMuon"],
    "years" : {
        "2017" : {
            "productions" : ["UL2017"],
            "eras" : ["B"]
        },
        "2018" : {
            "productions" : ["UL2018"],
            "runs" : ["317488", "317481"]
        }
    }
}
``` 
This will find all DQM files for the `SingleMuon` PDs from 2017 and 2018. For 2017/2018 only files with the substring `"UL2017"`/`"UL2018"` will be considered, and for 2017 only era B ("Run2017B") will be considered, while for 2018, only runs `317488` and `317481` will be considered. 

One could also at this point specify a list of runs to mark as either "good" (not anomalous) or "bad" (anomalous), by adding the following to your "datasets" `json`:
```json
{
    "2018" : {
            "productions" : ["UL2018"],
            "runs" : ["317488", "317481"],
            "bad_runs" : ["317488"],
            "good_runs" : ["317481"]
        }
}
```
If at least one of "bad_runs" or "good_runs" is specified, a field "label" will be added to your output dataframe, with a value of 1 for bad runs and a value of 0 for good runs. If both "bad_runs" and "good_runs" are specified, any runs not specified in either list will be assigned a "label" of -1. If only one of "bad_runs" or "good_runs" is specified, all runs not explicitly listed as good (bad) will be assigned a value corresponding to bad (good).

Note that it is recommended that the paths to the "contents" and "datasets" `json` files should be given either as the full absolute path or as relative paths under `AutoDQM_ML`. For example, if my file was located in
```
/<path_to_AutoDQM_ML>/AutoDQM_ML/configs/datasets.json
```
I should pass this to `fetch_data.py` as
```
--datasets "configs/datasets.json"
```
You could also specify it as a relative path to your current directory, for example,
```
--datasets "../configs/datasets.json"
```
but the same command will not be guaranteed to work if you run the script from inside a different directory.


## 2. Training ML Algorithms
### 2.1 Introduction
Having now prepped some DQM data, we can now train some ML algorithms to help us detect anomalies.

The script [`scripts/train.py`]() does the following:
1. Loads and preps training data (from the output of `scripts/fetch_data.py`). Prepping data includes splitting events into training/testing sets, normalizing histograms, removing low-statistics runs, etc.
2. Trains and saves a specified ML algorithm(s)
3. Creates a new file with the predictions of the ML algorithm

For example, assume we have run `scripts/fetch_data.py` and now have an output file:
```
output/test_SingleMuon.parquet
```
Now, we would like to train a PCA on some of the histograms in this file. This can be done with:
```
python train.py
    --input_file "output/test_SingleMuon.parquet"
    --output_dir "output_withMLAlgos/"
    --algorithm "pca"
    --tag "pca" # this will identify saved model files and branches added to the histograms file
    --histograms "emtfTrackPhi,emtfTrackEta"
```
The `train.py` script will do the following:
1. Load histograms from `"output/test_SingleMuon.parquet"`, normalize the histograms, remove any runs with low statistics, and split them into training/testing sets.
2. Train (and save) PCAs for each of the specified histograms. The saved PCA models will be placed in the `output_dir` with the form `"pca_<hist_name>_<tag>"`.
3. Use the trained PCAs to create reconstructed histograms for each specified histogram and for each run.
4. Calculate the sum-of-squared errors (SSE) between the original and reconstructed histogram.
5. Write a new output file in the `output_dir` with fields added for the reconstructions of each histogram and their SSE wrt the original. These fields have the form `<hist_name>_reco_<tag>` and `<hist_name>_score_<tag>`, respectively.

Next, we would like to train an AutoEncdoer and compare its performance to the PCA. Rather than having two separate output files for the PCA and the AutoEncoder, we can save the results all in one file by using the output from the previous step:
```
python train.py
    --input_file "output_withMLAlgos/test_SingleMuon.parquet"
    --output_dir "output_withMLAlgos/"
    --algorithm "autoencoder"
    --tag "AE"
    --histograms "emtfTrackPhi,emtfTrackEta"
```
We can now access both the results of the PCA and the AutoEncoder scores in the output file. The reconstructed histograms from each algorithm and their respective SSE will be available through fields like `emtfTrackPhi_reco_pca` and `emtfTrackPhi_score_AE`.

In this way, it is possible to chain together the training of multiple different ML algorithms and have results stored in a common place.

We can also add the results of statistical tests, like a 1d KS-test, through the `train.py` script:
```
python train.py
    --input_file "output_withMLAlgos/test_SingleMuon.parquet"
    --output_dir "output_withMLAlgos/"
    --algorithm "statistical_tester"
    --tag "stat"
    --histograms "emtfTrackPhi,emtfTrackEta"
```
As statistical tests do not make reconstructions of the original histogram, this will not add fields like `emtfTrackPhi_reco_stat`, but will still add fields with the results of the statistical test, e.g. `emtfTrackPhi_score_stat`.

Note that if I tried to train an ML algorithm with the same tag, `AutoDQM_ML` will check for saved ML models matching that tag in the output directory. If they already exist, it will not retrain and overwrite them in order to prevent unintended overwriting of results.

A few options for specifying the details of the algorithms are provided through the CLI in `train.py`:
- `--algorithm` specify a statistical test, PCA, or AutoEncoder
- `--histograms` specify the list of histograms to train algorithms for (or simply evaluate, in the case of a statistical test)
- `--n_components` specify the dimensionality of the latent space used for reconstructing histograms (only applicable for ML algorithms)

In general, however, we will want greater control over the specifics of the algorithms we train. For example, we may want to vary the DNN architecture and training strategy for AutoEncoders. This can be achieved by specifying a `json` file for the `--algorithms` CLI.

### 2.2 Statistical Tests
TODO

### 2.3 Principal Component Analysis (PCA)
TODO

### 2.4 Autoencoders
TODO

## 3. Assessing Performance of ML Algorithms 
TODO
