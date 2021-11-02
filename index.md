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

Dependencies are listed in ```environment.yml```. Install with
```
conda env create -f environment.yml
```

Note: if you are running on `lxplus`, you may run into permissions errors. You can fix this manually by doing:
```
chmod 755 -R /afs/cern.ch/user/<first_letter_of_your_user_name>/<your_user_name>/.conda
```
and then rerunning the command to create the `conda` env.

Note: if you have already set up your `conda` environment but there have been changes to `environment.yml`, you can update your existing environment with:
```
conda env update --file environment.yml --prune
```

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


## 2. Training ML Algorithms
TODO

### 2.1 PCA
TODO

### 2.2 Autoencoder
TODO

## 3. Assessing Performance of ML Algorithms 
TODO
