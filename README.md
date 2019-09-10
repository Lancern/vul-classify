# vul-classify

A malicious code detector leveraging machine learning approaches and plugged in to the Viper platform.

## Approaches

Feature extraction: [`asm2vec`](https://github.com/lancern/asm2vec)

Classification Models:

* Naive model
* <del>`node2vec`</del>
* XGboost
* LSTM

## Requirements

This tool is written in pure python and python 3.7+ is recommended.

### Dependencies

Execute the following command to install necessary dependencies for this repo:

```shell
python3 -m pip install -r requirements.txt
```

## Run

Before start, please make sure that you have `asm2vec` installed on your `PYTHONPATH`. If not, execute the following commands to install it:

```shell
git clone https://github.com/lancern/asm2vec.git
export PYTHONPATH="$PYTHONPATH:path/to/asm2vec"
```

`vul-classify` need a configuration file to run. Create a file named `vul-classify.config.json` and copy-and-paste the following content (which is the default configuration):

```json
{
  "logging": {
    "level": "DEBUG",
    "file": "vul-classify.log"
  },
  "thread_pool": {
    "workers": 10
  },
  "models": [
    {
      "file": "models/naive.py",
      "module": "vulcls.models.naive",
      "name": "NaiveModel",
      "data_file": "data/models/naive.bin"
    },
    {
      "file": "models/lstm.py",
      "module": "vulcls.models.lstm",
      "name": "LSTMModel",
      "data_file": "data/models/lstm.bin"
    },
    {
      "file": "models/xgb.py",
      "module": "vulcls.models.xgb",
      "name": "XGBModel",
      "data_file": "data/models/xgb.bin"
    }
  ],
  "asm2vec": {
    "memento": "path/to/asm2vec-memento-file"
  },
  "daemon": {
    "address": "127.0.0.1",
    "port": 8080
  },
  "repo": {
    "filename": "path/to/repo-file"
  }
}
```


