# Ultrasound Cardiac 3D Reconstraction from 2D Segmentated Images

## Plans:
1. Collect CAMUS ds
2. Try universeg
2. Parse EchoNet-Dynamic ds
2.1 Parse from Kaggle
2.2 Parse from https://echonet.github.io/dynamic/index.html#code
3. Train U2Net


## 📋 Requirements

* DVC
* Python3 and pip

## 🏃🏻 Running Project


```


### ✅ Pre-commit Testings

In order to activate pre-commit testing you need ```pre-commit```

Installing pre-commit with pip
```
pip install pre-commit
```

Installing pre-commit on your local repository. Keep in mind this creates a Github Hook.
```
pre-commit install
```

Now everytime you make a commit, it will run some tests defined on ```.pre-commit-config.yaml``` before allowing your commit.

**Example**
```
$ git commit -m "Example commit"

black....................................................................Passed
pytest-check.............................................................Passed
```


### ⚗️ Using DVC

Download data from the DVC repository(analog to ```git pull```)
```
dvc pull
```

Reproduces the pipeline using DVC
```
dvc repro
```