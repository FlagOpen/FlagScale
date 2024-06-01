# Requirements

This directory contains the various requirement files used for different environments and purposes in the project. 

## Requirements files
The requirements are divided into the following categories:

### Common requirements 

- `requirements.txt`: The main requirements file for the project. It lists the Python packages that are necessary for the project to run.

- `requirements-dev.txt`: This file lists the additional packages needed for development purposes, such as testing and building documentation.

### Training requirements

- `requirements-train.txt`: This file lists the packages needed for training.

- `requirements-train-dev.txt`: This file lists the additional packages needed for training development.

### Inference requirements
- `requirements-inf.txt`: This file lists the packages needed for inference.

- `requirements-inf-dev.txt`: This file lists the additional packages needed for inference development.

## Usage 

Please install the appropriate requirements based on your use case and environment.

### For training

To install the training requirements, run:

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt
```

### For training development

To install the training development requirements, run:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-train.txt
pip install -r requirements-train-dev.txt
```

### For inference 

To install the inference requirements, run:

```bash
pip install -r requirements.txt
pip install -r requirements-inf.txt
```

### For inference development

To install the inference development requirements, run:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-train.txt
pip install -r requirements-train-dev.txt
```
