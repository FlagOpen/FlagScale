## Overview

The test module supports:

1. Unit testing for different backends and operation modes.
2. Functional testing for multiple models, strategies, and heterogeneous hardware.
3. Monitoring incremental code test coverage and viewing test reports online.
4. Ensuring code style consistency.

This section introduces how to use these features.

---

## Unit Testing

### Run Specific Unit Tests Locally

```bash
tests/scripts/unit_tests/test_subset.sh --backend ${BACKEND} --subset ${SUBSET}
```

Please set the following variables:

- `BACKEND`: Specifies the backend for unit testing, either `megatron` or `flagscale`.
- `SUBSET`: The directory for unit tests. Check the directories within `tests/unit_tests` and `megatron/tests/unit_tests` for specific folders. Note: `./` represents the root directory of the above folders.

### Run All Unit Tests Locally

```bash
tests/scripts/unit_tests/test_all.sh
```

### Run Unit Tests Online

When you create a PR using your forked repository, the testing workflow will automatically trigger. Find the corresponding action for your [PR](https://github.com/FlagOpen/FlagScale/pulls) in [All Tests](https://github.com/FlagOpen/FlagScale/actions/workflows/all-tests.yml) to view the results.

### Adding Unit Tests

- **Adding a Single Unit Test File**

  - Directly add a file named `test_${NEW_TEST}.py` in the appropriate directory. `NEW_TEST` refers to the name of the new test.

- **Adding a Unit Test Directory**

  1. Add a test directory and files in the appropriate location. For `flagscale`, the path is `tests/unit_tests/${NEW_FOLD}`. For `megatron`, it's `megatron/tests/unit_tests/${NEW_FOLD}`. `NEW_FOLD` refers to the name of the new test folder.

  2. Update the configuration file `tests/scripts/unit_tests/config.yml` to include configuration for the directory, specifying `ignore`, `type`, and `depth` as needed. Unspecified parameters will default to pre-defined settings. Below is the **configuration file explanation:**

     ```yaml
     # backend: The backend for unit testing, either flagscale or megatron
     megatron:
       # Set the environment required before running unit tests
       set_environment:
         cd megatron; export PYTHONPATH=..:$PYTHONPATH
       # Specify the target folder for test coverage
       coverage:
         core
       # Select different tests for different test directories
       subset:
         # Use default configuration if not shown
         dist_checkpointing:
           # Files to ignore during testing
           ignore: models/test_mamba.py
         models:
           ignore: test_mamba_model.py
         transformer/moe:
           # Test mode:
           # batch (default): Run all test files at once
           # single: Run each test file individually
           # NOTE: Batch mode runs faster, single mode avoids environment interference among tests
           type: single
           ignore: test_upcycling.py
         transformer:
           # Test depth
           # all (default): All test files within the directory
           # Integer: Test files at the specified path depth
           # NOTE: Useful for running test files within a folder, rather than in subdirectories
           depth: 1
         ...
     ```

  3. Online Test Configuration

     Modify the workflow configuration in `.github/workflows/all-tests.yml` to activate online testing:

     ```yaml
     ...

     # Megatron Unit Tests with Matrix
     megatron-unit-tests:
       uses: ./.github/workflows/unit-tests.yml
       strategy:
         matrix:
           subset:
             # Add your new folder if you have a new test directory
             - {NEW_FOLD}
             - data
             - dist_checkpointing
             - distributed
             - fusions
             - inference
             - models
             - pipeline_parallel
             - tensor_parallel
             - transformer/moe
             - transformer
             - ./
       name: "megatron-${{ matrix.subset == './' && 'root' || matrix.subset }}"
       with:
         backend: megatron
         subset: ${{ matrix.subset }}

     ...
     ```

### Viewing Unit Test Coverage Report

- **View Locally:**

  Open the following in a browser:
  `/workspace/report/${ID}/cov-report-${BACKEND}/index.html`

  - `ID`: Use `0` when running locally.
  - `BACKEND`: `flagscale` or `megatron`.

- **View Online:**

  Find the corresponding action for your [PR](https://github.com/FlagOpen/FlagScale/pulls) in [All Tests](https://github.com/FlagOpen/FlagScale/actions/workflows/all-tests.yml), open any unit test under `flagscale` or `megatron`, and click the address provided under `Unit Test Coverage Online Report` to view the test report.

### Viewing Incremental Code Test Coverage Report

- **View Locally:**

  1. Run the command:
     ```bash
     # Ensure unit tests have been run locally before executing this command
     ./tests/scripts/unit_tests/test_coverage.sh --backend ${BACKEND} --status ${STATUS}
     ```

     Please set the following variables:

     - `BACKEND`: `flagscale` or `megatron`.
     - `STATUS`: `online` or `offline`.

  2. View the report:

     Open the following in a browser:
     `/workspace/report/${ID}/diff-cover-report-${BACKEND}.html`
     Use these variables:
     - `ID`: Use `0` when running locally.
     - `BACKEND`: `flagscale` or `megatron`.

- **View Online:**

  Find the corresponding action for your [PR](https://github.com/FlagOpen/FlagScale/pulls) in [All Tests](https://github.com/FlagOpen/FlagScale/actions/workflows/all-tests.yml), open the `flagscale-coverage-test` or `megatron-coverage-test` jobs, and click on the address under `Coverage Online Report` to view the test report online.

---

## Functional Testing

### Run Specific Functional Tests Locally

```bash
tests/scripts/functional_tests/test_model.sh --type ${TYPE} --model ${MODEL}
```

Please set the following variables:

- `TYPE`: The type of functional testing, supporting `train` or `hetero_train`.
- `MODEL`: The model used for functional testing, in conjunction with `TYPE`. Specific models can be found under the `tests/functional_tests/test_cases` directory.

### Run All Functional Tests Locally

```bash
tests/scripts/functional_tests/test_all.sh
```

### Run Functional Tests Online

Find the corresponding action for your [PR](https://github.com/FlagOpen/FlagScale/pulls) in [All Tests](https://github.com/FlagOpen/FlagScale/actions/workflows/all-tests.yml) to view the results.

### Adding Functional Tests

1. Update the functional test configuration file `tests/scripts/functional_tests/config.yml` to include relevant experiment configurations:

   ```yaml
   ...
   # Hardware mode: homogeneous or heterogeneous
   train:
     # Models used
     aquila:
       test_cases:
         # Parallel modes
         -tp2_pp2
         -tp4_pp2
   ...
   ```

2. Add configuration files and test results in the appropriate directory. Directory file structure:

   ```bash
   tests/functional_tests/test_cases/train/aquila
   ├── conf
   │   ├── tp2_pp2.yaml # Environment configuration file
   │   ├── tp4_pp2.yaml
   │   └── train
   │       ├── tp2_pp2.yaml # Model configuration file
   │       └── tp4_pp2.yaml
   ├── results_gold
   │   ├── tp2_pp2.json # Test result file
   │   └── tp4_pp2.json
   └── results_test
   ```

   *Note: We have included data and model files that you can use. For more details, consult the training configuration file of the respective test case. If you need to add your own test data or model files, please contact us.*

3. Modify the yml configuration file in the workflow to enable online testing:

   ```yaml
   ...

   # Functional Tests with Model and Type Matrix
   functional-tests-train:
     needs:
       - megatron-unit-tests
       - flagscale-unit-tests
     uses: ./.github/workflows/functional-tests.yml
     strategy:
       matrix:
         model:
           # Add the new model if applicable
           - {NEW_MODEL}
           - aquila
           - mixtral
     name: "train-${{ matrix.model }}"
     with:
       model: ${{ matrix.model }}
       type: train

   ...
   ```

---

## Code Style Check

1. **Run Manually:**

   ```bash
   ./tests/scripts/format_tests/test_format.sh
   ```

2. **Run via Pre-commit:**

   ```bash
   pre-commit install
   ```
   Code format checks will run automatically upon committing.

3. **Online Format Check:**

   Find the corresponding action for your [PR](https://github.com/FlagOpen/FlagScale/pulls) in the [Format Check](https://github.com/FlagOpen/FlagScale/actions/workflows/format.yml).
