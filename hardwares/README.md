## Background

- Vendors: Simplify and expedite the adaptation process for FlagScale.

- Users: Easily utilize FlagScale versions that have already been adapted by vendors.

- FlagScale: Provide guidelines and tools to assist vendors in better adapting FlagScale and help users in seamlessly using these adaptations.

## Introduction

[FlagScale](https://github.com/FlagOpen/FlagScale.git) will adopt a patch mechanism similar to Linux for managing and applying vendor-specific adaptation code, ensuring a simple, safe, and reliable process.

Additionally, FlagScale will provide tools to:
- Generate patches from vendor-adapted code automatically
- Assist users in automatically applying vendor-adapted code.

Vendors must ensure the correctness of the adapted code, and in the future, FlagScale will also implement automated CI checks during integration.

## Operation Process

### Vendor Adaptation

#### Homogeneous Scenario

1. Adapt Code: Vendor A selects a specific commit from the FlagScale main branch for adaptation. The selected commit serves as the base commit ID (e.g., aaaa). After completing the adaptation and validation, all modified code is merged into the local main branch, forming a new current commit ID (e.g., bbbb).

2. Generate Patch: Vendor A uses the tools provided by FlagScale to automatically generate a patch containing the adaptation code between the base commit ID and the current commit ID. This patch is created using the git format-patch command, and the `device-type` option should specify the vendor name and chip model, such as `A_X100`. Example code for generating the patch:

```
cd FlagScale
python tools/patch/patch.py --device-type A_X100 --base-commit-id aaaa --current-commit-id bbbb
```

After generating the patch, the file structure will appear as follows. You can see that Vendor A’s adaptation code is placed in the `FlagScale/hardwares/A_X100` directory, under a folder named after the base commit ID (i.e., aaaa). The patch file contains the actual adaptation content, with the base commit ID being the name of the patch file. If the `current-commit-id` option is not provided, the tool defaults to using the latest commit ID of the current branch as the current-commit-id.

Example of the generated file structure:

```
FlagScale/
├── hardwares/
│   └── A_X100/
│       └── aaaa/
│           └── aaaa.patch
```

3. Submit Patch: After the tool successfully generates the patch, you can directly submit a pull request to the FlagScale main branch. The commit message should follow the commit message of the current-commit-id.

#### Heterogeneous Scenario

1. Adapt Code: Manufacturer B selects a specific commit-id of a required heterogeneous chip A from the FlagScale/hardwares directory (e.g., aaaa) as the `base-commit-id` for adaptation. After completing the adaptation and verification, all modified code is merged into the local main branch, forming a new `current-commit-id` (e.g., bbbb).

2. Generate Patch: The manufacturer uses the tools provided by FlagScale to automatically generate a standardized patch based on the code changes between the base-commit-id and the current-commit-id. Here is an example command:

```
cd FlagScale
python tools/patch/patch.py --device-type A_X100 B_Y100 --base-commit-id aaaa --current-commit-id bbbb
```

The tool will also automatically update the heterogeneous information in FlagScale/flagscale/tools/patch/hetero.txt with the following format:

```
aaaa: A_X100 B_Y100
```

The generated file structure will look like this:

```
FlagScale/
|-- tools/
|   |-- patch/
|       |-- patch.py
|       |-- unpatch.py
|       |-- hetero.txt
|-- hardwares/
|   |-- A_X100/
|       |-- aaaa/
|           |-- aaaa.patch
|   |-- B_Y100/
|       |-- aaaa/
|           |-- aaaa.patch
```

3. Submit Patch: After the patch is successfully generated, the manufacturer can directly submit a pull request to the FlagScale main branch.

## User Workflow

#### Homogeneous Scenario

Users select the desired `commit-id` (e.g., aaaa) from the FlagScale/hardwares/A directory and use the provided tool to automatically generate code that can execute on the vendor's hardware in a specified directory `dir` (e.g., build). Example command:

```
cd FlagScale
python tools/patch/unpatch.py --device-type A_X100 --commit-id aaaa --dir build
```
The generated code structure is as follows. If `dir` is not provided, the tool defaults to placing the generated code in the FlagScale source directory.

Example of the generated file structure:

```
FlagScale/
|-- tools/
|   |-- patch/
|       |-- patch.py
|       |-- unpatch.py
|-- hardwares
|   |-- A_X100/
|       |-- aaaa/
|           |-- aaaa.patch
|   |-- B_Y100/
|       |-- aaaa/
|           |-- aaaa.patch
|-- build/
|   |-- A_X100/
|       |-- FlagScale/
```

#### Heterogeneous Scenario

Users select the appropriate commit-id from `FlagScale/flagscale/patch/hetero.txt` based on the heterogeneous training chip configuration. For example, if using chips A and B for heterogeneous training, select `aaaa: device_type_A device_type_B` from hetero.txt. Then use the provided tool to automatically generate code that can execute on the vendor's hardware in a specified directory `dir` (e.g., build). Example command:

```
cd FlagScale
python tools/patch/unpatch.py --device-type A_X100 B_Y100 --commit-id aaaa --dir build
```

The generated code structure is as follows. The `dir` is required for each unpatch operation. It is recommended to use `build` as the `dir` input; otherwise, the unpatch process may be slower.

```
FlagScale/
|-- tools/
|   |-- patch/
|       |-- patch.py
|       |-- unpatch.py
|-- hardwares
|   |-- A_X100/
|       |-- aaaa/
|           |-- aaaa.patch
|   |-- B_Y100/
|       |-- aaaa/
|           |-- aaaa.patch
|-- build/
|   |-- A_X100/
|       |-- FlagScale/
|   |-- B_Y100/
|       |-- FlagScale/
```

## Q & A

* Question: How to iterate adaptation?

Vendors only need to select the desired `base-commit-id` from the FlagScale main branch for adaptation. The tool will create a new folder named after the commit-id under the vendor directory in hardwares to store the new adaptation content. If the `base-commit-id` has been adapted before, the new content will overwrite the previous adaptation.

* Question: What to do if the tool failed?

Please first check if the operational procedures are followed correctly. If there are no issues with the process, please open an issue in the GitHub FlagScale repository or contact us directly.

* Question: What if the required heterogeneous configuration is not found?

Please open an issue in the GitHub FlagScale repository or contact us directly. We will actively update and encourage vendors to adapt.

* Question: How to ensure the correctness of the generated patch?

During the patch generation process, the tool automatically reverses the patch and compares it with the original adaptation. Adaptation is only confirmed as successful if there are no differences found.
