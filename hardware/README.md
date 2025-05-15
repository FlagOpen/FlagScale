# FlagScale Hardware Adaptation Mechanism

[中文](./README_CN.md)

## Legacy Version

If you want to use the old version (i.e., patches before 2025.05.06), please checkout to commit `8151afd3cc8ea7076b73844989b6b42c816ea945` and refer to the usage instructions under that commit.

## Background
- **Vendor**: Quickly and easily adapt to FlagScale under the new backend management mechanism (from subtree management to submodule management).
- **User**: More conveniently use vendor-adapted FlagScale under different task scenarios.
- **Framework**: Provide adaptation specifications and tools to help vendors better adapt and users better use FlagScale.

## Overview
Starting from 0.8.0, FlagScale has adopted a new backend management approach. Each backend exists as a submodule under `FlagScale/third_party/<submodule>`. Performance optimizations and new feature developments for different backends are stored in `FlagScale/backends/<submodule>`. Therefore, for users, the workflow for using and developing FlagScale is as follows:

1. **Use FlagScale adaptation, i.e., `unpatch`**  
```bash
python tools/patch/unpatch.py --backend submodules
```
`unpatch` will automatically force-update the submodule to the specified commit and overwrite files in `third_party/<submodule>` with symlinks or copies from the corresponding `FlagScale/backends/<submodule>` path.

2. **Develop FlagScale**  
Develop FlagScale based on step 1. If modifying submodule-related code, it's recommended to modify directly inside `third_party/<submodule>` (i.e., inplace development). FlagScale provides patch tools to sync inplace changes in `third_party/<submodule>` back to `FlagScale/backends/<submodule>`.

3. **Submit PR to FlagScale**  
If you made inplace modifications inside `third_party/<submodule>`, execute `patch` first, then proceed with normal git operations. (FlagScale already added `.gitignore` rules to filter `third_party` content, so no need to worry about local submodule commit changes affecting FlagScale's submodule commit.)
```bash
python tools/patch/patch.py --backend submodules
```

For vendor adaptation, FlagScale still adopts a patch file mechanism similar to Linux to manage and apply vendor-specific adaptation code. Development was done on `unpatch` and `patch` interfaces, providing:
- Automatic generation of patch files from vendor-adapted code.
- Helping users automatically apply vendor-adapted patch files.
- In the future, FlagScale will also perform automated CI checks before merging.

## Vendor Adaptation Workflow
The vendor adaptation process is similar to the user process, with a few additional parameters for `patch`, and submitting PRs with patch files. Below is a concrete example:

**Example: Adapting Megatron-LM backend for a training scenario with modifications to both FlagScale and Megatron-LM.**

1. **Use FlagScale adaptation (unpatch) as the base for vendor adaptation:**
```bash
cd FlagScale
python tools/patch/unpatch.py --backend Megatron-LM
```

2. **Make inplace modifications inside `third_party/Megatron-LM` and modify other FlagScale files.**  

3. **Use `patch` to package vendor adaptation as a patch file.** Patch files are organized per backend; to vendors, FlagScale is also a backend.
```bash
cd FlagScale
python tools/patch/patch.py --backend Megatron-LM FlagScale --task train --device-type Chip_Vendor --commit <commit>
```
Parameter explanation:
- `backend`: Backend(s). Support multiple backends separated by space.
- `task`: Task scenario(s). Supports multiple tasks separated by space.
- `device-type`: Chip model, named as `Vendor_Model`, with Vendor capitalized.
- `commit`: The FlagScale commit this adaptation is based on.
- `key-path`: If encryption of the patch file is required, specify the path to the key file. If no key file exists at the specified path, one will be generated automatically.

You’ll be prompted to enter 4 interactive inputs: backend version, adapted model, commit message (used for git commit), and contact info (optional).

Then push to a remote branch:
```bash
git push --force origin HEAD:refs/heads/<your_remote_branch>
```

## User Workflow
Once vendor PRs are merged, FlagScale will maintain `FlagScale/hardware/patch_history.yaml` recording all vendor patches.

Example `patch_history.yaml`:
```yaml
Chip_Vendor:
  train:
    FlagScale+Megatron-LM:
      - xxxxxxx
```

Users apply vendor adaptation using `unpatch`:
```bash
python tools/patch/unpatch.py --backend Megatron-LM FlagScale --task train --device-type Chip_Vendor
```
The unpatched FlagScale will be under `build/<Chip_Vendor>`.

To decrypt an encrypted patch, refer to the contact information in the corresponding YAML file of the patch and contact the vendor to obtain the key file. When running `unpatch`, you must specify the key file path using `key-path`.


## Q&A
**Q1: How to manage patches for multiple commits?**  
FlagScale’s main branch only keeps one patch per backend. Vendors are expected to keep backward-compatible upgrades. By default, `unpatch` uses the latest commit in `patch_history.yaml`. If needing an older patch for non-backward-compatible changes, specify commit manually:
```bash
python tools/patch/unpatch.py --backend Megatron-LM FlagScale --task train --device-type Chip_Vendor --commit <flagscale_commit>
```

**Q2: What if the tools fail?**  
First check if your steps follow the workflow. If still problematic, please open an issue in FlagScale GitHub repo or contact us directly.
