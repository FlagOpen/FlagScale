# FlagScale 硬件适配机制

[英文](./README.md)

## 旧版本

如果想使用旧版本，即2025.05.06之前的patch，请checkout到 `8151afd3cc8ea7076b73844989b6b42c816ea945` 上，查看该commit下的使用方式。

## 一、背景

- **厂商**：在新后端管理机制（由 `subtree` 管理变更为 `submodule` 管理）下简单快速适配 FlagScale。  
- **用户**：更方便地使用不同任务场景下厂商已经适配好的 FlagScale。  
- **框架**：提供适配规范和工具，帮助厂商更好适配和用户更好使用 FlagScale。

## 二、概述

FlagScale 在 **0.8.0** 开始启用新的后端管理方式，不同后端皆以 `submodule` 形式存在于 `FlagScale/third_party/<submodule>` 中。FlagScale 对不同后端做的性能优化、新特性开发等所有适配皆存放在 `FlagScale/backends/<submodule>` 中。  

因此对用户而言，使用和开发 FlagScale 的流程为：

1. **使用 FlagScale 的适配，即 `unpatch`**  
   ```bash
   python tools/patch/unpatch.py --backend submodules
   ```  
   `unpatch` 会自动强制 update `submodule` 至指定 `commit`，并且将 FlagScale 对应后端下的文件以软链或者拷贝形式覆盖 `third_party/<submodule>` 下的相同路径。

2. **开发 FlagScale**  
   在第 1 步基础上对 FlagScale 进行开发。如果是对 `submodule` 的相关修改，建议在 `third_party/<submodule>` 内直接修改，即以 `inplace` 方式进行开发。FlagScale 提供了相应的 `patch` 工具能够将 `third_party/<submodule>` 里 `inplace` 的修改同步回 `FlagScale/backends/<submodule>` 中。

3. **提交 PR 至 FlagScale**  
   如果在 `third_party/<submodule>` 内 `inplace` 修改了后端内容，则需要先执行 `patch`，此后正常进行 git 相关操作即可。（FlagScale 已在 `.gitignore` 中添加了过滤 `third_party` 内容，无需担心本地 `submodule` 的 commit 变化影响到 FlagScale 的 `submodule commit`）  
   ```bash
   python tools/patch/patch.py --backend submodules
   ```  

对于厂商适配，FlagScale 依旧借鉴了 Linux 的 `patch` 文件机制进行不同厂商适配代码的管理和应用。FlagScale 在 `unpatch` 和 `patch` 接口上进行开发，提供：

- 自动将厂商适配好的代码生成 `patch` 文件。
- 帮助用户自动使用厂商适配的 `patch` 文件。
- 未来 FlagScale 也会在合入之前通过自动化 CI 进行检测。

## 三、厂商适配流程

厂商适配流程与上述用户流程相似，仅第三步使用的 `patch` 工具增加了几个额外参数，最后提交 PR 的内容以 `patch` 文件形式进行提交。下面以一个具体示例进行说明：

**示例：适配训练场景下 Megatron-LM 这一训练后端，对 FlagScale 和 Megatron-LM 皆有修改。**

1. **使用 FlagScale 的适配，即 `unpatch`，以此为基础进行厂商的适配。**
   ```bash
   cd FlagScale
   python tools/patch/unpatch.py --backend Megatron-LM
   ```

2. **在 `third_party/Megatron-LM` 里 `inplace` 修改，以及修改 FlagScale 里其他内容。**

3. **使用 `patch`，将厂商适配打包成 `patch` 文件。**  
   `patch` 文件会按后端进行组织，对于厂商而言，FlagScale 也是后端。

   ```bash
   cd FlagScale
   python tools/patch/patch.py --backend Megatron-LM FlagScale --task train --device-type Chip_Vendor --commit <commit>
   ```

   **参数解释：**

   - `backend`：后端。支持多后端输入，如果需要 `patch` 多个后端修改，可输入如 `Megatron-LM FlagScale`。目前仅支持 `{Megatron-LM, vllm, Megatron-Energon, FlagScale}`
   - `task`：任务场景。支持多场景输入，如果该修改支持多个场景，可输入如 `train post_train`。目前仅支持 `{train, inference, post_train}`
   - `device-type`：芯片型号。以 `厂商名_具体型号` 为命名，厂商名开头需要大写。
   - `commit`：基于 FlagScale 某个 `commit` 进行的适配。
   - `key-path`：如果需要对patch文件进行加密，指定密钥文件路径，如果该路径下没有密钥文件，则会自动产生。

   执行该命令后，需要交互式输入以下四个信息，分别是：后端版本、适配的模型、`commit message`（将自动 `add patch` 文件，并以此 `message` 进行 `commit`）、联系方式（可选，适用于需要加密场景）  

   执行完后即可 `push` 到远程分支。  

4. **提交到远程分支，提交 PR，通知 FlagScale 团队进行 review。**

   ```bash
   git push --force origin HEAD:refs/heads/<your_remote_branch>
   ```  

## 四、用户使用流程

厂商 PR 合入后，FlagScale 会维护 `FlagScale/hardware/patch_history.yaml`，该 `yaml` 文件记录了所有厂商的 `patch` 信息。

**示例：patch_history.yaml**

```yaml
Chip_Vendor:
  train:
      FlagScale+Megatron-LM:
        - xxxxxxx
```

用户通过 `unpatch` 工具使用厂商的适配：

```bash
python tools/patch/unpatch.py --backend Megatron-LM FlagScale --task train --device-type Chip_Vendor
```

`unpatch` 出的 FlagScale 就在 `build/<Chip_Vendor>` 目录中。

如果需要对加密后的patch进行解密，需要参考patch文件对应的yaml文件中对应的联系方式，联系厂商获取密钥文件，`unpatch`时需要指定密钥文件路径`key-path`。

## 五、Q&A

**问题 1 ：多个不同 commit 的 patch 如何组织？**

FlagScale 的 `main` 分支上仅保留每个后端的一个 `patch`。对厂商而言，每个后端的升级也应该是兼容性升级。`unpatch` 默认寻找 `patch_history.yaml` 中最新合入的 `commit`，使用此 `commit` 下的 `patch` 文件进行 `unpatch`。如果非兼容性升级，又想使用旧的 `patch`，需要让用户 `unpatch` 时指定 `commit`。

```bash
python tools/patch/unpatch.py --backend Megatron-LM FlagScale --task train --device-type Chip_Vendor --commit <flagscale_commit>
```

**问题 2 ：使用工具失败怎么办？**

请先检查操作流程是否规范。如果没有问题，请在 GitHub FlagScale 仓库下提 `issue` 或者直接联系我们。
