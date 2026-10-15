# FlagScale 五项 CI 改进与验证步骤

本次修改基于 `6c45fac` 的公共 CI 框架，在 `feat/ci-five-improvements` 上实现。
保留 `ci.yml → _platform_pipeline.yml → prepare_dependencies.yml / all_tests_common.yml`
调用关系，平台镜像、runner、训练测试和依赖缓存继续由现有配置控制。

## 1. 改了什么，为什么改

| 文件 | 修改 | 原因 |
| --- | --- | --- |
| `.github/scripts/ci/generate_matrix.py` | 平台配置校验、文件路径匹配、PR 标签选择、选择原因输出 | 原脚本在所有配置缺少 `paths_watch` 时总是全量；解析失败还可能静默退回 |
| `.github/configs/{平台}.yml`、`template.yml` | 添加 `paths_watch` | 将平台专属路径的归属放在现有配置中，增加平台时有示例可循 |
| `.github/workflows/ci.yml` | 标签事件、CPU 控制逻辑测试、独立安装 PyYAML、`CI Complete` 总汇总 | 标签变化要重新调度；每个 job 环境独立；合并门禁需要固定名字 |
| `.github/scripts/ci/check_results.py` | 验证 lint、矩阵和平台流水线结果 | 防止失败、取消和意外跳过被当成成功 |
| `.github/workflows/all_tests_common.yml` | 配置读取失败纳入汇总 | 防止配置失败导致后续测试全部 skipped 后汇总仍成功 |
| `.github/workflows/test_failure_analysis.yml`、`retry_policy.py` | 有限重试已知临时故障 | 旧工作流监听的平台名称没有对接新 CI，且原方案依赖模型服务、缺少次数上限 |
| `.github/workflows/slash-command.yml`、`slash_command.js` | 校验评论命令、用户实际仓库权限，调度补测 | 不需要人工打开 Actions 页面选择平台 |
| `.github/workflows/ci-request.yml`、`request_source.js` | 校验 PR 当前 SHA，执行独立的定向补测 | 防止评论事件默认分支上下文导致测错代码，也避免补测替代正式门禁 |
| `_platform_pipeline.yml`、依赖准备和下游公共测试工作流 | 传递可选 `source_repository`、`source_sha` | PR 仓库和不可变提交必须传到每个实际 checkout，包括 checkout 重试 |
| `tests/ci/` | 不依赖 GPU、NPU、torch 的 Python 和 Node 测试 | 调度逻辑可以在本地和 GitHub 托管 CPU runner 上快速验收 |

新增来源输入默认留空，已有 PR、push、schedule 和旧手动入口不需要额外传参。
依赖准备现在也明确使用与测试一致的 PR head；Megatron/TE 自己的版本配置不受来源输入影响。
当前已禁用的推理、服务和 RL 测试维持原有启用状态。

## 2. 路径矩阵和标签规则

支持 8 个配置平台：ascend、cuda、enflame、hygon、kunlunxin、metax、musa、ppu。
平台专属路径覆盖以下目录或文件：

```yaml
paths_watch:
  - docker/ascend/
  - tools/install/ascend/
  - requirements/ascend/
  - .github/scripts/set_env_ascend.sh
  - tests/test_utils/config/platforms/ascend.yaml
  - .github/configs/ascend.yml
```

以 `/` 结尾的规则匹配仓库根目录下的目录前缀；其他规则使用区分大小写的
Python `fnmatchcase`，没有 glob 元字符时相当于匹配完整文件名。其 `*` 可以跨 `/`，
不是 gitignore 语义。修改配置时要与此语义保持一致。

选择规则：

1. push、schedule、普通 workflow_dispatch：全量。
2. PR 无 diff 或读取 diff 失败：全量，并记录原因。
3. PR 仅涉及 `docs/` 和 `.md`：没有平台标签则跳过平台测试，仍运行 lint 和总门禁。
4. 任一平台缺少路径规则：非文档改动保守全量。
5. 任一非文档文件不属于已声明的平台专属路径：视为公共/未知改动，全量。
6. 其余平台专属改动：有标签时按标签选择，无标签按路径选择；多个平台取并集。

标签为 `platform:ascend` 等，`platform:all` 表示全部平台。未知的 `platform:` 标签明确报错。
公共代码、公共测试、公共安装工具和 CI 工作流的改动不能用平台标签缩小范围。
平台配置允许维护者覆盖专属改动的测试范围，因此应由团队管理标签使用。
只增删普通标签也会触发一次 CI，以保持标签状态与运行一致。

不再在 PR 工作流顶层使用 `paths-ignore`，以防 required check 因整条工作流未启动而长期 Pending。
`examples/` 下非文档文件也保守全量，避免忽略测试所依赖的模型示例或配置。

## 3. 总门禁

正式工作流的固定检查名为 `CI Complete`。

- lint 和矩阵生成必须成功。
- 矩阵非空时，平台矩阵 job 必须成功。
- 矩阵为空时，只接受有明确选择结果的平台测试 skipped。
- 任一上游失败或取消，总门禁失败。
- 公共测试内部的配置读取失败也会让平台测试失败。

代码不会自动修改仓库 Rulesets / Branch protection。
管理员在组内验证后，应选择实际出现的 `CI Complete` check 作为 required check。
不要将 `Requested CI Complete` 配置成正式合并门禁。
如果仓库还单独要求旧检查，是否移除旧要求由管理员结合现有规则确认。

## 4. 自动重试

监听工作流显示名称 `CI` 和 `all_tests` 的 `completed` 事件，只处理第一次运行的 failure。
只在每个实际失败 job 的日志都有已知网络故障证据、且没有确定性错误标记时，执行
`gh run rerun RUN_ID --failed`。总汇总 job 的失败由实际失败 job 解释。

已知证据包括 ReadTimeout、ConnectTimeout、DNS 解析失败、连接重置和部分网关错误。
断言失败、语法错误、缺包、编译错误、显存不足，以及普通分布式超时不自动重试。
日志不可读、混合真实失败或证据不足也不自动重试。
执行前再次检查 attempt；自动流程最多再跑一次，第二次失败交由人工处理。

本方案用确定性规则替换原 Claude 分析/评论逻辑，不需要新增 ANTHROPIC 密钥。
它不宣称识别所有 flaky test；定向 `Requested CI` 补测暂由维护者按需重新请求。

重试控制器具有 `actions: write`，因此只 checkout 默认分支的可信策略脚本，
不 checkout PR 代码，不执行日志中的指令，不从 PR artifact 加载脚本。

## 5. 评论定向补测

维护者在 PR 普通评论区单独发送一行：

```text
/ci run ascend
```

或者 `/ci run cuda`、`/ci run ppu`、`/ci run all`。

- 只处理新建的 PR 评论，命令必须完整匹配。
- 查询评论者当前实际仓库权限；仅 write、maintain、admin 可发起。
- 从默认分支的配置验证平台名，不允许任意路径或脚本参数。
- 读取 PR 当前 head SHA，dispatch 默认分支上的 `ci-request.yml`。
- 补测开始时再次确认 PR 仍打开、SHA 未变化；如果变了，要求重新发命令。
- 来源仓库从 PR API 获取，不信任评论提供的仓库或 SHA。
- 测试中所有 FlagScale checkout 都使用这个固定 SHA；后续 PR 更新不会改变已启动补测的代码。
- 测试 job 使用只读仓库权限；有写权限的评论控制器只执行默认分支代码。
- 返回平台、完整 SHA 和工作流运行列表链接；每个补测运行名称也包含 PR、平台、SHA。
- 补测结果单独显示在 Requested CI 工作流中，不自动发布为 PR head 的正式 required check。

手动 dispatch `ci-request.yml` 也可以使用 `pr_number`、`head_sha`、`platform` 三个参数；
手动操作者本身需要具有调度 Actions 的权限。

`issue_comment` 和 `workflow_run` 入口需要先进入默认分支才能完整验收。
功能分支上的本地 mock 测试不等于 GitHub 上的真实评论事件已经可用。
先在组内仓库审核、受控启用，再考虑向官方提交。

## 6. 本地验证

以下命令都在仓库根目录运行，不需要加速卡：

```bash
python3 -m venv /tmp/flagscale-ci-venv
/tmp/flagscale-ci-venv/bin/python -m pip install PyYAML==6.0.2 pytest==8.3.5
/tmp/flagscale-ci-venv/bin/python -m pytest --confcutdir=tests/ci tests/ci -q
node --test tests/ci/slash_command.test.cjs
```

验证平台选择：

```bash
/tmp/flagscale-ci-venv/bin/python .github/scripts/ci/generate_matrix.py \
  --event-name pull_request --changed-files docker/ascend/Dockerfile.train

/tmp/flagscale-ci-venv/bin/python .github/scripts/ci/generate_matrix.py \
  --event-name pull_request --changed-files flagscale/runner/runner.py

/tmp/flagscale-ci-venv/bin/python .github/scripts/ci/generate_matrix.py \
  --event-name pull_request --changed-files README.md
```

预期分别为 Ascend、全部 8 平台、空平台矩阵，并输出 `reason`。
`--base-ref` 接受实际 Git revision，默认是 `origin/main`；PR 在 Actions 中使用事件携带的 base/head SHA。
使用 `--no-renames` 和 NUL 分隔的 diff 同时检查重命名前后的路径。

本地还应运行仓库 pre-commit 和 actionlint。现有公共工作流中动态 container volumes
及常量禁用条件可能被 actionlint 报告；需要与修改前比较，不能把这些原有告警当成全链路成功证明。

## 7. 对应服务器测试

这次修改涉及 CI 调度，不能只运行模型后就认定五项功能都通过。
推荐先检查 CPU 控制逻辑，再在团队已验证的容器/运行环境中执行原有平台测试。

1. 确认服务器工作目录没有待保留修改，然后获取本功能分支，记录 `git rev-parse HEAD`。
   未提交、未推送的本地修改不会自动出现在服务器上；需要先在本机提交并推送或使用明确的代码传输方式。
2. 核对 `.github/configs/<platform>.yml` 中的镜像、设备映射、数据卷、runner labels、依赖版本。
3. 按 `prepare_dependencies.yml` 准备并验证 Megatron/TE 运行时，使用既有平台测试环境初始化流程。
4. 在已经初始化好的容器内，运行原有单元测试入口。例如 Ascend：

   ```bash
   bash tests/test_utils/runners/run_unit_tests.sh --platform ascend
   ```

   这条命令依赖前面的镜像、设备、数据和运行时准备，不是宿主机上一条即可完成的环境安装命令。
5. 按 `all_tests_common.yml` 选择实际启用的训练、异构训练、benchmark 等功能测试；不要用单元测试代替全套功能测试。
6. 保存提交 SHA、镜像标识、执行命令、退出码和日志。网络故障与真实测试失败分开记录。

## 8. 组内 PR 与后续启用顺序

本次本地修改不自动 commit、push、创建 PR 或触发 Actions。
审核本地 diff 后，可提交并推送到现有 `feat/ci-five-improvements`，再创建组内 PR：

- base：`BrianPei/FlagScale:main`
- head：`BrianPei/FlagScale:feat/ci-five-improvements`

组内 runner 和 secrets 不会因为 fork 关系自动继承官方设置。
普通 PR CI 验证后，在组内默认分支启用控制器，再验证标签事件、评论命令、拒绝无权限用户、
固定提交、防止过期请求、一次重试和总门禁。先观察真实 check 名称，再配置 required check。

只有组内审核、服务器验证和事件链路验收完成后，再安排向 `flagos-ai/FlagScale:main` 提交 PR。
批准运行 fork PR 工作流与审核合并是两个动作，是否需要维护者批准运行由目标仓库设置决定。

参考：

- [GitHub 工作流触发事件](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
- [Required checks 与路径过滤](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)
- [批准 fork PR 的工作流运行](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/approve-runs-from-forks)
