# 覆盖率与 FlagCICD 报告

## 收集与上传是两个步骤

现有单元测试入口 `tests/test_utils/runners/run_unit_tests.sh` 已通过 coverage.py
执行各个 torchrun rank，独立写 coverage 数据，待所有进程结束后合并并输出 JSON。
本次沿用这个收集流程和统计范围，不额外执行一遍测试，不改变测试拓扑或覆盖率分母。

实际文件为 `/tmp/coverage-report/coverage.json`，由 `unit_tests_common.yml` 的
`COVERAGE_REPORT_DIR` 指定。GitHub artifact 名包含平台和设备，例如
`coverage-ascend-unit-910c`；FlagCICD job_name 则为
`coverage-ascend-unit-910c-unittest`，避免不同平台、设备的数据混在同一个名称下。
上传接口明确声明 `file_type: coverage`，不能把 coverage JSON 当成 pytest 用例报告。
上传动作固定在 2026-10-15 核对的 FlagOps v2 提交
`466de7d164bf6ca3989c0abaeba1c3a62c68a047`，避免移动标签改变接口。

覆盖率反映测试实际执行过多少被统计的可执行 Python 语句：
例如 100 行可执行语句中执行了 60 行，行覆盖率为 60%。它不是测试通过率，
也不等于正确性，更不等于原生 C++/设备算子内核覆盖率。
目前 coverage 配置的 source 为项目根目录，所以报告可能包含源码、测试和工具中的 Python 文件，
不是只统计某一种芯片专属代码。保留原统计范围便于与历史报告比较。
不同平台走到的代码路径不同，覆盖率可以不同；多个平台的百分比不能相加或直接平均，
也不能仅靠上传更多报告就提高覆盖率。若要统一改为仅统计 flagscale 包，应另行明确统计口径，
再更新基线；本次不借此变更历史覆盖率含义。

## 工作流与 secret

普通 CI 的调用链为：

```text
ci.yml → _platform_pipeline.yml → all_tests_common.yml
                                  ├─ unit_tests_common.yml
                                  └─ functional_tests_benchmark.yml
```

以上调用边逐层显式传递可选 `FLAGCICD_API_TOKEN`，只将该 token 传给报告相关子工作流，
不传给依赖构建、CLI 或训练工作流。`all_tests.yml` 的八个平台手动入口也补齐传递。
镜像构建入口已有 secrets: inherit，继续沿用。
secret 不会自动跨越多层 workflow_call；仓库已配置 token 也不代表末端 job 能收到。

fork PR 的 token 通常不可用，这是正常现象，批准运行不等于开放仓库 secret。
没有 token 时依然运行测试并保存 GitHub coverage artifact，跳过 FlagCICD 上传，
在 job summary 中说明 `Skipped: FLAGCICD_API_TOKEN unavailable`。
通过评论调度的 `ci-request.yml` 明确不传 token：它在默认分支上下文执行显式 PR 来源，
不能因此将默认分支可用的凭据交给待测 PR 代码。

有 token 且报告存在时才上报。真实测试失败时仍尽量保存已生成的覆盖率；
取消、无报告时跳过上传。上传错误使用 continue-on-error 和 fail_on_error: false，
并限制上报步骤为 3 分钟，避免网络请求长期阻塞；不把报告服务问题当成测试失败。测试自身错误仍保留，并由原有汇总门禁判断。
这不会让性能回归、断言失败变成成功。未新增最低覆盖率门槛。

## benchmark 认证修复

原上传接口的 user_id 已停用；现改用 api_token 引用同一个仓库 secret，
并固定 FlagOps 版本。缺 token 时跳过，上传服务异常不阻断测试结果。
这解决原先 `api_token is required and must not be empty` 导致 job 失败的问题，
不解决模型性能低于基线、算子错误或 runner 不可用。

## 内网域名

`/etc/hosts` 是文件。手动验证时，可以由管理员在实际执行上传的环境中配置团队提供的映射。
只修改宿主机 `/etc/hosts` 不保证 Docker job 容器能解析该域名；容器 DNS 和 hosts 是独立的。
也不能仅凭 ping 不通就判定是 DNS 问题，需要分别确认域名解析和 HTTP 端口连通。

CI 提供可选仓库/组织变量 `FLAGCICD_HOST_IP`。设置为团队确认的内网地址时，
只在 unit 和 benchmark 两类上传 job 的 container.options 中追加：

```text
--add-host flagcicd-inner.flagos.net:<FLAGCICD_HOST_IP>
```

保留原有所有设备映射和容器参数。变量留空则完全依赖现有 DNS；已有 DNS 可用时无需配置。
映射只解决域名解析，不建立内网路由，不保证 TCP 8000 可达，也不提供认证。
代码不修改任何服务器或本机的 /etc/hosts，不把内网 IP 固定到各平台公共配置。

## 验收与 PR 更新

1. 本地运行 tests/ci 的 Python 测试和 Node 测试、pre-commit，并与原工作流对比 actionlint。
2. 在无 token 的 fork PR 中确认测试继续、artifact 可下载、summary 明确说明上传跳过。
3. 由维护者在有 token、有内网访问的可信运行中检查上传 HTTP 状态和 FlagCICD 页面，
   确认对应平台/设备的报告确实被解析；CI 绿色本身不代表上传成功。
4. 只有真实测试结果才能证明平台通过；本地调用链测试不验证 token 值、网络和后端解析。

本地 commit 不会更新 GitHub PR；push 到 PR 的 head 分支才更新 PR #1309 并产生 synchronize 事件。
是否需要维护者再次批准 fork 工作流由目标仓库设置决定。
现有 CI 的 concurrency 使用 PR 编号且 cancel-in-progress: true，新运行进入同一组时会取消旧运行。
其他 workflow 的并发组单独计算；并不是所有 Actions 都会无条件停止。

参考：

- [GitHub 多层复用工作流与 secret](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)
- [FlagOps 固定版本的覆盖率上传接口](https://github.com/flagos-ai/FlagOps/blob/466de7d164bf6ca3989c0abaeba1c3a62c68a047/actions/post-pytest-report/action.yml)
- [FlagOps 固定版本的 benchmark 上传接口](https://github.com/flagos-ai/FlagOps/blob/466de7d164bf6ca3989c0abaeba1c3a62c68a047/actions/post-benchmark-report/action.yml)
