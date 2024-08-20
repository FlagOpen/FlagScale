# FlagScale 硬件适配机制

[英文](./README.md)

## 背景

- 厂商：简单快速适配FlagScale。

- 用户：方便使用厂商已经适配好的FlagScale。

- 框架：提供规范和工具，帮助厂商更好适配和用户更好使用FlagScale。

## 概述

[FlagScale](https://github.com/FlagOpen/FlagScale.git) 将借鉴Linux的patch机制进行不同厂商适配代码的管理和应用，流程简单，安全可靠。

同时，FlagScale将提供工具用于

- 自动将厂商适配好的代码生成patch。

- 帮助用户自动使用厂商适配的代码。

厂商需确保所适配代码的正确性，未来FlagScale也会在合入时通过自动化CI进行检测。

## 操作流程

### 厂商适配流程

#### 同构场景

1. 适配代码：厂商A选择FlagScale main分支的某一个commit进行适配，所选择的commit作为`base-commit-id`（假设aaaa）。适配和验证完成后，将所有修改代码合入到本地main分支，形成新的`current-commit-id` (假设bbbb)。

2. 进行patch: 厂商A使用FlagScale所提供的工具自动将`base-commit-id`到`urrent-commit-id`之间的适配代码，生成符合规范patch，该工具主要利用git format-patch命令。`device-type`需写明厂商名称和芯片型号，例如`A_X100`，示例代码如下：

```
cd FlagScale
python tools/patch/patch.py --device-type A_X100 --base-commit-id aaaa --current-commit-id bbbb
```

* `device-type`：芯片型号。
* `base-commit-id`：厂商适配时的FlagScale commit id。
* `current-commit-id`: 厂商适配时本地修改后的commit id。

生成的文件结构如下所示，可以看出厂商A的适配代码会放到`FlagScale/hardware/A_X100`中以`base-commit-id`为名的文件夹中（即aaaa），其中patch文件存放实际的适配内容，`base-commit-id`即是patch文件名。如果没有提供`current-commit-id`，工具默认以当前分支的最新commit id作为`current-commit-id`。
```
FlagScale/
├── hardware/
│   └── A_X100/
│       └── aaaa/
│           └── aaaa.patch
```

3. 提交patch：工具提示成功生成patch后，可以直接提交pull request到FlagScale main分支。commit-msg以`current-commit-id`的commit-msg为准。

#### 异构场景

1. 适配代码：厂商B从`FlagScale/hardware`目录中选择所需异构芯片A的某个commit-id (假设aaaa)进行适配，所选择的commit作为`base-commit-id`。适配和验证完成后，将所有修改代码合入到本地main分支，形成新的`current-commit-id` (假设bbbb)。

2. 进行patch：厂商使用FlagScale所提供的工具自动将`base-commit-id`到`current-commit-id`之间的适配代码生成符合规范patch，示例代码如下：

```
cd FlagScale
python tools/patch/patch.py --device-type A_X100 B_Y100 --base-commit-id aaaa --current-commit-id bbbb
```

同时，工具会自动将异构信息填入`FlagScale/flagscale/tools/patch/hetero.txt`中，格式如下

```
aaaa: A_X100 B_Y100
```

生成的文件组织结构如下：

```
FlagScale/
|-- tools/
|   |-- patch/
|       |-- patch.py
|       |-- unpatch.py
|       |-- hetero.txt
|-- hardware/
|   |-- A_X100/
|       |-- aaaa/
|           |-- aaaa.patch
|   |-- B_Y100/
|       |-- aaaa/
|           |-- aaaa.patch
```

3. 提交patch：工具提示成功生成patch后，可以直接提交pull request到FlagScale main分支。

### 用户使用流程

#### 同构场景

用户从`FlagScale/hardware/A`目录选择所需`commit-id`（假设aaaa），然后使用FlagScale所提供的工具在指定目录（假设build）自动生成能在厂商硬件上执行的代码。示例代码如下：

```
cd FlagScale
python tools/patch/unpatch.py --device-type A_X100 --commit-id aaaa --dir build
```

* `device-type`：芯片型号。
* `commit-id`：要unpatch回去的commit id。
* `dir`：放置unpatch后的FlagScale的目录地址。

生成代码结构如下所示，如果没有提供`dir`，工具默认将生成代码放在FlagScale源码目录中。

```
FlagScale/
|-- tools/
|   |-- patch/
|       |-- patch.py
|       |-- unpatch.py
|-- hardware
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

#### 异构场景

用户根据异构混训芯片配置，从`FlagScale/flagscale/patch/hetero.txt`中选择合适的`commit-id`，比如使用芯片A和芯片B进行异构混训，可以从hetero.txt中选择`aaaa: device_type_A device_type_B`。然后使用FlagScale所提供的工具在指定目录（假设build）自动将生成能在厂商硬件上执行的代码。示例代码如下：

```
cd FlagScale
python tools/patch/unpatch.py --device-type A_X100 B_Y100 --commit-id aaaa --dir build
```

生成代码结构如下所示

```
FlagScale/
|-- tools/
|   |-- patch/
|       |-- patch.py
|       |-- unpatch.py
|-- hardware
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

* 问题1 ：如何迭代适配？

厂商只需从FlagScale main分支选择所需`base-commit-id`进行适配，工具在`hardware`下的厂商目录生成以commit-id为名的新文件夹存放新的适配内容。如果`base-commit-id`已经适配过，将会覆盖上次适配内容。

* 问题2：使用工具失败怎么办？

请先检查操作流程是否规范。如果没有问题，请在Github FlagScale仓库下提issue或者直接联系我们。

* 问题3：没有找到异构所需配置？

请在Github FlagScale仓库下提issue或者直接联系我们，我们会积极更新并推动厂商适配。

* 问题4：如何确保生产patch的正确性？

工具在生成patch过程中，内部会基于patch自动reverse并与原适配进行对比，没有diff后才提示适配成功。
