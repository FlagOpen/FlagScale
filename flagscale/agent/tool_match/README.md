# Tool Match 模块

智能工具匹配模块，用于根据任务描述自动选择最相关的工具。

## 功能特性

- **多权重评分系统**: 结合语义相似度、关键词匹配和类别相关性进行综合评分
- **智能降级机制**: 当某些组件不可用时（如网络问题、依赖缺失），自动降级到其他评分方式
- **LRU缓存优化**: 使用最近最少使用缓存机制优化查询性能
- **类别管理**: 支持工具分类和按类别搜索
- **灵活配置**: 可配置最大工具数、最小相似度阈值等参数

## 核心组件

### ToolMatcher
智能工具匹配器，负责计算工具与任务的匹配度。

**主要功能:**
- 语义相似度计算（基于sentence-transformers）
- 关键词匹配评分
- 类别相关性评分
- 多权重综合评分
- 降级机制管理

### ToolRegistry
工具注册表，管理所有可用工具并提供搜索接口。

**主要功能:**
- 工具注册和管理
- 按类别组织工具
- 工具搜索和匹配
- 统计信息获取

## 快速开始

```python
from flagscale.agent.tool_match import ToolRegistry

# 创建工具注册表
registry = ToolRegistry(max_tools=5, min_similarity=0.1)

# 注册工具
tool = {
    "function": {
        "name": "read_file",
        "description": "read file content and return text data",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "file path to read"}
            }
        }
    }
}
registry.register_tool(tool, category="file")

# 搜索相关工具
results = registry.search_tools("read file content")
for tool_name, score in results:
    print(f"{tool_name}: {score:.3f}")
```

## 评分系统

### 权重配置
- **语义相似度**: 70% (需要sentence-transformers)
- **关键词匹配**: 20%
- **类别相关性**: 10%

### 降级机制
当某些组件不可用时，系统会自动降级：
- 网络不可用 → 禁用语义相似度
- 依赖缺失 → 禁用对应组件
- 权重自动重新归一化

## 工具格式

```python
tool = {
    "function": {
        "name": "tool_name",           # 工具名称
        "description": "tool description",  # 工具描述
        "parameters": {                # 参数定义
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "param description"}
            }
        }
    }
}
```

## 类别系统

支持以下预定义类别：
- `general`: 通用工具
- `file`: 文件操作
- `search`: 搜索相关
- `data`: 数据处理
- `network`: 网络操作
- `system`: 系统命令

## API 参考

### ToolRegistry

#### 主要方法
- `register_tool(tool, category)`: 注册单个工具
- `register_tools(tools, category)`: 批量注册工具
- `search_tools(query, category=None)`: 搜索工具
- `get_tool_by_name(name)`: 按名称获取工具
- `get_tools_by_category(category)`: 按类别获取工具
- `get_stats()`: 获取统计信息

#### 降级控制
- `set_degradation(component, degraded)`: 设置组件降级状态
- `get_degradation_status()`: 获取降级状态
- `reset_degradation()`: 重置所有降级标志

### ToolMatcher

#### 配置参数
- `max_tools`: 最大返回工具数 (默认: 3)
- `min_similarity`: 最小相似度阈值 (默认: 0.1)

## 依赖要求

- `sentence-transformers`: 语义相似度计算（可选，缺失时自动降级）
- `numpy`: 数值计算
- `torch`: 张量操作（可选）

## 安装依赖

```bash
# 完整功能（推荐）
pip install sentence-transformers torch numpy

# 基础功能（无语义匹配）
pip install numpy
```

## 使用示例

详细使用示例请参考 `fixed_test_tool_match.py` 测试脚本。

## 注意事项

1. 首次使用时会自动下载sentence-transformers模型
2. 网络不可用时系统会自动降级到关键词匹配
3. 建议使用英文关键词进行工具描述以获得更好的匹配效果
4. 缓存机制会自动管理内存使用，无需手动清理
