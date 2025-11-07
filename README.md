# SQLite MCP Server

一个简单的 Model Context Protocol (MCP) server，用于操作 SQLite 数据库。

## 功能特性

- 设置数据库文件路径
- 执行 SELECT 查询
- 执行 INSERT/UPDATE/DELETE 语句
- 列出所有表
- 查看表结构
- 创建新表

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 pip 直接安装
pip install mcp
```

## 使用方法

### 1. 直接运行

```bash
python sqlite_mcp_server.py
```

### 2. 在 Claude Desktop 中配置

编辑 Claude Desktop 配置文件：

**MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

添加以下配置：

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "python",
      "args": ["/path/to/ggg/sqlite_mcp_server.py"]
    }
  }
}
```

### 3. 在其他 MCP 客户端中使用

使用标准的 MCP stdio 协议连接此 server。

## 可用工具

### set_database

设置要操作的 SQLite 数据库文件路径。

```json
{
  "path": "/path/to/database.db"
}
```

### query

执行 SELECT 查询。

```json
{
  "sql": "SELECT * FROM users WHERE age > 18"
}
```

### execute

执行 INSERT、UPDATE 或 DELETE 语句。

```json
{
  "sql": "INSERT INTO users (name, age) VALUES ('Alice', 25)"
}
```

### list_tables

列出数据库中的所有表。

```json
{}
```

### describe_table

查看表的结构信息。

```json
{
  "table": "users"
}
```

### create_table

创建新表。

```json
{
  "sql": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
}
```

## 示例工作流

1. 设置数据库：`set_database` with path: `./test.db`
2. 创建表：`create_table` with SQL: `CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)`
3. 插入数据：`execute` with SQL: `INSERT INTO users (name, age) VALUES ('Alice', 25)`
4. 查询数据：`query` with SQL: `SELECT * FROM users`
5. 查看所有表：`list_tables`
6. 查看表结构：`describe_table` with table: `users`

## 注意事项

- 首次使用时必须先调用 `set_database` 设置数据库路径
- 如果数据库文件不存在，会自动创建
- 所有修改操作（INSERT/UPDATE/DELETE）会自动提交
- 查询结果以 JSON 格式返回

## 开发

```bash
# 安装开发依赖
pip install -e .

# 运行 server
python sqlite_mcp_server.py
```

## 许可证

MIT
