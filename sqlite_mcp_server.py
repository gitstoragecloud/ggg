#!/usr/bin/env python3
"""
SQLite MCP Server - A simple Model Context Protocol server for SQLite operations
"""

import sqlite3
import json
import os
from typing import Any
from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import mcp.server.stdio


# Global database connection
db_path = None
conn = None


def get_connection():
    """Get or create database connection"""
    global conn, db_path
    if conn is None:
        if db_path is None:
            raise ValueError("Database path not set. Use set_database tool first.")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    return conn


def close_connection():
    """Close database connection"""
    global conn
    if conn:
        conn.close()
        conn = None


# Create the MCP server
app = Server("sqlite-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="set_database",
            description="Set the SQLite database file path to use",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the SQLite database file"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="query",
            description="Execute a SELECT query on the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    }
                },
                "required": ["sql"]
            }
        ),
        Tool(
            name="execute",
            description="Execute an INSERT, UPDATE, or DELETE statement",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL statement to execute"
                    }
                },
                "required": ["sql"]
            }
        ),
        Tool(
            name="list_tables",
            description="List all tables in the database",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="describe_table",
            description="Show the schema/structure of a table",
            inputSchema={
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Name of the table to describe"
                    }
                },
                "required": ["table"]
            }
        ),
        Tool(
            name="create_table",
            description="Create a new table with specified schema",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "CREATE TABLE SQL statement"
                    }
                },
                "required": ["sql"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    global db_path, conn

    try:
        if name == "set_database":
            path = arguments.get("path")
            if not path:
                return [TextContent(type="text", text="Error: path is required")]

            # Expand user path
            path = os.path.expanduser(path)

            # Close existing connection
            close_connection()

            # Set new database path
            db_path = path

            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Test connection
            get_connection()

            return [TextContent(
                type="text",
                text=f"Database set to: {db_path}"
            )]

        elif name == "query":
            sql = arguments.get("sql", "")
            if not sql:
                return [TextContent(type="text", text="Error: sql query is required")]

            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()

            # Convert rows to list of dicts
            results = []
            for row in rows:
                results.append(dict(row))

            return [TextContent(
                type="text",
                text=json.dumps(results, indent=2, ensure_ascii=False)
            )]

        elif name == "execute":
            sql = arguments.get("sql", "")
            if not sql:
                return [TextContent(type="text", text="Error: sql statement is required")]

            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()

            return [TextContent(
                type="text",
                text=f"Success: {cursor.rowcount} row(s) affected"
            )]

        elif name == "list_tables":
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]

            return [TextContent(
                type="text",
                text=json.dumps(tables, indent=2)
            )]

        elif name == "describe_table":
            table = arguments.get("table")
            if not table:
                return [TextContent(type="text", text="Error: table name is required")]

            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

            # Format column information
            schema = []
            for col in columns:
                schema.append({
                    "cid": col[0],
                    "name": col[1],
                    "type": col[2],
                    "notnull": bool(col[3]),
                    "default": col[4],
                    "pk": bool(col[5])
                })

            return [TextContent(
                type="text",
                text=json.dumps(schema, indent=2)
            )]

        elif name == "create_table":
            sql = arguments.get("sql", "")
            if not sql:
                return [TextContent(type="text", text="Error: CREATE TABLE sql is required")]

            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()

            return [TextContent(
                type="text",
                text="Table created successfully"
            )]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


def main():
    """Run the server"""
    import asyncio

    async def run():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
