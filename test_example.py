#!/usr/bin/env python3
"""
Example test script to demonstrate SQLite MCP Server functionality
Note: This is a simple demonstration, not an actual MCP client test
"""

import sqlite3
import os

def test_sqlite_operations():
    """Test basic SQLite operations that the MCP server will support"""

    # Create a test database
    db_path = "test_example.db"

    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("1. Creating table...")
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            email TEXT
        )
    """)
    conn.commit()
    print("✓ Table created")

    print("\n2. Inserting data...")
    cursor.execute("INSERT INTO users (name, age, email) VALUES ('Alice', 25, 'alice@example.com')")
    cursor.execute("INSERT INTO users (name, age, email) VALUES ('Bob', 30, 'bob@example.com')")
    cursor.execute("INSERT INTO users (name, age, email) VALUES ('Charlie', 35, 'charlie@example.com')")
    conn.commit()
    print(f"✓ {cursor.rowcount} rows inserted")

    print("\n3. Listing tables...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"✓ Tables: {tables}")

    print("\n4. Describing table structure...")
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    print("✓ Table structure:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

    print("\n5. Querying data...")
    cursor.execute("SELECT * FROM users WHERE age > 25")
    rows = cursor.fetchall()
    print("✓ Query results:")
    for row in rows:
        print(f"  - {dict(row)}")

    print("\n6. Updating data...")
    cursor.execute("UPDATE users SET age = 31 WHERE name = 'Bob'")
    conn.commit()
    print(f"✓ {cursor.rowcount} row(s) updated")

    print("\n7. Deleting data...")
    cursor.execute("DELETE FROM users WHERE name = 'Charlie'")
    conn.commit()
    print(f"✓ {cursor.rowcount} row(s) deleted")

    print("\n8. Final data...")
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()
    print("✓ Final results:")
    for row in rows:
        print(f"  - {dict(row)}")

    # Cleanup
    conn.close()
    print(f"\n✓ Test completed successfully!")
    print(f"✓ Test database created at: {os.path.abspath(db_path)}")

if __name__ == "__main__":
    test_sqlite_operations()
