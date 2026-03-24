import sqlite3
import contextlib
from pathlib import Path

def create_connection(db_file: str) -> None:
    """Create a database connection to a SQLite database."""
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to {db_file} successfully.")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    finally:
        if conn:
            conn.close()

def create_table(db_file: str) -> None:
    """Create the users table."""
    query = '''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            email TEXT
        );
    '''
    try:
        with contextlib.closing(sqlite3.connect(db_file)) as conn:
            with conn:
                conn.execute(query)
                print("Users table created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

def setup_database(name: str) -> None:
    """Create database and table if it does not exist."""
    if Path(name).exists():
        print(f"Database {name} already exists.")
        return

    create_connection(name)
    create_table(name)
    print(f"Database {name} setup complete.")

# Example usage
if __name__ == "__main__":
    setup_database("users.db")
