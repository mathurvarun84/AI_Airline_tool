"""Test Supabase PostgreSQL connectivity. Run with env vars set — do not hardcode passwords."""

import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT", "5432")
    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    dbname = os.getenv("SUPABASE_DB_NAME", "postgres")

    missing = [
        name
        for name, value in {
            "SUPABASE_DB_HOST": host,
            "SUPABASE_DB_USER": user,
            "SUPABASE_DB_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        print("Missing env vars:", ", ".join(missing))
        return 1

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=10,
            sslmode="require",
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        print("CONNECTION_OK", cur.fetchone())
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'flights'"
        )
        print("flights_table_exists=", cur.fetchone()[0] == 1)
        cur.close()
        conn.close()
        return 0
    except Exception as exc:
        print("CONNECTION_FAILED", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
