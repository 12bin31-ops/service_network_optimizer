"""SQLite 저장소 어댑터.

파이프라인 단계 간 데이터 전달을 메모리가 아닌 DB 로 물질화한다.
- 각 단계를 독립 실행/재실행할 수 있고
- 대시보드와 에이전트가 동일한 단일 소스를 읽는다.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@contextmanager
def get_conn(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """커넥션 컨텍스트 매니저. 외래키 제약을 켜고 커밋/롤백을 보장한다."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    """스키마를 적용한다 (멱등)."""
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_conn(db_path) as conn:
        conn.executescript(ddl)


def write_df(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table: str,
    *,
    replace: bool = True,
) -> int:
    """DataFrame 을 테이블에 적재한다.

    replace=True 이면 기존 행을 지우고 다시 넣는다(단계 재실행 대비).
    스키마에 정의된 컬럼만 남기므로 상류에서 붙은 파생 컬럼은 자동으로 버려진다.
    """
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if not cols:
        raise ValueError(f"알 수 없는 테이블: {table} — init_db 를 먼저 실행하세요")

    missing = [c for c in cols if c not in df.columns]
    payload = df.copy()
    for col in missing:
        payload[col] = None
    payload = payload[cols]

    if replace:
        conn.execute(f"DELETE FROM {table}")
    payload.to_sql(table, conn, if_exists="append", index=False)
    return len(payload)


def read_table(db_path: str | Path, table: str) -> pd.DataFrame:
    with get_conn(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def read_sql(db_path: str | Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    with get_conn(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def table_exists(db_path: str | Path, table: str) -> bool:
    if not Path(db_path).exists():
        return False
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
    return row is not None


def row_count(db_path: str | Path, table: str) -> int:
    if not table_exists(db_path, table):
        return 0
    with get_conn(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
