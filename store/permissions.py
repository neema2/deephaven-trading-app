"""
Permissions helpers — share/unshare entities between users.
Sharing updates readers/writers on ALL versions of an entity.
All operations run as the entity owner (enforced by RLS).
"""

from __future__ import annotations

import psycopg2.extensions



def share_read(conn: psycopg2.extensions.connection, entity_id: str, to_user: str) -> bool:
    """Grant read access on all versions of an entity to another user.
    Only the owner (or a writer) can do this — RLS enforces."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE object_events
            SET readers = array_append(readers, %s)
            WHERE entity_id = %s
              AND NOT (%s = ANY(readers))
            RETURNING entity_id
            """,
            (to_user, entity_id, to_user),
        )
        return cur.fetchone() is not None


def share_write(conn: psycopg2.extensions.connection, entity_id: str, to_user: str) -> bool:
    """Grant read+write access on all versions of an entity to another user.
    Only the owner (or a writer) can do this — RLS enforces."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE object_events
            SET writers = array_append(writers, %s)
            WHERE entity_id = %s
              AND NOT (%s = ANY(writers))
            RETURNING entity_id
            """,
            (to_user, entity_id, to_user),
        )
        return cur.fetchone() is not None


def unshare_read(conn: psycopg2.extensions.connection, entity_id: str, from_user: str) -> bool:
    """Revoke read access from a user on all versions."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE object_events
            SET readers = array_remove(readers, %s)
            WHERE entity_id = %s
            RETURNING entity_id
            """,
            (from_user, entity_id),
        )
        return cur.fetchone() is not None


def unshare_write(conn: psycopg2.extensions.connection, entity_id: str, from_user: str) -> bool:
    """Revoke write access from a user on all versions."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE object_events
            SET writers = array_remove(writers, %s)
            WHERE entity_id = %s
            RETURNING entity_id
            """,
            (from_user, entity_id),
        )
        return cur.fetchone() is not None


def list_shared_with(conn: psycopg2.extensions.connection, entity_id: str) -> dict | None:
    """List who has read/write access to an entity (from latest version).
    Returns dict with readers/writers lists. Only visible if you can see the entity."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT readers, writers FROM object_events
            WHERE entity_id = %s
            ORDER BY version DESC LIMIT 1
            """,
            (entity_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {"readers": row[0] or [], "writers": row[1] or []}
