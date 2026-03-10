"""
Database schema: object_events table (bi-temporal event sourcing),
indexes, RLS policies, and user provisioning.
All DDL runs as app_admin (the table owner).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2.extensions


GROUP_ROLE = "app_user"
ADMIN_ROLE = "app_admin"


def bootstrap_schema(admin_conn: psycopg2.extensions.connection) -> None:
    """Create the object_events table, indexes, and RLS policies. Idempotent."""
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        # ── Table: append-only bi-temporal event log ─────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS object_events (
                event_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_id   UUID NOT NULL,
                version     INT NOT NULL,
                type_name   TEXT NOT NULL,
                owner       TEXT NOT NULL DEFAULT current_user,
                updated_by  TEXT NOT NULL DEFAULT current_user,
                readers     TEXT[] NOT NULL DEFAULT '{}',
                writers     TEXT[] NOT NULL DEFAULT '{}',
                data        JSONB NOT NULL,
                state       TEXT,
                event_type  TEXT NOT NULL DEFAULT 'CREATED',
                event_meta  JSONB,
                tx_time     TIMESTAMPTZ NOT NULL DEFAULT now(),
                valid_from  TIMESTAMPTZ NOT NULL DEFAULT now(),
                valid_to    TIMESTAMPTZ,
                UNIQUE (entity_id, version)
            );
        """)

        # ── Indexes ──────────────────────────────────────────────────
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_entity_version
                ON object_events (entity_id, version DESC);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
                ON object_events (type_name);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_owner
                ON object_events (owner);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_data
                ON object_events USING GIN (data);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_tx_time
                ON object_events (tx_time);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_valid_from
                ON object_events (valid_from);
        """)

        # ── Enable RLS ───────────────────────────────────────────────
        cur.execute("ALTER TABLE object_events ENABLE ROW LEVEL SECURITY;")
        cur.execute("ALTER TABLE object_events FORCE ROW LEVEL SECURITY;")

        # ── Drop existing policies (idempotent re-create) ────────────
        for policy in [
            "admin_all", "user_select", "user_insert", "user_update"
        ]:
            cur.execute(f"DROP POLICY IF EXISTS {policy} ON object_events;")

        # ── Admin policy: full access ────────────────────────────────
        cur.execute(f"""
            CREATE POLICY admin_all ON object_events
                FOR ALL
                TO {ADMIN_ROLE}
                USING (true)
                WITH CHECK (true);
        """)

        # ── User SELECT: owner, or listed in readers/writers ─────────
        cur.execute(f"""
            CREATE POLICY user_select ON object_events
                FOR SELECT
                TO {GROUP_ROLE}
                USING (
                    owner = current_user
                    OR current_user = ANY(readers)
                    OR current_user = ANY(writers)
                );
        """)

        # ── User INSERT: own events, OR writer appending new versions
        # updated_by is always current_user (enforced by DEFAULT, unforgeable)
        # owner stays as original entity owner for RLS visibility
        cur.execute(f"""
            CREATE POLICY user_insert ON object_events
                FOR INSERT
                TO {GROUP_ROLE}
                WITH CHECK (
                    updated_by = current_user
                    AND (
                        owner = current_user
                        OR (
                            current_user = ANY(writers)
                            AND version > 1
                        )
                    )
                );
        """)

        # ── User UPDATE: owner or writer (for sharing operations) ────
        cur.execute(f"""
            CREATE POLICY user_update ON object_events
                FOR UPDATE
                TO {GROUP_ROLE}
                USING (
                    owner = current_user
                    OR current_user = ANY(writers)
                )
                WITH CHECK (
                    owner = current_user
                    OR current_user = ANY(writers)
                );
        """)

        # No DELETE policy — append-only, nobody deletes events

        # ── Grant table permissions to group role ────────────────────
        cur.execute(f"GRANT SELECT, INSERT, UPDATE ON object_events TO {GROUP_ROLE};")

        # ── NOTIFY trigger: fires on every INSERT ────────────────────
        cur.execute("""
            CREATE OR REPLACE FUNCTION notify_object_event() RETURNS trigger AS $$
            BEGIN
                PERFORM pg_notify('object_events', json_build_object(
                    'entity_id', NEW.entity_id,
                    'version', NEW.version,
                    'event_type', NEW.event_type,
                    'type_name', NEW.type_name,
                    'updated_by', NEW.updated_by,
                    'state', NEW.state,
                    'tx_time', NEW.tx_time
                )::text);
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        cur.execute("""
            DROP TRIGGER IF EXISTS object_event_notify ON object_events;
        """)
        cur.execute("""
            CREATE TRIGGER object_event_notify
                AFTER INSERT ON object_events
                FOR EACH ROW EXECUTE FUNCTION notify_object_event();
        """)

        # ── Subscription checkpoints table ───────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS subscription_checkpoints (
                subscriber_id   TEXT PRIMARY KEY,
                last_tx_time    TIMESTAMPTZ NOT NULL,
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute(f"GRANT SELECT, INSERT, UPDATE ON subscription_checkpoints TO {GROUP_ROLE};")


def _provision_user(admin_conn: psycopg2.extensions.connection, username: str, password: str) -> None:
    """
    Create a new PG role for a user. Zero-trust: NOSUPERUSER, NOCREATEDB,
    NOCREATEROLE, NOBYPASSRLS, LOGIN with password, inherits app_user.
    """
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        # Check if role already exists
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (username,))
        if cur.fetchone() is None:
            # Use format string for role name (can't parameterize identifiers)
            # but validate the username first
            _validate_identifier(username)
            cur.execute(
                f"CREATE ROLE \"{username}\" LOGIN PASSWORD %s "
                f"NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS",
                (password,),
            )
            cur.execute(f"GRANT {GROUP_ROLE} TO \"{username}\";")
        else:
            # Update password
            _validate_identifier(username)
            cur.execute(
                f"ALTER ROLE \"{username}\" PASSWORD %s", (password,)
            )


def _validate_identifier(name: str) -> None:
    """Prevent SQL injection in role names."""
    if not name.isalnum() and not all(c.isalnum() or c == '_' for c in name):
        raise ValueError(f"Invalid identifier: {name!r}")
    if len(name) > 63:
        raise ValueError(f"Identifier too long: {name!r}")
