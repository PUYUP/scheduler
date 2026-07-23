from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from atlazer.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    host: str = settings.db_host
    port: int = settings.db_port
    database: str = settings.db_name
    user: str = settings.db_user
    password: str = settings.db_password

    min_size: int = 2       # -> pool_size
    max_size: int = 10      # -> max_overflow diturunkan dari ini
    command_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls()  # semua sudah datang dari `settings`

    @property
    def dsn(self) -> str:
        # psycopg2 driver, sync
        return f"postgresql+psycopg2://{self.user}:***@{self.host}:{self.port}/{self.database}"

    def _dsn_with_password(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def _dsn_with_aws(self) -> str:
        return f"postgresql+psycopg2://postgres.onasgmdatajeogvsbsom:{self.password}@aws-0-eu-west-1.pooler.supabase.com:6543/{self.database}"

    def _dsn_with_own_db(self) -> str:
        return f"postgresql+psycopg2://postgres.atlanize:{self.password}@{self.host}:{self.port}/{self.database}"

# ---------------------------------------------------------------------------
# DDL — schema definitions
# ---------------------------------------------------------------------------

_DDL = """\
-- Enable extensions (idempotent)
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- convert_to() bawaan Postgres ditandai STABLE (bukan IMMUTABLE) di katalog,
-- padahal untuk encoding tetap ('UTF8') hasilnya deterministic. Kolom
-- GENERATED mensyaratkan ekspresi IMMUTABLE, jadi kita bungkus dengan
-- wrapper ini agar bisa dipakai di content_hash di bawah.
CREATE OR REPLACE FUNCTION immutable_utf8_bytes(text)
RETURNS bytea
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$ SELECT convert_to($1, 'UTF8'); $$;

-- -----------------------------------------------------------------------
-- papers: rich bibliographic metadata
-- -----------------------------------------------------------------------
CREATE TABLE papers (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    doi                     TEXT,
    repository              TEXT            NOT NULL,
    identifier              TEXT            NOT NULL,
    attributes              JSONB           NOT NULL DEFAULT '{}'::jsonb,

    title                   TEXT            NOT NULL,
    abstract                TEXT,
    year                    SMALLINT,
    date_published          DATE,

    authors                 JSONB           NOT NULL DEFAULT '[]'::jsonb,
    affiliations            JSONB           NOT NULL DEFAULT '[]'::jsonb,

    venue                   TEXT,
    venue_type              TEXT,
    publisher               TEXT,
    volume                  TEXT,
    issue                   TEXT,
    pages                   TEXT,

    keywords                TEXT[],
    fields_of_study         TEXT[],
    language                TEXT            NOT NULL DEFAULT 'en',

    pdf_url                 TEXT,
    open_access             BOOLEAN,
    license                 TEXT,

    references_count        INTEGER,
    citations_count         INTEGER,

    processing_tool         TEXT,
    processing_version      TEXT,
    processing_status       TEXT            NOT NULL DEFAULT 'pending',
    error_message           TEXT,

    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT papers_status_check
        CHECK (processing_status IN ('pending', 'indexing', 'done', 'failed')),
    CONSTRAINT papers_venue_type_check
        CHECK (venue_type IN ('journal', 'conference', 'preprint', 'book', 'thesis', 'report', NULL)),

    -- Table constraint langsung (bukan CREATE UNIQUE INDEX terpisah), supaya
    -- bisa dijadikan target composite FK dari document_chunks. Postgres
    -- otomatis buat backing unique index dengan nama yang sama.
    CONSTRAINT papers_repo_identifier_uniq UNIQUE (repository, identifier)
);

-- Indexes on papers -------------------------------------------------------
CREATE INDEX idx_papers_repository ON papers (repository);
CREATE INDEX idx_papers_status     ON papers (processing_status);
CREATE INDEX idx_papers_year        ON papers (year)          WHERE year IS NOT NULL;
CREATE INDEX idx_papers_date_pub    ON papers (date_published) WHERE date_published IS NOT NULL;
CREATE INDEX idx_papers_title_trgm ON papers USING gin (title gin_trgm_ops);
CREATE INDEX idx_papers_attributes   ON papers USING gin (attributes);
CREATE INDEX idx_papers_authors_gin  ON papers USING gin (authors);
CREATE INDEX idx_papers_keywords_gin ON papers USING gin (keywords);
CREATE INDEX idx_papers_fos_gin      ON papers USING gin (fields_of_study);

-- -----------------------------------------------------------------------
-- document_chunks: BGE-M3-embedded text chunks
-- -----------------------------------------------------------------------
CREATE TABLE document_chunks (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id                UUID            NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    repository              TEXT            NOT NULL,
    identifier              TEXT            NOT NULL,

    section                 TEXT            NOT NULL,
    section_order           TEXT            NOT NULL,
    chunk                   TEXT            NOT NULL,

    chunk_type              TEXT            NOT NULL DEFAULT 'body',
    content                 TEXT            NOT NULL,

    word_count              INTEGER         NOT NULL
                                GENERATED ALWAYS AS (
                                    array_length(
                                        string_to_array(
                                            trim(regexp_replace(content, '\\s+', ' ', 'g')),
                                            ' '
                                        ),
                                        1
                                    )
                                ) STORED,

    -- FIXED: content::bytea gagal untuk string yang mengandung pola literal
    -- '\\x...' (mis. LaTeX seperti \\xi, \\times) karena Postgres mencoba
    -- memparsingnya sebagai hex-encoded bytea. convert_to() aman karena
    -- murni mengambil byte UTF-8 tanpa parsing escape, tapi convert_to()
    -- sendiri ditandai STABLE sehingga tidak bisa dipakai langsung di
    -- GENERATED column -> dibungkus via immutable_utf8_bytes().
    content_hash            TEXT            NOT NULL
                                GENERATED ALWAYS AS (
                                    encode(sha256(immutable_utf8_bytes(content)), 'hex')
                                ) STORED,

    -- BGE-M3 dense embedding = 1024 dimensi.
    embedding               vector(1024),
    embedding_model         TEXT,
    embedding_adapter       TEXT,
    embedding_normalized    BOOLEAN         NOT NULL DEFAULT TRUE,
    token_count             INTEGER,

    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chunks_type_check CHECK (
        chunk_type IN ('abstract', 'body', 'conclusion', 'caption', 'equation', 'other')
    ),
    CONSTRAINT chunks_token_count_positive  CHECK (token_count IS NULL OR token_count > 0),
    CONSTRAINT chunks_word_count_positive   CHECK (word_count > 0),

    -- Menjaga repository/identifier di sini selalu konsisten dengan papers,
    -- dan otomatis ikut ter-update kalau papers.repository/identifier
    -- pernah diedit. Valid karena papers_repo_identifier_uniq adalah
    -- constraint asli, bukan sekadar index.
    CONSTRAINT chunks_repo_identifier_fkey FOREIGN KEY (repository, identifier)
        REFERENCES papers (repository, identifier)
        ON UPDATE CASCADE ON DELETE CASCADE
);

-- Indexes on document_chunks -----------------------------------------------
CREATE UNIQUE INDEX idx_chunks_repo_identifier_section_chunk
    ON document_chunks (repository, identifier, section, chunk);

CREATE INDEX idx_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;

CREATE INDEX idx_chunks_paper_order
    ON document_chunks (paper_id, section_order, chunk);

CREATE INDEX idx_chunks_unembedded
    ON document_chunks (paper_id, created_at)
    WHERE embedding IS NULL;

CREATE INDEX idx_chunks_embedding_model
    ON document_chunks (embedding_model, embedding_adapter)
    WHERE embedding IS NOT NULL;

CREATE INDEX idx_chunks_content_hash
    ON document_chunks (content_hash)
    WHERE embedding IS NOT NULL;

CREATE INDEX idx_chunks_type
    ON document_chunks (chunk_type, paper_id);


-- =====================================================================
-- scrape_progress
-- Menyimpan progress pagination (offset "start") untuk scraping
-- increment per (repository, topic). Berfungsi sebagai source of
-- truth yang persistent, sebagai fallback/backup kalau state di Redis
-- hilang (mis. redeploy tanpa persistence, instance Redis baru, dll).
-- =====================================================================

-- gen_random_uuid() sudah built-in di Postgres 13+ (termasuk semua project
-- Supabase saat ini), jadi tidak perlu create extension tambahan.

create table if not exists public.scrape_progress (
    id             uuid primary key default gen_random_uuid(),
    repository     text not null,
    topic          text not null,
    start_offset   integer not null default 0,
    status         text not null default 'active'
                       check (status in ('active', 'done', 'paused')),
    last_result_count integer,           -- jumlah hasil dari fetch terakhir (opsional, untuk observability)
    last_error     text,                 -- pesan error terakhir kalau ada retry/failure
    created_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now(),

    constraint scrape_progress_repo_topic_uniq unique (repository, topic)
);

-- Index untuk query "ambil semua topic milik repository tertentu"
-- (mis. saat inisialisasi ulang serving_topics)
create index if not exists idx_scrape_progress_repository
    on public.scrape_progress (repository);

-- =====================================================================
-- Trigger: auto-update kolom updated_at setiap kali row di-UPDATE
-- =====================================================================

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_scrape_progress_updated_at on public.scrape_progress;

create trigger trg_scrape_progress_updated_at
    before update on public.scrape_progress
    for each row
    execute function public.set_updated_at();

-- =====================================================================
-- Upsert helper (opsional): satu statement untuk "insert atau update
-- start_offset" tanpa perlu SELECT dulu dari sisi aplikasi.
-- Dipanggil dari Python: select public.upsert_scrape_progress('arxiv','cs.AI', 5);
-- =====================================================================

create or replace function public.upsert_scrape_progress(
    p_repository text,
    p_topic text,
    p_start_offset integer,
    p_last_result_count integer default null,
    p_status text default 'active'
)
returns public.scrape_progress
language sql
as $$
    insert into public.scrape_progress (repository, topic, start_offset, last_result_count, status)
    values (p_repository, p_topic, p_start_offset, p_last_result_count, p_status)
    on conflict (repository, topic)
    do update set
        start_offset = excluded.start_offset,
        last_result_count = excluded.last_result_count,
        status = excluded.status
    returning *;
$$;

-- =====================================================================
-- Row Level Security
-- Supabase mengaktifkan akses publik lewat anon/authenticated key
-- secara default kalau RLS tidak diaktifkan. Karena tabel ini adalah
-- state internal untuk Celery worker (biasanya pakai service_role key,
-- yang otomatis bypass RLS), kita kunci total dari sisi anon/authenticated.
-- =====================================================================

alter table public.scrape_progress enable row level security;

-- Tidak ada policy untuk anon/authenticated -> akses hanya lewat
-- service_role key (dipakai backend/Celery worker), sesuai rekomendasi
-- Supabase untuk tabel yang bukan untuk diakses langsung dari client.


-- 1. Tipe data ENUM untuk status
CREATE TYPE answer_status AS ENUM ('DRAFT', 'SUBMITTED', 'EVALUATED');

-- 2. Tabel respons
CREATE TABLE IF NOT EXISTS answers (
    id              UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    challenge_id    UUID    NOT NULL,
    user_id         UUID    NOT NULL, 
    content         TEXT    NOT NULL,
    chunks          JSONB   DEFAULT '[]'::jsonb,
    status          answer_status   NOT NULL DEFAULT 'DRAFT',
    started_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    submitted_at    TIMESTAMPTZ,
    
    CONSTRAINT unique_user_challenge UNIQUE (challenge_id, user_id)
);

-- 3. Trigger untuk update waktu secara otomatis (sama seperti sebelumnya)
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_answers_modtime
BEFORE UPDATE ON answers
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();
"""


# ---------------------------------------------------------------------------
# Pool manager
# ---------------------------------------------------------------------------


class DatabasePool:
    """Sync SQLAlchemy engine + session factory. Thread-safe, aman
    dipakai lintas task Celery tanpa event loop khusus."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self._cfg = config or DatabaseConfig.from_env()
        self._engine = None
        self._SessionLocal: sessionmaker | None = None

    def start(self) -> None:
        if self._engine is not None:
            return
        cfg = self._cfg
        self._engine = create_engine(
            cfg._dsn_with_own_db(),
            poolclass=QueuePool,
            pool_size=cfg.min_size,
            max_overflow=max(cfg.max_size - cfg.min_size, 0),
            pool_pre_ping=True,          # buang koneksi mati otomatis
            pool_recycle=1800,           # hindari koneksi basi (30 menit)
            connect_args={"connect_timeout": int(cfg.command_timeout)},
        )
        self._SessionLocal = sessionmaker(bind=self._engine, expire_on_commit=False)

        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        logger.info("Database engine ready (pool_size=%d)", cfg.min_size)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._SessionLocal = None
            logger.info("Database engine disposed")

    @property
    def engine(self):
        if self._engine is None:
            raise RuntimeError("DatabasePool not started. Call start() first.")
        return self._engine

    def session(self) -> Session:
        if self._SessionLocal is None:
            raise RuntimeError("DatabasePool not started. Call start() first.")
        return self._SessionLocal()

    def ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(_DDL))
        logger.info("Database schema verified / created")

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.debug("Database ping failed: %s", exc)
            return False

    def __enter__(self) -> "DatabasePool":
        self.start()
        return self

    def __exit__(self, *_):
        self.close()
