from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from curiosift_miner.config.settings import settings

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


# ---------------------------------------------------------------------------
# DDL — schema definitions
# ---------------------------------------------------------------------------

_DDL = """\
-- Enable pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

-- -----------------------------------------------------------------------
-- papers: rich bibliographic metadata
--
-- Design decisions
-- ----------------
-- * paper_id   : caller-controlled stable key (e.g. SHA-256 of DOI / arXiv ID).
-- * doi / arxiv_id / pmid / pmcid / s2_id : well-known external identifiers,
--   nullable because many sources only expose a subset.
-- * authors    : JSONB array — flexible enough to store name, affiliation,
--   ORCID and email per author without a separate join table.
--   Shape: [{"name": "…", "first": "…", "last": "…",
--            "affiliation": "…", "orcid": "…", "email": "…"}, …]
-- * affiliations : JSONB array of institution objects extracted by GROBID.
--   Shape: [{"name": "…", "department": "…", "country": "…"}, …]
-- * references_count / citations_count : bibliometric signals, populated
--   asynchronously from Semantic Scholar / OpenAlex.
-- * processing_status : lifecycle state machine —
--   pending → indexing → done | failed
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS papers (
    -- Primary key --------------------------------------------------------
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),

    -- External identifiers (all optional, unique when set) ---------------
    doi                     TEXT,                            -- e.g. 10.1145/3442188.3445922
    repository              TEXT            NOT NULL,        -- repo source: arXiv, PubMed, Semantic Scholar
    identifier              TEXT            NOT NULL,        -- unique identifier within the repository e.g. 10.1145/3442188.3445922, 2301.07041, 12345678
    metadata                JSONB           NOT NULL DEFAULT '{}'::jsonb, -- arbitrary repository-specific raw fields for forward-compatibility

    -- Core bibliographic -------------------------------------------------
    title               TEXT            NOT NULL,
    abstract            TEXT,
    year                SMALLINT,                        -- publication year (fast range filter)
    date_published      DATE,                            -- full date when available

    -- People & institutions (structured JSONB) ---------------------------
    authors             JSONB           NOT NULL DEFAULT '[]'::jsonb,
    -- [{name, first, last, affiliation, orcid, email}, …]

    affiliations        JSONB           NOT NULL DEFAULT '[]'::jsonb,
    -- [{name, department, country}, …]

    -- Venue / publication ------------------------------------------------
    venue               TEXT,                            -- journal or conference name
    venue_type          TEXT,                            -- 'journal' | 'conference' | 'preprint' | 'book'
    publisher           TEXT,
    volume              TEXT,
    issue               TEXT,
    pages               TEXT,                            -- e.g. "1–14" or "1432"

    -- Classification & discovery -----------------------------------------
    keywords            TEXT[],                          -- author-supplied keywords
    fields_of_study     TEXT[],                          -- e.g. {'Computer Science','NLP'}
    language            TEXT            NOT NULL DEFAULT 'en',

    -- Access & availability ----------------------------------------------
    pdf_url             TEXT,
    open_access         BOOLEAN,
    license             TEXT,                            -- SPDX or free-text

    -- Bibliometric signals (populated asynchronously) --------------------
    references_count    INTEGER,
    citations_count     INTEGER,

    -- Processing lifecycle -----------------------------------------------
    processing_tool     TEXT,
    processing_version  TEXT,
    processing_status   TEXT            NOT NULL DEFAULT 'pending',
    -- 'pending' → 'indexing' → 'done' | 'failed'
    error_message       TEXT,

    -- Timestamps ---------------------------------------------------------
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Constraints --------------------------------------------------------
    CONSTRAINT papers_status_check
        CHECK (processing_status IN ('pending', 'indexing', 'done', 'failed')),
    CONSTRAINT papers_venue_type_check
        CHECK (venue_type IN ('journal', 'conference', 'preprint', 'book', 'thesis', 'report', NULL))
);

-- Indexes on papers ------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_repo_identifier
    ON papers (repository, identifier);

CREATE INDEX IF NOT EXISTS idx_papers_repository
    ON papers (repository);

CREATE INDEX IF NOT EXISTS idx_papers_metadata_gin
    ON papers USING gin (metadata);

-- Date / year range filtering
CREATE INDEX IF NOT EXISTS idx_papers_year        ON papers (year)          WHERE year IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_date_pub    ON papers (date_published) WHERE date_published IS NOT NULL;

-- Processing queue
CREATE INDEX IF NOT EXISTS idx_papers_status     ON papers (processing_status);

-- Full-text search on title (pg_trgm trigram index for ILIKE / similarity)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_papers_title_trgm
    ON papers USING gin (title gin_trgm_ops);

-- GIN index for JSONB authors array (supports @> containment queries)
CREATE INDEX IF NOT EXISTS idx_papers_authors_gin
    ON papers USING gin (authors);

-- GIN index for keyword and fields-of-study arrays
CREATE INDEX IF NOT EXISTS idx_papers_keywords_gin
    ON papers USING gin (keywords);
CREATE INDEX IF NOT EXISTS idx_papers_fos_gin
    ON papers USING gin (fields_of_study);

-- -----------------------------------------------------------------------
-- document_chunks: BGE-M3-embedded text chunks
--
-- Production design decisions
-- ---------------------------
-- * section_order : 0-based global reading order of sections within the
--   paper. Lets callers reconstruct document structure without re-parsing.
--
-- * content_hash  : SHA-256 of the raw content string, stored as a
--   generated column. Enables skip-on-no-change re-indexing:
--     SELECT id FROM document_chunks
--     WHERE paper_id = $1 AND content_hash = $2 AND embedding IS NOT NULL;
--
-- * word_count    : stored computed; avoids recomputing at query time and
--   supports filtering short/long chunks without loading content.
--
-- * embedding_model / embedding_adapter : provenance columns. When the
--   model is upgraded, stale chunks are found instantly:
--     SELECT * FROM document_chunks
--     WHERE embedding_model != 'allenai/specter2_base';
--
-- * token_count   : actual tokens fed to the encoder. A value equal to
--   max_length (default 512) means the chunk was hard-truncated and may
--   have lost trailing content. Useful for quality auditing.
--
-- * embedding_normalized : explicit flag; TRUE means dot-product distance
--   equals cosine distance, which halves query cost in pgvector.
--
-- * HNSW vs IVFFlat : IVFFlat requires a separate VACUUM+ANALYZE training
--   step, delivers poor recall on small tables, and needs lists retuned as
--   the table grows. HNSW (available since pgvector 0.5 / Postgres 15) is
--   trained incrementally, works from the first row, and gives consistently
--   better recall at the same ef_search budget.
--   m=16 (connections per graph node) and ef_construction=64 (build-time
--   candidate list) are conservative, high-quality defaults.
--
-- * Partial indexes   : all expensive indexes are filtered WHERE
--   embedding IS NOT NULL so they never bloat on un-embedded rows.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id                UUID            NOT NULL REFERENCES papers(id) ON DELETE CASCADE,

    -- Document position -----------------------------------------------
    section                 TEXT            NOT NULL,
    section_order           TEXT            NOT NULL,
    -- 0-based global position of this section across the paper
    chunk                   TEXT            NOT NULL,
    -- 0-based index of this chunk within the section

    -- Content ---------------------------------------------------------
    chunk_type              TEXT            NOT NULL DEFAULT 'body',
    -- 'abstract' | 'body' | 'conclusion' | 'caption' | 'equation' | 'other'
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
    -- Recomputed from content on every write; never stale.

    content_hash            TEXT            NOT NULL
                                GENERATED ALWAYS AS (
                                    encode(sha256(content::bytea), 'hex')
                                ) STORED,
    -- SHA-256 hex digest of the raw content string.

    -- Embedding -------------------------------------------------------
    embedding               vector(768),
    -- NULL until the embedding worker completes this chunk.

    embedding_model         TEXT,
    -- HuggingFace base model ID, e.g. 'allenai/specter2_base'

    embedding_adapter       TEXT,
    -- Active adapter, e.g. 'allenai/specter2' or 'allenai/specter2_adhoc_query'

    embedding_normalized    BOOLEAN         NOT NULL DEFAULT TRUE,
    -- TRUE ⟹ L2-normalised; cosine sim == dot product (faster at query time)

    token_count             INTEGER,
    -- Tokens consumed by the tokenizer. token_count = 512 ⟹ truncation occurred.

    -- Timestamps ------------------------------------------------------
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Constraints -----------------------------------------------------
    UNIQUE (paper_id, section, chunk),

    CONSTRAINT chunks_type_check CHECK (
        chunk_type IN ('abstract', 'body', 'conclusion', 'caption', 'equation', 'other')
    ),
    CONSTRAINT chunks_token_count_positive  CHECK (token_count IS NULL OR token_count > 0),
    CONSTRAINT chunks_word_count_positive   CHECK (word_count > 0)
);

-- Indexes on document_chunks ------------------------------------------

-- HNSW vector index for cosine ANN search (partial: embedded rows only).
-- No pre-training required. Consistent high recall from first insert.
--   m=16              → connections per graph node (memory ↔ recall)
--   ef_construction=64 → build-time candidate list (quality ↔ speed)
-- At query time tune SET hnsw.ef_search = 100; for higher recall.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;

-- Reading-order retrieval: reconstruct a paper's full chunk sequence.
CREATE INDEX IF NOT EXISTS idx_chunks_paper_order
    ON document_chunks (paper_id, section_order, chunk);

-- Embedding queue: find chunks still waiting for a vector.
CREATE INDEX IF NOT EXISTS idx_chunks_unembedded
    ON document_chunks (paper_id, created_at)
    WHERE embedding IS NULL;

-- Model staleness scan: find chunks embedded with an old model/adapter.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model
    ON document_chunks (embedding_model, embedding_adapter)
    WHERE embedding IS NOT NULL;

-- Deduplication: skip re-embedding when content is unchanged.
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash
    ON document_chunks (content_hash)
    WHERE embedding IS NOT NULL;

-- Chunk-type filtering (e.g. abstract-only retrieval).
CREATE INDEX IF NOT EXISTS idx_chunks_type
    ON document_chunks (chunk_type, paper_id);
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
            cfg._dsn_with_password(),
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
