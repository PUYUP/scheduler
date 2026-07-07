"""
config/settings.py
───────────────────
Centralised settings — loaded from environment / .env file.
"""

from __future__ import annotations

from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Redis ──────────────────────────────────────────────────
    redis_host: str     = "redis"
    redis_port: int     = 6379
    redis_db: int       = 0
    redis_password: str = ""

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── ArXiv Scraping ─────────────────────────────────────────
    arxiv_topics: List[str] = Field(
        default=[
            "cs.AI",    # (Artificial Intelligence)
            # "cs.AR",    # (Hardware Architecture)
            # "cs.CC",    # (Computational Complexity)
            # "cs.CE",    # (Computational Engineering, Finance, and Science)
            # "cs.CG",    # (Computational Geometry)
            # "cs.CL",    # (Computation and Language - NLP)
            # "cs.CR",    # (Cryptography and Security)
            # "cs.CV",    # (Computer Vision)
            # "cs.CY",    # (Cybersecurity)
            # "cs.DB",    # (Databases)
            # "cs.DC",    # (Distributed, Parallel, and Cluster Computing)
            # "cs.DL",    # (Deep Learning)
            # "cs.DM",    # (Discrete Mathematics)
            "cs.DS",    # (Data Structures and Algorithms)
            # "cs.ET",    # (Emerging Technologies)
            # "cs.FL",    # (Formal Languages and Automata Theory)
            # "cs.GL",    # (General Literature)
            # "cs.GR",    # (Graphics)
            # "cs.GT",    # (Computer Science and Game Theory)
            # "cs.HC",    # (Human-Computer Interaction)
            # "cs.IR",    # (Information Retrieval)
            # "cs.IT",    # (Information Theory)
            # "cs.LG",    # (Machine Learning)
            # "cs.LO",    # (Logic in Computer Science)
            # "cs.MA",    # (Mathematical Software)
            # "cs.MM",    # (Multimedia)
            # "cs.MS",    # (Mathematical Software)
            # "cs.NA",    # (Numerical Analysis)
            # "cs.NE",    # (Neural and Evolutionary Computing)
            # "cs.NI",    # (Networking and Internet Architecture)
            # "cs.OH",    # (Other Computer Science Fields)
            # "cs.OS",    # (Operating Systems)
            # "cs.PF",    # (Performance)
            # "cs.PL",    # (Programming Languages)
            # "cs.RO",    # (Robotics)
            # "cs.SD",    # (Sound)
            # "cs.SE",    # (Software Engineering)
            # "cs.SI",    # (Social and Information Networks)
            # "cs.SY",    # (Systems and Control Theory)

            # "econ.EM",  # (Econometrics)
            # "econ.GN",  # (General Economics)
            # "econ.TH",  # (Theoretical Economics)

            # "eess.AS",  # (Audio and Speech Processing)
            # "eess.IV",  # (Image and Video Processing)
            # "eess.SP",  # (Signal Processing)
            # "eess.SY",  # (Systems and Control Theory)

            # "math.AC",  # (Commutative Algebra)
            # "math.AG",  # (Algebraic Geometry)
            # "math.AP",  # (Analysis of PDEs)
            # "math.AT",  # (Algebraic Topology)
            # "math.CA",  # (Classical Analysis and ODEs)
            # "math.CO",  # (Combinatorics)
            # "math.CT",  # (Category Theory)
            # "math.CV",  # (Complex Variables)
            # "math.DG",  # (Differential Geometry)
            "math.DS",  # (Dynamical Systems)
            # "math.FA",  # (Functional Analysis)
            # "math.GM",  # (General Mathematics)
            # "math.GN",  # (General Topology)
            # "math.GR",  # (Group Theory)
            # "math.GT",  # (Geometric Topology)
            # "math.HO",  # (History and Overview)
            # "math.IT",  # (Information Theory)
            # "math.KT",  # (K-Theory and Homology)
            # "math.LO",  # (Logic)
            # "math.MG",  # (Metric Geometry)
            # "math.MP",  # (Mathematical Physics)
            # "math.NA",  # (Numerical Analysis)
            # "math.NT",  # (Number Theory)
            # "math.OA",  # (Operator Algebras)
            # "math.OC",  # (Optimization and Control)
            # "math.PR",  # (Probability)
            # "math.QA",  # (Quantum Algebra)
            # "math.RA",  # (Rings and Algebras)
            # "math.RT",  # (Representation Theory)
            # "math.SG",  # (Symplectic Geometry)
            # "math.SP",  # (Spectral Theory)
            # "math.ST",  # (Statistics Theory)

            # "astro-ph.CO",  # (Cosmology and Nongalactic Astrophysics)
            # "astro-ph.EP",  # (Earth and Planetary Astrophysics)
            # "astro-ph.GA",  # (Astrophysics of Galaxies)
            # "astro-ph.HE",  # (High Energy Astrophysical Phenomena)
            # "astro-ph.IM",  # (Instrumentation and Methods for Astrophysics)
            # "astro-ph.SR",  # (Solar and Stellar Astrophysics)

            # "cond-mat.dis-nn",     # (Disordered Systems and Neural Networks)
            # "cond-mat.mes-hall",   # (Mesoscale and Nanoscale Physics)
            # "cond-mat.mtrl-sci",   # (Materials Science)
            # "cond-mat.other",      # (Other Condensed Matter)
            # "cond-mat.quant-gas",  # (Quantum Gases)
            # "cond-mat.soft",       # (Soft Condensed Matter)
            # "cond-mat.stat-mech",  # (Statistical Mechanics)
            # "cond-mat.str-el",     # (Strongly Correlated Electrons)
            # "cond-mat.supr-con",   # (Superconductivity)

            # "gr-qc",     # (General Relativity and Quantum Cosmology)
            # "hep-ex",    # (High Energy Physics - Experiment)
            # "hep-lat",   # (High Energy Physics - Lattice)
            # "hep-ph",    # (High Energy Physics - Phenomenology)
            # "hep-th",    # (High Energy Physics - Theory)
            # "math-ph",   # (Mathematical Physics)
            # "nlin.AO",   # (Adaptation and Self-Organizing Systems)
            # "nlin.CD",   # (Chaotic Dynamics)
            # "nlin.CG",   # (Cellular Automata and Lattice Gases)
            # "nlin.PS",   # (Pattern Formation and Solitons)
            # "nlin.SI",   # (Exactly Solvable and Integrable Systems)
            # "nucl-ex",   # (Nuclear Experiment)
            # "nucl-th",   # (Nuclear Theory)

            # "physics.acc-ph",    # (Accelerator Physics)
            # "physics.ao-ph",     # (Atmospheric and Oceanic Physics)
            # "physics.app-ph",    # (Applied Physics)
            # "physics.atm-clus",  # (Atomic and Molecular Clusters)
            # "physics.atom-ph",   # (Atomic Physics)
            # "physics.bio-ph",    # (Biological Physics)
            # "physics.chem-ph",   # (Chemical Physics)
            # "physics.class-ph",  # (Classical Physics)
            # "physics.comp-ph",   # (Computational Physics)
            # "physics.data-an",   # (Data Analysis, Statistics and Probability)
            # "physics.ed-ph",     # (Physics Education)
            # "physics.flu-dyn",   # (Fluid Dynamics)
            # "physics.gen-ph",    # (General Physics)
            # "physics.geo-ph",    # (Geophysics)
            # "physics.hist-ph",   # (History and Philosophy of Physics)
            # "physics.ins-det",   # (Instrumentation and Detectors)
            # "physics.med-ph",    # (Medical Physics)
            # "physics.optics",    # (Optics)
            # "physics.plasm-ph",  # (Plasma Physics)
            # "physics.pop-ph",    # (Popular Physics)
            # "physics.soc-ph",    # (Physics and Society)
            # "physics.space-ph",  # (Space Physics)
            # "quant-ph",          # (Quantum Physics)

            # "q-bio.BM",  # (Biomolecules)
            # "q-bio.CB",  # (Cell Behavior)
            # "q-bio.GN",  # (Genomics)
            # "q-bio.MN",  # (Molecular Networks)
            # "q-bio.NC",  # (Neurons and Cognition)
            # "q-bio.OT",  # (Other Quantitative Biology)
            # "q-bio.PE",  # (Populations and Evolution)
            # "q-bio.QM",  # (Quantitative Methods)
            # "q-bio.SC",  # (Subcellular Processes)
            # "q-bio.TO",  # (Tissues and Organs)

            # "q-fin.CP",  # (Computational Finance)
            # "q-fin.EC",  # (Economics)
            # "q-fin.GN",  # (General Finance)
            # "q-fin.MF",  # (Mathematical Finance)
            # "q-fin.PM",  # (Portfolio Management)
            # "q-fin.PR",  # (Pricing of Securities)
            # "q-fin.RM",  # (Risk Management)
            # "q-fin.ST",  # (Statistical Finance)
            # "q-fin.TR",  # (Trading and Market Microstructure)

            # "stat.AP",  # (Applications)
            # "stat.CO",  # (Computation)
            # "stat.ME",  # (Methodology)
            # "stat.ML",  # (Machine Learning)
            # "stat.OT",  # (Other Statistics)
            # "stat.TH",  # (Statistics Theory)
        ]
    )
    arxiv_base_url: str     = "http://export.arxiv.org/api/query"
    max_results_per_topic: int = 1
    scrape_interval_seconds: float = 43_200    # 12 hours
    scrape_backfill_interval_seconds: float = 1800 # 30 minutes
    download_timeout_seconds: int  = 120

    # ── PDF Processing ─────────────────────────────────────────
    pdf_download_dir: str       = "/app/downloads"
    pdf_max_size_mb: int        = 150
    grobid_server_url: str      = "http://grobid:8070"

    # ── Chunking ───────────────────────────────────────────────
    default_section: str        = "Supplementary Information"
    chunk_size_tokens: int      = 1024
    chunk_overlap_tokens: int   = 128
    min_chunk_chars: int        = 100   # discard tiny chunks

    # ── Embeddings ─────────────────────────────────────────────
    worker_proc_alive_timeout: int = 60              # seconds
    hf_token: str               = ""
    hf_home: str                = "/home/atlazer/.cache/huggingface"
    embedding_provider: str     = "onnx"       # "local" | "tei" | "onnx"
    onnx_provider: str          = "CPUExecutionProvider"  # "CUDAExecutionProvider" / "CPUExecutionProvider"
    truncate_dim: int           = 1024
    embedding_batch_size: int   = 1            # texts per API call
    local_embedding_model: str  = "BAAI/bge-m3"
    tei_base_url: str           = "http://tei:80"
    onnx_cache_dir: str         = "/app/data/onnx_cache"
    
    # ── Logging ────────────────────────────────────────────────
    log_level: str          = "INFO"
    log_format: str         = "json"             # "json" | "console"

    # ── Storage ───────────────────────────────────────────────
    db_host: str            = "db"
    db_port: int            = 5432
    db_user: str            = "postgres"
    db_password: str        = ""
    db_name: str            = "atlazer"

    @property
    def db_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @field_validator("arxiv_topics", mode="before")
    @classmethod
    def parse_topics(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",")]
        return v


settings = Settings()