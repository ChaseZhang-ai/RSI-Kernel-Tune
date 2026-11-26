"""klib: the reference library's index (rsi_kernel_tune.klib.schema).

Only the schema is here: what an entry is, how it is checked, how the index is read and
written. Building the index from sources and finding or scoring entries is task seedlib's.
"""

from rsi_kernel_tune.klib.schema import (
    ALLOWED_LICENSES,
    ENTRY,
    FIELDS,
    INDEX_VERSION,
    LANGUAGES,
    KernelEntry,
    SchemaError,
    license_allowed,
    load_index,
    save_index,
    submittable,
    validate,
)

__all__ = [
    "ALLOWED_LICENSES",
    "ENTRY",
    "FIELDS",
    "INDEX_VERSION",
    "LANGUAGES",
    "KernelEntry",
    "SchemaError",
    "license_allowed",
    "load_index",
    "save_index",
    "submittable",
    "validate",
]
