"""A small MapReduce implementation: one boss, many workers, shared storage.

Modules:
    storage   -- resolve user paths inside shared storage, list input files
    splits    -- cut input into byte-range splits and read line records from them
    kv        -- intermediate key/value encoding, partitioning, sorted runs, output format
    jobcode   -- validate and load user job files (mapper / reducer / combiner)
    tasks     -- execute one map task or one reduce task
    local     -- sequential reference implementation (``mr local``, test oracle)
"""
