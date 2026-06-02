# Lorcana Oracle Freeze

Create immutable TypeScript oracle artifacts before Rust gameplay code.

Required artifacts:
- oracle/source-hash.txt
- oracle/card-catalog-hash.txt
- oracle/ruleset-hash.txt
- oracle/replay-corpus/
- oracle/snapshot-schema/

Do not modify oracle runtime behavior. Adding exporters/tests is allowed only when clearly isolated.
