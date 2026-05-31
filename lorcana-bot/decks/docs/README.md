# Final Lorcanito Tournament Deck Pack

This pack merges the previous 25-deck v2 zip with the manually supplied failed/new deck markdown.

## Contents

- Deck files: 25
- Retained from v2 zip: 17
- Added from manual markdown: 8
- Removed from v2 zip: 8

## Deck JSON Shape

Every deck is formatted as the Lorcanito simulator fixture shape:

```json
{
  "name": "Deck Name",
  "cards": "4 Card - Version\n4 Another Card - Version"
}
```

## Audit Summary

- All decks have 60+ cards: True
- All recorded placements are top 5 or better: True
- Parse errors: 0
- Banned-card hits: 0
- Screened banned names: Hiram Flaversham, Fortisphere

See `provenance_audit.json` and `AUDIT.md`.
