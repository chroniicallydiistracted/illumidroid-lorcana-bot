//! DAC (Deck-As-Code) format-validation algorithm and the `LORCANA_FORMATS`
//! data table (blueprint **Step 2** — "Port deck-validation rules").
//!
//! Byte-for-byte port of
//! `oracle/source/packages/lorcana/lorcana-types/src/decks/validate-deck.ts`
//! (`validateDeckForFormat` / `validateDeck` / `getDeckFormats`, the private
//! `groupByCanonical` / `resolveMaxCopies` helpers, and the `LORCANA_FORMATS`
//! record). The data **types** it operates on (`LorcanaFormat`, `DeckCard`,
//! `CardFormatData`, `FormatRuleResult`, `DeckFormatResult`,
//! `FormatValidationKind`, `LorcanaSetCode`, `LorcanaFormatId`) were ported in
//! Step 1 and live in [`crate::deck`]; this module adds only the algorithm and
//! the format data.
//!
//! ## Parity-critical behavior preserved
//! * **Rule order**: `DECK_SIZE`, `INK_TYPES`, `CARD_QUANTITY`, `CARD_SET`, then
//!   the conditional `BANNED_CARD` and `REQUIRES_ANY_SET` rules.
//! * **Order-sensitive output**: format order ([`LORCANA_FORMATS`] insertion
//!   order), failure-list order (deck order / first-seen canonical order), and
//!   ink-type insertion order (an [`IndexSet`] mirrors the oracle's `Set`).
//! * **Message construction**: every message string is reproduced verbatim.
//! * **Unknown-lookup handling**: a `None` lookup is *skipped* for canonical
//!   grouping, ink counting, and the set check; contributes `false` to
//!   `requiresAnySet`; and falls back to the raw `cardId` for banned-card naming.
//! * **Copy limits** grouped by `canonicalId` across reprints; `"no-limit"` is
//!   unlimited; an absent limit defaults to 4.
//!
//! The dependency-injected `lookup` is modeled as
//! `&dyn Fn(&str) -> Option<CardFormatData>`, matching the oracle's
//! `lookup: (id) => CardFormatData | undefined` (lorcana-types stays free of any
//! runtime card-data dependency).

use std::sync::LazyLock;

use indexmap::{IndexMap, IndexSet};

use crate::card::CardCopyLimit;
use crate::deck::{
    CardFormatData, DeckCard, DeckFormatResult, FormatRuleResult, FormatValidationKind,
    LorcanaFormat, LorcanaFormatId, LorcanaSetCode,
};

/// A card-data lookup by shortId. Mirrors the oracle's
/// `lookup: (id: string) => CardFormatData | undefined`.
pub type CardLookup<'a> = dyn Fn(&str) -> Option<CardFormatData> + 'a;

// ---------------------------------------------------------------------------
// Banned card shortIds (sourced from canonical-cards.json + cardsAuxKv)
// ---------------------------------------------------------------------------

/// Hiram Flaversham - Toymaker (set2-149).
const HIRAM_FLAVERSHAM_TOYMAKER: &str = "LsX";

/// Fortisphere (set4-200).
const FORTISPHERE: &str = "PSk";

// ---------------------------------------------------------------------------
// Format definitions
// ---------------------------------------------------------------------------

/// Build a [`LorcanaFormat`] with the three required fields and all optional
/// rules defaulted to absent; callers mutate the optional fields they need.
fn base_format(
    id: LorcanaFormatId,
    label: &str,
    description: &str,
    allowed_sets: Vec<LorcanaSetCode>,
) -> LorcanaFormat {
    LorcanaFormat {
        id,
        label: label.to_string(),
        description: Some(description.to_string()),
        allowed_sets,
        banned_card_ids: None,
        min_deck_size: None,
        max_ink_types: None,
        special_allowed_card_ids: None,
        required_rotation_state: None,
        requires_any_set: None,
        excluded_sets: None,
    }
}

/// The named Lorcana formats and their legality rules, in oracle declaration
/// order (`infinity`, `core-constructed`, `archazias-island`, `shimmering-skies`,
/// `azurite-sea`). Iteration order is the order `validateDeck`/`getDeckFormats`
/// check formats in, mirroring the TypeScript `Object.values(LORCANA_FORMATS)`.
pub static LORCANA_FORMATS: LazyLock<IndexMap<LorcanaFormatId, LorcanaFormat>> =
    LazyLock::new(|| {
        use LorcanaFormatId::*;
        use LorcanaSetCode::*;
        let mut formats: IndexMap<LorcanaFormatId, LorcanaFormat> = IndexMap::new();

        // All released sets legal; only Hiram Flaversham - Toymaker banned.
        let mut infinity = base_format(
            Infinity,
            "Infinity",
            "All released sets are legal. One card is banned.",
            vec![Tfc, Rof, Iti, Urr, Ssk, Azs, Arc, Roj, Fab, Wiw, Wsp, Wun],
        );
        infinity.banned_card_ids = Some(vec![HIRAM_FLAVERSHAM_TOYMAKER.to_string()]);
        formats.insert(Infinity, infinity);

        // Rotating constructed format: SSK through WUN; Hiram + Fortisphere banned.
        let mut core = base_format(
            CoreConstructed,
            "Core Constructed",
            "Current rotating format. Sets SSK through WUN are legal.",
            vec![Ssk, Azs, Arc, Roj, Fab, Wiw, Wsp, Wun],
        );
        core.banned_card_ids = Some(vec![
            HIRAM_FLAVERSHAM_TOYMAKER.to_string(),
            FORTISPHERE.to_string(),
        ]);
        core.required_rotation_state = Some("CoreConstructed".to_string());
        formats.insert(CoreConstructed, core);

        // Archazia's Island: SSK through FAB (five sets).
        formats.insert(
            ArchaziasIsland,
            base_format(
                ArchaziasIsland,
                "Archazia's Island",
                "Sets SSK through FAB are legal. No banned cards.",
                vec![Ssk, Azs, Arc, Roj, Fab],
            ),
        );

        // Shimmering Skies: legacy format covering the first five sets.
        formats.insert(
            ShimmeringSkies,
            base_format(
                ShimmeringSkies,
                "Shimmering Skies",
                "Sets TFC through SSK are legal. No banned cards.",
                vec![Tfc, Rof, Iti, Urr, Ssk],
            ),
        );

        // Azurite Sea: TFC through AZS, plus select Fabled promo cards.
        let mut azurite = base_format(
            AzuriteSea,
            "Azurite Sea",
            "Sets TFC through AZS are legal, plus select Fabled promo cards.",
            vec![Tfc, Rof, Iti, Urr, Ssk, Azs],
        );
        // Populate with Fabled promotional card shortIds when available.
        azurite.special_allowed_card_ids = Some(Vec::new());
        formats.insert(AzuriteSea, azurite);

        formats
    });

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Group deck cards by `canonicalId`, summing quantities for reprints. Unknown
/// lookups are skipped; the stored [`CardFormatData`] is the first-seen
/// printing's (mirroring the oracle's "insert once, then only add quantity").
/// Returns an [`IndexMap`] so iteration follows first-seen canonical order.
fn group_by_canonical(
    cards: &[DeckCard],
    lookup: &CardLookup<'_>,
) -> IndexMap<String, (CardFormatData, i64)> {
    let mut groups: IndexMap<String, (CardFormatData, i64)> = IndexMap::new();
    for card in cards {
        let Some(data) = lookup(&card.card_id) else {
            continue;
        };
        if let Some((_, total_qty)) = groups.get_mut(&data.canonical_id) {
            *total_qty += card.quantity;
        } else {
            let canonical_id = data.canonical_id.clone();
            groups.insert(canonical_id, (data, card.quantity));
        }
    }
    groups
}

/// Resolve the maximum allowed copies for a card. `None` means **unlimited**
/// (the oracle's `Infinity`), `Some(n)` a finite cap. Mirrors `resolveMaxCopies`:
/// `"no-limit"` ⇒ unlimited, an explicit number ⇒ that number, absent ⇒ 4.
fn resolve_max_copies(limit: Option<&CardCopyLimit>) -> Option<i64> {
    match limit {
        Some(CardCopyLimit::NoLimit(_)) => None,
        Some(CardCopyLimit::Count(n)) => Some(*n),
        None => Some(4),
    }
}

// ---------------------------------------------------------------------------
// Core validation
// ---------------------------------------------------------------------------

/// Validate a deck against a single format. Each rule produces a
/// [`FormatRuleResult`]; `valid` is `true` only when every rule passes. Mirrors
/// `validateDeckForFormat`.
pub fn validate_deck_for_format(
    cards: &[DeckCard],
    lookup: &CardLookup<'_>,
    format: &LorcanaFormat,
) -> DeckFormatResult {
    let mut rules: Vec<FormatRuleResult> = Vec::new();
    let min_size = format.min_deck_size.unwrap_or(60);
    let max_ink_types = format.max_ink_types.unwrap_or(2);

    // DECK_SIZE
    let total_cards: i64 = cards.iter().map(|c| c.quantity).sum();
    rules.push(FormatRuleResult {
        kind: FormatValidationKind::DeckSize,
        passed: total_cards >= min_size,
        message: if total_cards >= min_size {
            format!("Deck has {total_cards} cards (minimum {min_size}).")
        } else {
            format!("Deck has {total_cards} cards but requires at least {min_size}.")
        },
    });

    // INK_TYPES — distinct ink strings across the whole deck, insertion-ordered.
    let mut ink_types: IndexSet<String> = IndexSet::new();
    for card in cards {
        if let Some(data) = lookup(&card.card_id) {
            for ink in &data.ink_types {
                ink_types.insert(ink.clone());
            }
        }
    }
    let ink_count = ink_types.len() as i64;
    rules.push(FormatRuleResult {
        kind: FormatValidationKind::InkTypes,
        passed: ink_count <= max_ink_types,
        message: if ink_count <= max_ink_types {
            format!("Deck uses {ink_count} ink type(s) (maximum {max_ink_types}).")
        } else {
            let joined: Vec<&str> = ink_types.iter().map(String::as_str).collect();
            format!(
                "Deck uses {ink_count} ink types ({}), but at most {max_ink_types} are allowed.",
                joined.join(", ")
            )
        },
    });

    // CARD_QUANTITY — enforced per canonical card across all printings.
    let groups = group_by_canonical(cards, lookup);
    let mut quantity_failures: Vec<String> = Vec::new();
    for (data, total_qty) in groups.values() {
        // A failure can only occur against a finite limit (unlimited is never
        // exceeded), so the oracle's "unlimited" branch is unreachable here.
        if let Some(max) = resolve_max_copies(data.card_copy_limit.as_ref())
            && *total_qty > max
        {
            quantity_failures.push(format!(
                "{}: {total_qty} copies (maximum {max})",
                data.full_name
            ));
        }
    }
    rules.push(FormatRuleResult {
        kind: FormatValidationKind::CardQuantity,
        passed: quantity_failures.is_empty(),
        message: if quantity_failures.is_empty() {
            "All card quantities are within the allowed limits.".to_string()
        } else {
            format!("Too many copies: {}.", quantity_failures.join("; "))
        },
    });

    // CARD_SET — a card passes if any printing is in an allowed set, or any
    // printing's rotation state matches the format's required state; cards whose
    // printings live entirely in `excludedSets` are rejected regardless.
    let mut set_failures: Vec<String> = Vec::new();
    for card in cards {
        if let Some(special) = &format.special_allowed_card_ids
            && special.iter().any(|id| id == &card.card_id)
        {
            continue;
        }
        let Some(data) = lookup(&card.card_id) else {
            continue;
        };
        let in_allowed_set = data.sets.iter().any(|s| format.allowed_sets.contains(s));
        let matches_rotation = match &format.required_rotation_state {
            Some(required) => data
                .rotation_states
                .as_ref()
                .is_some_and(|states| states.iter().any(|s| s == required)),
            None => false,
        };
        let all_in_excluded = match &format.excluded_sets {
            Some(excluded) if !excluded.is_empty() => {
                !data.sets.is_empty() && data.sets.iter().all(|s| excluded.contains(s))
            }
            _ => false,
        };
        let passes = !all_in_excluded && (in_allowed_set || matches_rotation);
        if !passes {
            let sets = if data.sets.is_empty() {
                "unknown".to_string()
            } else {
                data.sets
                    .iter()
                    .map(LorcanaSetCode::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            set_failures.push(format!("{} (sets: {sets})", data.full_name));
        }
    }
    rules.push(FormatRuleResult {
        kind: FormatValidationKind::CardSet,
        passed: set_failures.is_empty(),
        message: if set_failures.is_empty() {
            "All cards are legal for this format.".to_string()
        } else {
            format!(
                "Cards not legal in {}: {}.",
                format.label,
                set_failures.join("; ")
            )
        },
    });

    // BANNED_CARD — only emitted when the format declares banned ids.
    if let Some(banned) = &format.banned_card_ids
        && !banned.is_empty()
    {
        let mut banned_failures: Vec<String> = Vec::new();
        for card in cards {
            if banned.iter().any(|id| id == &card.card_id) {
                let name = lookup(&card.card_id)
                    .map(|d| d.full_name)
                    .unwrap_or_else(|| card.card_id.clone());
                banned_failures.push(name);
            }
        }
        rules.push(FormatRuleResult {
            kind: FormatValidationKind::BannedCard,
            passed: banned_failures.is_empty(),
            message: if banned_failures.is_empty() {
                "No banned cards in deck.".to_string()
            } else {
                format!(
                    "Banned in {}: {}.",
                    format.label,
                    banned_failures.join(", ")
                )
            },
        });
    }

    // REQUIRES_ANY_SET — only emitted when the format declares required sets.
    if let Some(required) = &format.requires_any_set
        && !required.is_empty()
    {
        let has_required_set = cards.iter().any(|card| {
            lookup(&card.card_id).is_some_and(|data| data.sets.iter().any(|s| required.contains(s)))
        });
        let joined = required
            .iter()
            .map(LorcanaSetCode::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        rules.push(FormatRuleResult {
            kind: FormatValidationKind::RequiresAnySet,
            passed: has_required_set,
            message: if has_required_set {
                format!("Deck contains at least one card from the required sets ({joined}).")
            } else {
                format!(
                    "{} requires at least one card from: {joined}.",
                    format.label
                )
            },
        });
    }

    let valid = rules.iter().all(|r| r.passed);
    DeckFormatResult {
        format_id: format.id,
        valid,
        rules,
    }
}

// ---------------------------------------------------------------------------
// Multi-format validation
// ---------------------------------------------------------------------------

/// The default format set checked when none is supplied: every entry in
/// [`LORCANA_FORMATS`], in declaration order. Mirrors the oracle default
/// `Object.values(LORCANA_FORMATS)`.
pub fn default_formats() -> Vec<&'static LorcanaFormat> {
    LORCANA_FORMATS.values().collect()
}

/// Validate a deck against the given formats (or [`default_formats`] when
/// `formats` is `None`), returning one [`DeckFormatResult`] per format in order.
/// Mirrors `validateDeck`.
pub fn validate_deck(
    cards: &[DeckCard],
    lookup: &CardLookup<'_>,
    formats: Option<&[&LorcanaFormat]>,
) -> Vec<DeckFormatResult> {
    let owned_default;
    let formats: &[&LorcanaFormat] = match formats {
        Some(formats) => formats,
        None => {
            owned_default = default_formats();
            &owned_default
        }
    };
    formats
        .iter()
        .map(|format| validate_deck_for_format(cards, lookup, format))
        .collect()
}

/// Return the format ids for which the deck is fully legal, checked against
/// every [`LORCANA_FORMATS`] entry in order. Mirrors `getDeckFormats`.
pub fn get_deck_formats(cards: &[DeckCard], lookup: &CardLookup<'_>) -> Vec<LorcanaFormatId> {
    validate_deck(cards, lookup, None)
        .into_iter()
        .filter(|result| result.valid)
        .map(|result| result.format_id)
        .collect()
}
