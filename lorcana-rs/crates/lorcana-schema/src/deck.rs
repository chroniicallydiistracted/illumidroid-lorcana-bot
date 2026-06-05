//! Deck schema: deck-building constants, validation error/result/stats shapes,
//! and DAC (Deck-As-Code) format types.
//!
//! Mirrors `oracle/source/packages/lorcana/lorcana-types/src/cards/deck-validation.ts`
//! and `.../decks/validate-deck.ts`.
//!
//! ## Step boundary (Step 1 vs Step 2)
//! Per `docs/port/port-status.md`, the deck-validation **rules/algorithm** are
//! Step 2. Step 1 ports only the shared **constants, types, and error shapes**
//! that the schema round-trip tests need. The `validateDeck` /
//! `validateDeckForFormat` / `getDeckFormats` algorithms and the
//! `LORCANA_FORMATS` data table are intentionally **not** ported here — they
//! land in Step 2.

use crate::card::CardCopyLimit;
use crate::ink::InkType;
use crate::macros::str_enum;
use crate::serde_util::optional_non_null;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Minimum cards required in a deck (Rule 2.1.1.1).
pub const MIN_DECK_SIZE: i64 = 60;

/// Maximum different ink types allowed in a deck (Rule 2.1.1.2).
pub const MAX_INK_TYPES: i64 = 2;

/// Maximum copies of the same card by full name (Rule 2.1.1.3).
pub const MAX_COPIES_PER_CARD: i64 = 4;

/// `DeckValidationError` — the standard-rules validation error union, tagged on
/// `type` with the oracle's UPPER_SNAKE discriminators.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DeckValidationError {
    /// Fewer than [`MIN_DECK_SIZE`] cards.
    #[serde(rename = "TOO_FEW_CARDS")]
    TooFewCards {
        /// Actual card count.
        count: i64,
        /// The minimum (always [`MIN_DECK_SIZE`]).
        minimum: i64,
    },
    /// More than [`MAX_INK_TYPES`] ink types.
    #[serde(rename = "TOO_MANY_INK_TYPES", rename_all = "camelCase")]
    TooManyInkTypes {
        /// Ink types present.
        ink_types: Vec<InkType>,
        /// The maximum (always [`MAX_INK_TYPES`]).
        maximum: i64,
    },
    /// More than the allowed copies of one full name.
    #[serde(rename = "TOO_MANY_COPIES", rename_all = "camelCase")]
    TooManyCopies {
        /// The offending card's full name.
        full_name: String,
        /// Actual copy count.
        count: i64,
        /// The maximum allowed copies.
        maximum: i64,
    },
}

/// `DeckValidationResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeckValidationResult {
    /// Whether the deck is valid.
    pub valid: bool,
    /// Any validation errors.
    pub errors: Vec<DeckValidationError>,
}

/// `DeckStats.cardTypeBreakdown`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CardTypeBreakdown {
    /// Character count.
    pub characters: i64,
    /// Action count.
    pub actions: i64,
    /// Item count.
    pub items: i64,
    /// Location count.
    pub locations: i64,
}

/// `DeckStats` — informational deck statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeckStats {
    /// Total cards.
    pub total_cards: i64,
    /// Ink types present.
    pub ink_types: Vec<InkType>,
    /// Copies keyed by full name.
    pub card_counts: IndexMap<String, i64>,
    /// Breakdown by card type.
    pub card_type_breakdown: CardTypeBreakdown,
    /// Number of inkable cards.
    pub inkable_cards: i64,
    /// Average ink cost (may be fractional).
    pub average_cost: f64,
}

// ---------------------------------------------------------------------------
// DAC format types (validate-deck.ts) — types only; algorithm is Step 2.
// ---------------------------------------------------------------------------

str_enum! {
    /// `LorcanaSetCode` — all Lorcana set codes.
    pub enum LorcanaSetCode {
        Tfc = "TFC",
        Rof = "ROF",
        Iti = "ITI",
        Urr = "URR",
        Ssk = "SSK",
        Azs = "AZS",
        Arc = "ARC",
        Roj = "ROJ",
        Fab = "FAB",
        Wiw = "WIW",
        Wsp = "WSP",
        Wun = "WUN",
    }
}

str_enum! {
    /// `LorcanaFormatId` — named format identifiers.
    pub enum LorcanaFormatId {
        Infinity = "infinity",
        CoreConstructed = "core-constructed",
        ArchaziasIsland = "archazias-island",
        ShimmeringSkies = "shimmering-skies",
        AzuriteSea = "azurite-sea",
    }
}

str_enum! {
    /// `FormatValidationKind` — the kind of rule checked during format validation.
    pub enum FormatValidationKind {
        DeckSize = "DECK_SIZE",
        InkTypes = "INK_TYPES",
        CardQuantity = "CARD_QUANTITY",
        CardSet = "CARD_SET",
        BannedCard = "BANNED_CARD",
        RequiresAnySet = "REQUIRES_ANY_SET",
    }
}

/// `DeckCard` — a single card entry in a deck.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeckCard {
    /// Canonical shortId from `@tcg/lorcana-cards`.
    pub card_id: String,
    /// Number of copies.
    pub quantity: i64,
}

/// `CardFormatData` — card data required for format validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CardFormatData {
    /// Canonical identifier grouping printings.
    pub canonical_id: String,
    /// Display name for validation messages.
    pub full_name: String,
    /// All set codes where this canonical card has a printing.
    pub sets: Vec<LorcanaSetCode>,
    /// Ink types for this card.
    pub ink_types: Vec<String>,
    /// Override the default 4-copy limit (`number | "no-limit"`).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub card_copy_limit: Option<CardCopyLimit>,
    /// Ravensburger rotation states across printings.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub rotation_states: Option<Vec<String>>,
}

/// `LorcanaFormat` — a named format with its legality rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LorcanaFormat {
    /// Format identifier.
    pub id: LorcanaFormatId,
    /// Display label.
    pub label: String,
    /// Optional description.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,
    /// Set codes whose printings are legal.
    pub allowed_sets: Vec<LorcanaSetCode>,
    /// Card shortIds banned in this format.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub banned_card_ids: Option<Vec<String>>,
    /// Minimum deck size override (default 60).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub min_deck_size: Option<i64>,
    /// Maximum distinct ink types override (default 2).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_ink_types: Option<i64>,
    /// Card shortIds bypassing the set restriction.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub special_allowed_card_ids: Option<Vec<String>>,
    /// Rotation-state designation that also admits a card.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub required_rotation_state: Option<String>,
    /// At least one card must belong to one of these sets.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub requires_any_set: Option<Vec<LorcanaSetCode>>,
    /// Sets rejected even when rotation state would admit them.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub excluded_sets: Option<Vec<LorcanaSetCode>>,
}

/// `FormatRuleResult` — result of a single validation rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormatRuleResult {
    /// Which rule was checked.
    pub kind: FormatValidationKind,
    /// Whether it passed.
    pub passed: bool,
    /// Human-readable message.
    pub message: String,
}

/// `DeckFormatResult` — all validation results for one format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeckFormatResult {
    /// The format checked.
    pub format_id: LorcanaFormatId,
    /// Whether the deck is fully legal in this format.
    pub valid: bool,
    /// Per-rule results.
    pub rules: Vec<FormatRuleResult>,
}
