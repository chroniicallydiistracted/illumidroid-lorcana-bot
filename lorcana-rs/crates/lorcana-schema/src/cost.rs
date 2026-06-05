//! Ability cost schema.
//!
//! Mirrors `oracle/source/packages/lorcana/lorcana-types/src/abilities/cost-types.ts`.
//! `AbilityCost` is an optional-field "bag" in the oracle (the TS file itself
//! flags it for a future discriminated-union refactor), so it is ported as a
//! struct of optional fields plus a flattened `rest` map that losslessly
//! preserves the less-common shapes (`components`, `discard`,
//! `banishCharacterTargetDsl`) and any field not yet broken out.

use crate::macros::str_enum;
use crate::node::JsonObject;
use crate::serde_util::optional_non_null;
use serde::{Deserialize, Serialize};

str_enum! {
    /// The `type` discriminator of every `CostComponent` variant.
    pub enum CostComponentType {
        Exert = "exert",
        Ink = "ink",
        Banish = "banish",
        Discard = "discard",
        DamageSelf = "damage-self",
        ReturnToHand = "return-to-hand",
        PutUnder = "put-under",
        ExertOther = "exert-other",
    }
}

/// `boolean | number` — the exact union of `AbilityCost.banishItem` (`true` to
/// banish one item, or a count). An untagged enum so each form round-trips
/// byte-for-byte and any other JSON shape (string, `null`, float) fails closed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BoolOrNumber {
    /// The boolean form (`true` ⇒ one).
    Bool(bool),
    /// The integer-count form.
    Number(i64),
}

str_enum! {
    /// `CardType | "song"` — the closed union of `AbilityCost.discardCardType`.
    /// (`"character" | "action" | "item" | "location" | "song"`.) Modeled as its
    /// own enum so an out-of-vocabulary value fails closed rather than being
    /// accepted as a raw string.
    pub enum CardTypeOrSong {
        Character = "character",
        Action = "action",
        Item = "item",
        Location = "location",
        Song = "song",
    }
}

/// Complete ability cost. All fields optional; `rest` captures complex
/// components losslessly (see module docs).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AbilityCost {
    /// Whether to exert this card.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub exert: Option<bool>,
    /// Ink to pay from inkwell.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub ink: Option<i64>,
    /// Banish this card.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub banish_self: Option<bool>,
    /// Banish your items — `true` for one, or a number for several
    /// (`boolean | number`).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub banish_item: Option<BoolOrNumber>,
    /// Banish one of your characters.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub banish_character: Option<bool>,
    /// Restricts which characters can be banished (`"another"`).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub banish_character_target: Option<String>,
    /// Banish another card (generic).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub banish_other: Option<bool>,
    /// Number of cards to discard from hand.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub discard_cards: Option<i64>,
    /// Singular alias for `discardCards`.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub discard_card: Option<i64>,
    /// Whether discarded cards are chosen.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub discard_chosen: Option<bool>,
    /// Specific card type required for discard (`CardType | "song"`).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub discard_card_type: Option<CardTypeOrSong>,
    /// Specific card name required for discard.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub discard_card_name: Option<String>,
    /// Damage to deal to this character.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub damage_self: Option<i64>,
    /// Return this card to hand.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub return_self_to_hand: Option<bool>,
    /// Return another character to hand.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub return_character_to_hand: Option<bool>,
    /// Number of items to exert (other than self).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub exert_items: Option<i64>,
    /// Number of characters to exert (other than self).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub exert_characters: Option<i64>,
    /// Restricts which characters can be exerted to pay the cost.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub exert_characters_classification: Option<String>,
    /// Exert a single character (singular form).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub exert_character: Option<bool>,
    /// Exert another card (generic).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub exert_other: Option<bool>,
    /// Target for cost (e.g. which character to exert).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub target: Option<String>,

    /// All other fields (`components`, `discard`, `banishCharacterTargetDsl`,
    /// and any not yet broken out), preserved losslessly in input order.
    #[serde(flatten)]
    pub rest: JsonObject,
}
