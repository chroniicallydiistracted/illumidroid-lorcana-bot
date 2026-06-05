//! Card schema: card-type vocabulary and the `LorcanaCardDefinition` shape.
//!
//! Mirrors `oracle/source/packages/lorcana/lorcana-types/src/cards/card-types.ts`
//! (`CARD_TYPES`/`CardType`, `LANGUAGES`/`Languages`, `ActionSubtype`,
//! `CardText`, `I18nProperties`, and the unified `LorcanaCardDefinition`).
//!
//! Integer-valued stats use `i64` so that an input like `"cost":3` round-trips
//! byte-for-byte (an `f64` would re-emit `3.0`). Card-type-specific and
//! rarely-used fields are preserved losslessly via the flattened `rest` map.

use crate::ability::AbilityDefinition;
use crate::classification::Classification;
use crate::ink::InkType;
use crate::macros::str_enum;
use crate::node::JsonObject;
use crate::serde_util::optional_non_null;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

str_enum! {
    /// `CardType` (Section 6) — the four card types.
    pub enum CardType {
        Character = "character",
        Action = "action",
        Item = "item",
        Location = "location",
    }
}

str_enum! {
    /// `Languages` — supported localization languages.
    pub enum Language {
        En = "en",
        De = "de",
        Fr = "fr",
        It = "it",
    }
}

str_enum! {
    /// `ActionSubtype` non-null form. The oracle type is `"song" | null`; model a
    /// present subtype as `Some(ActionSubtype::Song)` and an absent/`null` one as
    /// `None`.
    pub enum ActionSubtype {
        Song = "song",
    }
}

str_enum! {
    /// Card `rarity` values (card-types.ts).
    pub enum Rarity {
        Common = "common",
        Uncommon = "uncommon",
        Rare = "rare",
        SuperRare = "super_rare",
        Legendary = "legendary",
        Enchanted = "enchanted",
        Epic = "epic",
        Iconic = "iconic",
        Promo = "promo",
        Special = "special",
    }
}

str_enum! {
    /// Card `specialRarity` values (card-types.ts).
    pub enum SpecialRarity {
        Enchanted = "enchanted",
        Epic = "epic",
        Iconic = "iconic",
        Promo = "promo",
        Challenge = "challenge",
    }
}

str_enum! {
    /// The `"no-limit"` sentinel used by [`CardCopyLimit`].
    pub enum NoLimitTag {
        NoLimit = "no-limit",
    }
}

/// `number | "no-limit"` — the exact union of `cardCopyLimit`. An untagged enum
/// so the numeric and sentinel forms each round-trip byte-for-byte and any other
/// JSON value fails closed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CardCopyLimit {
    /// A custom numeric copy limit (e.g. `99` for Dalmatian Puppy, `2` for The
    /// Glass Slipper).
    Count(i64),
    /// The `"no-limit"` sentinel — unlimited copies (e.g. Microbots).
    NoLimit(NoLimitTag),
}

/// A structured card-text entry (`CardTextEntry`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CardTextEntry {
    /// Entry title.
    pub title: String,
    /// Optional entry description.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,
}

/// Card rules text: raw string or structured entries (`CardText`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CardText {
    /// Raw printed text.
    Raw(String),
    /// Structured entries.
    Entries(Vec<CardTextEntry>),
}

/// Embedded localized card metadata (`I18nProperties`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct I18nProperties {
    /// Localized name.
    pub name: String,
    /// Localized version.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub version: Option<String>,
    /// Localized rules text.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub text: Option<CardText>,
    /// Localized choice-option labels.
    #[serde(
        rename = "optionTexts",
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub option_texts: Option<Vec<String>>,
}

/// External IDs from various systems (`BaseCardProperties.externalIds`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExternalIds {
    /// Ravensburger system ID.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub ravensburger: Option<String>,
    /// Culture-invariant ID.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub culture_invariant_id: Option<i64>,
    /// Lorcast ID.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub lorcast: Option<String>,
    /// TCGPlayer ID.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub tcg_player: Option<i64>,
}

/// Unified card definition (`LorcanaCardDefinition`).
///
/// All card-type-specific properties are optional; the discriminated
/// [`CardType`] selects which apply. Any field not broken out here (e.g.
/// `printings`, `vanilla`, `missingImplementation`) is preserved in `rest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LorcanaCardDefinition {
    /// Unique identifier for the card.
    pub id: String,
    /// Canonical ID grouping reprints/alternate art.
    pub canonical_id: String,
    /// Other printing IDs for this card.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub reprints: Option<Vec<String>>,
    /// Card name (Rule 6.2.4).
    pub name: String,
    /// Card version (Rule 6.2.5).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub version: Option<String>,
    /// Full name (name + version) for deck-building limits.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub full_name: Option<String>,
    /// Ink type(s) (Rule 6.2.3) — single or dual.
    pub ink_type: Vec<InkType>,
    /// Ink cost (Rule 6.2.7).
    pub cost: i64,
    /// Inkwell symbol (Rule 6.2.8).
    pub inkable: bool,
    /// Card type (Rule 6.1).
    pub card_type: CardType,
    /// Strength (Rule 6.2.9) — characters.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub strength: Option<i64>,
    /// Willpower (Rule 6.2.10) — characters / locations.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub willpower: Option<i64>,
    /// Lore value (Rule 6.2.11) — characters / locations.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub lore: Option<i64>,
    /// Character classifications (Rule 6.2.6).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub classifications: Option<Vec<Classification>>,
    /// Action subtype. The oracle type is `"song" | null`, so this field is one
    /// of the few that legitimately accepts an explicit `null` (⇒ `None`, same
    /// as absent); it therefore keeps the default `Option` semantics rather than
    /// the non-null helper.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_subtype: Option<ActionSubtype>,
    /// Move cost (Rule 6.5.5) — locations.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub move_cost: Option<i64>,
    /// Card abilities (includes keywords).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub abilities: Option<Vec<AbilityDefinition>>,
    /// Rules text as printed.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub text: Option<CardText>,
    /// Embedded localized card metadata (keyed by language).
    pub i18n: IndexMap<Language, I18nProperties>,
    /// Flavor text (not mechanically relevant).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub flavor_text: Option<String>,
    /// Set information.
    pub set: String,
    /// Card number in set.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub card_number: Option<i64>,
    /// Rarity.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub rarity: Option<Rarity>,
    /// Special rarity variant.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub special_rarity: Option<SpecialRarity>,
    /// Copy-limit override (`number | "no-limit"`).
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub card_copy_limit: Option<CardCopyLimit>,
    /// Franchise the card belongs to.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub franchise: Option<String>,
    /// External IDs from various systems.
    #[serde(
        default,
        deserialize_with = "optional_non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub external_ids: Option<ExternalIds>,

    /// Any field not broken out above, preserved losslessly in input order.
    #[serde(flatten)]
    pub rest: JsonObject,
}

impl LorcanaCardDefinition {
    /// The full name used for deck-building limits: `fullName` if present, else
    /// `"{name} - {version}"`, else `name`. Mirrors `getFullName`.
    pub fn full_name(&self) -> String {
        if let Some(full) = &self.full_name {
            return full.clone();
        }
        match &self.version {
            Some(version) => format!("{} - {}", self.name, version),
            None => self.name.clone(),
        }
    }

    /// Whether the card has dual ink types (Rule 6.2.3.1). Mirrors `isDualInk`.
    pub fn is_dual_ink(&self) -> bool {
        self.ink_type.len() == 2
    }
}
