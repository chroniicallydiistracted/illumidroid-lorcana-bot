//! `lorcana-schema` — Serde enums/structs mirroring the frozen
//! `@tcg/lorcana-types` DSL (headless Lorcana port, blueprint **Step 1**).
//!
//! This crate is the primitive type/schema layer. It contains **no gameplay
//! logic** — no reducer, RNG, zones, legal actions, or card effects. Its job is
//! to give the rest of the port a faithful, fail-closed Serde model of the
//! card / ability / condition / effect / cost / target / keyword / deck schema,
//! with discriminator names mirrored exactly from the frozen TypeScript oracle.
//!
//! ## Representation strategy
//! * **Closed string-literal unions** (ink, classification, card type, keywords,
//!   trigger events, comparison operators, condition/effect discriminators, …)
//!   are ported as fieldless [`macros`]-generated enums whose JSON wire string
//!   is written out explicitly per variant. Unknown values fail closed.
//! * **Recursive DSL nodes** (conditions, effects, triggers) are ported as a
//!   typed discriminator + a flattened, order-preserving `IndexMap` of every
//!   other field (see [`node`]). This preserves exact serialization — required
//!   by the blueprint (§5.2) — while still typing and validating the
//!   discriminator. Full field-level typing is layered in by later steps as the
//!   engine consumes each field.
//!
//! See `docs/port/headless_lorcana_engine_porting_symbol_registry.md` for the
//! authoritative symbol table.

#![forbid(unsafe_code)]

mod macros;
mod serde_util;

pub mod ability;
pub mod card;
pub mod classification;
pub mod condition;
pub mod cost;
pub mod deck;
pub mod deck_format;
pub mod effect;
pub mod error;
pub mod expressions;
pub mod ids;
pub mod ink;
pub mod keyword;
pub mod node;
pub mod target;
pub mod trigger;

// Convenience re-exports of the most-used items.
pub use ability::{
    AbilityDefinition, AbilityType, ReplacementAbilityReplaces, ReplacementEffectReplaces,
    RestrictionType, TriggerRestrictionType,
};
pub use card::{
    ActionSubtype, CardCopyLimit, CardText, CardTextEntry, CardType, ExternalIds, I18nProperties,
    Language, LorcanaCardDefinition, NoLimitTag, Rarity, SpecialRarity,
};
pub use classification::Classification;
pub use condition::{Condition, ConditionType};
pub use cost::{AbilityCost, BoolOrNumber, CardTypeOrSong, CostComponentType};
pub use deck::{
    CardFormatData, DeckCard, DeckFormatResult, DeckValidationError, DeckValidationResult,
    FormatRuleResult, FormatValidationKind, LorcanaFormat, LorcanaFormatId, LorcanaSetCode,
    MAX_COPIES_PER_CARD, MAX_INK_TYPES, MIN_DECK_SIZE,
};
pub use deck_format::{
    LORCANA_FORMATS, default_formats, get_deck_formats, validate_deck, validate_deck_for_format,
};
pub use effect::{Effect, EffectType};
pub use error::SchemaError;
pub use expressions::{AmountRef, EffectDuration, ForEachCounterType, VariableAmountType};
pub use ids::{CardInstanceId, CardPublicId, GameId, PlayerId, ZoneId};
pub use ink::InkType;
pub use keyword::{KeywordType, ParameterizedKeywordType, SimpleKeywordType, ValueKeywordType};
pub use node::JsonObject;
pub use target::{
    CardStatus, ComparisonOperator, ConditionComparisonOperator, LorcanaZoneId, OwnerScope,
    PlayerTargetScope, SelectorScope, TargetController, TargetZone,
};
pub use trigger::{Trigger, TriggerCardType, TriggerEvent, TriggerSubjectEnum, TriggerTiming};

use indexmap::IndexMap;

/// The full discriminator manifest: a stable mapping from category name to the
/// exact set of JSON wire strings the Rust enums accept.
///
/// This is the programmatic source of truth used by the discriminator-coverage
/// conformance test to prove the Rust vocabulary equals the frozen TypeScript
/// oracle's. Exposing it as data (rather than a committed JSON file) keeps the
/// manifest impossible to desynchronize from the enums it describes.
pub fn discriminator_manifest() -> IndexMap<&'static str, &'static [&'static str]> {
    let mut m: IndexMap<&'static str, &'static [&'static str]> = IndexMap::new();
    m.insert("InkType", InkType::NAMES);
    m.insert("Classification", Classification::NAMES);
    m.insert("CardType", CardType::NAMES);
    m.insert("Language", Language::NAMES);
    m.insert("ActionSubtype", ActionSubtype::NAMES);
    m.insert("Rarity", Rarity::NAMES);
    m.insert("SpecialRarity", SpecialRarity::NAMES);
    m.insert("SimpleKeywordType", SimpleKeywordType::NAMES);
    m.insert("ParameterizedKeywordType", ParameterizedKeywordType::NAMES);
    m.insert("ValueKeywordType", ValueKeywordType::NAMES);
    m.insert("KeywordType", KeywordType::NAMES);
    m.insert("AbilityType", AbilityType::NAMES);
    m.insert(
        "ReplacementAbilityReplaces",
        ReplacementAbilityReplaces::NAMES,
    );
    m.insert(
        "ReplacementEffectReplaces",
        ReplacementEffectReplaces::NAMES,
    );
    m.insert("RestrictionType", RestrictionType::NAMES);
    m.insert("TriggerRestrictionType", TriggerRestrictionType::NAMES);
    m.insert("ConditionType", ConditionType::NAMES);
    m.insert("EffectType", EffectType::NAMES);
    m.insert("CostComponentType", CostComponentType::NAMES);
    m.insert("TriggerEvent", TriggerEvent::NAMES);
    m.insert("TriggerTiming", TriggerTiming::NAMES);
    m.insert("TriggerCardType", TriggerCardType::NAMES);
    m.insert("TriggerSubjectEnum", TriggerSubjectEnum::NAMES);
    m.insert("ComparisonOperator", ComparisonOperator::NAMES);
    m.insert(
        "ConditionComparisonOperator",
        ConditionComparisonOperator::NAMES,
    );
    m.insert("CardStatus", CardStatus::NAMES);
    m.insert("TargetZone", TargetZone::NAMES);
    m.insert("LorcanaZoneId", LorcanaZoneId::NAMES);
    m.insert("TargetController", TargetController::NAMES);
    m.insert("SelectorScope", SelectorScope::NAMES);
    m.insert("OwnerScope", OwnerScope::NAMES);
    m.insert("PlayerTargetScope", PlayerTargetScope::NAMES);
    m.insert("AmountRef", AmountRef::NAMES);
    m.insert("EffectDuration", EffectDuration::NAMES);
    m.insert("ForEachCounterType", ForEachCounterType::NAMES);
    m.insert("VariableAmountType", VariableAmountType::NAMES);
    m.insert("LorcanaSetCode", LorcanaSetCode::NAMES);
    m.insert("LorcanaFormatId", LorcanaFormatId::NAMES);
    m.insert("FormatValidationKind", FormatValidationKind::NAMES);
    m
}
