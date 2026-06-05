//! Schema JSON round-trip tests (blueprint Step 1 validation:
//! "same accepted card definitions"; lossless serialization per blueprint §5.2).

use lorcana_schema::{
    AbilityCost, AbilityType, CardType, Condition, ConditionType, DeckValidationError, Effect,
    EffectType, InkType, KeywordType, LorcanaCardDefinition,
};
use serde_json::Value;

/// Compact JSON deserializes, re-serializes byte-for-byte, and re-parses equal.
/// Inputs are authored in canonical (discriminator-first) key order.
fn assert_byte_exact<T>(json: &str)
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    let parsed: T = serde_json::from_str(json).expect("deserialize");
    let reser = serde_json::to_string(&parsed).expect("serialize");
    assert_eq!(reser, json, "byte-exact round-trip mismatch");
}

/// Data-preserving round-trip regardless of key order: parse → re-serialize →
/// the result carries exactly the same data (compared order-insensitively).
fn assert_lossless<T>(json: &str)
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    let original: Value = serde_json::from_str(json).expect("parse original as value");
    let parsed: T = serde_json::from_str(json).expect("deserialize typed");
    let reser = serde_json::to_string(&parsed).expect("serialize typed");
    let roundtripped: Value = serde_json::from_str(&reser).expect("parse reserialized as value");
    assert_eq!(original, roundtripped, "data lost in round-trip");
}

#[test]
fn condition_node_byte_exact() {
    assert_byte_exact::<Condition>(
        r#"{"type":"has-named-character","name":"Elsa","controller":"you"}"#,
    );
    assert_byte_exact::<Condition>(
        r#"{"type":"and","conditions":[{"type":"used-shift"},{"type":"is-exerted"}]}"#,
    );
}

#[test]
fn condition_typed_discriminator_and_nesting() {
    let c: Condition =
        serde_json::from_str(r#"{"type":"resource-count","what":"characters","controller":"you","comparison":"greater-or-equal","value":3}"#)
            .unwrap();
    assert_eq!(c.kind(), ConditionType::ResourceCount);
    assert_eq!(c.rest.get("what").unwrap(), "characters");
    assert_eq!(c.rest.get("value").unwrap(), 3);
}

#[test]
fn effect_node_byte_exact_and_nested() {
    assert_byte_exact::<Effect>(r#"{"type":"draw","amount":2,"target":"CONTROLLER"}"#);
    // Nested effect inside a control-flow effect, with a nested condition.
    let json = r#"{"type":"conditional","condition":{"type":"used-shift"},"effect":{"type":"banish","target":"CHOSEN_CHARACTER"}}"#;
    assert_byte_exact::<Effect>(json);
    let e: Effect = serde_json::from_str(json).unwrap();
    assert_eq!(e.kind(), EffectType::Conditional);
}

#[test]
fn ability_cost_byte_exact() {
    assert_byte_exact::<AbilityCost>(r#"{"exert":true,"ink":2}"#);
    assert_byte_exact::<AbilityCost>(r#"{"banishSelf":true}"#);
    // boolean|number union for banishItem preserved exactly (number form).
    assert_byte_exact::<AbilityCost>(r#"{"banishItem":2}"#);
    // Complex component list preserved losslessly via `rest`.
    assert_lossless::<AbilityCost>(
        r#"{"exert":true,"components":[{"type":"discard","amount":1,"chosen":true}]}"#,
    );
}

#[test]
fn typed_scalar_unions_byte_exact() {
    // banishItem: boolean | number — both forms round-trip byte-for-byte.
    assert_byte_exact::<AbilityCost>(r#"{"banishItem":true}"#);
    assert_byte_exact::<AbilityCost>(r#"{"banishItem":3}"#);
    // discardCardType: CardType | "song".
    assert_byte_exact::<AbilityCost>(r#"{"discardCardType":"song"}"#);
    assert_byte_exact::<AbilityCost>(r#"{"discardCardType":"item"}"#);
    // cardCopyLimit: number | "no-limit".
    let limited = r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"item","cardCopyLimit":99,"i18n":{"en":{"name":"X"}},"set":"TFC"}"#;
    let nolimit = r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"item","cardCopyLimit":"no-limit","i18n":{"en":{"name":"X"}},"set":"TFC"}"#;
    assert_lossless::<LorcanaCardDefinition>(limited);
    assert_lossless::<LorcanaCardDefinition>(nolimit);
}

#[test]
fn deck_validation_error_byte_exact() {
    assert_byte_exact::<DeckValidationError>(r#"{"type":"TOO_FEW_CARDS","count":58,"minimum":60}"#);
    assert_byte_exact::<DeckValidationError>(
        r#"{"type":"TOO_MANY_INK_TYPES","inkTypes":["amber","ruby","steel"],"maximum":2}"#,
    );
    assert_byte_exact::<DeckValidationError>(
        r#"{"type":"TOO_MANY_COPIES","fullName":"Elsa - Ice Queen","count":5,"maximum":4}"#,
    );
}

#[test]
fn character_card_fixture_loads_losslessly() {
    let json = include_str!("fixtures/valid_character_card.json");
    assert_lossless::<LorcanaCardDefinition>(json);

    let card: LorcanaCardDefinition = serde_json::from_str(json).unwrap();
    assert_eq!(card.card_type, CardType::Character);
    assert_eq!(card.ink_type, vec![InkType::Amethyst]);
    assert_eq!(card.cost, 6);
    assert_eq!(card.full_name(), "Elsa - Ice Queen");
    assert!(!card.is_dual_ink());

    let abilities = card.abilities.as_ref().unwrap();
    assert_eq!(abilities.len(), 2);
    assert_eq!(abilities[0].kind(), AbilityType::Keyword);
    assert_eq!(abilities[0].keyword(), Some(KeywordType::Ward));
    assert_eq!(abilities[0].id(), Some("set1-001-ability-1"));
    assert_eq!(abilities[1].kind(), AbilityType::Triggered);
}

#[test]
fn song_action_card_fixture_loads_losslessly() {
    let json = include_str!("fixtures/valid_song_action_card.json");
    assert_lossless::<LorcanaCardDefinition>(json);

    let card: LorcanaCardDefinition = serde_json::from_str(json).unwrap();
    assert_eq!(card.card_type, CardType::Action);
    assert_eq!(
        card.action_subtype,
        Some(lorcana_schema::ActionSubtype::Song)
    );
    // A card with no version: full name falls back to the bare name.
    assert_eq!(card.full_name(), "Let It Go");
}

#[test]
fn dual_ink_card_detected() {
    let json = r#"{"id":"x","canonicalId":"ci_x","name":"Dual","inkType":["amber","steel"],"cost":3,"inkable":true,"cardType":"item","i18n":{"en":{"name":"Dual"}},"set":"TFC"}"#;
    let card: LorcanaCardDefinition = serde_json::from_str(json).unwrap();
    assert!(card.is_dual_ink());
    assert_eq!(card.card_type, CardType::Item);
}
