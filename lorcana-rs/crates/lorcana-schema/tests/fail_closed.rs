//! Negative / fail-closed tests (blueprint Step 1 validation: "same rejected
//! malformed definitions"; CLAUDE.md: invalid input must fail closed).

use lorcana_schema::{
    AbilityCost, ActionSubtype, Condition, DeckValidationError, Effect, InkType,
    LorcanaCardDefinition,
};

fn rejects<T: serde::de::DeserializeOwned>(json: &str) {
    let result: Result<T, _> = serde_json::from_str(json);
    assert!(result.is_err(), "expected rejection, but parsed: {json}");
}

/// A minimal valid character card with `{ABILITIES}` spliced in, for exercising
/// nested-DSL validation against an otherwise-valid card.
fn character_with(abilities: &str) -> String {
    format!(
        r#"{{"id":"x","canonicalId":"ci","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"character","strength":1,"willpower":1,"lore":1,"i18n":{{"en":{{"name":"X"}}}},"set":"001","abilities":{abilities}}}"#
    )
}

#[test]
fn unknown_condition_discriminator_is_rejected() {
    rejects::<Condition>(r#"{"type":"totally-not-a-condition"}"#);
}

#[test]
fn condition_missing_discriminator_is_rejected() {
    rejects::<Condition>(r#"{"name":"Elsa","controller":"you"}"#);
}

#[test]
fn unknown_effect_discriminator_is_rejected() {
    rejects::<Effect>(r#"{"type":"vaporize-everything","amount":99}"#);
}

#[test]
fn unknown_ink_type_is_rejected() {
    rejects::<InkType>(r#""plaid""#);
}

#[test]
fn unknown_deck_error_tag_is_rejected() {
    rejects::<DeckValidationError>(r#"{"type":"DECK_ON_FIRE","count":1}"#);
}

#[test]
fn ability_cost_wrong_scalar_type_is_rejected() {
    // `ink` is numeric; a string must fail closed rather than be silently coerced.
    rejects::<AbilityCost>(r#"{"ink":"two"}"#);
}

#[test]
fn card_unknown_card_type_is_rejected() {
    rejects::<LorcanaCardDefinition>(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"spell","i18n":{"en":{"name":"X"}},"set":"TFC"}"#,
    );
}

#[test]
fn card_unknown_ink_type_is_rejected() {
    rejects::<LorcanaCardDefinition>(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["plaid"],"cost":1,"inkable":true,"cardType":"item","i18n":{"en":{"name":"X"}},"set":"TFC"}"#,
    );
}

#[test]
fn card_unknown_classification_is_rejected() {
    rejects::<LorcanaCardDefinition>(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"character","classifications":["Wizard"],"i18n":{"en":{"name":"X"}},"set":"TFC"}"#,
    );
}

#[test]
fn card_missing_required_cost_is_rejected() {
    rejects::<LorcanaCardDefinition>(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"inkable":true,"cardType":"item","i18n":{"en":{"name":"X"}},"set":"TFC"}"#,
    );
}

#[test]
fn card_missing_required_i18n_is_rejected() {
    rejects::<LorcanaCardDefinition>(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"item","set":"TFC"}"#,
    );
}

// ---------------------------------------------------------------------------
// Null-rejection: `field?: T` (optional, non-nullable) must reject explicit null
// ---------------------------------------------------------------------------

#[test]
fn explicit_null_for_optional_non_null_scalar_is_rejected() {
    // `ink?: number` does not admit null.
    rejects::<AbilityCost>(r#"{"ink":null}"#);
    // `discardCards?: number`, `target?: string` likewise.
    rejects::<AbilityCost>(r#"{"discardCards":null}"#);
    rejects::<AbilityCost>(r#"{"target":null}"#);
}

#[test]
fn explicit_null_for_optional_non_null_card_field_is_rejected() {
    // `version?: string` is optional-but-not-nullable in the oracle.
    rejects::<LorcanaCardDefinition>(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","version":null,"inkType":["amber"],"cost":1,"inkable":true,"cardType":"item","i18n":{"en":{"name":"X"}},"set":"TFC"}"#,
    );
}

#[test]
fn action_subtype_null_is_accepted_and_becomes_none() {
    // `actionSubtype?: "song" | null` is one of the few fields that DOES admit
    // an explicit null (semantically "not a song"); it must parse, not reject.
    let card: LorcanaCardDefinition = serde_json::from_str(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"action","actionSubtype":null,"i18n":{"en":{"name":"X"}},"set":"TFC"}"#,
    )
    .expect("explicit actionSubtype null must be accepted");
    assert_eq!(card.action_subtype, None);

    let song: LorcanaCardDefinition = serde_json::from_str(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"action","actionSubtype":"song","i18n":{"en":{"name":"X"}},"set":"TFC"}"#,
    )
    .unwrap();
    assert_eq!(song.action_subtype, Some(ActionSubtype::Song));
}

// ---------------------------------------------------------------------------
// Closed scalar unions must reject out-of-vocabulary values
// ---------------------------------------------------------------------------

#[test]
fn bad_banish_item_union_is_rejected() {
    // `banishItem?: boolean | number` — a string is neither.
    rejects::<AbilityCost>(r#"{"banishItem":"two"}"#);
    rejects::<AbilityCost>(r#"{"banishItem":null}"#);
    // Valid forms still parse.
    assert!(serde_json::from_str::<AbilityCost>(r#"{"banishItem":true}"#).is_ok());
    assert!(serde_json::from_str::<AbilityCost>(r#"{"banishItem":2}"#).is_ok());
}

#[test]
fn bad_discard_card_type_union_is_rejected() {
    // `discardCardType?: CardType | "song"` — "spell" is neither.
    rejects::<AbilityCost>(r#"{"discardCardType":"spell"}"#);
    assert!(serde_json::from_str::<AbilityCost>(r#"{"discardCardType":"song"}"#).is_ok());
    assert!(serde_json::from_str::<AbilityCost>(r#"{"discardCardType":"character"}"#).is_ok());
}

#[test]
fn bad_card_copy_limit_union_is_rejected() {
    // `cardCopyLimit?: number | "no-limit"` — any other string is rejected.
    rejects::<LorcanaCardDefinition>(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"item","cardCopyLimit":"unlimited","i18n":{"en":{"name":"X"}},"set":"TFC"}"#,
    );
    // Both valid forms parse.
    assert!(serde_json::from_str::<LorcanaCardDefinition>(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"item","cardCopyLimit":"no-limit","i18n":{"en":{"name":"X"}},"set":"TFC"}"#
    )
    .is_ok());
    assert!(serde_json::from_str::<LorcanaCardDefinition>(
        r#"{"id":"x","canonicalId":"ci_x","name":"X","inkType":["amber"],"cost":1,"inkable":true,"cardType":"item","cardCopyLimit":99,"i18n":{"en":{"name":"X"}},"set":"TFC"}"#
    )
    .is_ok());
}

// ---------------------------------------------------------------------------
// Nested-DSL validation: a card with a bad nested discriminator fails closed
// ---------------------------------------------------------------------------

#[test]
fn card_bad_keyword_ability_is_rejected() {
    rejects::<LorcanaCardDefinition>(&character_with(
        r#"[{"type":"keyword","keyword":"Flying"}]"#,
    ));
}

#[test]
fn card_bad_nested_effect_type_is_rejected() {
    rejects::<LorcanaCardDefinition>(&character_with(
        r#"[{"type":"action","effect":{"type":"vaporize"}}]"#,
    ));
}

#[test]
fn card_bad_deeply_nested_effect_type_is_rejected() {
    // effect-inside-effect: recursion must reach the inner bad discriminator.
    rejects::<LorcanaCardDefinition>(&character_with(
        r#"[{"type":"action","effect":{"type":"conditional","effect":{"type":"vaporize"}}}]"#,
    ));
}

#[test]
fn card_bad_nested_condition_type_is_rejected() {
    rejects::<LorcanaCardDefinition>(&character_with(
        r#"[{"type":"static","condition":{"type":"phase-of-moon"},"effect":{"type":"draw"}}]"#,
    ));
}

#[test]
fn card_bad_nested_trigger_event_is_rejected() {
    rejects::<LorcanaCardDefinition>(&character_with(
        r#"[{"type":"triggered","trigger":{"event":"sneeze"},"effect":{"type":"draw"}}]"#,
    ));
}

#[test]
fn card_bad_effects_array_member_is_rejected() {
    rejects::<LorcanaCardDefinition>(&character_with(
        r#"[{"type":"action","effect":{"type":"sequence","effects":[{"type":"draw"},{"type":"vaporize"}]}}]"#,
    ));
}

#[test]
fn trigger_bad_event_is_rejected_directly() {
    rejects::<lorcana_schema::Trigger>(r#"{"event":"sneeze","timing":"when"}"#);
    // A valid event parses.
    assert!(
        serde_json::from_str::<lorcana_schema::Trigger>(r#"{"event":"play","timing":"when"}"#)
            .is_ok()
    );
}

/// Splice a single action-ability whose `effect` is `{ROOT}` into a card.
fn character_with_root_effect(root: &str) -> String {
    character_with(&format!(r#"[{{"type":"action","effect":{root}}}]"#))
}

#[test]
fn card_bad_control_flow_effect_array_containers_are_rejected() {
    // Effect[] containers that are NOT named `effects`: validated by parent kind.
    // choice/or/modal -> options/choices ; sequence -> steps.
    for root in [
        r#"{"type":"choice","options":[{"type":"draw"},{"type":"vaporize"}]}"#,
        r#"{"type":"choice","choices":[{"type":"vaporize"}]}"#,
        r#"{"type":"or","options":[{"type":"vaporize"}]}"#,
        r#"{"type":"modal","options":[{"type":"vaporize"}]}"#,
        r#"{"type":"sequence","steps":[{"type":"draw"},{"type":"vaporize"}]}"#,
    ] {
        rejects::<LorcanaCardDefinition>(&character_with_root_effect(root));
    }
}

#[test]
fn card_bad_control_flow_effect_singular_containers_are_rejected() {
    // Effect containers that are NOT named `effect`: then/else/ifTrue/ifFalse.
    for root in [
        r#"{"type":"conditional","then":{"type":"vaporize"}}"#,
        r#"{"type":"conditional","else":{"type":"vaporize"}}"#,
        r#"{"type":"conditional","ifTrue":{"type":"vaporize"}}"#,
        r#"{"type":"conditional","ifFalse":{"type":"vaporize"}}"#,
        r#"{"type":"reveal-and-conditional","ifTrue":{"type":"vaporize"}}"#,
        r#"{"type":"reveal-and-conditional","ifFalse":{"type":"vaporize"}}"#,
    ] {
        rejects::<LorcanaCardDefinition>(&character_with_root_effect(root));
    }
}

#[test]
fn card_bad_nested_in_container_dsl_is_rejected() {
    // Nested DSL fields living inside a container array/object (not a direct key).
    for root in [
        // reveal-and-route: routes[*].condition (Condition) and sideEffects (Effect[])
        r#"{"type":"reveal-and-route","routes":[{"condition":{"type":"phase-of-moon"},"destination":{"zone":"hand"}}]}"#,
        r#"{"type":"reveal-and-route","routes":[{"condition":{"type":"used-shift"},"destination":{"zone":"hand"},"sideEffects":[{"type":"vaporize"}]}]}"#,
        // create-triggered-ability: ability.trigger (Trigger), ability.condition (Condition), ability.effect (Effect)
        r#"{"type":"create-triggered-ability","ability":{"trigger":{"event":"sneeze"},"effect":{"type":"draw"}}}"#,
        r#"{"type":"create-triggered-ability","ability":{"trigger":{"event":"quest"},"condition":{"type":"phase-of-moon"},"effect":{"type":"draw"}}}"#,
        r#"{"type":"create-triggered-ability","ability":{"trigger":{"event":"quest"},"effect":{"type":"vaporize"}}}"#,
    ] {
        rejects::<LorcanaCardDefinition>(&character_with_root_effect(root));
    }
}

#[test]
fn nested_in_container_valid_dsl_is_accepted() {
    for root in [
        r#"{"type":"reveal-and-route","routes":[{"condition":{"type":"used-shift"},"destination":{"zone":"hand"},"sideEffects":[{"type":"gain-lore"}]}],"fallback":{"zone":"deck-top"}}"#,
        r#"{"type":"create-triggered-ability","ability":{"trigger":{"event":"quest"},"condition":{"type":"used-shift"},"effect":{"type":"draw"}},"lifecycle":{"kind":"floating","duration":"this-turn"}}"#,
    ] {
        let json = character_with_root_effect(root);
        assert!(
            serde_json::from_str::<LorcanaCardDefinition>(&json).is_ok(),
            "valid nested-in-container DSL was wrongly rejected: {root}"
        );
    }
}

#[test]
fn control_flow_containers_are_not_over_rejected() {
    // The `options` key is `Effect[]` only for choice/or/modal — on `put-on-deck`
    // it is a positioning list, so it must NOT be validated as effects.
    for root in [
        r#"{"type":"put-on-deck","options":[{"position":"top"},"top-of-deck"]}"#,
        // `replacement` (Effect | "prevent") admits the literal and non-Effect-
        // union objects (e.g. `prevent-remove-damage`) seen in real cards.
        r#"{"type":"replacement","replaces":"damage","replacement":"prevent"}"#,
        r#"{"type":"replacement","replaces":"damage","replacement":{"type":"prevent-remove-damage"}}"#,
        // Valid nested control-flow effects parse.
        r#"{"type":"conditional","then":{"type":"draw"},"else":{"type":"gain-lore"}}"#,
        r#"{"type":"choice","options":[{"type":"draw"},{"type":"banish"}]}"#,
        r#"{"type":"sequence","steps":[{"type":"draw"}]}"#,
    ] {
        let json = character_with_root_effect(root);
        assert!(
            serde_json::from_str::<LorcanaCardDefinition>(&json).is_ok(),
            "valid/overloaded control-flow container was wrongly rejected: {root}"
        );
    }
}

#[test]
fn good_nested_dsl_card_is_accepted() {
    // Mirror of the rejection cases, but all discriminators valid: must parse.
    let json = character_with(
        r#"[{"type":"keyword","keyword":"Rush"},{"type":"triggered","trigger":{"event":"play"},"effect":{"type":"conditional","condition":{"type":"used-shift"},"effect":{"type":"draw"}}}]"#,
    );
    assert!(
        serde_json::from_str::<LorcanaCardDefinition>(&json).is_ok(),
        "valid nested DSL should be accepted"
    );
}
