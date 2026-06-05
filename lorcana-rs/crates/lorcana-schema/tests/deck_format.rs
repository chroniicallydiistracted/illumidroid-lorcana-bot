//! DAC format-validation conformance (blueprint Step 2 — "Port deck-validation
//! rules"). Mirrors `oracle/.../decks/validate-deck.test.ts` case-for-case, then
//! adds field-exact `LORCANA_FORMATS` checks, message-exact checks, order and
//! unknown-lookup checks, and Serde fail-closed checks for the format/card types.

use std::collections::BTreeMap;

use lorcana_schema::card::CardCopyLimit;
use lorcana_schema::{
    CardFormatData, DeckCard, DeckFormatResult, FormatRuleResult, FormatValidationKind,
    LORCANA_FORMATS, LorcanaFormat, LorcanaFormatId, LorcanaSetCode, get_deck_formats,
    validate_deck, validate_deck_for_format,
};

// ---------------------------------------------------------------------------
// Helpers (mirroring the oracle test helpers)
// ---------------------------------------------------------------------------

fn sets(codes: &[LorcanaSetCode]) -> Vec<LorcanaSetCode> {
    codes.to_vec()
}

/// Minimal format for testing: 60-card minimum, 2-ink max, CoreConstructed rotation.
fn core_format() -> LorcanaFormat {
    LorcanaFormat {
        id: LorcanaFormatId::CoreConstructed,
        label: "Core Constructed".to_string(),
        description: None,
        allowed_sets: sets(&[
            LorcanaSetCode::Ssk,
            LorcanaSetCode::Azs,
            LorcanaSetCode::Arc,
            LorcanaSetCode::Roj,
            LorcanaSetCode::Fab,
            LorcanaSetCode::Wiw,
        ]),
        banned_card_ids: None,
        min_deck_size: None,
        max_ink_types: None,
        special_allowed_card_ids: None,
        required_rotation_state: Some("CoreConstructed".to_string()),
        requires_any_set: None,
        excluded_sets: None,
    }
}

/// Historical snapshot format with no `requiredRotationState`.
fn history_format() -> LorcanaFormat {
    LorcanaFormat {
        id: LorcanaFormatId::ShimmeringSkies,
        label: "Shimmering Skies".to_string(),
        description: None,
        allowed_sets: sets(&[
            LorcanaSetCode::Tfc,
            LorcanaSetCode::Rof,
            LorcanaSetCode::Iti,
            LorcanaSetCode::Urr,
            LorcanaSetCode::Ssk,
        ]),
        banned_card_ids: None,
        min_deck_size: None,
        max_ink_types: None,
        special_allowed_card_ids: None,
        required_rotation_state: None,
        requires_any_set: None,
        excluded_sets: None,
    }
}

/// A `CardFormatData` for shortId `id` with `canonicalId = "ci_{id}"`, default
/// SSK printing, amber ink. Callers tweak fields after construction.
fn card(id: &str) -> CardFormatData {
    CardFormatData {
        canonical_id: format!("ci_{id}"),
        full_name: format!("Test Card {id}"),
        sets: sets(&[LorcanaSetCode::Ssk]),
        ink_types: vec!["amber".to_string()],
        card_copy_limit: None,
        rotation_states: None,
    }
}

fn deck(entries: &[(&str, i64)]) -> Vec<DeckCard> {
    entries
        .iter()
        .map(|(id, qty)| DeckCard {
            card_id: (*id).to_string(),
            quantity: *qty,
        })
        .collect()
}

/// Build an owned-clone lookup closure backed by a card map.
fn lookup_from(cards: BTreeMap<String, CardFormatData>) -> impl Fn(&str) -> Option<CardFormatData> {
    move |id: &str| cards.get(id).cloned()
}

fn rule(result: &DeckFormatResult, kind: FormatValidationKind) -> &FormatRuleResult {
    result
        .rules
        .iter()
        .find(|r| r.kind == kind)
        .unwrap_or_else(|| panic!("missing rule of kind {kind:?}"))
}

fn map(entries: Vec<(&str, CardFormatData)>) -> BTreeMap<String, CardFormatData> {
    entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

// ===========================================================================
// CARD_SET rule (mirrors oracle "CARD_SET rule" describe block)
// ===========================================================================

#[test]
fn card_set_passes_when_printing_in_allowed_set() {
    let lookup = lookup_from(map(vec![("a", card("a"))])); // SSK is in core
    let result = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &core_format());
    assert!(rule(&result, FormatValidationKind::CardSet).passed);
}

#[test]
fn card_set_fails_when_no_printing_in_allowed_set() {
    let mut a = card("a");
    a.sets = sets(&[LorcanaSetCode::Tfc]); // not in core
    let lookup = lookup_from(map(vec![("a", a)]));
    let result = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &core_format());
    let set_rule = rule(&result, FormatValidationKind::CardSet);
    assert!(!set_rule.passed);
    assert!(set_rule.message.contains("Test Card a"));
}

#[test]
fn card_set_passes_via_rotation_state() {
    let mut a = card("a");
    a.sets = sets(&[LorcanaSetCode::Tfc]); // old set, not in allowedSets
    a.rotation_states = Some(vec!["CoreConstructed".to_string()]); // rotation matches
    let lookup = lookup_from(map(vec![("a", a)]));
    let result = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &core_format());
    assert!(rule(&result, FormatValidationKind::CardSet).passed);
}

#[test]
fn card_set_falls_back_to_set_only_when_rotation_undefined() {
    let mut a = card("a");
    a.sets = sets(&[LorcanaSetCode::Tfc]); // not in core
    a.rotation_states = None; // no rotation data
    let lookup = lookup_from(map(vec![("a", a)]));
    let result = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &core_format());
    assert!(!rule(&result, FormatValidationKind::CardSet).passed);
}

#[test]
fn card_set_ignores_rotation_for_historical_formats() {
    let mut a = card("a");
    a.sets = sets(&[LorcanaSetCode::Azs]); // NOT in shimmering-skies allowed sets
    a.rotation_states = Some(vec!["CoreConstructed".to_string()]);
    let lookup = lookup_from(map(vec![("a", a)]));
    let result = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &history_format());
    assert!(!rule(&result, FormatValidationKind::CardSet).passed);
}

#[test]
fn card_set_rejects_only_excluded_set_printings_even_when_rotation_matches() {
    let mut a = card("a");
    a.sets = sets(&[LorcanaSetCode::Wun]);
    a.rotation_states = Some(vec!["CoreConstructed".to_string()]);
    let lookup = lookup_from(map(vec![("a", a)]));
    let mut format = core_format();
    format.excluded_sets = Some(sets(&[LorcanaSetCode::Wun]));
    let result = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &format);
    assert!(!rule(&result, FormatValidationKind::CardSet).passed);
}

#[test]
fn card_set_allows_excluded_set_card_with_an_allowed_printing() {
    let mut a = card("a");
    a.sets = sets(&[LorcanaSetCode::Wun, LorcanaSetCode::Ssk]);
    a.rotation_states = Some(vec!["CoreConstructed".to_string()]);
    let lookup = lookup_from(map(vec![("a", a)]));
    let mut format = core_format();
    format.excluded_sets = Some(sets(&[LorcanaSetCode::Wun]));
    let result = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &format);
    assert!(rule(&result, FormatValidationKind::CardSet).passed);
}

#[test]
fn card_set_special_allowed_ids_bypass_set_and_rotation() {
    let mut promo = card("promo");
    promo.sets = sets(&[LorcanaSetCode::Tfc]); // not in core
    promo.rotation_states = Some(vec!["InfinityConstructed".to_string()]); // doesn't match
    let lookup = lookup_from(map(vec![("promo", promo)]));
    let mut format = core_format();
    format.special_allowed_card_ids = Some(vec!["promo".to_string()]);
    let result = validate_deck_for_format(&deck(&[("promo", 60)]), &lookup, &format);
    assert!(rule(&result, FormatValidationKind::CardSet).passed);
}

// ===========================================================================
// CARD_QUANTITY rule (copy limits across reprints)
// ===========================================================================

#[test]
fn card_quantity_enforces_limit_across_shortids_same_canonical() {
    // Two different shortIds, same canonical card (both "ci_shared").
    let lookup = lookup_from(map(vec![("v1", card("shared")), ("v2", card("shared"))]));
    let mut format = core_format();
    format.min_deck_size = Some(6);
    let result = validate_deck_for_format(&deck(&[("v1", 3), ("v2", 3)]), &lookup, &format);
    let qty_rule = rule(&result, FormatValidationKind::CardQuantity);
    assert!(!qty_rule.passed);
    assert!(qty_rule.message.contains("6 copies"));
}

#[test]
fn card_quantity_passes_when_combined_within_limit() {
    let lookup = lookup_from(map(vec![("v1", card("shared")), ("v2", card("shared"))]));
    let mut format = core_format();
    format.min_deck_size = Some(4);
    let result = validate_deck_for_format(&deck(&[("v1", 2), ("v2", 2)]), &lookup, &format);
    assert!(rule(&result, FormatValidationKind::CardQuantity).passed);
}

#[test]
fn card_quantity_honors_raised_limit_99() {
    let mut wagger = card("wagger");
    wagger.card_copy_limit = Some(CardCopyLimit::Count(99));
    let lookup = lookup_from(map(vec![("wagger", wagger)]));
    let result = validate_deck_for_format(&deck(&[("wagger", 60)]), &lookup, &core_format());
    assert!(rule(&result, FormatValidationKind::CardQuantity).passed);
}

#[test]
fn card_quantity_honors_no_limit() {
    let mut bots = card("bots");
    bots.card_copy_limit = Some(CardCopyLimit::NoLimit(lorcana_schema::NoLimitTag::NoLimit));
    let lookup = lookup_from(map(vec![("bots", bots)]));
    let result = validate_deck_for_format(&deck(&[("bots", 60)]), &lookup, &core_format());
    assert!(rule(&result, FormatValidationKind::CardQuantity).passed);
}

#[test]
fn card_quantity_honors_lowered_limit_2() {
    let mut slipper = card("slipper");
    slipper.card_copy_limit = Some(CardCopyLimit::Count(2));
    let lookup = lookup_from(map(vec![("slipper", slipper)]));
    let mut format = core_format();
    format.min_deck_size = Some(3);
    let result = validate_deck_for_format(&deck(&[("slipper", 3)]), &lookup, &format);
    let qty_rule = rule(&result, FormatValidationKind::CardQuantity);
    assert!(!qty_rule.passed);
    assert!(qty_rule.message.contains("3 copies (maximum 2)"));
}

#[test]
fn card_quantity_aggregates_reprints_against_override_limit() {
    let mut base = card("slipper");
    base.card_copy_limit = Some(CardCopyLimit::Count(2));
    let mut enchanted = card("slipper");
    enchanted.card_copy_limit = Some(CardCopyLimit::Count(2));
    let lookup = lookup_from(map(vec![("base", base), ("enchanted", enchanted)]));
    let mut format = core_format();
    format.min_deck_size = Some(3);
    let result =
        validate_deck_for_format(&deck(&[("base", 2), ("enchanted", 1)]), &lookup, &format);
    let qty_rule = rule(&result, FormatValidationKind::CardQuantity);
    assert!(!qty_rule.passed);
    assert!(qty_rule.message.contains("3 copies (maximum 2)"));
}

// ===========================================================================
// Format definitions (mirrors oracle "format definitions" describe block)
// ===========================================================================

#[test]
fn format_core_constructed_has_required_rotation_state() {
    assert_eq!(
        LORCANA_FORMATS[&LorcanaFormatId::CoreConstructed].required_rotation_state,
        Some("CoreConstructed".to_string())
    );
}

#[test]
fn format_infinity_has_no_required_rotation_state() {
    assert_eq!(
        LORCANA_FORMATS[&LorcanaFormatId::Infinity].required_rotation_state,
        None
    );
}

#[test]
fn format_historical_formats_have_no_required_rotation_state() {
    for id in [
        LorcanaFormatId::ShimmeringSkies,
        LorcanaFormatId::AzuriteSea,
        LorcanaFormatId::ArchaziasIsland,
    ] {
        assert_eq!(LORCANA_FORMATS[&id].required_rotation_state, None);
    }
}

#[test]
fn format_infinity_includes_wun() {
    assert!(
        LORCANA_FORMATS[&LorcanaFormatId::Infinity]
            .allowed_sets
            .contains(&LorcanaSetCode::Wun)
    );
}

#[test]
fn format_core_constructed_includes_wun_and_does_not_exclude_it() {
    let cc = &LORCANA_FORMATS[&LorcanaFormatId::CoreConstructed];
    assert!(cc.allowed_sets.contains(&LorcanaSetCode::Wun));
    assert!(
        !cc.excluded_sets
            .clone()
            .unwrap_or_default()
            .contains(&LorcanaSetCode::Wun)
    );
}

#[test]
fn wun_only_card_is_legal_in_infinity_and_core_constructed() {
    let mut wun = card("wun");
    wun.sets = sets(&[LorcanaSetCode::Wun]);
    wun.rotation_states = Some(vec!["CoreConstructed".to_string()]);
    let mut ssk = card("ssk");
    ssk.sets = sets(&[LorcanaSetCode::Ssk]);
    ssk.rotation_states = Some(vec!["CoreConstructed".to_string()]);
    let lookup = lookup_from(map(vec![("wun", wun), ("ssk", ssk)]));
    let deck_cards = deck(&[("wun", 4), ("ssk", 56)]);

    let infinity = validate_deck_for_format(
        &deck_cards,
        &lookup,
        &LORCANA_FORMATS[&LorcanaFormatId::Infinity],
    );
    assert!(rule(&infinity, FormatValidationKind::CardSet).passed);

    let cc = validate_deck_for_format(
        &deck_cards,
        &lookup,
        &LORCANA_FORMATS[&LorcanaFormatId::CoreConstructed],
    );
    assert!(rule(&cc, FormatValidationKind::CardSet).passed);
}

// ===========================================================================
// Field-exact LORCANA_FORMATS definitions
// ===========================================================================

#[test]
fn lorcana_formats_declaration_order_is_exact() {
    let ids: Vec<LorcanaFormatId> = LORCANA_FORMATS.keys().copied().collect();
    assert_eq!(
        ids,
        vec![
            LorcanaFormatId::Infinity,
            LorcanaFormatId::CoreConstructed,
            LorcanaFormatId::ArchaziasIsland,
            LorcanaFormatId::ShimmeringSkies,
            LorcanaFormatId::AzuriteSea,
        ]
    );
}

#[test]
fn lorcana_formats_fields_are_exact() {
    use LorcanaSetCode::*;

    let infinity = &LORCANA_FORMATS[&LorcanaFormatId::Infinity];
    assert_eq!(infinity.label, "Infinity");
    assert_eq!(
        infinity.description.as_deref(),
        Some("All released sets are legal. One card is banned.")
    );
    assert_eq!(
        infinity.allowed_sets,
        vec![Tfc, Rof, Iti, Urr, Ssk, Azs, Arc, Roj, Fab, Wiw, Wsp, Wun]
    );
    assert_eq!(infinity.banned_card_ids, Some(vec!["LsX".to_string()]));
    assert_eq!(infinity.required_rotation_state, None);
    assert_eq!(infinity.special_allowed_card_ids, None);

    let cc = &LORCANA_FORMATS[&LorcanaFormatId::CoreConstructed];
    assert_eq!(cc.label, "Core Constructed");
    assert_eq!(
        cc.description.as_deref(),
        Some("Current rotating format. Sets SSK through WUN are legal.")
    );
    assert_eq!(
        cc.allowed_sets,
        vec![Ssk, Azs, Arc, Roj, Fab, Wiw, Wsp, Wun]
    );
    assert_eq!(
        cc.banned_card_ids,
        Some(vec!["LsX".to_string(), "PSk".to_string()])
    );
    assert_eq!(
        cc.required_rotation_state.as_deref(),
        Some("CoreConstructed")
    );
    assert_eq!(cc.excluded_sets, None);

    let arch = &LORCANA_FORMATS[&LorcanaFormatId::ArchaziasIsland];
    assert_eq!(arch.label, "Archazia's Island");
    assert_eq!(arch.allowed_sets, vec![Ssk, Azs, Arc, Roj, Fab]);
    assert_eq!(arch.banned_card_ids, None);

    let ss = &LORCANA_FORMATS[&LorcanaFormatId::ShimmeringSkies];
    assert_eq!(ss.label, "Shimmering Skies");
    assert_eq!(ss.allowed_sets, vec![Tfc, Rof, Iti, Urr, Ssk]);

    let azs = &LORCANA_FORMATS[&LorcanaFormatId::AzuriteSea];
    assert_eq!(azs.label, "Azurite Sea");
    assert_eq!(azs.allowed_sets, vec![Tfc, Rof, Iti, Urr, Ssk, Azs]);
    // specialAllowedCardIds is an explicit empty array (not absent) in the oracle.
    assert_eq!(azs.special_allowed_card_ids, Some(Vec::new()));
}

// ===========================================================================
// Message-exact, rule-order, ink-order, unknown-lookup, banned, requiresAnySet
// ===========================================================================

#[test]
fn deck_size_messages_are_exact() {
    let lookup = lookup_from(map(vec![("a", card("a"))]));
    let pass = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &core_format());
    assert_eq!(
        rule(&pass, FormatValidationKind::DeckSize).message,
        "Deck has 60 cards (minimum 60)."
    );
    let fail = validate_deck_for_format(&deck(&[("a", 4)]), &lookup, &core_format());
    let r = rule(&fail, FormatValidationKind::DeckSize);
    assert!(!r.passed);
    assert_eq!(r.message, "Deck has 4 cards but requires at least 60.");
}

#[test]
fn ink_types_pass_message_and_fail_message_with_insertion_order() {
    // Pass: single ink.
    let lookup = lookup_from(map(vec![("a", card("a"))]));
    let pass = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &core_format());
    assert_eq!(
        rule(&pass, FormatValidationKind::InkTypes).message,
        "Deck uses 1 ink type(s) (maximum 2)."
    );

    // Fail: three inks, listed in first-seen insertion order (ruby, amber, steel).
    let mut ruby = card("r");
    ruby.ink_types = vec!["ruby".to_string()];
    let amber = card("a"); // amber
    let mut steel = card("s");
    steel.ink_types = vec!["steel".to_string()];
    let lookup3 = lookup_from(map(vec![("r", ruby), ("a", amber), ("s", steel)]));
    let fail = validate_deck_for_format(
        &deck(&[("r", 20), ("a", 20), ("s", 20)]),
        &lookup3,
        &core_format(),
    );
    let r = rule(&fail, FormatValidationKind::InkTypes);
    assert!(!r.passed);
    assert_eq!(
        r.message,
        "Deck uses 3 ink types (ruby, amber, steel), but at most 2 are allowed."
    );
}

#[test]
fn pass_messages_and_set_fail_message_are_exact() {
    // CARD_QUANTITY + CARD_SET pass messages (verified byte-identical to the TS
    // oracle), using a within-limit single-ink deck.
    let lookup = lookup_from(map(vec![("a", card("a"))]));
    let mut format = core_format();
    format.min_deck_size = Some(4);
    let pass = validate_deck_for_format(&deck(&[("a", 4)]), &lookup, &format);
    assert_eq!(
        rule(&pass, FormatValidationKind::CardQuantity).message,
        "All card quantities are within the allowed limits."
    );
    assert_eq!(
        rule(&pass, FormatValidationKind::CardSet).message,
        "All cards are legal for this format."
    );

    // CARD_SET fail message — full string (not just a substring).
    let mut tfc = card("a");
    tfc.sets = sets(&[LorcanaSetCode::Tfc]);
    let lookup2 = lookup_from(map(vec![("a", tfc)]));
    let fail = validate_deck_for_format(&deck(&[("a", 60)]), &lookup2, &core_format());
    assert_eq!(
        rule(&fail, FormatValidationKind::CardSet).message,
        "Cards not legal in Core Constructed: Test Card a (sets: TFC)."
    );

    // CARD_QUANTITY fail message — full string.
    let mut slipper = card("slipper");
    slipper.card_copy_limit = Some(CardCopyLimit::Count(2));
    let lookup3 = lookup_from(map(vec![("slipper", slipper)]));
    let mut f3 = core_format();
    f3.min_deck_size = Some(3);
    let qfail = validate_deck_for_format(&deck(&[("slipper", 3)]), &lookup3, &f3);
    assert_eq!(
        rule(&qfail, FormatValidationKind::CardQuantity).message,
        "Too many copies: Test Card slipper: 3 copies (maximum 2)."
    );
}

#[test]
fn rule_order_is_deck_size_ink_quantity_set_then_conditionals() {
    let lookup = lookup_from(map(vec![("a", card("a"))]));
    // core_format has no banned/requiresAnySet -> 4 rules, in fixed order.
    let result = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &core_format());
    let kinds: Vec<FormatValidationKind> = result.rules.iter().map(|r| r.kind).collect();
    assert_eq!(
        kinds,
        vec![
            FormatValidationKind::DeckSize,
            FormatValidationKind::InkTypes,
            FormatValidationKind::CardQuantity,
            FormatValidationKind::CardSet,
        ]
    );
}

#[test]
fn banned_card_rule_absent_without_banned_ids_present_with() {
    let lookup = lookup_from(map(vec![("a", card("a"))]));

    // No bannedCardIds on the format -> no BANNED_CARD rule emitted at all.
    let no_ban = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &core_format());
    assert!(
        no_ban
            .rules
            .iter()
            .all(|r| r.kind != FormatValidationKind::BannedCard)
    );

    // With a banned id present in the deck -> failing BANNED_CARD with fullName.
    let mut format = core_format();
    format.banned_card_ids = Some(vec!["a".to_string()]);
    let banned = validate_deck_for_format(&deck(&[("a", 60)]), &lookup, &format);
    let r = rule(&banned, FormatValidationKind::BannedCard);
    assert!(!r.passed);
    assert_eq!(r.message, "Banned in Core Constructed: Test Card a.");
}

#[test]
fn banned_card_unknown_lookup_falls_back_to_card_id() {
    // Empty lookup -> banned card id is unknown -> message uses the raw cardId.
    let lookup = lookup_from(BTreeMap::new());
    let mut format = core_format();
    format.banned_card_ids = Some(vec!["ghost".to_string()]);
    let result = validate_deck_for_format(&deck(&[("ghost", 60)]), &lookup, &format);
    let r = rule(&result, FormatValidationKind::BannedCard);
    assert!(!r.passed);
    assert_eq!(r.message, "Banned in Core Constructed: ghost.");
}

#[test]
fn requires_any_set_rule_pass_and_fail() {
    let mut ssk = card("ssk");
    ssk.sets = sets(&[LorcanaSetCode::Ssk]);
    let mut wsp = card("wsp");
    wsp.sets = sets(&[LorcanaSetCode::Wsp]);
    let lookup = lookup_from(map(vec![("ssk", ssk), ("wsp", wsp)]));

    let mut format = core_format();
    format.requires_any_set = Some(sets(&[LorcanaSetCode::Wsp]));

    // Deck has no WSP card -> REQUIRES_ANY_SET fails.
    let fail = validate_deck_for_format(&deck(&[("ssk", 60)]), &lookup, &format);
    let rf = rule(&fail, FormatValidationKind::RequiresAnySet);
    assert!(!rf.passed);
    assert_eq!(
        rf.message,
        "Core Constructed requires at least one card from: WSP."
    );

    // Deck has a WSP card -> passes.
    let pass = validate_deck_for_format(&deck(&[("ssk", 56), ("wsp", 4)]), &lookup, &format);
    let rp = rule(&pass, FormatValidationKind::RequiresAnySet);
    assert!(rp.passed);
    assert_eq!(
        rp.message,
        "Deck contains at least one card from the required sets (WSP)."
    );

    // Without requiresAnySet -> the rule is not emitted.
    let none = validate_deck_for_format(&deck(&[("ssk", 60)]), &lookup, &core_format());
    assert!(
        none.rules
            .iter()
            .all(|r| r.kind != FormatValidationKind::RequiresAnySet)
    );
}

#[test]
fn unknown_cards_are_skipped_in_set_and_ink_checks() {
    // A deck made entirely of cards the lookup doesn't know: skipped for ink/set,
    // so CARD_SET passes (no failures) and INK_TYPES sees zero inks.
    let lookup = lookup_from(BTreeMap::new());
    let result = validate_deck_for_format(&deck(&[("ghost", 60)]), &lookup, &core_format());
    assert!(rule(&result, FormatValidationKind::CardSet).passed);
    let ink = rule(&result, FormatValidationKind::InkTypes);
    assert!(ink.passed);
    assert_eq!(ink.message, "Deck uses 0 ink type(s) (maximum 2).");
}

// ===========================================================================
// validate_deck / get_deck_formats (multi-format)
// ===========================================================================

#[test]
fn validate_deck_default_returns_one_result_per_format_in_order() {
    let lookup = lookup_from(map(vec![("a", card("a"))]));
    let results = validate_deck(&deck(&[("a", 60)]), &lookup, None);
    let ids: Vec<LorcanaFormatId> = results.iter().map(|r| r.format_id).collect();
    assert_eq!(
        ids,
        vec![
            LorcanaFormatId::Infinity,
            LorcanaFormatId::CoreConstructed,
            LorcanaFormatId::ArchaziasIsland,
            LorcanaFormatId::ShimmeringSkies,
            LorcanaFormatId::AzuriteSea,
        ]
    );
}

#[test]
fn validate_deck_explicit_formats_subset() {
    let lookup = lookup_from(map(vec![("a", card("a"))]));
    let cc = core_format();
    let results = validate_deck(&deck(&[("a", 60)]), &lookup, Some(&[&cc]));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].format_id, LorcanaFormatId::CoreConstructed);
}

#[test]
fn get_deck_formats_returns_legal_format_ids() {
    // An SSK card with CoreConstructed rotation, 60 copies via no-limit, single
    // ink. Legal in infinity, core-constructed, archazias-island, azurite-sea?
    // SSK is in all of those allowedSets except none excludes it; shimmering-skies
    // also allows SSK. So it is legal in every format -> all five ids, in order.
    let mut c = card("only");
    c.sets = sets(&[LorcanaSetCode::Ssk]);
    c.rotation_states = Some(vec!["CoreConstructed".to_string()]);
    c.card_copy_limit = Some(CardCopyLimit::NoLimit(lorcana_schema::NoLimitTag::NoLimit));
    let lookup = lookup_from(map(vec![("only", c)]));
    let formats = get_deck_formats(&deck(&[("only", 60)]), &lookup);
    assert_eq!(
        formats,
        vec![
            LorcanaFormatId::Infinity,
            LorcanaFormatId::CoreConstructed,
            LorcanaFormatId::ArchaziasIsland,
            LorcanaFormatId::ShimmeringSkies,
            LorcanaFormatId::AzuriteSea,
        ]
    );

    // A TFC-only card with no rotation is legal in every format whose allowedSets
    // include TFC: infinity (all sets), shimmering-skies, and azurite-sea — but
    // NOT core-constructed or archazias-island (TFC not in their sets, no rotation
    // match). (no-limit so the 60-copy deck passes CARD_QUANTITY, isolating CARD_SET.)
    let mut t = card("tfc");
    t.sets = sets(&[LorcanaSetCode::Tfc]);
    t.card_copy_limit = Some(CardCopyLimit::NoLimit(lorcana_schema::NoLimitTag::NoLimit));
    let lookup2 = lookup_from(map(vec![("tfc", t)]));
    let formats2 = get_deck_formats(&deck(&[("tfc", 60)]), &lookup2);
    assert_eq!(
        formats2,
        vec![
            LorcanaFormatId::Infinity,
            LorcanaFormatId::ShimmeringSkies,
            LorcanaFormatId::AzuriteSea
        ]
    );
}

// ===========================================================================
// Serde fail-closed checks for the format/card data types
// ===========================================================================

#[test]
fn card_format_data_rejects_unknown_set_code() {
    let json = r#"{"canonicalId":"ci_x","fullName":"X","sets":["XXX"],"inkTypes":["amber"]}"#;
    assert!(serde_json::from_str::<CardFormatData>(json).is_err());
}

#[test]
fn lorcana_format_rejects_unknown_format_id_and_explicit_null_label() {
    // Unknown LorcanaFormatId.
    assert!(
        serde_json::from_str::<LorcanaFormat>(
            r#"{"id":"bogus-format","label":"X","allowedSets":["SSK"]}"#
        )
        .is_err()
    );
    // Unknown set code in allowedSets.
    assert!(
        serde_json::from_str::<LorcanaFormat>(
            r#"{"id":"infinity","label":"X","allowedSets":["NOPE"]}"#
        )
        .is_err()
    );
    // `description?: string` is optional-but-not-nullable: explicit null must fail.
    assert!(
        serde_json::from_str::<LorcanaFormat>(
            r#"{"id":"infinity","label":"X","allowedSets":["SSK"],"description":null}"#
        )
        .is_err()
    );
}

#[test]
fn card_copy_limit_rejects_bad_string_in_card_format_data() {
    let json = r#"{"canonicalId":"ci_x","fullName":"X","sets":["SSK"],"inkTypes":["amber"],"cardCopyLimit":"unlimited"}"#;
    assert!(serde_json::from_str::<CardFormatData>(json).is_err());
    // Valid forms parse.
    let ok = r#"{"canonicalId":"ci_x","fullName":"X","sets":["SSK"],"inkTypes":["amber"],"cardCopyLimit":"no-limit"}"#;
    assert!(serde_json::from_str::<CardFormatData>(ok).is_ok());
}
