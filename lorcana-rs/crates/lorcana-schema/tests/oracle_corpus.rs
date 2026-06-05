//! Real-card conformance (blueprint Step 1 validation: "same accepted card
//! definitions", lossless serialization, and the fail-closed nested-DSL
//! validation proven against *real* oracle data rather than only hand-authored
//! examples).
//!
//! Fixtures are derived from the frozen oracle card catalog
//! (`oracle/source/.../lorcana-cards`, serialized via the package's own
//! `allCards` export):
//! * `oracle_mickey_mouse_artful_rogue.json` — one documented real card.
//! * `oracle_cards_sample.json` — a 130-card greedy cover that exercises every
//!   discriminator value present in the full 2754-card catalog (all 200 effect/
//!   condition `type:` values, every trigger `event`, every keyword, plus the
//!   `cardCopyLimit` / `banishItem` / `discardCardType` / `actionSubtype` forms).
//!
//! The full catalog can additionally be checked by pointing
//! `LORCANA_CARD_CORPUS` at a JSON array of cards.

use lorcana_schema::{AbilityType, CardType, InkType, KeywordType, LorcanaCardDefinition};
use serde_json::Value;

/// Deserialize every card and assert a data-preserving (order-insensitive)
/// round-trip. Returns the parsed cards for further assertions.
fn load_all(json: &str) -> Vec<LorcanaCardDefinition> {
    let originals: Vec<Value> = serde_json::from_str(json).expect("parse corpus as JSON array");
    let mut out = Vec::with_capacity(originals.len());
    for (i, original) in originals.into_iter().enumerate() {
        let id = original
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let card: LorcanaCardDefinition = serde_json::from_value(original.clone())
            .unwrap_or_else(|e| panic!("card #{i} (id={id}) failed to deserialize: {e}"));
        let reserialized: Value = serde_json::from_str(&serde_json::to_string(&card).unwrap())
            .expect("re-parse reserialized card");
        assert_eq!(
            reserialized, original,
            "card #{i} (id={id}) did not round-trip losslessly"
        );
        out.push(card);
    }
    out
}

#[test]
fn oracle_card_sample_deserializes_losslessly() {
    let cards = load_all(include_str!("fixtures/oracle_cards_sample.json"));
    assert_eq!(cards.len(), 130, "sample size changed unexpectedly");
    // Every card has a valid typed card_type (the discriminator parsed).
    for card in &cards {
        let _ = card.card_type; // typed; would have failed to parse otherwise
    }
}

#[test]
fn oracle_mickey_mouse_artful_rogue_typed_fields() {
    let json = include_str!("fixtures/oracle_mickey_mouse_artful_rogue.json");
    let original: Value = serde_json::from_str(json).unwrap();
    let card: LorcanaCardDefinition = serde_json::from_value(original.clone()).unwrap();

    // Round-trip is lossless.
    let reser: Value = serde_json::from_str(&serde_json::to_string(&card).unwrap()).unwrap();
    assert_eq!(reser, original);

    assert_eq!(card.card_type, CardType::Character);
    assert_eq!(card.ink_type, vec![InkType::Emerald]);
    assert_eq!(card.full_name(), "Mickey Mouse - Artful Rogue");

    let abilities = card.abilities.as_ref().unwrap();
    assert_eq!(abilities.len(), 2);
    // Keyword ability: Shift (validated as a KeywordType on deserialize).
    assert_eq!(abilities[0].kind(), AbilityType::Keyword);
    assert_eq!(abilities[0].keyword(), Some(KeywordType::Shift));
    // Triggered ability with a nested validated effect + trigger.
    assert_eq!(abilities[1].kind(), AbilityType::Triggered);
    assert_eq!(abilities[1].name(), Some("MISDIRECTION"));
}

/// Optional whole-catalog check: set `LORCANA_CARD_CORPUS` to a JSON array of
/// every oracle card to prove zero rejections / zero drift across the catalog.
#[test]
fn full_corpus_deserializes_losslessly_when_present() {
    let Ok(path) = std::env::var("LORCANA_CARD_CORPUS") else {
        eprintln!("LORCANA_CARD_CORPUS not set; skipping full-catalog check");
        return;
    };
    let json = std::fs::read_to_string(&path).expect("read corpus file");
    let cards = load_all(&json);
    assert!(
        cards.len() > 2000,
        "expected the full catalog (>2000 cards), got {}",
        cards.len()
    );
}
