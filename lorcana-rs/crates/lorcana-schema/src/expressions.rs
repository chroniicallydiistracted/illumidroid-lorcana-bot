//! Amount / duration / variable-amount expression vocabularies.
//!
//! Mirrors the closed unions in
//! `oracle/source/packages/lorcana/lorcana-types/src/expressions/index.ts`
//! (`AmountRef`, `ForEachCounterType`, `EffectDuration` string forms, and the
//! `VariableAmount` discriminator set). The structured `VariableAmount` payloads
//! live inside DSL node `rest` maps; this module fixes the discriminators.

use crate::macros::str_enum;

str_enum! {
    /// `AmountRef` — string-keyed amount references.
    pub enum AmountRef {
        All = "all",
        Full = "full",
        DiscardedCount = "DISCARDED_COUNT",
        DiscardedCardLore = "DISCARDED_CARD_LORE",
        ReturnedCardCost = "RETURNED_CARD_COST",
        DamageDealt = "DAMAGE_DEALT",
        OpponentsDamagedCharacterCount = "OPPONENTS_DAMAGED_CHARACTER_COUNT",
        X = "X",
        DamageRemoved = "DAMAGE_REMOVED",
        DrawnCount = "DRAWN_COUNT",
        Hand = "HAND",
        TargetCost = "TARGET_COST",
        TargetStrength = "TARGET_STRENGTH",
        TargetWillpower = "TARGET_WILLPOWER",
    }
}

str_enum! {
    /// `ForEachCounterType` — counter kinds for for-each amounts.
    pub enum ForEachCounterType {
        Characters = "characters",
        DamagedCharacters = "damaged-characters",
        Items = "items",
        Locations = "locations",
        CardsInHand = "cards-in-hand",
        CardsInDiscard = "cards-in-discard",
        CardsInInkwellOverLimit = "cards-in-inkwell-over-limit",
        DamageOnSelf = "damage-on-self",
        DamageOnTarget = "damage-on-target",
        LastEffectTargetCount = "last-effect-target-count",
        CardsUnderSelf = "cards-under-self",
        ExertedCharacters = "exerted-characters",
        CharactersThatSang = "characters-that-sang",
    }
}

str_enum! {
    /// `EffectDuration` — the string-literal duration forms. (The oracle union
    /// also admits an object form `{ type: string }`, which is not enumerable
    /// and is carried as raw JSON where it appears.)
    pub enum EffectDuration {
        ThisTurn = "this-turn",
        UntilStartOfNextTurn = "until-start-of-next-turn",
        UntilEndOfTurn = "until-end-of-turn",
        DuringChallenge = "during-challenge",
        Permanent = "permanent",
        WhileCondition = "while-condition",
        NextPlayThisTurn = "next-play-this-turn",
        NextTurn = "next-turn",
        TheirNextTurn = "their-next-turn",
        WhileInPlay = "while-in-play",
    }
}

str_enum! {
    /// The `type` discriminator of every structured `VariableAmount` variant.
    pub enum VariableAmountType {
        TargetAttribute = "target-attribute",
        SourceAttribute = "source-attribute",
        TriggerTargetAttribute = "trigger-target-attribute",
        TargetLocationAttribute = "target-location-attribute",
        FilteredCount = "filtered-count",
        Difference = "difference",
        Reducer = "reducer",
        Clamp = "clamp",
        TriggerAmount = "trigger-amount",
        DamageOnTarget = "damage-on-target",
        DamageOnSelf = "damage-on-self",
        LastEffectTargetCount = "last-effect-target-count",
        CardsInHand = "cards-in-hand",
        CharactersInPlay = "characters-in-play",
        ItemsInPlay = "items-in-play",
        CardsInDiscard = "cards-in-discard",
        Lore = "lore",
        StrengthOf = "strength-of",
        WillpowerOf = "willpower-of",
        LoreValueOf = "lore-value-of",
        CostOf = "cost-of",
        CardsUnderSelf = "cards-under-self",
        LocationLoreFromCharacter = "location-lore-from-character",
        ClassificationCharacterCount = "classification-character-count",
        NameCharacterCount = "name-character-count",
        LocationsInPlay = "locations-in-play",
        TurnMetric = "turn-metric",
        ForEach = "for-each",
        Count = "count",
        Variable = "VARIABLE",
        LoreLost = "lore-lost",
        Stat = "stat",
    }
}
