//! Condition discriminators and the lossless `Condition` node.
//!
//! Mirrors the `Condition` discriminated union in
//! `oracle/source/packages/lorcana/lorcana-types/src/abilities/condition-types.ts`.
//! Declaration order follows the `Condition` union. Per-variant payload fields
//! are preserved losslessly in the node's `rest` map (see [`crate::node`]).

use crate::macros::str_enum;
use crate::node::tagged_node;

str_enum! {
    /// The `type` discriminator of every [`Condition`] variant.
    ///
    /// `TurnCondition` carries `type: "turn" | "during-turn"`, so both spellings
    /// appear here.
    pub enum ConditionType {
        HasNamedCharacter = "has-named-character",
        HasCharacterWithClassification = "has-character-with-classification",
        HasCharacterWithKeyword = "has-character-with-keyword",
        HasCharacterCount = "has-character-count",
        HasNamedItem = "has-named-item",
        HasItemCount = "has-item-count",
        HasNamedLocation = "has-named-location",
        HasLocationCount = "has-location-count",
        AtLocation = "at-location",
        HasAnyDamage = "has-any-damage",
        DamageComparison = "damage-comparison",
        NoDamage = "no-damage",
        IsExerted = "is-exerted",
        IsReady = "is-ready",
        HasCardUnder = "has-card-under",
        TriggerSubjectHadCardUnder = "trigger-subject-had-card-under",
        PutCardUnderSelfThisTurn = "put-card-under-self-this-turn",
        PutCardUnderAnyThisTurn = "put-card-under-any-this-turn",
        InInkwell = "in-inkwell",
        InPlay = "in-play",
        ResourceCount = "resource-count",
        KeywordCharacterCount = "keyword-character-count",
        ClassificationCharacterCount = "classification-character-count",
        Comparison = "comparison",
        UsedShift = "used-shift",
        ThisTurnHappened = "this-turn-happened",
        ThisTurnCount = "this-turn-count",
        Turn = "turn",
        DuringTurn = "during-turn",
        YourTurn = "your-turn",
        BanishedInChallengeThisTurn = "banished-in-challenge-this-turn",
        FirstThisTurn = "first-this-turn",
        Zone = "zone",
        HasCharacterHere = "has-character-here",
        RevealedMatchesNamed = "revealed-matches-named",
        RevealedMatchesChosenName = "revealed-matches-chosen-name",
        IfYouDo = "if-you-do",
        InChallenge = "in-challenge",
        PlayerChoice = "player-choice",
        And = "and",
        Or = "or",
        Not = "not",
        TargetQuery = "target-query",
        TargetAggregateComparison = "target-aggregate-comparison",
        TurnMetric = "turn-metric",
        PlayContext = "play-context",
        FirstTurnNonOtp = "first-turn-non-otp",
        If = "if",
        Resolution = "resolution",
        Exerted = "exerted",
        HandCount = "hand-count",
        StatThreshold = "stat-threshold",
        PlayedThisTurn = "played-this-turn",
        HaveCharacter = "have-character",
        HaveCard = "have-card",
        Name = "name",
        CharacterCount = "character-count",
        Target = "target",
        HasAnotherCharacter = "has-another-character",
        HasCaptainCharacter = "has-captain-character",
        SelfExerted = "self-exerted",
        IsVillain = "is-villain",
        IsPrincess = "is-princess",
        IsNamed = "is-named",
        InkwellCount = "inkwell-count",
        HasCharacterNamed = "has-character-named",
        HasLocationInPlay = "has-location-in-play",
        HasFewerCharacters = "has-fewer-characters",
        OpponentHasMoreCards = "opponent-has-more-cards",
        OpponentHasDamagedCharacter = "opponent-has-damaged-character",
        TargetIsVillain = "target-is-villain",
        HasNoDamage = "has-no-damage",
        HasCharacterWithStrength = "has-character-with-strength",
        ReturnedCardIsPrincess = "returned-card-is-princess",
        ReturnedCardIsNamed = "returned-card-is-named",
        ReturnedCardHasClassification = "returned-card-has-classification",
        RevealedHasSameName = "revealed-has-same-name",
        HasCharactersHere = "has-characters-here",
        HasCharacterAtLocation = "has-character-at-location",
        BeingChallenged = "being-challenged",
        SelfHasDamage = "self-has-damage",
        RevealedIsCharacterNamed = "revealed-is-character-named",
        SecondInkwellThisTurn = "second-inkwell-this-turn",
        WhileInPlay = "while-in-play",
        PlayedCardThisTurn = "played-card-this-turn",
        OpponentHasMoreThanCards = "opponent-has-more-than-cards",
        OpponentHasLore = "opponent-has-lore",
        HasStrongestCharacter = "has-strongest-character",
        HasDamagedCharacterHere = "has-damaged-character-here",
        HasItemInPlay = "has-item-in-play",
        Unless = "unless",
        LoreComparison = "lore-comparison",
        SecondInTurn = "second-in-turn",
        TargetIsDamaged = "target-is-damaged",
        DiscardedCardHasClassification = "discarded-card-has-classification",
        HasGrantedAbility = "has-granted-ability",
        RevealedIsCardType = "revealed-is-card-type",
    }
}

tagged_node! {
    /// A Lorcana condition node: typed `type` discriminator plus all other
    /// fields preserved losslessly.
    Condition, ConditionType, crate::node::NodeFamily::Condition
}
