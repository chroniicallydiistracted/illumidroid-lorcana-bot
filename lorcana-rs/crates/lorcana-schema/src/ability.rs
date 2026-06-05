//! Ability schema: top-level ability discriminator and the `AbilityDefinition`
//! node, plus the restriction / replacement vocabularies.
//!
//! Mirrors `oracle/source/packages/lorcana/lorcana-types/src/abilities/ability-types.ts`
//! and the card-definition ability wrapper in `.../cards/card-types.ts`
//! (`AbilityDefinition = KeywordAbilityDefinition | TriggeredAbilityDefinition |
//! ActivatedAbilityDefinition | StaticAbilityDefinition | ActionAbilityDefinition
//! | ReplacementAbilityDefinition`).
//!
//! The per-ability-type payload (keyword / trigger / effect / cost / condition /
//! restrictions …) and the `BaseAbilityDefinition` metadata (`id`, `name`,
//! `text`) are carried losslessly in the node's `rest` map; typed accessors are
//! provided for the common fields.

use crate::keyword::KeywordType;
use crate::macros::str_enum;
use crate::node::tagged_node;

str_enum! {
    /// The `type` discriminator shared by all card-definition abilities.
    pub enum AbilityType {
        Keyword = "keyword",
        Triggered = "triggered",
        Activated = "activated",
        Static = "static",
        Action = "action",
        Replacement = "replacement",
    }
}

str_enum! {
    /// `ReplacementAbility.replaces` — events a replacement ability can replace
    /// (ability-types.ts).
    pub enum ReplacementAbilityReplaces {
        DamageToSelf = "damage-to-self",
        DamageToCharacter = "damage-to-character",
        BanishSelf = "banish-self",
        DrawCard = "draw-card",
        GainLore = "gain-lore",
        LoseLore = "lose-lore",
        RemoveDamage = "remove-damage",
        Discard = "discard",
    }
}

str_enum! {
    /// `ReplacementEffect.replaces` — events a replacement *effect* can replace
    /// (effect-types/combined-types.ts).
    pub enum ReplacementEffectReplaces {
        Damage = "damage",
        Banish = "banish",
        Quest = "quest",
        DamageToCharacter = "damage-to-character",
    }
}

str_enum! {
    /// `Restriction.type` — activated-ability usage restrictions (ability-types.ts).
    pub enum RestrictionType {
        OncePerTurn = "once-per-turn",
        OncePerGame = "once-per-game",
        FirstTimeEachTurn = "first-time-each-turn",
        DuringTurn = "during-turn",
        WhileExerted = "while-exerted",
        WhileReady = "while-ready",
        WhileAtLocation = "while-at-location",
        WhileDamaged = "while-damaged",
        WhileUndamaged = "while-undamaged",
        InChallenge = "in-challenge",
        NotInChallenge = "not-in-challenge",
    }
}

str_enum! {
    /// `TriggerRestriction.type` — restrictions specific to triggers (trigger-types.ts).
    pub enum TriggerRestrictionType {
        OncePerTurn = "once-per-turn",
        FirstTimeEachTurn = "first-time-each-turn",
        NTimesPerTurn = "n-times-per-turn",
        OncePerSong = "once-per-song",
        DuringTurn = "during-turn",
        FromLocation = "from-location",
        InChallenge = "in-challenge",
        FromDiscard = "from-discard",
        ToHand = "to-hand",
        DefenderIsCharacter = "defender-is-character",
    }
}

tagged_node! {
    /// A card-definition ability: typed `type` discriminator plus all other
    /// fields (`keyword`/`trigger`/`effect`/`cost`/`id`/`name`/`text`/…)
    /// preserved losslessly in input order.
    AbilityDefinition, AbilityType, crate::node::NodeFamily::Ability
}

impl AbilityDefinition {
    /// `id` metadata, if present.
    pub fn id(&self) -> Option<&str> {
        self.rest.get("id").and_then(|v| v.as_str())
    }

    /// `name` metadata, if present.
    pub fn name(&self) -> Option<&str> {
        self.rest.get("name").and_then(|v| v.as_str())
    }

    /// Raw `text` metadata when it is the string form (the structured
    /// `CardTextEntry[]` form returns `None` here and is read via `rest`).
    pub fn text(&self) -> Option<&str> {
        self.rest.get("text").and_then(|v| v.as_str())
    }

    /// The typed `keyword`, present on keyword abilities.
    pub fn keyword(&self) -> Option<KeywordType> {
        self.rest
            .get("keyword")
            .and_then(|v| serde_json::from_value::<KeywordType>(v.clone()).ok())
    }
}
