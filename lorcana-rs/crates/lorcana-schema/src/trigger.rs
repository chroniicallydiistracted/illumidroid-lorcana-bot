//! Trigger discriminators and the lossless `Trigger` node.
//!
//! Mirrors `oracle/source/packages/lorcana/lorcana-types/src/abilities/trigger-types.ts`
//! (`TriggerEvent`, `TriggerTiming`, `TriggerCardType`, `TriggerSubjectEnum`).
//!
//! Unlike conditions/effects, a `Trigger` has **no single required
//! discriminator**: `BaseTrigger.event` is optional (a trigger may instead carry
//! `events`, `timing`, `on`, etc.). The node is therefore a fully lossless,
//! order-preserving map with typed accessors for the common fields.

use crate::macros::str_enum;
use crate::node::JsonObject;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

str_enum! {
    /// `TriggerEvent` — the fundamental game actions a trigger can watch for.
    pub enum TriggerEvent {
        Play = "play",
        Banish = "banish",
        LeavePlay = "leave-play",
        Quest = "quest",
        Challenge = "challenge",
        Challenged = "challenged",
        ChallengedAndBanished = "challenged-and-banished",
        Damage = "damage",
        Exert = "exert",
        Ready = "ready",
        Move = "move",
        Sing = "sing",
        BeChosen = "be-chosen",
        BanishInChallenge = "banish-in-challenge",
        DealDamage = "deal-damage",
        Draw = "draw",
        Discard = "discard",
        Ink = "ink",
        GainLore = "gain-lore",
        LoseLore = "lose-lore",
        StartTurn = "start-turn",
        EndTurn = "end-turn",
        RemoveDamage = "remove-damage",
        ReturnToHand = "return-to-hand",
        StartOfTurn = "start-of-turn",
        EndOfTurn = "end-of-turn",
        PutIntoInkwell = "put-into-inkwell",
        AddToInkwell = "add-to-inkwell",
        PutCardUnder = "put-card-under",
        Support = "support",
        Inkwell = "inkwell",
        Boost = "boost",
        LeaveDiscard = "leave-discard",
    }
}

str_enum! {
    /// `TriggerTiming` — timing word. (The `timing` field additionally admits the
    /// parser-compat value `"when-or-whenever"`, which is not part of this named
    /// type and is preserved as raw JSON where it appears.)
    pub enum TriggerTiming {
        When = "when",
        Whenever = "whenever",
        At = "at",
    }
}

str_enum! {
    /// `TriggerCardType` — card-type filter for triggers.
    pub enum TriggerCardType {
        Character = "character",
        Action = "action",
        Item = "item",
        Location = "location",
        Song = "song",
        Card = "card",
    }
}

str_enum! {
    /// `TriggerSubjectEnum` — the string-literal trigger subjects.
    pub enum TriggerSubjectEnum {
        SelfCard = "SELF",
        YourCharacters = "YOUR_CHARACTERS",
        YourOtherCharacters = "YOUR_OTHER_CHARACTERS",
        OpponentCharacters = "OPPONENT_CHARACTERS",
        OpposingCharacters = "OPPOSING_CHARACTERS",
        OtherCharacters = "OTHER_CHARACTERS",
        AnyCharacter = "ANY_CHARACTER",
        YourItems = "YOUR_ITEMS",
        YourOtherItems = "YOUR_OTHER_ITEMS",
        AnyItem = "ANY_ITEM",
        YourLocations = "YOUR_LOCATIONS",
        YourActions = "YOUR_ACTIONS",
        YourSongs = "YOUR_SONGS",
        You = "YOU",
        Opponent = "OPPONENT",
        AnyPlayer = "ANY_PLAYER",
        FloodbornCharacters = "FLOODBORN_CHARACTERS",
        SelfOrSevenDwarfsCharacters = "SELF_OR_SEVEN_DWARFS_CHARACTERS",
        CinderellaCharacters = "CINDERELLA_CHARACTERS",
        YourCharactersCost4OrMore = "YOUR_CHARACTERS_COST_4_OR_MORE",
        Songs = "SONGS",
        YourBroomCharacters = "YOUR_BROOM_CHARACTERS",
        YourMusketeerCharacters = "YOUR_MUSKETEER_CHARACTERS",
        YourBodyguardCharacters = "YOUR_BODYGUARD_CHARACTERS",
        Controller = "CONTROLLER",
        CharactersHere = "CHARACTERS_HERE",
        YourOtherSteelCharacters = "YOUR_OTHER_STEEL_CHARACTERS",
        YourOtherSapphireCharacters = "YOUR_OTHER_SAPPHIRE_CHARACTERS",
        CharactersAtLocation = "CHARACTERS_AT_LOCATION",
        CharactersMovedHere = "CHARACTERS_MOVED_HERE",
        OpponentsCards = "OPPONENTS_CARDS",
        YourPirateCharacters = "YOUR_PIRATE_CHARACTERS",
        CharacterHere = "CHARACTER_HERE",
        Song = "SONG",
        YourCharactersOrLocations = "YOUR_CHARACTERS_OR_LOCATIONS",
        YourOtherAmethystCharacters = "YOUR_OTHER_AMETHYST_CHARACTERS",
        YourCharactersOrLocationsWithCardUnder = "YOUR_CHARACTERS_OR_LOCATIONS_WITH_CARD_UNDER",
    }
}

/// A Lorcana trigger node, preserved losslessly in input order.
///
/// A trigger has no single required discriminator (`event` is optional), so the
/// body is a lossless map. On deserialization, a present `event` is validated
/// against [`TriggerEvent`] and any recognized nested DSL sub-node is validated
/// (fail-closed); the raw fields are still preserved for byte-exact round-trip.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Trigger(pub JsonObject);

impl<'de> Deserialize<'de> for Trigger {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map = JsonObject::deserialize(deserializer)?;
        if let Some(event) = map.get("event") {
            serde_json::from_value::<TriggerEvent>(event.clone()).map_err(D::Error::custom)?;
        }
        crate::node::validate_dsl_children(&map, crate::node::NodeFamily::Other, None)
            .map_err(D::Error::custom)?;
        Ok(Trigger(map))
    }
}

impl Trigger {
    /// An empty trigger object.
    pub fn new() -> Self {
        Self(JsonObject::new())
    }

    /// The typed `event`, if present and recognized.
    pub fn event(&self) -> Option<TriggerEvent> {
        self.0
            .get("event")
            .and_then(|v| serde_json::from_value::<TriggerEvent>(v.clone()).ok())
    }

    /// The typed `timing`, if present and recognized as a [`TriggerTiming`].
    pub fn timing(&self) -> Option<TriggerTiming> {
        self.0
            .get("timing")
            .and_then(|v| serde_json::from_value::<TriggerTiming>(v.clone()).ok())
    }
}

impl Default for Trigger {
    fn default() -> Self {
        Self::new()
    }
}
