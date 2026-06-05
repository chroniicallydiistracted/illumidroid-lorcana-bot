//! Character classifications (Rule 6.2.6).
//!
//! Mirrors `oracle/source/packages/lorcana/lorcana-types/src/cards/classifications.ts`
//! (`CLASSIFICATIONS` / `Classification`), preserving the exact set and order.

use crate::macros::str_enum;

str_enum! {
    /// Classifications that appear on character cards. A character may have
    /// several. Note: several wire values contain spaces (e.g. "Seven Dwarfs").
    pub enum Classification {
        Alien = "Alien",
        Ally = "Ally",
        Broom = "Broom",
        Captain = "Captain",
        Deity = "Deity",
        Detective = "Detective",
        Dinosaur = "Dinosaur",
        Dragon = "Dragon",
        Dreamborn = "Dreamborn",
        Entangled = "Entangled",
        Fairy = "Fairy",
        Floodborn = "Floodborn",
        Gargoyle = "Gargoyle",
        Ghost = "Ghost",
        Giant = "Giant",
        Hero = "Hero",
        Hunny = "Hunny",
        Hyena = "Hyena",
        Illusion = "Illusion",
        Inventor = "Inventor",
        King = "King",
        Knight = "Knight",
        Madrigal = "Madrigal",
        Mentor = "Mentor",
        Monster = "Monster",
        Musketeer = "Musketeer",
        Pirate = "Pirate",
        Prince = "Prince",
        Princess = "Princess",
        Puppy = "Puppy",
        Queen = "Queen",
        Racer = "Racer",
        Robot = "Robot",
        SevenDwarfs = "Seven Dwarfs",
        Sorcerer = "Sorcerer",
        Storyborn = "Storyborn",
        Super = "Super",
        Tigger = "Tigger",
        Titan = "Titan",
        Toy = "Toy",
        Villain = "Villain",
        Whisper = "Whisper",
    }
}
