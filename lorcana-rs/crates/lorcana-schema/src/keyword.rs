//! Keyword ability discriminators.
//!
//! Mirrors the keyword unions in
//! `oracle/source/packages/lorcana/lorcana-types/src/abilities/ability-types.ts`
//! (`SimpleKeywordType`, `ParameterizedKeywordType`, `ValueKeywordType`,
//! `KeywordType`). The full `KeywordAbility` payload (value / cost / condition /
//! shiftTarget) is carried losslessly by [`crate::ability::AbilityDefinition`]'s
//! flattened `rest` map; this module fixes the closed discriminator vocabulary.

use crate::macros::str_enum;

str_enum! {
    /// Keywords with no parameters.
    pub enum SimpleKeywordType {
        Rush = "Rush",
        Ward = "Ward",
        Evasive = "Evasive",
        Bodyguard = "Bodyguard",
        Support = "Support",
        Reckless = "Reckless",
        Vanish = "Vanish",
        Alert = "Alert",
        QuestWhileDrying = "QuestWhileDrying",
    }
}

str_enum! {
    /// Keywords with a numeric value and an optional condition.
    pub enum ParameterizedKeywordType {
        Challenger = "Challenger",
        Resist = "Resist",
    }
}

str_enum! {
    /// Keywords with a required numeric value (not conditional).
    pub enum ValueKeywordType {
        Singer = "Singer",
        SingTogether = "SingTogether",
        Boost = "Boost",
    }
}

str_enum! {
    /// All keyword types: simple + parameterized + value + `Shift`.
    ///
    /// Declaration order follows the oracle composition
    /// `SimpleKeywordType | ParameterizedKeywordType | ValueKeywordType | "Shift"`.
    pub enum KeywordType {
        Rush = "Rush",
        Ward = "Ward",
        Evasive = "Evasive",
        Bodyguard = "Bodyguard",
        Support = "Support",
        Reckless = "Reckless",
        Vanish = "Vanish",
        Alert = "Alert",
        QuestWhileDrying = "QuestWhileDrying",
        Challenger = "Challenger",
        Resist = "Resist",
        Singer = "Singer",
        SingTogether = "SingTogether",
        Boost = "Boost",
        Shift = "Shift",
    }
}
