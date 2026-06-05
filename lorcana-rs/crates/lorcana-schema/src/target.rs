//! Targeting / filter primitive enums.
//!
//! Mirrors closed string-literal unions from
//! `oracle/source/packages/lorcana/lorcana-types/src/abilities/target-types.ts`,
//! `.../targeting/target-dsl.ts`, `.../targeting/lorcana-target-dsl.ts`, and
//! `.../expressions/index.ts`.
//!
//! The recursive `CharacterTarget` / `LorcanaCardTarget` / `FilterExpr` shapes
//! themselves are carried losslessly inside DSL node `rest` maps (see
//! [`crate::node`]); this module fixes the closed scalar vocabularies that the
//! engine and parity tests reason about.

use crate::macros::str_enum;

str_enum! {
    /// `SelectorScope` — how targets are selected from valid options.
    pub enum SelectorScope {
        SelfCard = "self",
        Chosen = "chosen",
        All = "all",
        Each = "each",
        Any = "any",
        Random = "random",
    }
}

str_enum! {
    /// `OwnerScope` — whose cards can be selected.
    pub enum OwnerScope {
        You = "you",
        Opponent = "opponent",
        Any = "any",
    }
}

str_enum! {
    /// `PlayerTargetScope` — player targeting selector.
    pub enum PlayerTargetScope {
        You = "you",
        Opponent = "opponent",
        EachPlayer = "each-player",
        Chosen = "chosen",
        ChallengingPlayer = "challenging-player",
    }
}

str_enum! {
    /// `TargetController` — who controls the target (target-types.ts).
    pub enum TargetController {
        You = "you",
        Opponent = "opponent",
        Any = "any",
        CurrentTurn = "CURRENT_TURN",
    }
}

str_enum! {
    /// `TargetZone` — zone where targets can be found (target-types.ts).
    pub enum TargetZone {
        Play = "play",
        Hand = "hand",
        Discard = "discard",
        Deck = "deck",
        Inkwell = "inkwell",
    }
}

str_enum! {
    /// `LorcanaZoneId` — Lorcana zone IDs (lorcana-target-dsl.ts). Superset of
    /// [`TargetZone`] with the internal `limbo` zone.
    pub enum LorcanaZoneId {
        Deck = "deck",
        Hand = "hand",
        Play = "play",
        Discard = "discard",
        Inkwell = "inkwell",
        Limbo = "limbo",
    }
}

str_enum! {
    /// `CardStatus` — card state filters (expressions/index.ts; superset of the
    /// target-types.ts `CardStatus` with the additional `dry` state).
    pub enum CardStatus {
        Damaged = "damaged",
        Undamaged = "undamaged",
        Exerted = "exerted",
        Ready = "ready",
        Dry = "dry",
    }
}

str_enum! {
    /// `ComparisonOperator` — the canonical operator union exported by
    /// `expressions/index.ts` (superset of the target-types.ts spelling; adds
    /// the short `eq`/`ne`/`gt`/`gte`/`lt`/`lte` aliases). Declaration order
    /// matches the oracle union.
    pub enum ComparisonOperator {
        Eq = "eq",
        Ne = "ne",
        Gt = "gt",
        Gte = "gte",
        Lt = "lt",
        Lte = "lte",
        Equal = "equal",
        NotEqual = "not-equal",
        Greater = "greater",
        GreaterThan = "greater-than",
        MoreThan = "more-than",
        GreaterOrEqual = "greater-or-equal",
        OrMore = "or-more",
        Less = "less",
        LessThan = "less-than",
        LessOrEqual = "less-or-equal",
        OrLess = "or-less",
    }
}

str_enum! {
    /// `ConditionComparisonOperator` — the restricted six-operator union used by
    /// the structured condition comparisons (condition-types.ts).
    pub enum ConditionComparisonOperator {
        Eq = "eq",
        Ne = "ne",
        Gt = "gt",
        Gte = "gte",
        Lt = "lt",
        Lte = "lte",
    }
}
