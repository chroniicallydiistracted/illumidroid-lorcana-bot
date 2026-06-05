//! Ink types (Rule 2.1.1.2).
//!
//! Mirrors `oracle/source/packages/lorcana/lorcana-types/src/cards/ink-types.ts`
//! (`INK_TYPES` / `InkType`). The UI-only `INK_COLORS` map and helper functions
//! are intentionally not ported (display concern, not schema).

use crate::macros::str_enum;

str_enum! {
    /// Six ink colors that define card identity and deck-building constraints.
    pub enum InkType {
        Amber = "amber",
        Amethyst = "amethyst",
        Emerald = "emerald",
        Ruby = "ruby",
        Sapphire = "sapphire",
        Steel = "steel",
    }
}
