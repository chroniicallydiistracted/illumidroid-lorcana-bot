//! Branded identifier newtypes.
//!
//! Mirrors `oracle/source/packages/lorcana/lorcana-types/src/branded.ts`.
//! In TypeScript these are compile-time `Brand<string, ...>` aliases; at the
//! serialization boundary they are plain strings. The Rust port preserves that
//! exact wire shape via `#[serde(transparent)]` newtypes, which keeps the brand
//! distinction in the type system without changing the JSON representation.

use serde::{Deserialize, Serialize};

/// Macro to declare a `Brand<string, "...">` newtype that serializes as a bare
/// string (byte-identical to the TypeScript value).
macro_rules! branded_string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            /// Borrow the underlying string.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

branded_string_id!(
    /// Card instance identifier — unique ID for a card instance in a match.
    CardInstanceId
);
branded_string_id!(
    /// Public card definition identifier — unique ID for static card definitions.
    CardPublicId
);
branded_string_id!(
    /// Player identifier — unique ID for a player in a match.
    PlayerId
);
branded_string_id!(
    /// Game identifier — unique ID for a game session.
    GameId
);
branded_string_id!(
    /// Zone identifier — unique ID for a zone.
    ZoneId
);
