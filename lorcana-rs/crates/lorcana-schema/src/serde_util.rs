//! Serde helpers that enforce the frozen oracle's field nullability semantics.
//!
//! The TypeScript oracle distinguishes two optional-field flavors:
//! * `field?: T` — may be **absent**, but if present must be a `T`; explicit
//!   JSON `null` is **not** a legal value.
//! * `field?: T | null` — may be absent, a `T`, or explicit `null`.
//!
//! A plain Serde `Option<T>` collapses both `null` and absent to `None`, which
//! (a) silently accepts an explicit `null` the oracle type forbids and (b) is
//! lossy (re-serializes the `null` as absent). [`optional_non_null`] restores
//! the `field?: T` contract: absent ⇒ `None` (via `#[serde(default)]`), present
//! ⇒ the value is deserialized as `T` so an explicit `null` fails closed.
//!
//! Fields whose oracle type genuinely includes `null` (e.g. `ActionSubtype =
//! "song" | null`) must keep the default `Option` behavior and must **not** use
//! this helper.

use serde::{Deserialize, Deserializer};

/// Deserialize a `field?: T` (optional, non-nullable) oracle field.
///
/// Pair with `#[serde(default, skip_serializing_if = "Option::is_none")]`:
/// Serde only invokes this function when the key is present, so an absent key
/// uses the `Default` (`None`) and a present key is deserialized as `T` —
/// rejecting an explicit `null`, which `T` itself does not accept.
pub(crate) fn optional_non_null<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}
