//! Lossless DSL node representation.
//!
//! The Lorcana DSL (conditions, effects, triggers, targets) is a deeply nested,
//! dynamically-shaped object graph. The blueprint (§5.2 "Dynamic JS object
//! semantics") is explicit that the port must *preserve exact serialization*
//! during parity validation and normalize only later. Hand-expanding every
//! optional field of ~150 effect variants and ~90 condition variants into typed
//! structs at this layer would risk silent drift (dropped/renamed fields) — the
//! one failure mode CLAUDE.md weighs as worse than slowness.
//!
//! So recursive DSL nodes are modeled as a **typed discriminator + a flattened,
//! order-preserving map of every other field** ([`IndexMap`] + `serde_json`'s
//! `preserve_order`). This guarantees:
//! * lossless round-trip of all non-discriminator fields and their order,
//! * a typed, fail-closed discriminator (unknown/missing `type` ⇒ deser error),
//! * forward-compatibility with the later steps that consume each field.
//!
//! Re-serialization emits the `type` discriminator first (canonical form);
//! all remaining fields keep their input order.
//!
//! ## Recursive discriminator validation
//! Losslessly storing every child field as raw [`serde_json::Value`] means a
//! nested DSL sub-node (an `effect`, `condition`, `trigger`, …) would *not* be
//! validated by Serde on its own — its bad `type`/`event` would survive inside
//! `rest`. To stay fail-closed without abandoning the lossless representation,
//! every tagged node (and [`crate::trigger::Trigger`], and
//! [`crate::ability::AbilityDefinition`]) runs [`validate_dsl_children`] during
//! deserialization: it re-parses the well-known structural sub-node keys into
//! their typed schema, which recurses to full depth. The raw value is still
//! kept in `rest`; only the discriminator is checked.

use indexmap::IndexMap;
use serde::Deserialize;
use serde::de::Error as _;

/// Alias for a lossless JSON object body preserving key insertion order.
pub type JsonObject = IndexMap<String, serde_json::Value>;

/// Which schema node family is being validated. Only [`NodeFamily::Effect`]
/// carries parent-`kind`-specific `Effect`/`Effect[]` container fields (see
/// [`validate_effect_kind_children`]); all families share the generic keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeFamily {
    /// An [`Effect`](crate::effect::Effect) node.
    Effect,
    /// A [`Condition`](crate::condition::Condition) node.
    Condition,
    /// An [`AbilityDefinition`](crate::ability::AbilityDefinition) node.
    Ability,
    /// A node with no `kind`-specific child containers (e.g.
    /// [`Trigger`](crate::trigger::Trigger)).
    Other,
}

/// Validate (and discard) `value` parsed as `T`, attributing failures to `key`.
fn check_child<T: serde::de::DeserializeOwned>(
    key: &str,
    value: &serde_json::Value,
) -> Result<(), String> {
    serde_json::from_value::<T>(value.clone())
        .map(|_| ())
        .map_err(|e| format!("invalid `{key}` sub-node: {e}"))
}

/// Recursively validate the discriminators of nested DSL sub-node keys.
///
/// Two layers, both verified against the full 2754-card frozen catalog (no
/// over-rejection) and matched to the oracle field types:
///
/// 1. **Generic keys** (every node family) — keys that hold the same DSL node
///    type *wherever* they appear:
///    * `effect` ⇒ [`Effect`], `effects` ⇒ `Vec<Effect>`
///    * `condition` ⇒ [`Condition`], `conditions` ⇒ `Vec<Condition>`
///    * `trigger` ⇒ [`Trigger`] (which validates its own `event`)
///    * `keyword` ⇒ [`KeywordType`]
/// 2. **Parent-`kind`-specific containers** (Effect family only) — see
///    [`validate_effect_kind_children`]; gated on the parent effect's `type` so
///    overloaded keys (`options` on `put-on-deck`/scry, `cost` on abilities) are
///    not misvalidated. Includes nested-in-container fields such as
///    `reveal-and-route.routes[*].condition`/`sideEffects` and
///    `create-triggered-ability.ability.trigger`/`condition`/`effect`.
///
/// The parsed value is discarded — the lossless raw value stays in `rest`.
/// `keywords` (a list of keyword-ability *objects*, not bare keyword strings),
/// and the union-typed `Effect | "prevent" | …` fields (`with`/`replacement`/
/// `cost`, which legitimately hold strings/objects outside the `Effect` union in
/// real cards) are deliberately *not* validated here, to avoid rejecting valid
/// oracle data.
///
/// [`Effect`]: crate::effect::Effect
/// [`Condition`]: crate::condition::Condition
/// [`Trigger`]: crate::trigger::Trigger
/// [`KeywordType`]: crate::keyword::KeywordType
pub(crate) fn validate_dsl_children(
    rest: &JsonObject,
    family: NodeFamily,
    kind: Option<&str>,
) -> Result<(), String> {
    use crate::condition::Condition;
    use crate::effect::Effect;
    use crate::keyword::KeywordType;
    use crate::trigger::Trigger;

    for (key, value) in rest {
        match key.as_str() {
            "effect" => check_child::<Effect>(key, value)?,
            "effects" => check_child::<Vec<Effect>>(key, value)?,
            "condition" => check_child::<Condition>(key, value)?,
            "conditions" => check_child::<Vec<Condition>>(key, value)?,
            "trigger" => check_child::<Trigger>(key, value)?,
            "keyword" => check_child::<KeywordType>(key, value)?,
            _ => {}
        }
    }

    if family == NodeFamily::Effect {
        validate_effect_kind_children(kind, rest)?;
    }
    Ok(())
}

/// Validate the `Effect`/`Effect[]` (and nested `Condition`/`Trigger`) container
/// fields that exist only on specific control-flow effects, keyed by the parent
/// effect's discriminator. Mirrors the oracle effect-type interfaces
/// (`control-flow.ts`, `combined-types.ts`):
///
/// | parent `type`              | direct `Effect[]`   | direct `Effect`              | nested container |
/// |----------------------------|---------------------|------------------------------|------------------|
/// | `sequence`                 | `steps`             | —                            | — |
/// | `choice` / `or`            | `options`,`choices` | —                            | — |
/// | `modal`                    | `options`           | —                            | — |
/// | `conditional`              | —                   | `then`,`else`,`ifTrue`,`ifFalse` | — |
/// | `reveal-and-conditional`   | —                   | `ifTrue`,`ifFalse`           | — |
/// | `reveal-and-route`         | —                   | —                            | `routes[*].condition` (Condition), `routes[*].sideEffects` (Effect[]) |
/// | `create-triggered-ability` | —                   | —                            | `ability.trigger` (Trigger), `ability.condition` (Condition), `ability.effect` (Effect) |
///
/// (`effect`/`effects`/`condition`/`conditions`/`trigger` as *direct* keys on
/// these and other effects are covered by the generic keys.) Gating on `kind` is
/// essential: `options` is `Effect[]` here but a positioning/string list on
/// `put-on-deck`/scry effects, so it must only be validated for `choice`/`or`/
/// `modal`.
fn validate_effect_kind_children(kind: Option<&str>, rest: &JsonObject) -> Result<(), String> {
    use crate::condition::Condition;
    use crate::effect::Effect;
    use crate::trigger::Trigger;

    let Some(kind) = kind else { return Ok(()) };
    let (effect_fields, effect_array_fields): (&[&str], &[&str]) = match kind {
        "sequence" => (&[], &["steps"]),
        "choice" | "or" => (&[], &["options", "choices"]),
        "modal" => (&[], &["options"]),
        "conditional" => (&["then", "else", "ifTrue", "ifFalse"], &[]),
        "reveal-and-conditional" => (&["ifTrue", "ifFalse"], &[]),
        _ => (&[], &[]),
    };
    for field in effect_fields {
        if let Some(value) = rest.get(*field) {
            check_child::<Effect>(field, value)?;
        }
    }
    for field in effect_array_fields {
        if let Some(value) = rest.get(*field) {
            check_child::<Vec<Effect>>(field, value)?;
        }
    }

    // Nested DSL fields that live inside a container object/array rather than as
    // a direct key of the effect node.
    match kind {
        // `RevealAndRouteEffect.routes: RevealRoute[]`, each route carrying a
        // `condition: Condition` and optional `sideEffects: Effect[]`.
        "reveal-and-route" => {
            if let Some(routes) = rest.get("routes").and_then(|v| v.as_array()) {
                for route in routes {
                    if let Some(route) = route.as_object() {
                        if let Some(value) = route.get("condition") {
                            check_child::<Condition>("routes[].condition", value)?;
                        }
                        if let Some(value) = route.get("sideEffects") {
                            check_child::<Vec<Effect>>("routes[].sideEffects", value)?;
                        }
                    }
                }
            }
        }
        // `CreateTriggeredAbilityEffect.ability: { trigger; condition?; effect }`.
        "create-triggered-ability" => {
            if let Some(ability) = rest.get("ability").and_then(|v| v.as_object()) {
                if let Some(value) = ability.get("trigger") {
                    check_child::<Trigger>("ability.trigger", value)?;
                }
                if let Some(value) = ability.get("condition") {
                    check_child::<Condition>("ability.condition", value)?;
                }
                if let Some(value) = ability.get("effect") {
                    check_child::<Effect>("ability.effect", value)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Split a `type`-tagged JSON object into its typed discriminator and the
/// lossless remainder, validating both the discriminator and every recognized
/// nested DSL sub-node (generic keys + the node family's `kind`-specific
/// `Effect` containers). Shared by all tagged nodes.
pub(crate) fn deserialize_tagged<'de, D, T>(
    deserializer: D,
    family: NodeFamily,
) -> Result<(T, JsonObject), D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let mut rest = JsonObject::deserialize(deserializer)?;
    let type_val = rest
        .shift_remove("type")
        .ok_or_else(|| D::Error::missing_field("type"))?;
    let kind_str = type_val.as_str().map(str::to_owned);
    let kind: T = serde_json::from_value(type_val).map_err(D::Error::custom)?;
    validate_dsl_children(&rest, family, kind_str.as_deref()).map_err(D::Error::custom)?;
    Ok((kind, rest))
}

/// Declare a tagged lossless node: a required typed `type` discriminator plus a
/// flattened map of every other field. The discriminator and nested DSL
/// sub-nodes are validated on deserialization (fail-closed); all non-`type`
/// fields are preserved losslessly in input order for byte-exact round-trip.
macro_rules! tagged_node {
    (
        $(#[$meta:meta])*
        $name:ident, $disc:ty, $family:expr
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, ::serde::Serialize)]
        pub struct $name {
            /// The typed discriminator (`type` field). Fails closed on an
            /// unknown or missing value during deserialization.
            #[serde(rename = "type")]
            pub kind: $disc,
            /// All other fields, preserved losslessly in input order.
            #[serde(flatten)]
            pub rest: crate::node::JsonObject,
        }

        impl $name {
            /// Construct a node with the given discriminator and no extra fields.
            pub fn new(kind: $disc) -> Self {
                Self { kind, rest: crate::node::JsonObject::new() }
            }

            /// The typed discriminator for this node.
            pub fn kind(&self) -> $disc {
                self.kind
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> ::core::result::Result<Self, D::Error>
            where
                D: ::serde::Deserializer<'de>,
            {
                let (kind, rest) =
                    crate::node::deserialize_tagged::<D, $disc>(deserializer, $family)?;
                Ok(Self { kind, rest })
            }
        }
    };
}

pub(crate) use tagged_node;
