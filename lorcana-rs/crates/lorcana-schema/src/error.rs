//! Schema-layer error type.

use thiserror::Error;

/// Errors raised when inspecting or validating a schema node.
///
/// Deserialization failures themselves surface as [`serde_json::Error`]; this
/// type covers post-parse inspection (e.g. reading a discriminator out of a
/// lossless map node).
#[derive(Debug, Error)]
pub enum SchemaError {
    /// A node was missing a required discriminator field.
    #[error("node is missing required `{0}` discriminator")]
    MissingDiscriminator(&'static str),

    /// A discriminator field held a non-string JSON value.
    #[error("`{0}` discriminator was not a string")]
    NonStringDiscriminator(&'static str),

    /// A discriminator string was not a member of the known oracle vocabulary.
    #[error("unknown `{field}` discriminator: {value:?}")]
    UnknownDiscriminator {
        /// The discriminator field name.
        field: &'static str,
        /// The unrecognized value.
        value: String,
    },
}
