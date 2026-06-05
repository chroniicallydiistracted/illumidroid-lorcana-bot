//! Internal macros for declaring string-literal discriminator enums.

/// Declare a fieldless enum whose Serde representation is an explicit JSON
/// string literal per variant (mirroring a TypeScript string-literal union).
///
/// Each variant's wire string is written out explicitly rather than derived
/// from `rename_all`, so the mapping to the frozen oracle is auditable and
/// cannot silently drift through a casing-convention change. The generated
/// `NAMES` constant is the discriminator manifest used by the coverage tests.
///
/// Deserialization of an unknown string fails closed (Serde rejects unknown
/// unit variants), which is the fail-closed behavior the port requires.
macro_rules! str_enum {
    (
        $(#[$emeta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$vmeta:meta])*
                $variant:ident = $wire:literal
            ),+ $(,)?
        }
    ) => {
        $(#[$emeta])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash,
            ::serde::Serialize, ::serde::Deserialize,
        )]
        $vis enum $name {
            $(
                $(#[$vmeta])*
                #[serde(rename = $wire)]
                $variant,
            )+
        }

        impl $name {
            /// The exact JSON wire string for this variant.
            $vis fn as_str(&self) -> &'static str {
                match self {
                    $( Self::$variant => $wire, )+
                }
            }

            /// All variants, in declaration order.
            $vis const ALL: &'static [$name] = &[ $( $name::$variant ),+ ];

            /// All wire strings, in declaration order (the discriminator manifest).
            $vis const NAMES: &'static [&'static str] = &[ $( $wire ),+ ];
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

pub(crate) use str_enum;
