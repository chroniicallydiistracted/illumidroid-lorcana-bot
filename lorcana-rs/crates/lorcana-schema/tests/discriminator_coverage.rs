//! Discriminator-coverage conformance test (blueprint Step 1 validation:
//! "same keyword/effect/condition discriminator names").
//!
//! Proves the Rust discriminator vocabulary matches the **frozen** TypeScript
//! oracle under `oracle/source/.../lorcana-types/src`, in both directions:
//!
//! 1. **Forward containment (all categories):** every wire string the Rust
//!    enums accept appears verbatim, as a quoted literal, in the oracle file(s)
//!    that define that category. Catches typos and renames.
//! 2. **Exact set-equality (all closed categories):** the Rust discriminator set
//!    for each closed category equals the set extracted from the oracle's own
//!    declaration of that category — catching both **missing** members (oracle
//!    has a discriminator the port lacks) and **extra** members (the port
//!    invented one). Covers keyword, ability, cost, condition, effect, trigger,
//!    target, expression, and deck-format categories.
//! 3. **No duplicate wire strings** within any enum.
//!
//! The oracle sets are extracted by a small principled TypeScript reader (no
//! regex dep): it resolves a named union/`as const` array to each member's own
//! `type:` discriminant (recursing into sub-unions, descending into inline
//! object members at brace depth 1 only), after stripping comments. The reader
//! is deliberately *independent* of the Rust enums so it can detect drift.

use std::collections::BTreeSet;
use std::path::PathBuf;

use lorcana_schema::{CardType, Classification, InkType, discriminator_manifest};

fn types_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../oracle/source/packages/lorcana/lorcana-types/src")
        .canonicalize()
        .expect("frozen oracle lorcana-types/src must exist")
}

fn read(rel: &str) -> String {
    let path = types_src_dir().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Concatenate every `.ts` file directly under `rel` (non-recursive).
fn read_dir_concat(rel: &str) -> String {
    let dir = types_src_dir().join(rel);
    let mut out = String::new();
    for entry in std::fs::read_dir(&dir).expect("read effect-types dir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("ts") {
            out.push_str(&std::fs::read_to_string(&path).unwrap());
            out.push('\n');
        }
    }
    out
}

// ===========================================================================
// Principled TypeScript discriminator reader (comment-stripped, dependency-free)
// ===========================================================================

/// Strip `//` line comments and `/* */` block comments.
fn strip_comments(s: &str) -> String {
    let c: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < c.len() {
        if c[i] == '/' && i + 1 < c.len() && c[i + 1] == '/' {
            i += 2;
            while i < c.len() && c[i] != '\n' {
                i += 1;
            }
        } else if c[i] == '/' && i + 1 < c.len() && c[i + 1] == '*' {
            i += 2;
            while i + 1 < c.len() && !(c[i] == '*' && c[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(c.len());
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn is_ident(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// All double-quoted string literals in `s` (used by forward containment).
fn quoted_strings(s: &str) -> Vec<String> {
    let cs = chars(s);
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        if cs[i] == '"' {
            let mut j = i + 1;
            let mut lit = String::new();
            while j < cs.len() && cs[j] != '"' {
                lit.push(cs[j]);
                j += 1;
            }
            if j < cs.len() {
                out.push(lit);
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Index just past a `kw name` declaration token, with identifier boundaries.
fn find_decl(cs: &[char], kw: &str, name: &str) -> Option<usize> {
    let pat: Vec<char> = format!("{kw} {name}").chars().collect();
    let mut i = 0;
    while i + pat.len() <= cs.len() {
        if cs[i..i + pat.len()] == pat[..] {
            let before = if i == 0 { ' ' } else { cs[i - 1] };
            let after = cs.get(i + pat.len()).copied().unwrap_or(' ');
            if !is_ident(before) && !is_ident(after) {
                return Some(i + pat.len());
            }
        }
        i += 1;
    }
    None
}

/// The brace-matched body (including the outer braces) of `interface NAME`.
fn iface_body(cs: &[char], name: &str) -> Option<String> {
    let mut i = find_decl(cs, "interface", name)?;
    while i < cs.len() && cs[i] != '{' {
        i += 1;
    }
    if i >= cs.len() {
        return None;
    }
    let mut depth = 0i32;
    let mut out = String::new();
    while i < cs.len() {
        let ch = cs[i];
        if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
        }
        out.push(ch);
        if depth == 0 {
            return Some(out);
        }
        i += 1;
    }
    Some(out)
}

/// The RHS of `type NAME = ...;`, terminated at the `;` that sits at bracket
/// depth 0 (so object members' internal `;` do not truncate it).
fn alias_rhs(cs: &[char], name: &str) -> Option<String> {
    let mut i = find_decl(cs, "type", name)?;
    while i < cs.len() && cs[i] != '=' {
        i += 1;
    }
    i += 1;
    let mut depth = 0i32;
    let mut out = String::new();
    while i < cs.len() {
        let ch = cs[i];
        match ch {
            '{' | '[' | '(' => depth += 1,
            '}' | ']' | ')' => depth -= 1,
            ';' if depth == 0 => return Some(out),
            _ => {}
        }
        out.push(ch);
        i += 1;
    }
    Some(out)
}

/// Collect `type: "a" | "b"` discriminant literals at brace depth 1 of `body`.
fn depth1_type_lits(body: &str) -> BTreeSet<String> {
    let cs = chars(body);
    let mut out = BTreeSet::new();
    let mut depth = 0i32;
    let mut i = 0;
    while i < cs.len() {
        let ch = cs[i];
        if ch == '{' {
            depth += 1;
            i += 1;
            continue;
        }
        if ch == '}' {
            depth -= 1;
            i += 1;
            continue;
        }
        if depth == 1 && cs[i..].starts_with(&['t', 'y', 'p', 'e']) {
            let prev = if i == 0 { ' ' } else { cs[i - 1] };
            let mut j = i + 4;
            while j < cs.len() && cs[j].is_whitespace() {
                j += 1;
            }
            if !is_ident(prev) && j < cs.len() && cs[j] == ':' {
                j += 1;
                collect_literal_union(&cs, &mut j, &mut out);
                i = j;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// From position `*j`, consume a `"a" | "b" | ...` literal sequence (allowing a
/// leading `|`), inserting each literal and advancing `*j` past the run.
fn collect_literal_union(cs: &[char], j: &mut usize, out: &mut BTreeSet<String>) {
    loop {
        while *j < cs.len() && (cs[*j].is_whitespace() || cs[*j] == '|') {
            *j += 1;
        }
        if *j < cs.len() && cs[*j] == '"' {
            *j += 1;
            let mut s = String::new();
            while *j < cs.len() && cs[*j] != '"' {
                s.push(cs[*j]);
                *j += 1;
            }
            *j += 1;
            out.insert(s);
        } else {
            break;
        }
    }
}

/// Split a union RHS into its top-level members (split on `|` at bracket depth 0).
fn split_top_members(rhs: &str) -> Vec<String> {
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut out = Vec::new();
    for ch in chars(rhs) {
        match ch {
            '{' | '[' | '(' => {
                depth += 1;
                cur.push(ch);
            }
            '}' | ']' | ')' => {
                depth -= 1;
                cur.push(ch);
            }
            '|' if depth == 0 => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Resolve the discriminator set of a named union/interface: literal members are
/// taken verbatim, object members contribute their depth-1 `type:`, and
/// identifier members recurse (so sub-unions like `KeywordAbility` expand).
fn resolve(cs: &[char], name: &str, seen: &mut BTreeSet<String>) -> BTreeSet<String> {
    if !seen.insert(name.to_string()) {
        return BTreeSet::new();
    }
    if let Some(body) = iface_body(cs, name) {
        return depth1_type_lits(&body);
    }
    let Some(rhs) = alias_rhs(cs, name) else {
        return BTreeSet::new();
    };
    let mut out = BTreeSet::new();
    for member in split_top_members(&rhs) {
        let mc = chars(&member);
        match mc.first() {
            Some('"') => {
                let mut j = 0usize;
                collect_literal_union(&mc, &mut j, &mut out);
            }
            Some('{') => out.extend(depth1_type_lits(&member)),
            _ => {
                let id: String = mc.iter().take_while(|c| is_ident(**c)).collect();
                if !id.is_empty() {
                    out.extend(resolve(cs, &id, seen));
                }
            }
        }
    }
    out
}

fn resolve_top(text: &str, name: &str) -> BTreeSet<String> {
    resolve(&chars(text), name, &mut BTreeSet::new())
}

/// Extract a string-literal union assigned to `field` inside `iface`.
fn field_union(text: &str, iface: &str, field: &str) -> BTreeSet<String> {
    let cs = chars(text);
    let Some(body) = iface_body(&cs, iface) else {
        return BTreeSet::new();
    };
    let bc = chars(&body);
    let fpat: Vec<char> = field.chars().collect();
    let mut i = 0;
    while i + fpat.len() <= bc.len() {
        if bc[i..i + fpat.len()] == fpat[..] {
            let before = if i == 0 { ' ' } else { bc[i - 1] };
            let mut k = i + fpat.len();
            while k < bc.len() && (bc[k] == '?' || bc[k].is_whitespace()) {
                k += 1;
            }
            if !is_ident(before) && k < bc.len() && bc[k] == ':' {
                k += 1;
                let mut out = BTreeSet::new();
                collect_literal_union(&bc, &mut k, &mut out);
                if !out.is_empty() {
                    return out;
                }
            }
        }
        i += 1;
    }
    BTreeSet::new()
}

/// Extract the quoted members of an `export const NAME = [ ... ] as const;` array.
fn const_array_members(text: &str, const_name: &str) -> BTreeSet<String> {
    let marker = format!("{const_name} = [");
    let start = text
        .find(&marker)
        .unwrap_or_else(|| panic!("const `{const_name}` not found in oracle"))
        + marker.len();
    let end = start
        + text[start..]
            .find(']')
            .expect("array close bracket not found");
    quoted_strings(&text[start..end]).into_iter().collect()
}

// ===========================================================================
// Forward containment (every Rust wire string appears in the oracle source)
// ===========================================================================

/// Map each manifest category to the oracle source it must be found in.
fn oracle_text_for(category: &str) -> String {
    match category {
        "InkType" => read("cards/ink-types.ts"),
        "Classification" => read("cards/classifications.ts"),
        "CardType" | "Language" | "ActionSubtype" | "Rarity" | "SpecialRarity" => {
            read("cards/card-types.ts")
        }
        "SimpleKeywordType"
        | "ParameterizedKeywordType"
        | "ValueKeywordType"
        | "KeywordType"
        | "ReplacementAbilityReplaces"
        | "RestrictionType"
        // Each ability `type:` literal (keyword/triggered/activated/static/
        // action/replacement) appears as code in ability-types.ts.
        | "AbilityType" => read("abilities/ability-types.ts"),
        "ReplacementEffectReplaces" | "EffectType" => read_dir_concat("abilities/effect-types"),
        "CostComponentType" => read("abilities/cost-types.ts"),
        "ConditionType" | "ConditionComparisonOperator" => read("abilities/condition-types.ts"),
        "TriggerEvent"
        | "TriggerTiming"
        | "TriggerCardType"
        | "TriggerSubjectEnum"
        | "TriggerRestrictionType" => read("abilities/trigger-types.ts"),
        "ComparisonOperator" | "CardStatus" | "AmountRef" | "EffectDuration"
        | "ForEachCounterType" | "VariableAmountType" => read("expressions/index.ts"),
        "TargetZone" | "TargetController" => read("abilities/target-types.ts"),
        "LorcanaZoneId" => read("targeting/lorcana-target-dsl.ts"),
        "SelectorScope" | "OwnerScope" | "PlayerTargetScope" => read("targeting/target-dsl.ts"),
        "LorcanaSetCode" | "LorcanaFormatId" | "FormatValidationKind" => {
            read("decks/validate-deck.ts")
        }
        other => panic!("no oracle-source mapping for manifest category `{other}`"),
    }
}

#[test]
fn every_rust_discriminator_exists_in_frozen_oracle() {
    for (category, names) in discriminator_manifest() {
        let text = oracle_text_for(category);
        for name in names {
            let needle = format!("\"{name}\"");
            assert!(
                text.contains(&needle),
                "category `{category}`: discriminator {needle} not found in frozen oracle source"
            );
        }
    }
}

#[test]
fn no_duplicate_wire_strings_within_a_category() {
    for (category, names) in discriminator_manifest() {
        let mut seen = BTreeSet::new();
        for name in names {
            assert!(
                seen.insert(*name),
                "category `{category}`: duplicate wire string {name:?}"
            );
        }
    }
}

// ===========================================================================
// Exact set-equality (oracle declaration <-> Rust enum), both directions
// ===========================================================================

/// The Rust discriminator set for `category`, read from the live manifest.
fn rust_set(category: &str) -> BTreeSet<String> {
    let manifest = discriminator_manifest();
    let names = manifest
        .get(category)
        .unwrap_or_else(|| panic!("manifest missing category `{category}`"));
    names.iter().map(|s| s.to_string()).collect()
}

/// All closed categories paired with the set extracted from the oracle's own
/// declaration. Independent of the Rust enums (drift would surface as a diff).
fn oracle_sets() -> Vec<(&'static str, BTreeSet<String>)> {
    // The structured reader requires comment-free text (comments may contain
    // braces/pipes/`type:` examples that would corrupt union parsing).
    let ability = strip_comments(&read("abilities/ability-types.ts"));
    let cond = strip_comments(&read("abilities/condition-types.ts"));
    let cost = strip_comments(&read("abilities/cost-types.ts"));
    let effect = strip_comments(&read_dir_concat("abilities/effect-types"));
    let trigger = strip_comments(&read("abilities/trigger-types.ts"));
    let target = strip_comments(&read("abilities/target-types.ts"));
    let expr = strip_comments(&read("expressions/index.ts"));
    let tdsl = strip_comments(&read("targeting/target-dsl.ts"));
    let ltdsl = strip_comments(&read("targeting/lorcana-target-dsl.ts"));
    let deck = strip_comments(&read("decks/validate-deck.ts"));
    let card = strip_comments(&read("cards/card-types.ts"));
    let ink = read("cards/ink-types.ts");
    let classif = read("cards/classifications.ts");

    let mut effect_set = resolve_top(&effect, "Effect");
    effect_set.extend(resolve_top(&effect, "StaticEffect"));

    vec![
        // const arrays
        ("InkType", const_array_members(&ink, "INK_TYPES")),
        (
            "Classification",
            const_array_members(&classif, "CLASSIFICATIONS"),
        ),
        ("CardType", const_array_members(&card, "CARD_TYPES")),
        ("Language", const_array_members(&card, "LANGUAGES")),
        // card-property scalar unions
        ("ActionSubtype", resolve_top(&card, "ActionSubtype")),
        (
            "Rarity",
            field_union(&card, "LorcanaCardDefinition", "rarity"),
        ),
        (
            "SpecialRarity",
            field_union(&card, "LorcanaCardDefinition", "specialRarity"),
        ),
        // keyword group
        (
            "SimpleKeywordType",
            resolve_top(&ability, "SimpleKeywordType"),
        ),
        (
            "ParameterizedKeywordType",
            resolve_top(&ability, "ParameterizedKeywordType"),
        ),
        (
            "ValueKeywordType",
            resolve_top(&ability, "ValueKeywordType"),
        ),
        ("KeywordType", resolve_top(&ability, "KeywordType")),
        // ability group
        ("AbilityType", resolve_top(&ability, "Ability")),
        ("RestrictionType", resolve_top(&ability, "Restriction")),
        (
            "ReplacementAbilityReplaces",
            field_union(&ability, "ReplacementAbility", "replaces"),
        ),
        (
            "TriggerRestrictionType",
            resolve_top(&trigger, "TriggerRestriction"),
        ),
        (
            "ReplacementEffectReplaces",
            field_union(&effect, "ReplacementEffect", "replaces"),
        ),
        // cost group
        ("CostComponentType", resolve_top(&cost, "CostComponent")),
        // condition group
        ("ConditionType", resolve_top(&cond, "Condition")),
        (
            "ConditionComparisonOperator",
            resolve_top(&cond, "ConditionComparisonOperator"),
        ),
        // effect group
        ("EffectType", effect_set),
        // trigger group
        ("TriggerEvent", resolve_top(&trigger, "TriggerEvent")),
        ("TriggerTiming", resolve_top(&trigger, "TriggerTiming")),
        ("TriggerCardType", resolve_top(&trigger, "TriggerCardType")),
        (
            "TriggerSubjectEnum",
            resolve_top(&trigger, "TriggerSubjectEnum"),
        ),
        // target group
        ("TargetZone", resolve_top(&target, "TargetZone")),
        ("TargetController", resolve_top(&target, "TargetController")),
        ("LorcanaZoneId", resolve_top(&ltdsl, "LorcanaZoneId")),
        ("SelectorScope", resolve_top(&tdsl, "SelectorScope")),
        ("OwnerScope", resolve_top(&tdsl, "OwnerScope")),
        ("PlayerTargetScope", resolve_top(&tdsl, "PlayerTargetScope")),
        // expression group (canonical declarations live in expressions/index.ts)
        (
            "ComparisonOperator",
            resolve_top(&expr, "ComparisonOperator"),
        ),
        ("CardStatus", resolve_top(&expr, "CardStatus")),
        ("AmountRef", resolve_top(&expr, "AmountRef")),
        ("EffectDuration", resolve_top(&expr, "EffectDuration")),
        (
            "ForEachCounterType",
            resolve_top(&expr, "ForEachCounterType"),
        ),
        ("VariableAmountType", resolve_top(&expr, "VariableAmount")),
        // deck-format group
        ("LorcanaSetCode", resolve_top(&deck, "LorcanaSetCode")),
        ("LorcanaFormatId", resolve_top(&deck, "LorcanaFormatId")),
        (
            "FormatValidationKind",
            resolve_top(&deck, "FormatValidationKind"),
        ),
    ]
}

#[test]
fn oracle_discriminator_sets_equal_rust_exactly() {
    let mut failures = Vec::new();
    let covered: BTreeSet<&str> = oracle_sets().iter().map(|(c, _)| *c).collect();

    for (category, oracle) in oracle_sets() {
        assert!(
            !oracle.is_empty(),
            "category `{category}`: oracle extraction returned an empty set (parser/anchor bug)"
        );
        let rust = rust_set(category);
        let missing: Vec<_> = oracle.difference(&rust).cloned().collect();
        let extra: Vec<_> = rust.difference(&oracle).cloned().collect();
        if !missing.is_empty() || !extra.is_empty() {
            failures.push(format!(
                "category `{category}`: in oracle but not rust = {missing:?}; in rust but not oracle = {extra:?}"
            ));
        }
    }

    // Every manifest category must be covered by an exact-equality check, so a
    // newly added enum cannot silently escape bidirectional validation.
    for category in discriminator_manifest().keys() {
        assert!(
            covered.contains(category),
            "manifest category `{category}` has no exact set-equality check in oracle_sets()"
        );
    }

    assert!(
        failures.is_empty(),
        "discriminator set mismatches vs frozen oracle:\n{}",
        failures.join("\n")
    );
}

// Keep the explicit, isolated const-array equalities as targeted regressions.
#[test]
fn ink_types_set_matches_oracle_exactly() {
    let oracle = const_array_members(&read("cards/ink-types.ts"), "INK_TYPES");
    let rust: BTreeSet<String> = InkType::NAMES.iter().map(|s| s.to_string()).collect();
    assert_eq!(rust, oracle, "InkType set differs from oracle INK_TYPES");
}

#[test]
fn classifications_set_matches_oracle_exactly() {
    let oracle = const_array_members(&read("cards/classifications.ts"), "CLASSIFICATIONS");
    let rust: BTreeSet<String> = Classification::NAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        rust, oracle,
        "Classification set differs from oracle CLASSIFICATIONS"
    );
}

#[test]
fn card_types_set_matches_oracle_exactly() {
    let oracle = const_array_members(&read("cards/card-types.ts"), "CARD_TYPES");
    let rust: BTreeSet<String> = CardType::NAMES.iter().map(|s| s.to_string()).collect();
    assert_eq!(rust, oracle, "CardType set differs from oracle CARD_TYPES");
}
