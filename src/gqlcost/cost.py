"""Alias-aware GraphQL query costing and admission checks.

The public entry points operate on a small dictionary based syntax tree.  A
document has the following shape::

    {
        "kind": "document",
        "operations": [
            {
                "kind": "operation",
                "operation": "query",
                "name": "Q" | None,
                "selection": [node, ...],
            },
        ],
        "fragments": {
            "Details": {
                "kind": "fragment",
                "name": "Details",
                "type_condition": "User",
                "selection": [node, ...],
            },
        },
    }

A field node is::

    {
        "kind": "field",
        "name": "user",
        "alias": "owner" | None,
        "arguments": {...},
        "directives": [...],
        "selection": [node, ...],
    }

Two auxiliary nodes are used for fragment spreads::

    {"kind": "spread", "name": "Details"}
    {"kind": "inline_fragment", "type_condition": "User", "selection": [...]}

Policy keys are ordinary mappings.  max_cost is the normal query budget and
max_aliases is the alias ceiling.  field_weights maps a field name to its
integer weight; default_weight applies when no entry exists.  Introspection
fields, whose names begin with two underscores, use introspection_weight
instead of the normal field table.  Every explicit alias adds alias_weight to
the cost of that occurrence.  Missing weights use the values documented in the
function signatures.

enforce returns a list of human readable problems.  An empty list means the
query is admissible; a non-empty list is a rejection.  The helpers never
mutate the caller's document or policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "parse_query",
    "collect_aliases",
    "field_weight",
    "cost_of",
    "check_alias_limit",
    "check_cost",
    "enforce",
    "cost_batch",
]

_NAME_START = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
)
_NAME_CONTINUE = _NAME_START | frozenset("0123456789")
_PUNCTUATION = frozenset("!$&():=@[]{|}")
_OPERATION_WORDS = frozenset({"query", "mutation", "subscription"})


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Split a GraphQL document into a compact token stream."""

    tokens: list[tuple[str, str]] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in " \t\r\n,":
            index += 1
            continue
        if char == "#":
            newline = text.find("\n", index)
            index = length if newline == -1 else newline + 1
            continue
        if text.startswith("...", index):
            tokens.append(("SPREAD", "..."))
            index += 3
            continue
        if char == '"':
            if text.startswith('"""', index):
                end = text.find('"""', index + 3)
                if end == -1:
                    raise ValueError("unterminated block string")
                tokens.append(("STRING", text[index + 3 : end]))
                index = end + 3
                continue
            cursor = index + 1
            value: list[str] = []
            while cursor < length:
                current = text[cursor]
                if current == "\\":
                    cursor += 1
                    if cursor >= length:
                        raise ValueError("unterminated escape sequence")
                    value.append(text[cursor])
                    cursor += 1
                    continue
                if current == '"':
                    break
                value.append(current)
                cursor += 1
            else:
                raise ValueError("unterminated string")
            tokens.append(("STRING", "".join(value)))
            index = cursor + 1
            continue
        if char in _PUNCTUATION:
            tokens.append(("PUNCT", char))
            index += 1
            continue
        if char in _NAME_START:
            cursor = index + 1
            while cursor < length and text[cursor] in _NAME_CONTINUE:
                cursor += 1
            tokens.append(("NAME", text[index:cursor]))
            index = cursor
            continue
        if char.isdigit() or char == "-":
            cursor = index + 1
            while cursor < length and (
                text[cursor].isdigit() or text[cursor] in ".eE+-"
            ):
                cursor += 1
            tokens.append(("NUMBER", text[index:cursor]))
            index = cursor
            continue
        raise ValueError(f"unexpected character {char!r}")
    tokens.append(("EOF", ""))
    return tokens


class _Parser:
    """Recursive descent parser for the execution subset used by the engine."""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._index = 0

    def _peek(self) -> tuple[str, str]:
        return self._tokens[self._index]

    def _advance(self) -> tuple[str, str]:
        token = self._tokens[self._index]
        if token[0] != "EOF":
            self._index += 1
        return token

    def _at_punct(self, value: str) -> bool:
        kind, current = self._peek()
        return kind == "PUNCT" and current == value

    def _at_name(self, value: str) -> bool:
        kind, current = self._peek()
        return kind == "NAME" and current == value

    def _expect_punct(self, value: str) -> None:
        if not self._at_punct(value):
            raise ValueError(f"expected {value!r}")
        self._advance()

    def _expect_name(self) -> str:
        kind, value = self._peek()
        if kind != "NAME":
            raise ValueError("expected a name")
        self._advance()
        return value

    def parse_document(self) -> dict[str, Any]:
        operations: list[dict[str, Any]] = []
        fragments: dict[str, dict[str, Any]] = {}
        while self._peek()[0] != "EOF":
            if self._at_punct("{"):
                operations.append(
                    {
                        "kind": "operation",
                        "operation": "query",
                        "name": None,
                        "selection": self._parse_selection(),
                    }
                )
                continue
            kind, word = self._peek()
            if kind != "NAME":
                raise ValueError("expected an operation or fragment")
            if word in _OPERATION_WORDS:
                operations.append(self._parse_operation())
                continue
            if word == "fragment":
                fragment = self._parse_fragment()
                name = fragment["name"]
                if name in fragments:
                    raise ValueError(f"duplicate fragment {name}")
                fragments[name] = fragment
                continue
            raise ValueError(f"unexpected definition {word}")
        return {
            "kind": "document",
            "operations": operations,
            "fragments": fragments,
        }

    def _parse_operation(self) -> dict[str, Any]:
        operation = self._advance()[1]
        name = None
        if self._peek()[0] == "NAME":
            name = self._advance()[1]
        if self._at_punct("("):
            self._skip_balanced("(", ")")
        directives = self._parse_directives()
        return {
            "kind": "operation",
            "operation": operation,
            "name": name,
            "directives": directives,
            "selection": self._parse_selection(),
        }

    def _parse_fragment(self) -> dict[str, Any]:
        self._expect_name()
        name = self._expect_name()
        if not self._at_name("on"):
            raise ValueError("fragment is missing its type condition")
        self._advance()
        type_condition = self._expect_name()
        directives = self._parse_directives()
        return {
            "kind": "fragment",
            "name": name,
            "type_condition": type_condition,
            "directives": directives,
            "selection": self._parse_selection(),
        }

    def _parse_selection(self) -> list[dict[str, Any]]:
        self._expect_punct("{")
        selection: list[dict[str, Any]] = []
        while not self._at_punct("}"):
            if self._peek()[0] == "EOF":
                raise ValueError("unterminated selection set")
            if self._peek()[0] == "SPREAD":
                selection.append(self._parse_spread())
            else:
                selection.append(self._parse_field())
        self._expect_punct("}")
        return selection

    def _parse_spread(self) -> dict[str, Any]:
        self._advance()
        if self._at_name("on"):
            self._advance()
            type_condition: str | None = self._expect_name()
            directives = self._parse_directives()
            return {
                "kind": "inline_fragment",
                "type_condition": type_condition,
                "directives": directives,
                "selection": self._parse_selection(),
            }
        if self._at_punct("@"):
            directives = self._parse_directives()
            return {
                "kind": "inline_fragment",
                "type_condition": None,
                "directives": directives,
                "selection": self._parse_selection(),
            }
        if self._at_punct("{"):
            return {
                "kind": "inline_fragment",
                "type_condition": None,
                "directives": [],
                "selection": self._parse_selection(),
            }
        name = self._expect_name()
        directives = self._parse_directives()
        return {"kind": "spread", "name": name, "directives": directives}

    def _parse_field(self) -> dict[str, Any]:
        first = self._expect_name()
        alias = None
        name = first
        if self._at_punct(":"):
            self._advance()
            alias = first
            name = self._expect_name()
        arguments = self._parse_arguments() if self._at_punct("(") else {}
        directives = self._parse_directives()
        selection = self._parse_selection() if self._at_punct("{") else []
        return {
            "kind": "field",
            "name": name,
            "alias": alias,
            "arguments": arguments,
            "directives": directives,
            "selection": selection,
        }

    def _parse_directives(self) -> list[dict[str, Any]]:
        directives: list[dict[str, Any]] = []
        while self._at_punct("@"):
            self._advance()
            name = self._expect_name()
            arguments = self._parse_arguments() if self._at_punct("(") else {}
            directives.append({"name": name, "arguments": arguments})
        return directives

    def _parse_arguments(self) -> dict[str, Any]:
        self._expect_punct("(")
        arguments: dict[str, Any] = {}
        while not self._at_punct(")"):
            if self._peek()[0] == "EOF":
                raise ValueError("unterminated argument list")
            name = self._expect_name()
            self._expect_punct(":")
            arguments[name] = self._parse_value()
        self._expect_punct(")")
        return arguments

    def _parse_value(self) -> Any:
        kind, value = self._peek()
        if kind in {"STRING", "NUMBER", "NAME"}:
            self._advance()
            return value
        if kind == "PUNCT" and value == "$":
            self._advance()
            return "$" + self._expect_name()
        if kind == "PUNCT" and value == "[":
            self._advance()
            values = []
            while not self._at_punct("]"):
                if self._peek()[0] == "EOF":
                    raise ValueError("unterminated list value")
                values.append(self._parse_value())
            self._advance()
            return values
        if kind == "PUNCT" and value == "{":
            self._advance()
            fields: dict[str, Any] = {}
            while not self._at_punct("}"):
                if self._peek()[0] == "EOF":
                    raise ValueError("unterminated object value")
                name = self._expect_name()
                self._expect_punct(":")
                fields[name] = self._parse_value()
            self._advance()
            return fields
        raise ValueError("expected a value")

    def _skip_balanced(self, opening: str, closing: str) -> None:
        self._expect_punct(opening)
        depth = 1
        while depth:
            kind, value = self._peek()
            if kind == "EOF":
                raise ValueError(f"unterminated {opening}")
            self._advance()
            if kind == "PUNCT" and value == opening:
                depth += 1
            elif kind == "PUNCT" and value == closing:
                depth -= 1


def parse_query(text: str) -> dict[str, Any]:
    """Parse text into a document tree.

    Supports shorthand and named operations, arbitrary selection nesting,
    aliases, arguments, directives, named fragments, and inline fragments.
    Invalid text raises ValueError.
    """

    if not isinstance(text, str):
        raise TypeError("query text must be a string")
    if not text.strip():
        raise ValueError("query text is empty")
    return _Parser(_tokenize(text)).parse_document()


def _field_name(field: Any) -> str:
    if isinstance(field, Mapping):
        name = field.get("name")
    else:
        name = field
    if not isinstance(name, str) or not name:
        raise TypeError("field must have a name")
    return name


def _amount(value: Any, default: int = 1) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, number)


def field_weight(field: Any, policy: Mapping[str, Any] | None) -> int:
    """Return the weight of one field under policy.

    Introspection fields use introspection_weight.  Ordinary fields use the
    field_weights table, falling back to default_weight when absent.
    """

    name = _field_name(field)
    policy = policy or {}
    if name.startswith("__"):
        return _amount(policy.get("introspection_weight"), 3)
    weights = policy.get("field_weights")
    if not isinstance(weights, Mapping):
        weights = {}
    return _amount(weights.get(name, policy.get("default_weight")), 1)


def _cost_weight(field: Mapping[str, Any], policy: Mapping[str, Any]) -> int:
    return field_weight(field, policy)


def _document_operations(ast: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(ast, Mapping) or ast.get("kind") != "document":
        raise ValueError("expected a document node")
    operations = ast.get("operations")
    if not isinstance(operations, list):
        raise ValueError("document operations must be a list")
    return operations


def _document_fragments(ast: Mapping[str, Any]) -> Mapping[str, Any]:
    fragments = ast.get("fragments")
    if fragments is None:
        return {}
    if not isinstance(fragments, Mapping):
        raise ValueError("document fragments must be a mapping")
    return fragments


def _fragment(
    fragments: Mapping[str, Any], name: str, stack: tuple[str, ...]
) -> Mapping[str, Any]:
    if name in stack:
        raise ValueError(f"fragment cycle at {name}")
    fragment = fragments.get(name)
    if not isinstance(fragment, Mapping):
        raise ValueError(f"unknown fragment {name}")
    return fragment


def _collect_selection(
    selection: list[dict[str, Any]],
    fragments: Mapping[str, Any],
    found: list[dict[str, Any]],
    stack: tuple[str, ...],
) -> None:
    for node in selection:
        kind = node.get("kind")
        if kind == "field":
            if node.get("alias") is not None:
                found.append(node)
            _collect_selection(
                node.get("selection") or [], fragments, found, stack
            )
        elif kind == "inline_fragment":
            _collect_selection(
                node.get("selection") or [], fragments, found, stack
            )
        elif kind == "spread":
            name = node.get("name")
            fragment = _fragment(fragments, name, stack)
            _collect_selection(
                fragment.get("selection") or [],
                fragments,
                found,
                stack + (name,),
            )


def collect_aliases(ast: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every explicit alias field in request order.

    Fragment spreads are expanded where they occur.  A fragment used twice
    therefore contributes its aliases twice, and unreferenced fragment
    definitions contribute nothing.
    """

    fragments = _document_fragments(ast)
    found: list[dict[str, Any]] = []
    for operation in _document_operations(ast):
        _collect_selection(
            operation.get("selection") or [], fragments, found, ()
        )
    return found


def _cost_selection(
    selection: list[dict[str, Any]],
    fragments: Mapping[str, Any],
    policy: Mapping[str, Any],
    stack: tuple[str, ...],
) -> int:
    total = 0
    for node in selection:
        kind = node.get("kind")
        if kind == "field":
            # Every occurrence is a separate resolver invocation, so aliased
            # copies of the same field are metered separately rather than
            # de-duplicated.  The explicit alias itself adds a surcharge.
            cost = _cost_weight(node, policy)
            if node.get("alias") is not None:
                cost += _amount(policy.get("alias_weight"), 1)
            total += cost + _cost_selection(
                node.get("selection") or [], fragments, policy, stack
            )
        elif kind == "inline_fragment":
            total += _cost_selection(
                node.get("selection") or [], fragments, policy, stack
            )
        elif kind == "spread":
            name = node.get("name")
            fragment = _fragment(fragments, name, stack)
            total += _cost_selection(
                fragment.get("selection") or [],
                fragments,
                policy,
                stack + (name,),
            )
    return total


def cost_of(ast: Mapping[str, Any], policy: Mapping[str, Any] | None) -> int:
    """Return the total cost of a document.

    Each field occurrence contributes its weight plus any alias charge.  All
    occurrences and all selection levels are accumulated.  Fragments are
    charged every time their spread appears.
    """

    policy = policy or {}
    fragments = _document_fragments(ast)
    total = 0
    for operation in _document_operations(ast):
        total += _cost_selection(
            operation.get("selection") or [], fragments, policy, ()
        )
    return total


def check_alias_limit(
    ast: Mapping[str, Any], policy: Mapping[str, Any] | None
) -> bool:
    """Return whether the document stays within max_aliases."""

    policy = policy or {}
    limit = policy.get("max_aliases")
    if limit is None:
        return True
    return len(collect_aliases(ast)) <= _amount(limit, 0)


def check_cost(ast: Mapping[str, Any], policy: Mapping[str, Any] | None) -> bool:
    """Return whether the document cost stays within max_cost."""

    policy = policy or {}
    limit = policy.get("max_cost")
    if limit is None:
        return False
    return cost_of(ast, policy) <= _amount(limit, 0)


def _policy_issues(policy: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    cost = policy.get("max_cost")
    if cost is None:
        issues.append("policy max_cost is required")
    elif isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
        issues.append("policy max_cost must be a non-negative integer")
    aliases = policy.get("max_aliases")
    if aliases is not None:
        if isinstance(aliases, bool) or not isinstance(aliases, int) or aliases < 0:
            issues.append("policy max_aliases must be a non-negative integer")
    return issues


def enforce(
    ast: Mapping[str, Any], policy: Mapping[str, Any] | None
) -> list[str]:
    """Return the admission problems for one document.

    The result is empty only when the policy is usable and both the alias
    ceiling and the cost ceiling are respected.
    """

    policy = policy or {}
    problems = _policy_issues(policy)
    if problems:
        return problems
    if not check_alias_limit(ast, policy):
        problems.append("alias limit exceeded")
    if not check_cost(ast, policy):
        problems.append("cost limit exceeded")
    return problems


def cost_batch(
    queries: list[str], policy: Mapping[str, Any] | None
) -> dict[str, list[Any]]:
    """Price and check several query strings independently.

    The returned mapping has parallel lists: costs, allowed, and errors, all in
    input order.  An error list is empty exactly when that query is admissible.
    """

    costs: list[int] = []
    allowed: list[bool] = []
    errors: list[list[str]] = []
    for text in queries:
        ast = parse_query(text)
        cost = cost_of(ast, policy)
        issues = enforce(ast, policy)
        costs.append(cost)
        allowed.append(not issues)
        errors.append(issues)
    return {"costs": costs, "allowed": allowed, "errors": errors}