"""Public behaviour checks for alias-aware query costing."""

from gqlcost import (
    check_alias_limit,
    check_cost,
    collect_aliases,
    cost_batch,
    cost_of,
    enforce,
    parse_query,
)

POLICY = {
    "max_cost": 100,
    "max_aliases": 20,
    "default_weight": 1,
    "field_weights": {"user": 3, "id": 1, "post": 2},
    "introspection_weight": 4,
    "alias_weight": 1,
}


def test_parse_query_preserves_each_alias_and_nested_selection():
    ast = parse_query("query Q { first: user { id } second: user { id } }")
    fields = ast["operations"][0]["selection"]

    assert [field["alias"] for field in fields] == ["first", "second"]
    assert [field["name"] for field in fields] == ["user", "user"]
    assert fields[0]["selection"][0]["name"] == "id"
    assert cost_of(ast, POLICY) == 10


def test_collect_aliases_expands_fragment_spreads():
    ast = parse_query(
        "query { a: user ...Names } "
        "fragment Names on Query { b: user c: user }"
    )
    aliases = collect_aliases(ast)

    assert [field["alias"] for field in aliases] == ["a", "b", "c"]
    assert check_alias_limit(ast, {**POLICY, "max_aliases": 2}) is False
    assert enforce(ast, {**POLICY, "max_aliases": 2}) != []


def test_cost_accumulates_weights_aliases_and_repeated_fields():
    ast = parse_query("{ first: user second: user third: post }")

    assert cost_of(ast, POLICY) == 11


def test_enforce_rejects_a_query_over_the_cost_ceiling():
    ast = parse_query("{ user }")
    policy = {**POLICY, "max_cost": 0}

    assert check_cost(ast, policy) is False
    assert enforce(ast, policy) != []


def test_cost_batch_keeps_two_queries_independent():
    batch = cost_batch(
        ["{ user }", "{ user { posts } }"],
        {
            "max_cost": 10,
            "max_aliases": 10,
            "default_weight": 1,
            "alias_weight": 0,
        },
    )

    assert batch["costs"] == [1, 2]
    assert batch["allowed"] == [True, True]