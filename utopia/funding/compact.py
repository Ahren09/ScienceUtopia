"""Compact funding elicitation for new, prospectively versioned experiments.

The model selects the complete order of application indices. Rank numbers and
applicant strings are deterministic input metadata; they are expanded only
after the entire permutation passes validation. No missing index is filled,
no duplicate is removed, and no model-selected ordering is changed.

Changing the requested output can change the model's ranking distribution.
This is a shared new elicitation protocol, not demonstrated equivalence to the
legacy response format. Funding explanations are not consumed by the stock
simulation; an empty legacy reason records their deliberate omission.
"""

from copy import deepcopy


COMPACT_REPRESENTATION = "ordered_application_ids_v1"
COMPACT_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_application_ids": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 24},
            "minItems": 1,
            "maxItems": 25,
        },
    },
    "required": ["ranked_application_ids"],
    "additionalProperties": False,
}


def compact_response_format():
    return {
        "type": "json_schema",
        "json_object": {
            "name": "CompactFundingRanking",
            "description": "Complete application-index ordering, most preferred first",
            "schema": deepcopy(COMPACT_SCHEMA),
        },
    }


def compact_prompt(original_prompt, n_apps):
    if type(original_prompt) is not str:
        raise ValueError("Funding prompt must be a string")
    if type(n_apps) is not int or not 1 <= n_apps <= 25:
        raise ValueError("Compact funding requires a nonempty panel of at most 25 applications")
    prefix, marker, tail = original_prompt.rpartition("\n## Response Format\n")
    if not marker or '"ranked_applications"' not in tail:
        raise ValueError("Original funding response-format boundary was not found")
    return prefix + marker + (
        'Return only a JSON object with the single key "ranked_application_ids".\n'
        f"The value must be an array containing exactly {n_apps} integer application IDs, "
        f"from 0 through {n_apps - 1}, with every application ID appearing exactly once.\n"
        "Order the IDs from the most preferred application to the least preferred application. "
        "The first array entry receives rank 1, the second receives rank 2, and so on.\n"
        "Use the application indices shown in the application headings. "
        "Do not output applicant names, separate rank numbers, or explanations.\n"
        "Before returning the object, check that no application ID is missing or repeated.\n"
    )


def expand_compact_result(result, apps):
    if type(apps) not in (list, tuple) or not 1 <= len(apps) <= 25:
        raise ValueError("Expected a nonempty application panel of size at most 25")
    if any(type(app) is not dict or type(app.get("applicant_id")) is not str for app in apps):
        raise ValueError("Frozen application metadata lack exact applicant IDs")
    if type(result) is not dict or set(result) != {"ranked_application_ids"}:
        raise ValueError("Compact output must contain only ranked_application_ids")
    order = result["ranked_application_ids"]
    if type(order) is not list or len(order) != len(apps):
        raise ValueError("Compact ranking does not have exactly one entry per application")
    if any(type(index) is not int for index in order):
        raise ValueError("Application indices must be actual integers, not booleans or coerced values")
    if sorted(order) != list(range(len(apps))):
        raise ValueError("Application indices are not a complete unique permutation")
    return {
        "ranked_applications": [
            {
                "application_id": index,
                "applicant_id": apps[index]["applicant_id"],
                "rank": rank,
                "reason": "",
            }
            for rank, index in enumerate(order, 1)
        ],
    }
