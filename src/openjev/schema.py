"""The intentionally small categorical/boolean schema supported by this experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    description: str
    choices: tuple[str | bool, ...]


def same_value(a, b) -> bool:
    return type(a) is type(b) and a == b


def parse_schema(schema: dict) -> tuple[Field, ...]:
    if not isinstance(schema, dict) or not schema:
        raise ValueError("schema must be a nonempty object")
    fields = []
    for name, spec in schema.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
            raise ValueError("fields need nonempty names and object definitions")
        description = spec.get("description", name)
        if not isinstance(description, str):
            raise ValueError(f"{name}: description must be text")
        if spec.get("type") == "boolean":
            choices = (False, True)
        elif spec.get("type") == "enum":
            raw = spec.get("choices")
            if not isinstance(raw, list) or not 2 <= len(raw) <= 255:
                raise ValueError(f"{name}: provide between 2 and 255 enum choices")
            if any(not isinstance(c, str) or not c for c in raw) or len(set(raw)) != len(raw):
                raise ValueError(f"{name}: enum choices must be unique nonempty strings")
            choices = tuple(raw)
        else:
            raise ValueError(f"{name}: supported types are enum and boolean")
        fields.append(Field(name, description, choices))
    return tuple(fields)


def valid_answer(answer, fields: tuple[Field, ...]) -> bool:
    return (
        isinstance(answer, dict)
        and set(answer) == {f.name for f in fields}
        and all(any(same_value(answer[f.name], c) for c in f.choices) for f in fields)
    )
