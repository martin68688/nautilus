# Safe Text Processing Patterns

## Literal Character Counting

**Problem:** `Series.str.count("(")` crashes because `(` is a regex metacharacter.

**Solutions (in order of preference):**

1. **Python `str.count` (literal, no regex):**
   ```python
   paren_count = text_series.apply(lambda s: s.count("("))
   ```

2. **Escaped regex:**
   ```python
   paren_count = text_series.str.count(r"\(")
   ```

## Escaping Regex Metacharacters

Characters that MUST be escaped: `( ) [ ] { } * + ? . | ^ $ \`

Use `re.escape()` for dynamic patterns:
```python
import re
pattern = re.escape(user_input)
text_series.str.contains(pattern)
```

## Non-Regex Alternatives

| Task | Regex Approach (risky) | Non-Regex Approach (safe) |
|------|----------------------|--------------------------|
| Count literal char | `str.count(r"\(")` | `apply(lambda s: s.count("("))` |
| Replace literal string | `str.replace(r"(", "", regex=True)` | `apply(lambda s: s.replace("(", ""))` |
| Check substring | `str.contains(r"\(")` | `apply(lambda s: "(" in s)` |
