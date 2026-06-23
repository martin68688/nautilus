# Code Validation Patterns

## Merge Conflict Artifacts

Scan generated code for these patterns before execution:
- `<<<<<<<` — conflict start marker
- `=======` — conflict separator
- `>>>>>>>` — conflict end marker

## Syntax Validation

```python
import ast

with open(script_path, 'r') as f:
    source = f.read()
try:
    ast.parse(source)
except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}: {e.msg}")
```

Alternatively: `python -m py_compile script.py`

## Common Generation Pitfalls

1. **Incomplete code blocks** — unclosed parentheses, brackets, or quotes
2. **Placeholder text** — `TODO`, `FIXME`, or `...` left in executable sections
3. **Mixed indentation** — tabs and spaces mixed within the same block
4. **Duplicate definitions** — repeated `def` or `class` names from overlapping edits
