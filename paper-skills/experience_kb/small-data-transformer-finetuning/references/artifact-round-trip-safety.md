# Artifact Round-Trip Safety

Patterns for saving and loading intermediate pipeline artifacts without serialization crashes.

## String/object arrays

**Option A — CSV (preferred for string metadata):**
```python
pd.DataFrame({"id": test_ids}).to_csv("./working/test_ids.csv", index=False)
test_ids = pd.read_csv("./working/test_ids.csv")["id"].values
```

**Option B — np.load with allow_pickle:**
```python
np.save("./working/test_ids.npy", np.array(test_ids, dtype=object))
test_ids = np.load("./working/test_ids.npy", allow_pickle=True)
```

## Numeric arrays

```python
features = np.load("./working/features.npy")  # safe for numeric dtypes
```

## Round-trip validation checklist

1. Confirm the dtype produced by the save step.
2. Confirm the load call is compatible with that dtype.
3. For mixed-type or string data, prefer CSV/JSON over `.npy`.
