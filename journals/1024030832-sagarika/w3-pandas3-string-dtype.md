# Week 3 : Text Columns Not Detected Under pandas 3

# `object` vs `StringDtype`

## Error:

After the merge, the data loader stopped recognising
text columns.  Columns such as `"yes"/"no"` flags,
`"2024-01-15"` dates and numbers stored as strings were
left as plain text instead of being converted to
boolean, datetime and numeric types, and repetitive
string columns were no longer converted to `category`
to save memory.

No exception was raised; the type detection silently
did nothing.

## Relevant Context

In `ml_engine/data_loader.py`, every check for a text
column was written as:

``` python
if series.dtype == "object":
    ...  # try boolean / datetime / numeric coercion
```

Our virtual environment has **pandas 3.0.5**.  A quick
check shows what a text column looks like there:

``` python
>>> s = pd.Series(["a", "b"])
>>> s.dtype
<StringDtype(na_value=nan)>
>>> s.dtype == "object"
False
```

## Key Observation

From pandas 3, text is stored in the dedicated string
dtype by default, not in `object`.  The code was
written and tested against pandas 2, where text columns
were `object`, so `dtype == "object"` was the usual way
to find them.  Under pandas 3 that comparison is always
`False` for text, and every branch guarded by it is
skipped.

## Solution

Use one helper that accepts both representations
(merge commit `a5c0a4d`):

``` python
def _is_text_dtype(dtype) -> bool:
    """True for plain-object columns and pandas' string dtype
    (default for text in pandas >= 3)."""
    return dtype == "object" or isinstance(dtype, pd.StringDtype)
```

and replace each `series.dtype == "object"` check in
`detect_and_coerce_types` and `optimize_memory` with
`_is_text_dtype(series.dtype)`.

**Because**

`isinstance(dtype, pd.StringDtype)` matches the new
string dtype whichever missing-value marker it uses,
and the `== "object"` check still covers mixed columns
and older pandas versions.  The tests in
`ml_engine/tests/test_data_loader.py` now pass on
pandas 3.

_Lesson:_ `requirements.txt` did not pin the pandas
major version, so a fresh install brought in a new
major release with a changed default.  Pinning major
versions (e.g. `pandas>=2,<4`) and testing on the
version we deploy avoids this kind of silent change.
