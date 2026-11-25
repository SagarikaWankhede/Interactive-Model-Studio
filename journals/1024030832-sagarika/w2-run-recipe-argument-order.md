# Week 2 : Swapped Arguments to `run_recipe` After Merge

# Positional Arguments Across a Merge

## Error:

After merging `origin/main` into my Studio branch, the
recipe preview endpoint failed inside the ML engine.

> File "ml_engine/recipe_runner.py", line 46, in run_recipe<br>
> &nbsp;&nbsp;&nbsp;&nbsp;if target_column not in df.columns:<br>
> TypeError: unhashable type: 'list'

## Relevant Context

In `backend/services/ml_service.py`, the preview
function called the engine with positional arguments:

``` python
X_tr, X_te, y_tr, y_te, _ = run_recipe(df, recipe, target_column)
```

This matched the signature on my branch (`076919f`):

``` python
def run_recipe(df, recipe, target_col=None, target_column=None): ...
```

But the remote branch (`9668e37`, Hiral’s ML-engine
work) had changed the signature to:

``` python
def run_recipe(df, target_column, recipe): ...
```

The merge of `ml_service.py` had no textual conflict,
so git combined both sides without complaint.

## Key Observation

With the new signature the positional call binds

- `target_column` ← the recipe (a `list` of steps)
- `recipe` ← the column name (a `str`)

so `target_column not in df.columns` tries to hash a
list and raises `TypeError`.

Git only detects conflicts between *lines*.  A call
site in one file and a signature change in another file
are not a conflict to git, even though together they
break the program.

## Solution

Call the engine with keyword arguments (merge commit
`a5c0a4d`):

``` python
X_tr, X_te, y_tr, y_te, _ = run_recipe(df, target_column=target_column, recipe=recipe)
```

and make the engine accept both contracts, so older
callers keep working:

``` python
def run_recipe(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    recipe: Optional[List[Dict[str, Any]]] = None,
    target_col: Optional[str] = None,
) -> Tuple[...]:
```

**Because**

Keyword arguments bind by *name*, not by position, so
they keep working when parameters are reordered.  For
functions shared between team members (the
frontend/backend ↔ ML-engine boundary), we now pass
arguments by keyword, and run the full test suite
(`pytest backend/tests ml_engine/tests`) after every
merge rather than trusting a conflict-free merge.
