# Week 1 : Virtual Environment Committed to the Repository

# Committing `venv`

## Error:

The first setup commit (`c3e21e7`, 3 Sep 2026) was
far larger than the code it contained.

> 3724 files changed, 710854 insertions(+)

Almost every file was under `backend/venv/Lib/site-packages/`
or `venv/`, including compiled `.pyc` files and
`venv/Scripts/pythonw.exe`.

## Relevant Context

I created the Python virtual environment inside the
project folder, as usual,

``` shell
python -m venv backend/venv
backend/venv/Scripts/pip install -r backend/requirements.txt
```

and then staged everything for the first commit,

``` shell
git add .
git commit -m "frontend and backend setup and test"
```

There was no `.gitignore` in the repository at this
point.

## Key Observation

`git add .` stages *every* untracked file that is not
ignored.  With no `.gitignore`, that includes the whole
virtual environment (every installed package), the
`__pycache__` folders, and the local SQLite database
`backend/data.db`.

None of these belong in version control: they are
machine-specific, can be rebuilt from
`requirements.txt`, and a Windows `venv` will not even
work on a teammate’s Mac or Linux machine.

## Solution

Add a `.gitignore` at the repository root, then remove
the already-tracked files from the index (but not from
disk):

``` gitignore
# Virtual Environment & Python cache
venv/
backend/venv/
**/__pycache__/
*.py[cod]

# Database & Model Artifact Storage
backend/data.db
backend/storage/
*.joblib
*.pkl

# Node / Frontend
frontend/node_modules/
frontend/dist/

# Environment variables
.env
.env.*
!.env.example
```

``` shell
git rm -r --cached backend/venv venv
git add .gitignore
git commit
```

This was done in commit `276c3c9` (16 Sep 2026), which
shows `2280 files changed, … 431886 deletions(-)`.

**Because**

`.gitignore` only stops *untracked* files from being
added.  A file that is already tracked stays tracked
even after a matching pattern is added, so it must be
removed from the index explicitly with
`git rm --cached`.  The `--cached` flag keeps the local
copy, so the working `venv` is untouched.

_Note:_ the files are still in the history of
`c3e21e7`, so the clone size does not shrink.  Removing
them completely needs a history rewrite
(`git filter-repo`), which we did not do because the
branch was already shared.
