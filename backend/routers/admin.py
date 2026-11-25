"""Temporary, password-protected page listing registered users.

Disabled unless the ADMIN_KEY environment variable is set. Open it at:
    /api/admin/users?key=<ADMIN_KEY>
"""
import hmac
import os
from html import escape

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import UserDB

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.get("/users", response_class=HTMLResponse, include_in_schema=False)
def list_users(key: str = Query(""), db: Session = Depends(get_db)):
    admin_key = os.getenv("ADMIN_KEY", "")
    # Not configured -> behave as if the page doesn't exist.
    if not admin_key:
        raise HTTPException(status_code=404, detail="Not Found")
    if not hmac.compare_digest(key.encode(), admin_key.encode()):
        raise HTTPException(status_code=403, detail="Forbidden")

    users = db.query(UserDB).order_by(UserDB.created_at.asc()).all()
    rows = "".join(
        "<tr>"
        f"<td>{u.id}</td>"
        f"<td>{escape(u.username or '')}</td>"
        f"<td>{escape(u.email or '')}</td>"
        f"<td>{escape(u.full_name or '')}</td>"
        f"<td>{u.created_at.strftime('%Y-%m-%d %H:%M') + ' UTC' if u.created_at else ''}</td>"
        "</tr>"
        for u in users
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="robots" content="noindex">
<title>Registered users</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; padding: .4rem .8rem; text-align: left; }}
  th {{ background: #f3f3f3; }}
</style></head>
<body>
<h1>Registered users: {len(users)}</h1>
<table>
<tr><th>#</th><th>Username</th><th>Email</th><th>Full name</th><th>Signed up</th></tr>
{rows or '<tr><td colspan="5">No users yet.</td></tr>'}
</table>
</body></html>"""
