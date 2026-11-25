# Week 4 : Frontend Cannot Reach the Backend on Render

# Build-time API URL in a Static Site

## Error:

After deploying with the Render blueprint
(`render.yaml`), the frontend loaded but the navbar
status pill showed the backend as disconnected, and
every API call from the browser failed.  The backend’s
own `/health` URL worked when opened directly.

## Relevant Context

The blueprint deploys two services: the FastAPI backend
`model-studio-api` and the Vite frontend
`model-studio-web` (a static site).  The frontend reads
the API address from a build-time variable, which I
first wired to the backend service:

``` yaml
- key: VITE_API_BASE_URL
  fromService:
    type: web
    name: model-studio-api
    property: host
```

and `frontend/src/api/client.js` used it as is:

``` js
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
```

## Key Observation

There were two separate problems.

1. **No scheme.**  `property: host` gives a bare host
   name with no `https://`.  A URL without a scheme is
   treated by the browser as a *relative path*, so
   requests went to the frontend’s own domain
   (`…/model-studio-api/health`) instead of the backend.

2. **Private host name.**  Even with `https://` added,
   the value `fromService … host` returns is the
   service’s *private-network* name (`model-studio-api`).
   That name only resolves for other services inside
   Render’s network.  A static site has no server: the
   code runs in the user’s browser, on the public
   internet, which cannot resolve it.

## Solution

In `client.js`, accept a bare host by adding the scheme
and dropping any trailing slash (commit `1d4b320`):

``` js
const RAW_API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
// Accept a bare host (e.g. "my-api.onrender.com") as well as a full URL.
const API_BASE_URL = (/^https?:\/\//.test(RAW_API_BASE_URL)
  ? RAW_API_BASE_URL
  : `https://${RAW_API_BASE_URL}`).replace(/\/+$/, "");
```

and in `render.yaml`, set the backend’s **public** URL
instead of the private host:

``` yaml
# Must be the PUBLIC URL: fromService/property:host gives the private-network
# hostname ("model-studio-api"), which browsers cannot resolve.
- key: VITE_API_BASE_URL
  value: https://model-studio-api.onrender.com
```

**Because**

`VITE_*` variables are substituted into the JavaScript
bundle at *build time*; the browser later uses the
value exactly as written.  So the value must be a
complete URL that the *user’s browser* can reach, not
one that only works between servers.  Since Vite bakes
the value into the bundle, the frontend has to be
rebuilt and redeployed after changing it.

Also in this commit: the navbar now shows the
configured backend URL instead of a hard-coded
`http://127.0.0.1:8000`, which made the problem easy to
see, and `frontend/vercel.json` adds an SPA rewrite so
deep links such as `/studio` do not return 404 on
refresh.
