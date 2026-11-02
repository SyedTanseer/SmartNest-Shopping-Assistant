# Deploy SmartNest to Vercel: handoff for the next Claude session

You are picking up a half-finished task. **Goal: get SmartNest running on a public Vercel URL, using the user's own Vercel
account, and verify it works.** The code changes are already done and tested locally. What is left is account, browser and
secrets work that the previous session could not do (it had no browser and Vercel was not logged in).

Repo: `C:\Users\tanse\OneDrive\Desktop\Agent` (git, remote `https://github.com/SimranS2875/SmartNest.git`, branch `main`).
Shell is Windows; use Git Bash or PowerShell syntax accordingly. The user's name for you to use is "the user" (pronouns unknown: use they/them).

---

## 1. What SmartNest is (30 seconds)

- `frontend/`: React 19 + Vite + Tailwind. Calls the API with **relative** URLs (`/api/...`), so it must be served from the same origin as the API.
- `backend/`: Python 3.12 FastAPI app (`smartnest.api.app:get_app`), Gemini for language, MongoDB Atlas for sessions, saved real product snapshots in `backend/data/catalog` (336 KB, **git-ignored on purpose**).
- Live product search (paid SerpApi quota) is opt-in and **must stay OFF on the public site**.

## 2. What is already done (uncommitted in the working tree)

| File | Purpose |
|---|---|
| `api/index.py` | Vercel Python entry: adds `backend/src` to `sys.path`, exposes `app = get_app()` |
| `vercel.json` | installs/builds the frontend (`cd frontend && npm ci` / `npm run build`), output `frontend/dist`, rewrites `/api/(.*)` to `/api/index`, `maxDuration` 60, `includeFiles: backend/{src,data}/**` |
| `requirements.txt` (root) | Python deps for the function (fastapi, pydantic, pymongo, python-dotenv, google-genai, httpx, cryptography) |
| `.python-version` | `3.12` |
| `.vercelignore` | keeps `.env`, keys, `docs/`, tests, scratch folders, node_modules out of the upload; **deliberately does NOT list `backend/data`** |
| `backend/src/smartnest/api/app.py` | new `client_address()`: behind Vercel the rate limiter uses the first `X-Forwarded-For` address (only when env `VERCEL` is set) |
| `backend/tests/test_api.py` | two tests for that (mutation-checked: they fail when the fix is broken) |

Verified locally: `tests/test_api.py` 52 passed; `npm run build` OK; the function entry loads, `/api/catalog` returns the real
snapshots, and `POST /api/sessions` returns 201.

Not verified (could not be, without a Vercel login): the actual Vercel build, that the CLI uploads `backend/data/`, and that the
`/api/*` rewrite preserves the original path inside the function.

## 3. Hard rules (from the user: do not break these)

1. **Never commit or print secrets**: `backend/.env`, API keys, the MongoDB URI/password. Never paste them into chat, logs, files or the PR.
2. Put secrets into Vercel only via `vercel env add` (stdin) or the Vercel dashboard. When reading `backend/.env`, do it in code and never echo values.
3. **Do NOT set** `SERPAPI_API_KEY` or `SMARTNEST_ALLOW_LIVE_SEARCH` on Vercel. The public site must not be able to spend SerpApi quota.
4. Do not commit `New folder (17)/` (scratch folder containing an exposed key) or `backend/data/`.
5. Kill processes only by port, never all `python.exe`/`node.exe`.
6. Ask the user before: pushing to GitHub, adding custom domains, changing billing/plan, or anything that exposes the site beyond the default `*.vercel.app` URL.
7. Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (or your own model name if different, per the attribution reminder in your system prompt).
8. Report honestly: if something fails, say what failed with the real output. Do not claim a deployment works until step 7 passes.

## 4. Prerequisites the user must have approved or done

- A Vercel account (Hobby is fine). Logging in needs the user's browser: **either the user clicks the device-login link, or you do it in Chrome with the user's existing sign-in.**
- A MongoDB Atlas cluster (already used by the project). **Atlas Network Access must allow Vercel**: Vercel has no fixed IPs, so add `0.0.0.0/0` (Atlas dashboard > Security > Network Access > Add IP Address > Allow access from anywhere). Tell the user this widens exposure; recommend a database user limited to the SmartNest database. Ask the user to confirm before you change it.
- Sessions MUST persist in Atlas: serverless functions keep no memory between requests, so without `SMARTNEST_USE_ATLAS=1` a session disappears on the next click.

## 5. Steps

### Step 0: sanity check
```bash
cd C:\Users\tanse\OneDrive\Desktop\Agent
git status --short            # expect the files in section 2, plus untracked docs/assignments/
cd backend && python -m pytest -q -p no:cacheprovider tests/test_api.py   # 52 passed
```

### Step 1: log in to Vercel
```bash
npx vercel whoami
npx vercel login              # prints https://vercel.com/oauth/device?user_code=XXXX-XXXX ; open it in Chrome, check the code matches, approve
npx vercel whoami             # must print the user's username
```
(The previous session started a login that has probably expired; just run it again.) If you have Claude in Chrome tools, load them
with one ToolSearch call, call `tabs_context_mcp` first, work in a new tab, and stop and ask if the site needs a permission the user has not granted.

### Step 2: first deploy (creates and links the project)
```bash
cd C:\Users\tanse\OneDrive\Desktop\Agent
npx vercel --prod --yes       # scope: the user's personal account; project name: smartnest; framework: Other; root directory: ./
```
Use the **CLI upload route, not GitHub auto-deploy**: `backend/data/` is git-ignored and only the CLI upload carries it.
This deploy will start but `/api/*` will not work yet (no env vars). That is expected.

### Step 3: environment variables (Production, also Preview if the user wants)
Values come from the user's local `backend/.env`. Read them in Python and feed each to `vercel env add` on stdin so nothing is printed:
```python
# run from the repo root; prints only variable NAMES
import subprocess
from dotenv import dotenv_values
vals = dotenv_values(r"backend\.env")
names = ["MONGODB_URI", "MONGODB_DATABASE", "GEMINI_API_KEY", "GEMINI_MODEL"]
for n in names:
    if not vals.get(n):
        print("MISSING", n); continue
    subprocess.run(["npx", "vercel", "env", "add", n, "production"], input=vals[n].encode(), check=True, shell=True)
    print("set", n)
subprocess.run(["npx", "vercel", "env", "add", "SMARTNEST_USE_ATLAS", "production"], input=b"1", check=True, shell=True)
print("set SMARTNEST_USE_ATLAS")
```
Required set: `SMARTNEST_USE_ATLAS=1`, `MONGODB_URI`, `MONGODB_DATABASE`, `GEMINI_API_KEY`, `GEMINI_MODEL`.
Never set: `SERPAPI_API_KEY`, `SMARTNEST_ALLOW_LIVE_SEARCH`. The user said they rotated the Atlas and Gemini credentials after
they leaked earlier; if `backend/.env` still holds the old values the app will fail to connect: ask the user for the new ones.
Check names only with `npx vercel env ls`.

### Step 4: Atlas network access
Do section 4's Atlas step (with the user's confirmation).

### Step 5: redeploy so the variables take effect
```bash
npx vercel --prod --yes
```
Note the production URL it prints (`https://smartnest-xxxx.vercel.app`).

### Step 6: verify (all of these must pass; show the user the real output)
```bash
URL=https://<the-url>
curl -s $URL/api/health
curl -s $URL/api/catalog | head -c 400
curl -s -X POST $URL/api/sessions -H "Content-Type: application/json" \
  -d '{"message":"I just moved into a 2BHK. Budget 2 lakh, minimalist. Furnish the entire apartment."}' | head -c 600
```
Expected:
- `/api/health`: `"status":"ok"`, `"persistence"` is NOT `"memory"` (it should say atlas/mongo), `"gemini":true`, and `"live_search":{"enabled":false,...}`.
- `/api/catalog`: categories with `"origin":"serpapi"` and listings. If categories are empty or origin is simulated, `backend/data/catalog` was not uploaded (see Troubleshooting).
- `POST /api/sessions`: HTTP 201 with `has_plan: true`.
- Then in Chrome open the URL: the page loads, send the 2BHK message, a room-by-room plan with real product cards appears, the header shows no "live search" button, and **reloading the page keeps the session** (this proves Atlas persistence). Also click around Budget / Cart / Agents tabs.

### Step 7: finish
- Tell the user the live URL and what you verified, and what you could not.
- Offer to commit the Vercel files (`api/`, `vercel.json`, `requirements.txt`, `.python-version`, `.vercelignore`, `backend/src/smartnest/api/app.py`, `backend/tests/test_api.py`, this file) **without** `docs/assignments/` unless the user wants it. Do not push without being asked.
- Remind the user: the site is public and uses their Gemini key; the per-visitor rate limit is 120 POST/min. They can enable Vercel Deployment Protection to make it private.

## 6. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `/api/*` returns 404 or the SPA's index.html | rewrite not applied: check `vercel.json` is at the repo root and valid JSON; `npx vercel inspect <url>` and the function logs (`npx vercel logs <url>`) |
| `/api/health` works but routes inside 404 | the function sees `/api/index` instead of the original path. Fix: in `api/index.py` wrap the app so the path is restored, or change the rewrite to `{ "source": "/api/:path*", "destination": "/api/index" }`; as a last resort mount the app under a prefix. Re-run the local test (`api/index.py` through `TestClient`) after any change |
| `/api/catalog` empty / "simulated_allowed": false with no data | `backend/data/` not uploaded. `.vercelignore` must not list it, and it must not be excluded by `.gitignore` for CLI uploads (CLI uses `.vercelignore` when present). Check with `npx vercel inspect --files` or the deployment's source view. Fallback: set `SMARTNEST_CATALOG_DIR` to a path inside the bundle and make sure `includeFiles` matches it |
| `persistence: memory` or sessions vanish on reload | `SMARTNEST_USE_ATLAS` not `1`, or Mongo connection failed (see function logs; the app logs `MongoDB unavailable` without secrets). Check Atlas network access, URI, database name, that the DB user can read/write |
| `Function exceeded ... size` | trim `includeFiles`; the bundle should be far below 250 MB. `google-genai` is the biggest dependency |
| `Task timed out` on plan requests | raise `maxDuration` in `vercel.json` (Hobby allows up to 300 with Fluid compute; 60 is set) |
| Build fails in `tsc --noEmit` | run `cd frontend && npm run build` locally first; it passed locally on 2026-11-01 |
| Everyone gets 429 | rate limiter keyed on the proxy: confirm env `VERCEL` is set on the function (Vercel sets it) and `client_address()` is deployed |
| `ModuleNotFoundError: smartnest` | `api/index.py` must insert `backend/src` into `sys.path` and `includeFiles` must bundle `backend/src/**` |

## 7. Facts that may save you time

- Local dev: API `python src/serve.py` (port 8000, run from `backend/`), UI `npm run dev` (port 5173, proxies `/api`).
- Backend tests: `cd backend && python -m pytest -q` (2551 passed, 29 skipped at last count); frontend: `cd frontend && npm test` (64).
- Real-browser e2e scripts live in `frontend/e2e/*.mjs` (they need the local stack running); they can be pointed at a base URL argument, e.g. `node e2e/demo.mjs https://<url>`, but check each script before using it against production, and never trigger live search.
- Settings come from real environment variables first, then `backend/.env` (absent on Vercel, which is what we want).
- `SMARTNEST_CORS_ORIGINS` is not needed: the frontend and API share one origin.
- Do not "fix" things by weakening the app's honesty rules (no invented product data, hard budget ceiling, simulated data always labelled). They are the point of the project.
