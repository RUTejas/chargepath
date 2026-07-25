# ChargePath — Deployment Guide
## Make it Public + Install on Every Device

---

## How Auto-Training Works

**First launch:** Training starts automatically in background (~10–30 min depending on hardware).
Results stream live to the dashboard as they complete.

**Every launch after:** Results load from disk instantly (~30 seconds).
No buttons. No waiting. It just works.

---

## Run Locally

```bash
bash run.sh          # Linux / macOS
run.bat              # Windows

# Open in browser:
# Map App:    http://localhost:5000/map
# Dashboard:  http://localhost:5000
```

---

## Make It Public (Free Hosting)

### Option 1: Railway ⭐ Recommended — Easiest
```bash
# 1. Install Railway CLI
npm install -g @railway/cli       # or brew install railway

# 2. Login
railway login

# 3. Deploy (from project folder)
cd EV_CHARGING_Q1
railway init
railway up

# Done! Railway gives you a public URL like:
# https://chargepath-ev-ai-production.up.railway.app
```
- Free tier: 500 hours/month ($5 credit)
- Persistent disk: YES (your trained model stays between restarts)
- URL: auto-generated, looks professional
- Sign up: railway.app

---

### Option 2: Render — Best Free Tier
```bash
# 1. Push code to GitHub first
git init && git add . && git commit -m "ChargePath"
git remote add origin https://github.com/YOURNAME/chargepath.git
git push -u origin main

# 2. Go to render.com → New → Web Service
# 3. Connect your GitHub repo
# 4. Settings auto-detected from render.yaml
# 5. Click Deploy

# Free tier gives you:
# https://chargepath-ev-ai.onrender.com
```
- Free tier: 750 hours/month (always-on with paid plan $7/mo)
- Note: Free tier sleeps after 15min inactivity (first request takes ~30s)
- Sign up: render.com

---

### Option 3: Fly.io — Best for Global Performance
```bash
# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. Login
fly auth login

# 3. Deploy
cd EV_CHARGING_Q1
fly launch    # uses fly.toml automatically
fly deploy

# URL: https://chargepath-ev-ai.fly.dev
```
- Free tier: 3 shared-cpu VMs, 256MB RAM each
- Upgrade to 2GB RAM for comfortable training (~$3/month)
- Sign up: fly.io

---

### Option 4: Temporary Public URL (No Account Needed)
```bash
# Using ngrok — instant tunnel to your local server
# 1. Install: https://ngrok.com/download
# 2. Run your server: bash run.sh
# 3. In another terminal:
ngrok http 5000

# You get a URL like: https://abc123.ngrok.io
# Share this URL — works on any device!
# Note: URL changes every restart on free plan
```

---

## Install as App on Every Platform

### Android
1. Open the public URL in **Chrome**
2. Tap the **⋮ menu** → "Add to Home screen"
3. Or wait for the **"Install App"** banner to appear automatically
4. The app installs like a native app with its own icon

### iOS / iPadOS
1. Open the URL in **Safari** (must be Safari, not Chrome)
2. Tap the **Share button** (box with arrow) → "Add to Home Screen"
3. Tap "Add" → appears on home screen like a native app

### macOS
1. Open in **Chrome** or **Edge**
2. Click the **install icon** (⊕) in the address bar
3. Or: Chrome menu → "Save and Share" → "Install ChargePath"

### Windows
1. Open in **Chrome** or **Edge**
2. Click install icon in address bar, or:
   Edge: Settings → Apps → Install this site as an app

### Linux
1. Open in **Chrome** or **Chromium**
2. Click install icon in address bar
3. Or: Menu → More Tools → Create Shortcut → "Open as window"

---

## Custom Domain (Optional, Free)
```bash
# With Railway:
railway domain        # add your custom domain

# With Render:
# Dashboard → Settings → Custom Domains → Add

# Free domains from Freenom: .tk, .ml, .ga, .cf, .gq
# Or buy a .com for ~$10/year from Namecheap
```

---

## Environment Variables for Production
```bash
# Set these in your hosting platform's dashboard:
PORT=5000
FLASK_ENV=production

# Railway: railway variables set PORT=5000
# Render: set in Environment tab
# Fly.io: fly secrets set PORT=5000
```

---

## API Keys Used (All Free, No Account Needed)
| Service | Purpose | Cost | Key needed? |
|---|---|---|---|
| Open-Meteo | Historical weather data | Free forever | ❌ None |
| OpenStreetMap (Leaflet) | Map tiles | Free forever | ❌ None |
| OSRM | Driving directions/routing | Free forever | ❌ None |
| CartoDB Dark tiles | Dark map theme | Free forever | ❌ None |

---

## Production Checklist
- [ ] `bash run.sh` works locally
- [ ] Visit `http://localhost:5000/map` — map loads
- [ ] Visit `http://localhost:5000` — dashboard loads, auto-training starts
- [ ] Push to GitHub
- [ ] Deploy to Railway/Render/Fly.io
- [ ] Visit public URL — same as local
- [ ] Install on your phone using browser menu
