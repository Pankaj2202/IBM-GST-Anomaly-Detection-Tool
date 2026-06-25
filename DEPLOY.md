# Making it a LIVE tool — deployment guide

The app runs as a web service. "Live" = host it once on a machine your team can
reach, so anyone with the URL can use it without installing anything.

> **Data governance:** this handles vendor masters, GSTINs and invoice data.
> Host it **inside the JK Cement / IBM network** behind normal access controls.
> Do **not** deploy to a public cloud (Streamlit Community Cloud, etc.).

## Option A — internal server / VM (simplest)
On a shared Windows or Linux box that teammates can reach:
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
Share `http://<that-machine-ip>:8501`. To keep it running after you log off:
- **Linux:** run under `systemd`, `tmux`, or `nohup … &`
- **Windows:** use NSSM to install it as a service, or Task Scheduler "at startup"

## Option B — Docker (cleanest for IT to manage)
```bash
docker build -t gst-anomaly-tool .
docker run -d --restart unless-stopped -p 8501:8501 \
  -v $(pwd)/reference:/app/reference \
  --name gst-tool gst-anomaly-tool
```
Then browse `http://<host>:8501`. Mounting `reference/` as a volume lets you
refresh the masters without rebuilding the image.

## Turn on the login (recommended for a shared URL)
Create `.streamlit/secrets.toml`:
```toml
app_password = "choose-a-strong-password"
```
With that present, the app shows a password screen first. Remove it for open
local use. (For per-user logins / SSO, ask IT about putting it behind the
corporate reverse proxy or an auth gateway.)

## Branding
Drop your official logos in `assets/jkcl_logo.png` and `assets/ibm_logo.png`
(transparent PNG, ~200px tall). They appear top-right automatically; until then
styled text marks show. Theme colours live in `.streamlit/config.toml`.

## Keeping the model valid
Retrain on your machine after any scikit-learn upgrade:
```bash
python train.py     # rewrites gst_rf_model.joblib, then restart the app
```
