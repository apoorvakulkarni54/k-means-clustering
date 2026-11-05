# ⚡ Energy Consumption Segmentation (K-Means)

Groups hourly energy readings by **energy consumption** and **temperature** using K-Means, with a Streamlit web app.

## Project structure
```
energy-kmeans/
├── app.py                          # Streamlit web app
├── train_model.py                  # Trains & saves the model
├── energy_kmeans_clustering.ipynb  # Step-by-step notebook (steps 1-9)
├── requirements.txt
├── .streamlit/config.toml          # Theme
├── data/energy_data.csv            # Dataset (+ energy_clustered.csv)
└── model/kmeans_model.joblib       # Saved model + scaler
```

## Run locally (steps 10-11)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py            # optional: model is already included
streamlit run app.py
```
Open http://localhost:8501. Test: move the sliders, upload a CSV with `energy_consumption,temperature`, check every tab.

## Upload to GitHub (step 12)
```bash
git init
git add .
git commit -m "Energy K-Means project"
git branch -M main
git remote add origin https://github.com/<your-username>/energy-kmeans.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud (step 13)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
3. Click **Deploy**. First build takes 2-3 minutes.

## Access & monitor (step 14)
- Share the `*.streamlit.app` URL.
- **Manage app** shows logs and resource use, and lets you reboot.
- Pushing to GitHub auto-redeploys. Retrain with `python train_model.py` when new data arrives.

## Results
K = 4 (best silhouette ≈ 0.41). Clusters: Low/High usage × Cool/Hot weather.
