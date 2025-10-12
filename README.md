# XGen - Tweet & Reply Generator

A Streamlit app that generates engaging tweets and replies using OpenAI's API.

## Features
- **Generate Tweets**: Create multiple tweets based on your niche, topic, and style samples
- **Reply Generator**: Generate diverse, engaging replies to any tweet

## 🚀 Quick Start (Local Development)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd XGen
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**
   
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```
   
   ⚠️ **Important**: Never commit your `.env` file to git! It's already in `.gitignore`.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deploy to Streamlit Cloud (Recommended)

### Step 1: Prepare Your Repository
1. Make sure your `.env` file is **NOT** committed to git (it should be in `.gitignore`)
2. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository, branch (main), and main file path (`app.py`)

### Step 3: Configure Secrets (SECURE METHOD)
This is where you add your API key **securely** without exposing it in your code:

1. In the Streamlit Cloud dashboard, click on your deployed app
2. Click the **"⋮"** menu (three dots) → **"Settings"**
3. Go to the **"Secrets"** section
4. Add your secrets in TOML format:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-api-key-here"
   ```
5. Click **"Save"**
6. Your app will automatically restart with the secure API key

### Security Benefits
✅ API keys are encrypted at rest  
✅ Only your app can access them  
✅ Not visible in logs or to viewers  
✅ Separate from your source code  
✅ Can be updated without code changes  

## 📁 Project Structure
```
XGen/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Local API key (DO NOT COMMIT)
├── .gitignore         # Excludes sensitive files
└── README.md          # This file
```

## 🔒 Security Best Practices

1. **Never** commit API keys to your repository
2. **Never** hardcode API keys in your code
3. **Always** use `.env` for local development
4. **Always** use Streamlit secrets for cloud deployment
5. Keep your `.gitignore` file updated

## 🛠️ Troubleshooting

**"API key missing" error:**
- **Local**: Check that your `.env` file exists and contains `OPENAI_API_KEY=...`
- **Cloud**: Verify secrets are configured correctly in Streamlit Cloud settings

**App won't start:**
- Check that all dependencies in `requirements.txt` are installed
- Verify your OpenAI API key is valid

## 📝 License
MIT