from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import requests
from bs4 import BeautifulSoup
from yahoo_fin.stock_info import get_live_price
from werkzeug.security import generate_password_hash, check_password_hash
import openai
import os

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace this with a secure random key for production

# OpenAI API setup
openai.api_key = 'your_openai_api_key'  # Replace with your OpenAI key

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple user class
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# In-memory user store (use a real database for production)
users = {
    'user1': {'password': generate_password_hash('password1')}
}

# Load user callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id in users else None

# News sources
news_websites = [
    "https://www.reuters.com/business/finance",
    "https://www.bloomberg.com/markets",
    "https://www.cnbc.com/investing",
]

headers = {"User-Agent": "Mozilla/5.0"}

# Function to extract news headlines
def extract_news(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = []

    for tag in soup.find_all(['h2', 'h3', 'a']):
        text = tag.get_text(strip=True)
        if len(text.split()) > 3:  # Only keep meaningful headlines
            link = tag.get('href', '#')
            if link.startswith('/'):
                link = f"{url.rstrip('/')}{link}"
            headlines.append({'text': text, 'url': link})

    return headlines

# Function to fetch real-time stock prices
def get_stock_prices():
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    prices = {}
    for ticker in tickers:
        try:
            price = round(get_live_price(ticker), 2)
            prices[ticker] = price
        except Exception as e:
            print(f"Failed to get price for {ticker}: {e}")
            prices[ticker] = "N/A"
    return prices

# Function to ask OpenAI
def ask_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Sorry, I couldn't process your request."

# Home route
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('login'))

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and check_password_hash(users[username]['password'], password):
            user = User(username)
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users[username] = {'password': generate_password_hash(password)}
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# Dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    headlines = []
    for url in news_websites:
        headlines.extend(extract_news(url))

    stock_prices = get_stock_prices()

    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    valid_prices = [price for price in stock_prices.values() if price != "N/A"]
    average_price = sum(valid_prices) / len(valid_prices) if valid_prices else 0
    recommended_sector = "Technology" if average_price > 200 else "Finance"

    return render_template(
        'dashboard.html',
        headlines=headlines,
        stock_prices=stock_prices,
        recommended_sector=recommended_sector
    )

# Ask AI Assistant route
@app.route('/ask', methods=['GET', 'POST'])
@login_required
def ask():
    response = ""
    if request.method == 'POST':
        user_query = request.form.get('query')
        if user_query:
            response = ask_openai(user_query)
    return render_template('ask.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
