import datetime
import re
import sqlite3
from sqlite3 import IntegrityError
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import requests
import yfinance as yf
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
import os
from efficient_frontier import EF


app = Flask(__name__)
app.secret_key = os.environ['SEC_KEY']
db_path = "static/warehouse.db"
api = os.environ['API_KEY']


@app.route('/')
def welcome():
    return render_template("welcome.html")


@app.route("/register", methods=['Get', 'POST'])
def register():
    db = sqlite3.connect(db_path)
    c = db.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS user_auth (
        email VARCHAR(255) PRIMARY KEY,
        username VARCHAR(255),
        password VARCHAR(255),
        is_admin INTEGER NOT NULL DEFAULT 0
        )
    """)

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        is_valid = re.match(email_pattern, email)

        if not email or not username or not password:
            err_msg = "Please fill in all fields"
        elif not is_valid:
            err_msg = "Invalid Email Address"
        elif not re.search(r"^(?=.*[A-Z])", password):
            err_msg = "Password must contain an Uppercase letter"
        else:
            try:
                session['email'] = email
                c.execute("INSERT INTO user_auth (email, username, password, is_admin) VALUES (?, ?, ?, ?)",
                          (email, username, password, 0))
                db.commit()
                db.close()
                init_user_db()
                flash("Registration successful. Redirecting to login page in 3 seconds.")
                return redirect(url_for('register'))
            except IntegrityError:
                err_msg = "This email has been registered already."
        return render_template('auth/register.html', err_msg=err_msg)
    return render_template('auth/register.html', err_msg=None)


def check_user(email, password):
    db = sqlite3.connect(db_path)
    c = db.cursor()
    # Check if the user exists
    c.execute("SELECT password, is_admin FROM user_auth WHERE email=?", (email,))
    result = c.fetchone()
    db.close()
    if result is None:
        return "User does not exist"
    # Check if the password is correct
    if result[0] != password:
        return "Incorrect password"
    else:
        session['is_admin'] = result[1]
    # If everything checks out, return None (no error)
    return None


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        password = request.form["password"]
        session['email'] = email
        err_msg = check_user(email, password)
        if err_msg:
            return render_template("auth/login.html", err_msg=err_msg)

        if session['is_admin']:
            print("admin login")
            # Redirect to the admin page if the user is an admin
            return redirect(url_for('admin'))

        return redirect(url_for('finance_home'))

    return render_template('auth/login.html', err_msg=None)


@app.route('/logout')
def logout():
    session.clear()
    print("User logged out.")
    return redirect(url_for('login'))


@app.route("/admin", methods=['GET'])
def admin():
    if not session.get('is_admin'):
        flash("Registration successful. Redirecting to login page in 3 seconds.")
        return "You are not authorized to access this page.", 403

    email = session.get('email')
    # find username
    db = sqlite3.connect(db_path)
    c = db.cursor()
    c.execute("SELECT username FROM user_auth WHERE email=?", (email,))
    result = c.fetchone()
    session['username'] = result[0]

    # Get all non-admin users
    c.execute("SELECT email, username FROM user_auth WHERE is_admin = 0")
    users = c.fetchall()
    user_holdings = []

    for email, username in users:
        c.execute(
            f"SELECT symbol, SUM(volume) as total_volume, ROUND(SUM(price*volume)/SUM(volume), 2), name FROM [{email}] GROUP BY symbol HAVING total_volume > 0")
        holdings = c.fetchall()

        user_holdings.append({
            'email': email,
            'username': username,
            'holdings': holdings
        })

    db = sqlite3.connect(db_path)
    combined_portfolio = get_all_users_portfolios(db)

    statistics = get_portfolio_statistics(combined_portfolio, db, mode=1)
    db.close()

    # Get all current day's market orders
    today = datetime.date.today()

    db = sqlite3.connect(db_path)
    c = db.cursor()
    c.execute("SELECT email, username FROM user_auth WHERE is_admin = 0")
    users = c.fetchall()
    orders = []

    for email, username in users:
        c.execute(
            f"SELECT symbol, name, SUM(CASE WHEN trade_type = 'Buy' THEN volume ELSE 0 END) AS shares_bought, -SUM(CASE WHEN trade_type = 'Sell' THEN volume ELSE 0 END) AS shares_sold FROM [{email}]  WHERE date LIKE '{today}%' AND symbol != 'Initial Balance' GROUP BY date")
        data = c.fetchall()
        orders += data
    # print(orders)
    db.close()

    return render_template('auth/admin.html', user_holdings=user_holdings, orders = orders, statistics=statistics, username=session.get('username'))


@app.route("/finance_home", methods=['GET', 'POST'])
def finance_home():
    if request.method == 'POST':
        symbol = request.form['search_content']
        if is_valid_symbol(symbol):
            try:
                price = get_price(symbol, api)
            except ValueError as e:
                # Handle the error as needed, e.g., log the error message or show an error message to the user
                print(f"Error: {e}")
                return jsonify({'success': False})
            return jsonify({'success': True, 'symbol': symbol})
        else:
            return jsonify({'success': False})

    init_user_db()
    update_stock_data(api, symbol='SPY')
    spy_data = get_stock_data(symbol='SPY')
    
    email = session.get('email')
    # find username
    db = sqlite3.connect(db_path)
    c = db.cursor()
    c.execute("SELECT username FROM user_auth WHERE email=?", (email,))
    result = c.fetchone()
    session['username'] = result[0]
    c.execute(f'''SELECT * FROM [{email}]''')
    rows = c.fetchall()
    portfolio = {}
    for row in rows:
        symbol = row[0]
        name = row[1]
        volume = row[5]
        price = row[4]
        balance = row[6]
        if symbol not in portfolio and symbol != 'Initial Balance':
            portfolio[symbol] = {'total_volume': 0, 'total_value': 0}
            portfolio[symbol]['name'] = name
            portfolio[symbol]['average_cost_price'] = price
        if symbol != 'Initial Balance':
            portfolio[symbol]['total_volume'] += volume
            portfolio[symbol]['total_value'] += volume * price
            portfolio[symbol]['name'] = name
            if portfolio[symbol]['total_volume'] != 0:
                portfolio[symbol]['average_cost_price'] = portfolio[symbol]['total_value'] / portfolio[symbol]['total_volume']
    total_revenue = 0
    holdings_value = 0
    for symbol in portfolio.keys():  # 当portfolio为空会跳过这段运行-> 运气好绕过一个检测
        update_stock_data(api, symbol)
        cur_price = get_realtime_stock_price(symbol)
        portfolio[symbol]['current_price'] = cur_price
        portfolio[symbol]['revenue'] = (cur_price - portfolio[symbol]['average_cost_price']) * portfolio[symbol]['total_volume']
        total_revenue += portfolio[symbol]['revenue']
        holdings_value += cur_price * portfolio[symbol]['total_volume']
    if not portfolio:
        session['portfolio'] = {}
        session['total_revenue'] = 0
        session['statistics'] = {}

    else:
        for p in portfolio:
            fetch_stock_data(api, p)
        # If the Users portfolio has changed, delete all session values and waiting to recalculate
        if 'portfolio' in session:
            if portfolio != session['portfolio']:
                print('Need to recalculate')
                del session['portfolio']
                del session['total_revenue']
                del session['statistics']
            else:
                print('No longer need to recalculate')

        if 'portfolio' not in session:
            # print('Start calculation\n')
            statistics = get_portfolio_statistics(portfolio, db)
            session['portfolio'] = portfolio
            session['total_revenue'] = total_revenue
            session['statistics'] = statistics

    # Calculate the portfolio_value
    portfolio_value = balance + holdings_value
    
    return render_template('post/finance_home.html', username=session.get('username'), datas=rows, portfolio=session['portfolio'],
                        total_revenue=session['total_revenue'], stock_data=spy_data, symbol="SPY", 
                        statistics=session['statistics'], portfolio_value=portfolio_value, balance=balance
                        )


@app.route('/api/get_stock_price', methods=['GET'])
def get_stock_price():
    symbol = request.args.get('symbol', '')
    if symbol:
        try:
            stock = yf.Ticker(symbol)
            todays_data = stock.history(period="1d")
            open_price = round(todays_data["Open"][0], 2)
            current_price = round(todays_data["Close"][0], 2)
            return jsonify({'success': True, 'open_price': open_price, 'current_price': current_price})
        except ValueError as e:
            print(f"Error: {e}")
            return jsonify({'success': False})
    else:
        return jsonify({'success': False})



def init_user_db():
    email = session.get('email')
    db = sqlite3.connect(db_path)
    c = db.cursor()
    c.execute(f'''CREATE TABLE IF NOT EXISTS [{email}]
                 (symbol TEXT NOT NULL,
                  name TEXT NOT NULL,
                  date DATETIME NOT NULL,
                  trade_type TEXT NOT NULL,
                  price FLOAT NOT NULL,
                  volume INTEGER NOT NULL,

                  balance FLOAT NOT NULL)''')
    c.execute(f'SELECT * FROM [{email}]')
    if c.fetchone() is None:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        initial_data = ('Initial Balance', '', now,
                        'Initial Balance', 0, 0, 10 ** 6)
        c.execute(
            f"INSERT INTO [{email}] VALUES (?, ?, ?, ?, ?, ?, ?)", initial_data)
        db.commit()
        print("Table " + email + " created and initial data inserted.")
    else:
        print("User " + email + "'s Finance home.")
    db.close()
    return None


@app.route("/company_info/<symbol>", methods=['GET', 'POST'])
def company_info(symbol):
    db = sqlite3.connect(db_path)
    c = db.cursor()
    email = session.get('email')
    if request.method == 'POST':
        trade_type = request.form['buy_sell_select']
        trade_volume = int(request.form['trade_vol'])
        symbol = request.form['new_symbol_name']
        balance = float(request.form['current_balance'])
        order_price = float(request.form['price'])
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        name = get_profile(symbol, api)['Name']
        # print(name)
        if trade_type == "Buy":
            trade_volume = int(trade_volume) * 1
        elif trade_type == "Sell":
            trade_volume = int(trade_volume) * (-1)
        balance_change = float(order_price) * trade_volume
        post_trade_balance = round((balance - balance_change), 2)
        # print(post_trade_balance)
        c.execute(f'''INSERT INTO [{email}] ( symbol, name, date, trade_type, price, volume, 
           balance) VALUES (?, ?, ?, ?, ?, ?, ?) ''',
                  (symbol, name, now, trade_type, order_price, trade_volume, post_trade_balance))
        db.commit()
        db.close()
        return redirect(url_for('company_info', symbol=symbol))
    stock_data = get_stock_data(symbol)
    spy_data = get_stock_data('SPY')
    profile = get_profile(symbol, api)
    symbol = profile['Symbol']
    price = get_price(symbol, api)

    news = stock_news(symbol)
    c.execute(f'''SELECT balance FROM [{email}] ORDER BY date DESC LIMIT 1''')
    balance = c.fetchone()[0]
    c.execute(f'''SELECT volume FROM "{email}" WHERE symbol=?''', (symbol,))
    results = c.fetchall()
    if len(results) == 0:
        cur_vol = 0
    else:
        cur_vol = sum([result[0] for result in results])

    db.close()
    return render_template('search/company_info.html', stock_data=stock_data, spy_data=spy_data, symbol=symbol, profile=profile,
                           price=price, balance=balance, cur_vol=cur_vol, news=news)


def get_profile(symbol, api):
    profile = {}
    url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=' + \
        symbol + '&apikey=' + api
    r = requests.get(url)
    data = r.json()
    for key in data.keys():
        if key not in profile.keys():
            if key == "Symbol" or key == "Name" or key == "Description" or key == "Exchange" or key == "Sector" or key == "Industry" or key == "MarketCapitalization" or key == "PERatio" or key == "DividendPerShare" or key == "DividendYield" or key == "52WeekHigh" or key == "52WeekLow":
                profile[key] = data[key]
    # print(profile)
    return profile


def get_price(symbol, api):
    price = {}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api}"
    r = requests.get(url)
    data = r.json()
    price['previous_close'] = float(data["Global Quote"]["08. previous close"])
    price['open'] = float(data["Global Quote"]["02. open"])
    price['high'] = float(data["Global Quote"]["03. high"])
    price['low'] = float(data["Global Quote"]["04. low"])
    price['volume'] = int(data["Global Quote"]["06. volume"])
    price['current_price'] = get_realtime_stock_price(symbol)
    return price


def get_stock_data(symbol):
    fetch_stock_data(api, symbol)
    db = sqlite3.connect(db_path)
    c = db.cursor()
    symbol = symbol.upper()
    if symbol == 'SPY':
        c.execute(
            f"SELECT date, adj_close FROM Historical_{symbol}_Data ORDER BY date")
    else:
        c.execute(
            f"SELECT date, adj_close FROM Stock_{symbol}_Data ORDER BY date")
    results = c.fetchall()
    stock_data = [{'date': row[0], 'close': row[1]} for row in results]
    db.close()
    return stock_data


def stock_news(symbol):
    symbol = symbol.upper()
    news = {}
    news_url = requests.get(
        f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&sort=LATEST&apikey={api}').json()
    news['news'] = news_url['feed'][:5]
    return news


def get_realtime_stock_price(symbol):
    stock = yf.Ticker(symbol)
    todays_data = stock.history(period="1d")
    return round(todays_data["Close"][0],2)


def get_risk_free_rate():
    rf_url = requests.get(
        f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={api}').json()
    rf_rate = float(rf_url['data'][0]['value'])
    return rf_rate/100


def is_valid_symbol(symbol):
    try:
        stock_data = get_stock_data(symbol=symbol)
        if stock_data:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking stock symbol validity: {e}")
        return False


def fetch_stock_data(api, symbol):
    symbol = symbol.upper()
    if symbol == 'SPY':
        table_name = "Historical_SPY_Data"
    else:
        table_name = f"Stock_{symbol}_Data"

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api}"
    r = requests.get(url)
    data = r.json()

    # Check for an error message in the data
    if "Error Message" in data:
        print("Error or invalid symbol:", data)
        return None
    
    db = sqlite3.connect(db_path)
    c = db.cursor()
    c.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} (
                date DATETIME PRIMARY KEY,
                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,
                adj_close NUMERIC,
                volume INTEGER
            )
        ''')
    c.execute(f'SELECT * FROM {table_name}')
    if c.fetchone() is None:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api}"
        r = requests.get(url)
        data = r.json()
        historical_data = data["Time Series (Daily)"]
        latest_day = datetime.datetime.strptime(
            data['Meta Data']['3. Last Refreshed'][0:10], '%Y-%m-%d')
        five_years_ago = latest_day - datetime.timedelta(days=5 * 365)
        for date, data in historical_data.items():
            dt = datetime.datetime.strptime(date[0:10], '%Y-%m-%d')
            if dt >= five_years_ago:
                open_price = data['1. open']
                high_price = data['2. high']
                low_price = data['3. low']
                close_price = data['4. close']
                adj_close = data['5. adjusted close']
                volume = data['6. volume']

                c.execute(f'''
                    INSERT INTO {table_name} (date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (date[0:10], open_price, high_price, low_price, close_price, adj_close, volume))
        db.commit()
    else:
        c.execute(f"SELECT date FROM {table_name} ORDER BY date DESC LIMIT 1")
        result = c.fetchone()
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api}"
        r = requests.get(url)
        data = r.json()
        # print(data['Meta Data']['3. Last Refreshed'])
        latest_day = (data['Meta Data']['3. Last Refreshed'])[0:10]
        # print(latest_day)
        if result[0] != latest_day:
            info = data['Time Series (Daily)'][latest_day]
            c.execute(f'''
                               INSERT INTO {table_name} (date, open, high, low, close, adj_close, volume)
                               VALUES (?, ?, ?, ?, ?, ?, ?)
                           ''', (
                latest_day, info['1. open'], info['2. high'], info['3. low'], info['4. close'],
                info['5. adjusted close'],
                info['6. volume']))
            db.commit()
    db.close()
    return None

def update_stock_data(api, symbol):
    symbol = symbol.upper()
    if symbol == 'SPY':
        table_name = "Historical_SPY_Data"
    else:
        table_name = f"Stock_{symbol}_Data"

    db = sqlite3.connect(db_path)
    c = db.cursor()

    # Get the latest date in the database
    c.execute(f"SELECT date FROM {table_name} ORDER BY date DESC LIMIT 1")
    latest_date_in_db = c.fetchone()
    if latest_date_in_db:
        latest_date_in_db = datetime.datetime.strptime(latest_date_in_db[0], '%Y-%m-%d')
    else:
        latest_date_in_db = datetime.datetime.min

    # Get the latest data from Alpha Vantage
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api}"
    r = requests.get(url)
    data = r.json()

    if "Error Message" in data:
        print("Error or invalid symbol:", data)
        return

    historical_data = data["Time Series (Daily)"]
    latest_day = datetime.datetime.strptime(data['Meta Data']['3. Last Refreshed'][0:10], '%Y-%m-%d')

    # Fill in missing dates
    while latest_date_in_db < latest_day:
        latest_date_in_db += datetime.timedelta(days=1)
        date_str = latest_date_in_db.strftime('%Y-%m-%d')
        if date_str in historical_data:
            open_price = historical_data[date_str]['1. open']
            high_price = historical_data[date_str]['2. high']
            low_price = historical_data[date_str]['3. low']
            close_price = historical_data[date_str]['4. close']
            adj_close = historical_data[date_str]['5. adjusted close']
            volume = historical_data[date_str]['6. volume']

            c.execute(f'''
                INSERT INTO {table_name} (date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (date_str, open_price, high_price, low_price, close_price, adj_close, volume))

    db.commit()
    db.close()


def read_portfolio_data(conn, symbols):
    # Initialize an empty DataFrame to store price data
    portfolio_data = pd.DataFrame()

    for symbol in symbols:
        query = f"SELECT date, adj_close FROM Stock_{symbol}_Data ORDER BY date"
        df = pd.read_sql_query(query, conn)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        # Rename the 'adj_close' column to the stock symbol
        df.rename(columns={"adj_close": symbol}, inplace=True)

        # Merge the stock data with the portfolio data
        if portfolio_data.empty:
            portfolio_data = df
        else:
            portfolio_data = portfolio_data.merge(
                df, left_index=True, right_index=True, how="outer")

    return portfolio_data

def get_all_users_portfolios(db):
    c = db.cursor()
    c.execute("SELECT email FROM user_auth WHERE is_admin=0")
    users_emails = [row[0] for row in c.fetchall()]

    combined_portfolio = {}

    for email in users_emails:
        c.execute(f"SELECT * FROM [{email}]")
        rows = c.fetchall()
        for row in rows:
            symbol = row[0]
            volume = row[5]
            if symbol not in combined_portfolio and symbol != 'Initial Balance':
                combined_portfolio[symbol] = {'total_volume': 0}
            if symbol != 'Initial Balance':
                combined_portfolio[symbol]['total_volume'] += volume
                if combined_portfolio[symbol]['total_volume'] == 0:
                    del combined_portfolio[symbol]
    return combined_portfolio

def get_portfolio_statistics(portfolio, db, mode = 0):
    statistics = {}
    portfolio_data = read_portfolio_data(db,portfolio.keys())
    if mode == 0:
        message = "Your current portfolio"
        err = "The number of stocks in current portfolio is less than 2. Add more stocks to analyze the risk return." 
    else:
        message = "Overall portfolios of all Users"
        err = "There are less than 2 stocks in the overall portfolio. Wait more stocks in the overall portfolio to see risk return analysis."
    if len(portfolio.keys()) >= 2:
        returns = calculate_expected_returns(portfolio_data)
        cov_matrix = calculate_covariance_matrix(portfolio_data)
        holdings = np.array([v['total_volume']
                            for v in portfolio.values()])
        # print('holdings:', holdings)
        total_holdings = np.sum(holdings)
        weights = holdings / total_holdings
        rf_rate = get_risk_free_rate()
        port_vol = calculate_portfolio_volatility(weights, cov_matrix)
        port_ret = calculate_portfolio_returns(weights, returns)
        statistics['sharpe_ratio'] = round(calculate_sharpe_ratio(
            port_ret, port_vol, rf_rate), 2)
            
        # call the function in the module to calculate the intended ror and vol
        ror, volatility = EF.restrict(portfolio.keys(), db)
  
        # Create the efficient frontier plot
        efficient_frontier = go.Scatter(
            x=volatility,
            y=ror,
            mode="lines",
            # name="Efficient Frontier",
        )

        single_point = go.Scatter(
            x=[port_vol],
            y=[port_ret],
            mode="markers+text",
            text=[
                f"Expected return: {port_ret:.2f}, Volatility: {port_vol:.2f}"],
            textposition="top right",
            marker=dict(size=10, color="red"),
            name=message,
        )

        # Combine the plots
        data = [efficient_frontier, single_point]

        # Define the layout
        layout = go.Layout(
            # title="Efficient Frontier",
            xaxis=dict(title="Volatility", tickformat=".2f"),
            yaxis=dict(title="Rate of Return", tickformat=".2f"),
         width = 800
        )

        # Create a Figure object
        fig = go.Figure(data=data, layout=layout)

        # Generate the plot div
        statistics['make_plot'] = pyo.plot(fig, output_type='div',
                            include_plotlyjs=False)
    else:
        statistics['error'] = err
    return statistics

def calculate_expected_returns(assets_data):
    returns = assets_data.pct_change().dropna()
    return returns.mean() * 252


def calculate_covariance_matrix(assets_data):
    returns = assets_data.pct_change().dropna()
    return returns.cov() * 252


def calculate_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)


def calculate_portfolio_returns(weights, expected_returns):
    return np.dot(weights, expected_returns)


def calculate_sharpe_ratio(portfolio_returns, portfolio_volatility, risk_free_rate):
    return (portfolio_returns - risk_free_rate) / portfolio_volatility


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)