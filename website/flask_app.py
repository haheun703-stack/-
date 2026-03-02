"""
═══════════════════════════════════════════════════════════
  JARVIS Control Tower Dashboard — ppwangga.com
  자비스 컨트롤타워 대시보드

  v2.0 — 배너형 대시보드 + 시장 데이터 API
═══════════════════════════════════════════════════════════
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory
)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# ─── 설정 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')
DAILY_DIR = os.path.join(REPORTS_DIR, 'daily')
WEEKLY_DIR = os.path.join(REPORTS_DIR, 'weekly')
MONTHLY_DIR = os.path.join(REPORTS_DIR, 'monthly')
METRICS_FILE = os.path.join(DATA_DIR, 'metrics.json')
MARKET_FILE = os.path.join(DATA_DIR, 'market_data.json')
HOLDINGS_FILE = os.path.join(DATA_DIR, 'holdings.json')
BRAIN_FILE = os.path.join(DATA_DIR, 'brain_data.json')

# 디렉토리 자동 생성
for d in [DATA_DIR, REPORTS_DIR, DAILY_DIR, WEEKLY_DIR, MONTHLY_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── 사용자 설정 (원하는 대로 변경) ─────────────────────
USERNAME = "ppwangga"
PASSWORD_HASH = hashlib.sha256("gkgjs0907".encode()).hexdigest()
API_KEY = "e_Yws1RwLkUwg1vlXFqpbbRe-GMp7MRugnsrPfuF99M"

# ─── 로그인 관리 ────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if key != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

# ─── 헬퍼 함수 ─────────────────────────────────────────
def get_report_dates():
    """저장된 리포트 날짜 목록 반환"""
    dates = []
    if os.path.exists(DAILY_DIR):
        for f in os.listdir(DAILY_DIR):
            if f.endswith('.html'):
                date_str = f.replace('.html', '')
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    dates.append(date_str)
                except ValueError:
                    pass
    return sorted(dates, reverse=True)

def get_metrics():
    """메트릭스 데이터 로드"""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'total_pnl': 0,
        'total_pnl_pct': 0,
        'win_rate': 0,
        'pf': 0,
        'mdd': 0,
        'sharpe': 0,
        'total_trades': 0,
        'portfolio_value': 0,
        'daily_pnl': [],
        'strategy_a': {'name': 'Quantum v10.3 + v3 Brain', 'alloc': 60, 'pf': 0, 'mdd': 0},
        'strategy_b': {'name': 'ETF 3축 로테이션', 'alloc': 40, 'pf': 0, 'mdd': 0},
        'updated_at': ''
    }

def save_metrics(data):
    """메트릭스 데이터 저장"""
    data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_market_data():
    """시장 데이터 로드 (추천종목, US시그널, 레짐)"""
    if os.path.exists(MARKET_FILE):
        with open(MARKET_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'picks': [],
        'us_signal': {
            'grade': '-',
            'score': 0,
            'spy_return': 0,
            'qqq_return': 0,
            'vix_return': 0,
            'special_rules': [],
            'kill_sectors': []
        },
        'regime': 'UNKNOWN',
        'regime_slots': 0,
        'bat_status': {},
        'updated_at': ''
    }

def save_market_data(data):
    """시장 데이터 저장"""
    data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(MARKET_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_holdings():
    """보유주식 데이터 로드"""
    if os.path.exists(HOLDINGS_FILE):
        with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'holdings': [], 'total_eval': 0, 'total_pnl': 0, 'available_cash': 0, 'fetched_at': ''}

def save_holdings(data):
    """보유주식 데이터 저장"""
    with open(HOLDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_brain_data():
    """AI Brain 데이터 로드 (전략/섹터/릴레이/뉴스/v3)"""
    if os.path.exists(BRAIN_FILE):
        with open(BRAIN_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_brain_data(data):
    """AI Brain 데이터 저장"""
    data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(BRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_calendar_data(year, month):
    """달력 데이터 생성"""
    import calendar
    cal = calendar.Calendar(firstweekday=6)  # 일요일 시작
    dates = get_report_dates()

    weeks = []
    for week in cal.monthdatescalendar(year, month):
        week_data = []
        for day in week:
            date_str = day.strftime('%Y-%m-%d')
            week_data.append({
                'date': day,
                'date_str': date_str,
                'in_month': day.month == month,
                'has_report': date_str in dates,
                'is_today': day == datetime.now().date()
            })
        weeks.append(week_data)
    return weeks

def generate_weekly_report(year, week_num):
    """주간 보고서 생성"""
    dates = get_report_dates()
    start = datetime.strptime(f'{year}-W{week_num:02d}-1', '%Y-W%W-%w').date()
    end = start + timedelta(days=6)

    week_dates = [d for d in dates if start.strftime('%Y-%m-%d') <= d <= end.strftime('%Y-%m-%d')]

    return {
        'year': year,
        'week': week_num,
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'report_count': len(week_dates),
        'dates': week_dates
    }

def generate_monthly_report(year, month):
    """월간 보고서 생성"""
    dates = get_report_dates()
    month_prefix = f'{year}-{month:02d}'
    month_dates = [d for d in dates if d.startswith(month_prefix)]

    return {
        'year': year,
        'month': month,
        'report_count': len(month_dates),
        'dates': month_dates
    }

# ─── 라우트: 로그인 ─────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        pw_hash = hashlib.sha256(password.encode()).hexdigest()

        if username == USERNAME and pw_hash == PASSWORD_HASH:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('아이디 또는 비밀번호가 잘못되었습니다.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─── 라우트: 메인 대시보드 ──────────────────────────────
@app.route('/')
@login_required
def dashboard():
    now = datetime.now()
    year = request.args.get('year', now.year, type=int)
    month = request.args.get('month', now.month, type=int)

    calendar_data = get_calendar_data(year, month)
    metrics = get_metrics()
    market = get_market_data()
    holdings = get_holdings()
    brain = get_brain_data()
    recent_dates = get_report_dates()[:7]

    # 이전/다음 달 계산
    prev_month = month - 1 if month > 1 else 12
    prev_year = year if month > 1 else year - 1
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1

    return render_template('dashboard.html',
        calendar=calendar_data,
        metrics=metrics,
        market=market,
        holdings=holdings,
        brain=brain,
        recent_dates=recent_dates,
        year=year, month=month,
        prev_year=prev_year, prev_month=prev_month,
        next_year=next_year, next_month=next_month,
        month_name=f"{year}년 {month}월"
    )

# ─── 라우트: 일별 리포트 보기 ───────────────────────────
@app.route('/report/<date>')
@login_required
def view_report(date):
    filepath = os.path.join(DAILY_DIR, f'{date}.html')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        dates = get_report_dates()
        idx = dates.index(date) if date in dates else -1
        prev_date = dates[idx + 1] if idx + 1 < len(dates) else None
        next_date = dates[idx - 1] if idx > 0 else None

        return render_template('report.html',
            date=date, content=content,
            prev_date=prev_date, next_date=next_date
        )
    else:
        flash(f'{date} 리포트가 없습니다.', 'error')
        return redirect(url_for('dashboard'))

# ─── 라우트: 주간 보고서 ────────────────────────────────
@app.route('/weekly')
@app.route('/weekly/<int:year>/<int:week>')
@login_required
def weekly_report(year=None, week=None):
    now = datetime.now()
    if year is None:
        year = now.year
        week = now.isocalendar()[1]

    report = generate_weekly_report(year, week)
    metrics = get_metrics()

    return render_template('weekly.html', report=report, metrics=metrics)

# ─── 라우트: 월간 보고서 ────────────────────────────────
@app.route('/monthly')
@app.route('/monthly/<int:year>/<int:month>')
@login_required
def monthly_report(year=None, month=None):
    now = datetime.now()
    if year is None:
        year = now.year
        month = now.month

    report = generate_monthly_report(year, month)
    metrics = get_metrics()

    return render_template('monthly.html', report=report, metrics=metrics)

# ─── 라우트: 수동 업로드 ────────────────────────────────
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_report():
    if request.method == 'POST':
        date = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        file = request.files.get('report_file')

        if file and file.filename.endswith('.html'):
            filepath = os.path.join(DAILY_DIR, f'{date}.html')
            file.save(filepath)
            flash(f'{date} 리포트가 업로드되었습니다!', 'success')
            return redirect(url_for('view_report', date=date))
        else:
            flash('HTML 파일만 업로드 가능합니다.', 'error')

    return render_template('upload.html')

# ═══════════════════════════════════════════════════════════
#  봇 연동 API (자동 업로드용)
# ═══════════════════════════════════════════════════════════

@app.route('/api/upload', methods=['POST'])
@api_key_required
def api_upload_report():
    """봇에서 리포트를 자동 업로드하는 API"""
    date = None
    if request.is_json:
        date = request.json.get('date')
    else:
        date = request.form.get('date')
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')

    # 파일 업로드 방식
    if 'file' in request.files:
        file = request.files['file']
        filepath = os.path.join(DAILY_DIR, f'{date}.html')
        file.save(filepath)
        return jsonify({'status': 'ok', 'date': date, 'method': 'file'})

    # JSON 방식 (HTML 텍스트 직접 전송)
    if request.is_json and 'html' in request.json:
        filepath = os.path.join(DAILY_DIR, f'{date}.html')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(request.json['html'])
        return jsonify({'status': 'ok', 'date': date, 'method': 'json'})

    return jsonify({'error': 'No file or html provided'}), 400

@app.route('/api/metrics', methods=['POST'])
@api_key_required
def api_update_metrics():
    """봇에서 메트릭스를 업데이트하는 API"""
    if not request.is_json:
        return jsonify({'error': 'JSON required'}), 400

    metrics = get_metrics()
    metrics.update(request.json)

    # daily_pnl 히스토리 추가
    if 'today_pnl' in request.json:
        if 'daily_pnl' not in metrics:
            metrics['daily_pnl'] = []
        metrics['daily_pnl'].append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'pnl': request.json['today_pnl']
        })
        # 최근 90일만 유지
        metrics['daily_pnl'] = metrics['daily_pnl'][-90:]

    save_metrics(metrics)
    return jsonify({'status': 'ok', 'updated_at': metrics['updated_at']})

@app.route('/api/market', methods=['POST'])
@api_key_required
def api_update_market():
    """봇에서 시장 데이터를 업데이트하는 API

    수신 데이터:
      - picks: [{name, code, score, grade, signals}] 추천 종목
      - us_signal: {grade, score, spy_return, qqq_return, ...} US Overnight
      - regime: "BULL" | "CAUTION" | "BEAR" | "CRISIS"
      - regime_slots: 0~5
      - bat_status: {bat_a: "06:10 OK", bat_d: "16:30 OK", ...}
    """
    if not request.is_json:
        return jsonify({'error': 'JSON required'}), 400

    market = get_market_data()
    market.update(request.json)
    save_market_data(market)
    return jsonify({'status': 'ok', 'updated_at': market['updated_at']})

@app.route('/api/brain', methods=['POST'])
@api_key_required
def api_update_brain():
    """봇에서 AI Brain 데이터를 업데이트하는 API

    수신 데이터:
      - strategic: {regime, confidence, thesis[], sector_priority, risk_factors}
      - sector_focus: {focus_sectors[], boost[], suppress[], warnings[]}
      - group_relay: {fired_groups[], summary}
      - news: {sentiment, themes[], sector_outlook{}}
      - v3_picks: {buys[], regime, slots}
    """
    if not request.is_json:
        return jsonify({'error': 'JSON required'}), 400

    save_brain_data(request.json)
    return jsonify({'status': 'ok'})

@app.route('/api/holdings', methods=['POST'])
@api_key_required
def api_update_holdings():
    """봇에서 보유주식 데이터를 업데이트하는 API

    수신 데이터:
      - holdings: [{ticker, name, quantity, avg_price, current_price, eval_amount, pnl_amount, pnl_pct}]
      - total_eval: 총 평가금액
      - total_pnl: 총 손익
      - available_cash: 예수금
      - fetched_at: 조회 시각
    """
    if not request.is_json:
        return jsonify({'error': 'JSON required'}), 400

    save_holdings(request.json)
    return jsonify({'status': 'ok', 'count': len(request.json.get('holdings', []))})

@app.route('/api/status')
@api_key_required
def api_status():
    """서버 상태 확인 API"""
    metrics = get_metrics()
    market = get_market_data()
    dates = get_report_dates()
    return jsonify({
        'status': 'online',
        'total_reports': len(dates),
        'latest_report': dates[0] if dates else None,
        'metrics_updated': metrics.get('updated_at', 'never'),
        'market_updated': market.get('updated_at', 'never'),
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True)
