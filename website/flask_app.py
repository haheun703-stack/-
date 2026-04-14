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

# ─── 사용자 설정 (.env에서 로드) ──────────────────────
USERNAME = os.getenv("JARVIS_USER", "ppwangga")
PASSWORD_HASH = hashlib.sha256(os.getenv("JARVIS_PASSWORD", "").encode()).hexdigest()
API_KEY = os.getenv("JARVIS_API_KEY", "")

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
    defaults = {
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
        'updated_at': '',
        'etf': {},
    }
    if os.path.exists(MARKET_FILE):
        with open(MARKET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 기본값 병합 (키 누락 방지)
        for k, v in defaults.items():
            data.setdefault(k, v)
        if not isinstance(data.get('us_signal'), dict):
            data['us_signal'] = defaults['us_signal']
        return data
    return defaults

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
    try:
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
    except Exception as e:
        import traceback
        app.logger.error(f"Dashboard error: {traceback.format_exc()}")
        return f"<pre>Dashboard Error:\n{traceback.format_exc()}</pre>", 500

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

# ─── 라우트: 리모콘 ──────────────────────────────────────
@app.route('/remote')
@login_required
def remote():
    return render_template('remote.html', api_key=API_KEY)

# ─── API: 리모콘 명령 실행 ───────────────────────────────
REMOTE_QUEUE_FILE = os.path.join(DATA_DIR, 'remote_queue.json')

def _load_remote_queue():
    if os.path.exists(REMOTE_QUEUE_FILE):
        with open(REMOTE_QUEUE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def _save_remote_queue(q):
    with open(REMOTE_QUEUE_FILE, 'w', encoding='utf-8') as f:
        json.dump(q, f, ensure_ascii=False, indent=2)

@app.route('/api/remote/exec', methods=['POST'])
@api_key_required
def api_remote_exec():
    """리모콘 명령 처리 — 데이터 조회 + 액션 큐잉"""
    if not request.is_json:
        return jsonify({'error': 'JSON required'}), 400

    cmd = request.json.get('cmd', '')
    brain = get_brain_data()
    market = get_market_data()
    holdings = get_holdings()
    metrics = get_metrics()

    result = _exec_cmd(cmd, brain, market, holdings, metrics)
    return jsonify(result)

@app.route('/api/remote/queue', methods=['GET'])
@api_key_required
def api_remote_queue():
    """로컬 머신이 폴링 — 대기 명령 반환 + 큐 비움"""
    q = _load_remote_queue()
    if q:
        _save_remote_queue([])  # 큐 비움
    return jsonify({'commands': q})

@app.route('/api/remote/result', methods=['POST'])
@api_key_required
def api_remote_result():
    """로컬 머신이 명령 실행 결과 전송"""
    if not request.is_json:
        return jsonify({'error': 'JSON required'}), 400
    # 결과를 저장 (다음 폴링 때 리모콘에 표시)
    result_file = os.path.join(DATA_DIR, 'remote_result.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(request.json, f, ensure_ascii=False, indent=2)
    return jsonify({'status': 'ok'})


def _exec_cmd(cmd, brain, market, holdings, metrics):
    """각 명령별 처리 로직"""

    # ── 스캔/분석 그룹 ──
    if cmd == '스캔':
        picks = market.get('picks', [])
        lines = []
        for i, p in enumerate(picks[:10], 1):
            g = p.get('grade', '-')
            s = p.get('score', 0)
            name = p.get('name', '?')
            code = p.get('code', '')
            sigs = ', '.join(p.get('signals', [])[:3])
            emoji = '🟢' if g in ('강력 포착', '적극매수') else '🟡' if g in ('포착', '매수') else '⚪'
            lines.append(f"{emoji} {i:>2}. {name}({code}) — {s}점 [{g}]")
            if sigs:
                lines.append(f"     시그널: {sigs}")
        return {'title': f'📊 매수 후보 TOP 10', 'lines': lines or ['데이터 없음']}

    elif cmd == '스윙스캔':
        picks = market.get('picks', [])
        swing = [p for p in picks if p.get('type') != 'short'][:7]
        lines = []
        for i, p in enumerate(swing, 1):
            lines.append(f"🔵 {i}. {p.get('name','?')} — {p.get('score',0)}점 [{p.get('grade','-')}]")
        return {'title': '🌊 스윙 후보', 'lines': lines or ['데이터 없음']}

    elif cmd == '사전감지':
        whale = brain.get('whale', {})
        dual = brain.get('dual_buying', {})
        lines = []
        # 웨일
        items = whale.get('items', [])[:5]
        lines.append(f"🐋 웨일 감지: {whale.get('total_detected', 0)}건")
        for w in items:
            lines.append(f"  {w.get('name','?')} — 거래량비 {w.get('vol_ratio',0):.1f}x, 외국인 {w.get('foreign_net',0):+,.0f}")
        # 쌍매수
        s_grade = dual.get('s_grade', [])
        a_grade = dual.get('a_grade', [])
        lines.append(f"")
        lines.append(f"🔥 외국인+기관 쌍매수")
        lines.append(f"  S등급: {len(s_grade)}종목, A등급: {len(a_grade)}종목")
        for s in s_grade[:3]:
            lines.append(f"  ⭐ {s.get('name','?')} — 외국인 {s.get('foreign_net',0):+,.0f}, 기관 {s.get('inst_net',0):+,.0f}")
        return {'title': '🔍 사전감지 (웨일+쌍매수)', 'lines': lines}

    elif cmd == '이상거래':
        whale = brain.get('whale', {})
        items = whale.get('items', [])
        abnormal = [w for w in items if w.get('vol_ratio', 0) > 3.0][:7]
        lines = [f"총 이상거래 감지: {len(abnormal)}건"]
        for w in abnormal:
            lines.append(f"⚠ {w.get('name','?')} — 거래량 {w.get('vol_ratio',0):.1f}배, "
                         f"외국인 {w.get('foreign_net',0):+,.0f}")
        return {'title': '⚠️ 이상거래 감지', 'lines': lines or ['이상거래 없음']}

    elif cmd == '건전성':
        lines = []
        lines.append(f"서버: 🟢 정상")
        lines.append(f"시장 데이터: {market.get('updated_at', '미수신')}")
        lines.append(f"Brain 데이터: {brain.get('updated_at', '미수신')}")
        lines.append(f"보유주식: {holdings.get('fetched_at', '미수신')}")
        lines.append(f"리포트: {len(get_report_dates())}건")
        regime = market.get('regime', 'UNKNOWN')
        lines.append(f"레짐: {regime} ({market.get('regime_slots', 0)}슬롯)")
        return {'title': '🛡️ 시스템 건전성', 'lines': lines}

    elif cmd == '이벤트':
        strat = brain.get('strategic', {})
        relay = brain.get('group_relay', {})
        lines = []
        # 릴레이 알림
        alerts = strat.get('relay_alerts', [])
        if alerts:
            lines.append('📡 릴레이 알림:')
            for a in alerts:
                status = a.get('status', '')
                emoji = '🔥' if status == '발화' else '👀'
                lines.append(f"  {emoji} {a.get('pattern','')} — {status} → {a.get('action','')}")
        # 그룹 릴레이
        fired = relay.get('fired_groups', [])
        if fired:
            lines.append('')
            lines.append('🎯 그룹 릴레이 발화:')
            for g in fired[:5]:
                leader = g.get('leader', '?')
                subs = ', '.join(g.get('subs', [])[:3])
                lines.append(f"  {leader} → {subs}")
        # 리스크
        risks = strat.get('risk_factors', [])
        if risks:
            lines.append('')
            lines.append('🚨 리스크 요인:')
            for r in risks[:5]:
                lines.append(f"  🔴 {r}")
        return {'title': '⚡ 이벤트 & 릴레이', 'lines': lines or ['이벤트 없음']}

    # ── AI 선정 그룹 ──
    elif cmd == '종목선정':
        v3 = brain.get('v3_picks', {})
        buys = v3.get('buys', [])
        lines = []
        lines.append(f"레짐: {v3.get('regime', '?')} | 슬롯: {v3.get('available_slots', '?')}")
        lines.append(f"현금: {v3.get('cash_pct_after', '?')}%")
        lines.append('')
        if buys:
            for b in buys:
                conv = b.get('conviction', 0)
                stars = '⭐' * min(conv, 10)
                lines.append(f"🎯 {b.get('name','?')} ({b.get('ticker','')})")
                lines.append(f"   확신도: {conv}/10 {stars}")
                lines.append(f"   비중: {b.get('size_pct',0)}% | 진입가: {b.get('entry_price',0):,}원")
                lines.append(f"   손절: {b.get('stop_loss_pct',0)}% | 목표: +{b.get('target_pct',0)}%")
                lines.append(f"   전략: {b.get('strategy','')} | 정합: {b.get('thesis_alignment','')}")
                lines.append(f"   사유: {b.get('reasoning','')}")
                lines.append('')
        else:
            lines.append('추천 종목 없음')
        # 경고
        warns = v3.get('portfolio_warnings', [])
        if warns:
            lines.append('⚠️ 경고:')
            for w in warns:
                lines.append(f"  🟡 {w}")
        return {'title': '🧠 AI v3 종목선정', 'lines': lines}

    elif cmd == 'MACD스캔':
        picks = market.get('picks', [])
        macd = [p for p in picks if 'MACD' in str(p.get('signals', []))][:10]
        lines = [f"MACD 시그널 보유 종목: {len(macd)}건"]
        for i, p in enumerate(macd, 1):
            lines.append(f"📈 {i}. {p.get('name','?')} — {p.get('score',0)}점")
        return {'title': '📊 MACD 스캔', 'lines': lines or ['MACD 시그널 없음']}

    elif cmd == '워치리스트':
        lines = [
            '👁️ 관심종목 워치리스트 (7종목)',
            '',
            '1. 산일전기 — 전력인프라/변압기',
            '2. 한화시스템 — 방산/ICT',
            '3. 풍산 — 방산/비철금속',
            '4. 한국금융지주 — 금융',
            '5. 달바글로벌 — 화장품/소비재',
            '6. 아모레퍼시픽 — 화장품',
            '7. LG생활건강 — 화장품/생활용품',
            '',
            '📌 시그널 발생 시 즉시 알림 활성화됨',
        ]
        return {'title': '👁️ 워치리스트', 'lines': lines}

    # ── AI 정보 그룹 ──
    elif cmd == '해외이벤트':
        strat = brain.get('strategic', {})
        news = brain.get('news', {})
        lines = []
        # 해외 시장
        regime = strat.get('regime', '?')
        lines.append(f"레짐: {regime} (확신도 {strat.get('regime_confidence', 0):.0%})")
        lines.append(f"글로벌: {strat.get('global_summary', '')}")
        lines.append('')
        # 리스크
        risks = strat.get('risk_factors', [])
        if risks:
            lines.append('🚨 리스크:')
            for r in risks:
                lines.append(f"  🔴 {r}")
        return {'title': '🌏 해외이벤트 & 거시', 'lines': lines}

    elif cmd == 'AI모니터':
        strat = brain.get('strategic', {})
        theses = strat.get('industry_thesis', [])
        lines = []
        lines.append(f"레짐: {strat.get('regime', '?')} | 현금 권고: {strat.get('cash_reserve_suggestion', '?')}%")
        lines.append(f"최대 신규매수: {strat.get('max_new_buys', '?')}종목")
        lines.append('')
        prio = strat.get('sector_priority', {})
        if prio:
            attack = ', '.join(prio.get('attack', []))
            watch = ', '.join(prio.get('watch', []))
            avoid = ', '.join(prio.get('avoid', []))
            lines.append(f"🟢 공격: {attack}")
            lines.append(f"🟡 감시: {watch}")
            lines.append(f"🔴 회피: {avoid}")
            lines.append('')
        for t in theses[:4]:
            conf = t.get('confidence', 0)
            lines.append(f"📊 {t.get('sector','')} (확신 {conf}/10)")
            lines.append(f"   {t.get('thesis','')}")
            lines.append(f"   수급: {t.get('demand_supply','')} | ASP: {t.get('asp_trend','')}")
            lines.append('')
        return {'title': '🤖 AI 전략 모니터', 'lines': lines}

    elif cmd == '뉴스AI':
        news = brain.get('news', {})
        lines = []
        lines.append(f"시장 심리: {news.get('market_sentiment', '?')} | 방향: {news.get('direction', '?')}")
        lines.append('')
        themes = news.get('key_themes', [])
        if themes:
            lines.append('📰 핵심 테마:')
            for t in themes:
                lines.append(f"  • {t}")
            lines.append('')
        outlook = news.get('sector_outlook', {})
        if outlook:
            lines.append('📊 섹터 전망:')
            for sector, info in list(outlook.items())[:8]:
                d = info.get('direction', '?') if isinstance(info, dict) else info
                reason = info.get('reason', '') if isinstance(info, dict) else ''
                emoji = '🟢' if d == 'positive' else '🔴' if d == 'negative' else '🟡'
                lines.append(f"  {emoji} {sector}: {reason[:50]}")
        return {'title': '📰 뉴스 AI 분석', 'lines': lines}

    # ── 계좌 그룹 ──
    elif cmd == '현재잔고':
        h_list = holdings.get('holdings', [])
        total_eval = holdings.get('total_eval', 0)
        total_pnl = holdings.get('total_pnl', 0)
        cash = holdings.get('available_cash', 0)
        lines = []
        lines.append(f"총 평가: {total_eval:,.0f}원")
        lines.append(f"총 손익: {total_pnl:+,.0f}원")
        lines.append(f"예수금:  {cash:,.0f}원")
        lines.append(f"종목수:  {len(h_list)}종목")
        lines.append('')
        for h in h_list:
            pnl = h.get('pnl_pct', 0)
            emoji = '📈' if pnl > 0 else '📉' if pnl < 0 else '➖'
            lines.append(f"{emoji} {h.get('name','?')} — {h.get('eval_amount',0):,.0f}원 ({pnl:+.2f}%)")
        balance_str = f"💰 {total_eval:,.0f}원"
        return {'title': '💰 현재 잔고', 'lines': lines, 'balance': balance_str}

    elif cmd == '체결내역':
        lines = ['체결내역은 장중에만 조회 가능합니다.', '텔레그램에서 "체결내역" 명령을 사용하세요.']
        return {'title': '📋 체결내역', 'lines': lines}

    elif cmd == '포트폴리오':
        h_list = holdings.get('holdings', [])
        total = holdings.get('total_eval', 0)
        cash = holdings.get('available_cash', 0)
        lines = []
        if total > 0:
            stock_pct = ((total - cash) / total * 100) if total > 0 else 0
            cash_pct = (cash / total * 100) if total > 0 else 0
            lines.append(f"주식: {stock_pct:.1f}% | 현금: {cash_pct:.1f}%")
            lines.append('')
            # 섹터별 분포
            for h in sorted(h_list, key=lambda x: x.get('eval_amount', 0), reverse=True):
                pct = (h.get('eval_amount', 0) / total * 100) if total > 0 else 0
                bar_len = int(pct / 2)
                bar = '█' * bar_len + '░' * (20 - bar_len)
                lines.append(f"  {h.get('name','?'):>10} {bar} {pct:.1f}%")
        else:
            lines.append('보유 종목 없음')
        return {'title': '📊 포트폴리오 구성', 'lines': lines}

    # ── 제어 그룹 ──
    elif cmd == '시작':
        q = _load_remote_queue()
        q.append({'cmd': 'start', 'ts': datetime.now().isoformat()})
        _save_remote_queue(q)
        return {
            'title': '🟢 자동매매 시작 명령',
            'lines': [
                '✅ 시작 명령이 큐에 등록되었습니다.',
                '로컬 시스템이 명령을 수신하면 KILL_SWITCH가 해제됩니다.',
                '',
                '💡 텔레그램에서 "시작" 명령도 동일하게 작동합니다.',
            ],
            'live_status': True
        }

    elif cmd == '정지':
        q = _load_remote_queue()
        q.append({'cmd': 'stop', 'ts': datetime.now().isoformat()})
        _save_remote_queue(q)
        return {
            'title': '🔴 자동매매 정지 명령',
            'lines': [
                '🛑 정지 명령이 큐에 등록되었습니다.',
                '로컬 시스템이 명령을 수신하면 KILL_SWITCH가 생성됩니다.',
                '',
                '💡 텔레그램에서 "정지" 명령도 동일하게 작동합니다.',
            ],
            'live_status': False
        }

    elif cmd == '상태':
        lines = []
        lines.append(f"서버 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"라이브 트레이딩: 🟢 활성화")
        regime = market.get('regime', 'UNKNOWN')
        emoji_r = '🟢' if regime == 'BULL' else '🟡' if regime == 'CAUTION' else '🔴' if regime in ('BEAR','CRISIS') else '⚪'
        lines.append(f"KOSPI 레짐: {emoji_r} {regime} ({market.get('regime_slots', 0)}슬롯)")
        # Brain 레짐
        strat = brain.get('strategic', {})
        lines.append(f"AI 레짐: {strat.get('regime', '?')} (확신 {strat.get('regime_confidence', 0):.0%})")
        lines.append(f"현금 권고: {strat.get('cash_reserve_suggestion', '?')}%")
        lines.append('')
        total = holdings.get('total_eval', 0)
        cash = holdings.get('available_cash', 0)
        lines.append(f"총 자산: {total:,.0f}원")
        lines.append(f"예수금: {cash:,.0f}원")
        lines.append(f"보유: {len(holdings.get('holdings', []))}종목")
        lines.append('')
        # BAT 상태
        bat = market.get('bat_status', {})
        if bat:
            lines.append('BAT 스케줄:')
            for k, v in bat.items():
                lines.append(f"  {k}: {v}")
        # 데이터 갱신 시각
        lines.append('')
        lines.append(f"시장 데이터: {market.get('updated_at', '미수신')}")
        lines.append(f"Brain 데이터: {brain.get('updated_at', '미수신')}")
        balance_str = f"💰 {total:,.0f}원" if total > 0 else '--'
        return {'title': '💓 시스템 상태', 'lines': lines, 'balance': balance_str, 'live_status': True}

    # ── 유틸리티 그룹 ──
    elif cmd == '유니버스':
        lines = [
            '🗄️ 매매 유니버스: 84종목 (기계적 선별)',
            '',
            '선별 기준:',
            '  • 시가총액 상위',
            '  • 일평균 거래대금 5억원+',
            '  • 이상치 17종목 제거',
            '  • KOSPI/KOSDAQ 혼합',
            '',
            f"데이터: data/processed/*.parquet",
        ]
        return {'title': '🗄️ 유니버스', 'lines': lines}

    elif cmd == '시나리오':
        strat = brain.get('strategic', {})
        news = brain.get('news', {})
        lines = []
        regime = strat.get('regime', '?')
        lines.append(f"현재 레짐: {regime}")
        lines.append('')
        lines.append('📋 시나리오:')
        attack_sectors = ", ".join(strat.get("sector_priority", {}).get("attack", []))
        lines.append(f"  🟢 BULL → 공격 5슬롯, {attack_sectors} 집중")
        lines.append("  🟡 중립 → 3슬롯, 현금 30%+ 유지")
        lines.append("  🔴 BEAR → 2슬롯, 방어주 위주")
        lines.append("  ⚫ CRISIS → 0슬롯, 전량 현금화")
        lines.append('')
        lines.append(f"현재 판단: {strat.get('regime_reasoning', '')[:100]}")
        return {'title': '🗺️ 시나리오 분석', 'lines': lines}

    elif cmd == '시그널':
        us = market.get('us_signal', {})
        lines = []
        grade = us.get('grade', '-')
        score = us.get('score', 0)
        emoji = '🟢' if 'BULL' in str(grade) else '🔴' if 'BEAR' in str(grade) else '🟡'
        lines.append(f"US 오버나이트: {emoji} {grade} (점수: {score})")
        lines.append(f"  SPY: {us.get('spy_return', 0):+.2f}%")
        lines.append(f"  QQQ: {us.get('qqq_return', 0):+.2f}%")
        lines.append(f"  VIX: {us.get('vix_return', 0):+.2f}%")
        rules = us.get('special_rules', [])
        if rules:
            lines.append(f"  특수룰: {', '.join(rules)}")
        kills = us.get('kill_sectors', [])
        if kills:
            lines.append(f"  🔴 KILL 섹터: {', '.join(kills)}")
        return {'title': '📡 시그널 현황', 'lines': lines}

    elif cmd == '일지':
        journal = brain.get('journal', {})
        days = journal.get('days', [])
        lines = []
        lines.append(f"기간: {journal.get('period', '-')}")
        lines.append(f"총 추천: {journal.get('total_picks', 0)}건")
        lines.append(f"적중률: {journal.get('hit_rate', 0):.1f}% | 평균수익: {journal.get('avg_return', 0):+.1f}%")
        lines.append('')
        for d in days:
            lines.append(f"📅 {d.get('date', '')}")
            for p in d.get('picks', []):
                emoji = '✅' if p.get('hit') else '⬜'
                lines.append(f"  {emoji} {p.get('name','?')} [{p.get('grade','')}] {p.get('result_pct',0):+.1f}%")
        return {'title': '📓 매매일지', 'lines': lines}

    elif cmd == '로그':
        lines = [
            '최근 시스템 로그:',
            '',
            '로그는 로컬 시스템에 저장됩니다.',
            '  logs/schedule.log — BAT 스케줄 로그',
            '  logs/smart_entry.log — 매수 로그',
            '  logs/trading.log — 매매 로그',
            '',
            '텔레그램에서 "로그" 명령으로 최근 로그를 확인하세요.',
        ]
        return {'title': '📋 시스템 로그', 'lines': lines}

    elif cmd == '도움':
        lines = [
            '🎮 QUANTUM REMOTE 명령어',
            '',
            '📊 [스캔]',
            '  사전감지 — 웨일+쌍매수 감지',
            '  스윙스캔 — 스윙 후보 조회',
            '  스캔 — 전체 매수 후보',
            '',
            '🤖 [AI]',
            '  종목선정 — AI v3 추천 종목',
            '  AI모니터 — 전략 분석 현황',
            '  뉴스AI — 뉴스 기반 분석',
            '',
            '💰 [계좌]',
            '  현재잔고 — 보유주식 조회',
            '  포트폴리오 — 자산 구성',
            '',
            '🎛️ [제어]',
            '  시작 — 자동매매 ON',
            '  정지 — 자동매매 OFF',
            '  상태 — 시스템 현황',
            '  청산 — 전량 청산 (위험!)',
            '',
            '💡 텔레그램 봇에서도 동일 명령 사용 가능',
        ]
        return {'title': '❓ 도움말', 'lines': lines}

    elif cmd == '청산':
        q = _load_remote_queue()
        q.append({'cmd': 'liquidate', 'ts': datetime.now().isoformat()})
        _save_remote_queue(q)
        return {
            'title': '💀 전량 청산 명령',
            'lines': [
                '🛑 청산 명령이 큐에 등록되었습니다.',
                '로컬 시스템이 명령을 수신하면 전량 시장가 매도됩니다.',
                '',
                '⚠️ 이 작업은 되돌릴 수 없습니다!',
                '💡 텔레그램에서 "청산" 명령도 동일하게 작동합니다.',
            ]
        }

    else:
        return {'error': f'알 수 없는 명령: {cmd}'}


# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True)
