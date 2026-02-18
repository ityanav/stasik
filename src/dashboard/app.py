import logging

from aiohttp import web

from src.storage.database import Database

logger = logging.getLogger(__name__)


class Dashboard:
    def __init__(self, config: dict, db: Database, engine=None):
        self.config = config
        self.db = db
        self.engine = engine
        self.port = config.get("dashboard", {}).get("port", 8080)
        self.app = web.Application()
        self._setup_routes()
        self._runner: web.AppRunner | None = None

    def _setup_routes(self):
        self.app.router.add_get("/", self._index)
        self.app.router.add_get("/api/stats", self._api_stats)
        self.app.router.add_get("/api/trades", self._api_trades)
        self.app.router.add_get("/api/pnl", self._api_pnl)
        self.app.router.add_get("/api/positions", self._api_positions)

    async def start(self):
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        logger.info("Dashboard started on port %d", self.port)

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
            logger.info("Dashboard stopped")

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(text=_DASHBOARD_HTML, content_type="text/html")

    async def _api_stats(self, request: web.Request) -> web.Response:
        daily_pnl = await self.db.get_daily_pnl()
        total_pnl = await self.db.get_total_pnl()
        open_trades = await self.db.get_open_trades()
        stats = await self.db.get_trade_stats()

        balance = 0.0
        running = False
        if self.engine:
            try:
                balance = self.engine.client.get_balance()
            except Exception:
                pass
            running = self.engine._running

        total = stats["total"]
        wins = stats["wins"]
        data = {
            "balance": balance,
            "daily_pnl": daily_pnl,
            "total_pnl": total_pnl,
            "open_positions": len(open_trades),
            "total_trades": total,
            "wins": wins,
            "losses": stats["losses"],
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "running": running,
        }
        return web.json_response(data)

    async def _api_trades(self, request: web.Request) -> web.Response:
        limit = int(request.query.get("limit", "50"))
        trades = await self.db.get_recent_trades(limit)
        return web.json_response(trades)

    async def _api_pnl(self, request: web.Request) -> web.Response:
        days = int(request.query.get("days", "30"))
        data = await self.db.get_daily_pnl_history(days)
        return web.json_response(data)

    async def _api_positions(self, request: web.Request) -> web.Response:
        positions = []
        if self.engine:
            try:
                raw = self.engine.client.get_positions(category="linear")
                for p in raw:
                    positions.append({
                        "symbol": p["symbol"],
                        "side": p["side"],
                        "size": p["size"],
                        "entry_price": p["entry_price"],
                        "unrealised_pnl": p["unrealised_pnl"],
                    })
            except Exception:
                pass
        return web.json_response(positions)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8">
<title>Stasik Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#0f0f23;color:#e0e0e0;padding:20px}
h1{font-size:24px;margin-bottom:20px;color:#fff}
h2{font-size:18px;margin:30px 0 15px;color:#aaa}
.cards{display:flex;gap:12px;flex-wrap:wrap}
.card{background:#1a1a3e;border-radius:12px;padding:18px;min-width:140px;flex:1}
.card h3{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:1px}
.card .val{font-size:28px;font-weight:700;margin-top:6px}
.g{color:#00e676}.r{color:#ff5252}
.status{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600}
.status.on{background:#00e67633;color:#00e676}
.status.off{background:#ff525233;color:#ff5252}
#chart-box{max-width:900px;margin:30px 0;background:#1a1a3e;border-radius:12px;padding:20px}
table{width:100%;border-collapse:collapse;margin-top:10px}
th{text-align:left;padding:10px;color:#888;font-size:12px;text-transform:uppercase;border-bottom:2px solid #2a2a4e}
td{padding:10px;border-bottom:1px solid #1a1a3e;font-size:14px}
tr:hover{background:#1a1a3e}
.tbl-wrap{background:#12122e;border-radius:12px;padding:15px;overflow-x:auto}
</style>
</head><body>
<h1>Stasik Trading Bot <span id="status" class="status off">...</span></h1>
<div class="cards" id="stats"></div>
<div id="chart-box"><canvas id="pnlChart"></canvas></div>
<h2>Последние сделки</h2>
<div class="tbl-wrap">
<table><thead><tr>
<th>Пара</th><th>Сторона</th><th>Вход</th><th>Выход</th><th>PnL</th><th>Статус</th>
</tr></thead><tbody id="tbody"></tbody></table>
</div>
<script>
let chart=null;
async function load(){
  try{
    const s=await(await fetch('/api/stats')).json();
    document.getElementById('status').className='status '+(s.running?'on':'off');
    document.getElementById('status').textContent=s.running?'РАБОТАЕТ':'СТОП';
    const pc=v=>v>=0?'g':'r';
    const fm=v=>(v>=0?'+':'')+v.toFixed(2);
    document.getElementById('stats').innerHTML=`
      <div class="card"><h3>Баланс</h3><div class="val">${s.balance.toFixed(0)} USDT</div></div>
      <div class="card"><h3>За день</h3><div class="val ${pc(s.daily_pnl)}">${fm(s.daily_pnl)}</div></div>
      <div class="card"><h3>Всего PnL</h3><div class="val ${pc(s.total_pnl)}">${fm(s.total_pnl)}</div></div>
      <div class="card"><h3>Win Rate</h3><div class="val">${s.win_rate.toFixed(1)}%</div></div>
      <div class="card"><h3>Позиции</h3><div class="val">${s.open_positions}</div></div>
      <div class="card"><h3>Сделок</h3><div class="val">${s.total_trades}</div></div>
    `;
    const pnl=await(await fetch('/api/pnl?days=30')).json();
    let cum=0;const cumData=pnl.map(d=>{cum+=d.pnl;return cum});
    const labels=pnl.map(d=>d.trade_date);
    if(chart){chart.destroy()}
    chart=new Chart(document.getElementById('pnlChart'),{
      type:'line',
      data:{labels,datasets:[{
        label:'Cumulative PnL (USDT)',data:cumData,
        borderColor:cum>=0?'#00e676':'#ff5252',borderWidth:2,
        fill:true,backgroundColor:cum>=0?'rgba(0,230,118,0.08)':'rgba(255,82,82,0.08)',
        tension:0.3,pointRadius:3
      }]},
      options:{responsive:true,plugins:{legend:{labels:{color:'#888'}}},
        scales:{x:{ticks:{color:'#666'}},y:{ticks:{color:'#666'},grid:{color:'#1a1a3e'}}}}
    });
    const trades=await(await fetch('/api/trades?limit=20')).json();
    document.getElementById('tbody').innerHTML=trades.map(t=>{
      const p=t.pnl||0;const c=p>=0?'g':'r';
      return`<tr><td>${t.symbol}</td><td>${t.side==='Buy'?'ЛОНГ':'ШОРТ'}</td>
      <td>${t.entry_price||'-'}</td><td>${t.exit_price||'-'}</td>
      <td class="${c}">${p?p.toFixed(2):'-'}</td><td>${t.status}</td></tr>`;
    }).join('');
  }catch(e){console.error(e)}
}
load();setInterval(load,60000);
</script>
</body></html>"""
