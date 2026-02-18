import json
import logging
from dataclasses import dataclass, field

import httpx
import pandas as pd

from src.strategy.indicators import (
    calculate_bollinger,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_volume_signal,
    detect_candlestick_patterns,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Ты — опытный криптотрейдер-аналитик. Твоя задача — подтвердить или отклонить торговый сигнал \
и предложить оптимальные параметры для конкретной сделки.

Ты получаешь:
- Направление сигнала (BUY/SELL) и оценку от технических индикаторов
- Текущие значения индикаторов (RSI, EMA, MACD, Bollinger Bands, Volume)
- Обнаруженные свечные паттерны (hammer, engulfing, doji, morning/evening star)
- Последние 20 свечей (OHLCV)
- Данные с нескольких таймфреймов (5м, 15м, 1ч) для контекста тренда и уровней
- Текущие параметры риска (SL%, TP%)

Правила:
- Будь консервативен. Лучше пропустить сделку, чем потерять деньги.
- Если только 1 индикатор за вход — скорее всего REJECT.
- Подтверждай (CONFIRM) только если видишь реальную конвергенцию сигналов.
- Учитывай паттерны свечей, тренд, объём.
- Используй старшие таймфреймы для определения тренда и ключевых уровней.
- Сигнал на 1м ПРОТИВ тренда на 15м/1ч — повод для REJECT.
- Предлагай SL/TP исходя из волатильности и текущей ситуации.

Отвечай СТРОГО в формате JSON (без markdown, без ```):
{
  "decision": "CONFIRM" или "REJECT",
  "confidence": число от 1 до 10,
  "reasoning": "краткое объяснение на русском",
  "stop_loss": число (SL в процентах, например 1.2) или null (использовать дефолт),
  "take_profit": число (TP в процентах, например 2.5) или null (использовать дефолт),
  "position_size": число (множитель размера позиции: 0.5=половина, 1.0=норма, 1.5=увеличить) или null
}
"""

REVIEW_PROMPT = """\
Ты — опытный квант-аналитик. Тебе дают результаты торгового бота за последний период \
и текущие параметры стратегии. Проанализируй и предложи корректировки.

Текущие параметры стратегии:
{strategy_text}

Текущие параметры риска:
{risk_text}

Последние сделки:
{trades_text}

Правила:
- Меняй только то, что действительно нужно. Не трогай то, что работает.
- Если мало данных (< 5 сделок) — будь осторожен с выводами.
- Допустимые параметры для изменения:
  * rsi_oversold (20-45), rsi_overbought (55-80)
  * ema_fast (5-15), ema_slow (15-50)
  * bb_period (10-30), bb_std (1.5-3.0)
  * vol_threshold (1.0-3.0)
  * min_score (1-4)
  * stop_loss (0.5-3.0%), take_profit (1.0-5.0%)
  * risk_per_trade (1.0-10.0%)
- Не меняй macd — его параметры стандартные.

Отвечай СТРОГО в формате JSON (без markdown, без ```):
{{
  "changes": {{"имя_параметра": новое_значение, ...}},
  "reasoning": "объяснение на русском, что и почему меняешь"
}}
Если менять ничего не нужно — верни пустой changes: {{}}.
"""


@dataclass
class AIVerdict:
    confirmed: bool = False
    confidence: int = 0
    reasoning: str = ""
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: float | None = None
    error: str | None = None


@dataclass
class StrategyUpdate:
    changes: dict = field(default_factory=dict)
    reasoning: str = ""
    error: str | None = None


@dataclass
class AIAnalyst:
    api_key: str = ""
    model: str = "google/gemini-2.0-flash-001"
    min_confidence: int = 6
    timeout: int = 10
    enabled: bool = False
    review_interval: int = 60
    _client: httpx.AsyncClient = field(default=None, repr=False)

    def __post_init__(self):
        if self.enabled and self.api_key:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    @classmethod
    def from_config(cls, config: dict) -> "AIAnalyst":
        ai_cfg = config.get("ai", {})
        return cls(
            api_key=ai_cfg.get("api_key", ""),
            model=ai_cfg.get("model", "google/gemini-2.0-flash-001"),
            min_confidence=ai_cfg.get("min_confidence", 6),
            timeout=ai_cfg.get("timeout", 10),
            enabled=ai_cfg.get("enabled", False),
            review_interval=ai_cfg.get("review_interval", 60),
        )

    async def close(self):
        if self._client:
            await self._client.aclose()

    # ── Per-trade analysis ─────────────────────────────────

    async def analyze(
        self,
        signal: str,
        score: int,
        details: dict,
        indicator_text: str,
        candles_text: str,
        risk_text: str = "",
        mtf_data: dict | None = None,
        config: dict | None = None,
    ) -> AIVerdict:
        if not self.enabled or not self._client:
            return AIVerdict(error="AI disabled")

        user_prompt = (
            f"Сигнал: {signal} (score={score})\n"
            f"Детали индикаторов: {json.dumps(details)}\n\n"
            f"Текущие значения индикаторов:\n{indicator_text}\n\n"
            f"Последние 20 свечей (новые внизу):\n{candles_text}"
        )
        if risk_text:
            user_prompt += f"\n\nТекущие параметры риска:\n{risk_text}"

        # Multi-timeframe context
        if mtf_data and config:
            mtf_sections = []
            for tf in sorted(mtf_data, key=lambda x: int(x)):
                tf_df = mtf_data[tf]
                if len(tf_df) < 30:
                    continue
                tf_indicators = extract_indicator_values(tf_df, config)
                tf_candles = summarize_candles(tf_df, n=10)
                mtf_sections.append(
                    f"=== Таймфрейм {tf}м ===\n{tf_indicators}\n\nПоследние 10 свечей:\n{tf_candles}"
                )
            if mtf_sections:
                user_prompt += "\n\n--- МУЛЬТИ-ТАЙМФРЕЙМ КОНТЕКСТ ---\n" + "\n\n".join(mtf_sections)

        try:
            content = await self._call_api(SYSTEM_PROMPT, user_prompt)
            return self._parse_verdict(content)
        except httpx.TimeoutException:
            logger.warning("AI analyst timeout after %ds", self.timeout)
            return AIVerdict(error="timeout")
        except Exception:
            logger.exception("AI analyst error")
            return AIVerdict(error="api_error")

    # ── Periodic strategy review ───────────────────────────

    async def review_strategy(
        self,
        strategy_config: dict,
        risk_config: dict,
        recent_trades: list[dict],
    ) -> StrategyUpdate:
        if not self.enabled or not self._client:
            return StrategyUpdate(error="AI disabled")

        strategy_text = "\n".join(f"  {k}: {v}" for k, v in strategy_config.items())
        risk_text = "\n".join(f"  {k}: {v}" for k, v in risk_config.items())

        if not recent_trades:
            trades_text = "Нет закрытых сделок за период."
        else:
            lines = []
            for t in recent_trades:
                pnl = t.get("pnl") or 0
                direction = "ЛОНГ" if t["side"] == "Buy" else "ШОРТ"
                result = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
                lines.append(
                    f"  {direction} {t['symbol']} | вход={t.get('entry_price', '?')} "
                    f"выход={t.get('exit_price', '?')} | {result} USDT | {t.get('status', '?')}"
                )
            trades_text = "\n".join(lines)

        prompt = REVIEW_PROMPT.format(
            strategy_text=strategy_text,
            risk_text=risk_text,
            trades_text=trades_text,
        )

        try:
            content = await self._call_api(
                "Ты — квант-аналитик. Отвечай строго JSON.",
                prompt,
            )
            return self._parse_review(content)
        except httpx.TimeoutException:
            logger.warning("AI review timeout")
            return StrategyUpdate(error="timeout")
        except Exception:
            logger.exception("AI review error")
            return StrategyUpdate(error="api_error")

    # ── API call ───────────────────────────────────────────

    async def _call_api(self, system: str, user: str) -> str:
        resp = await self._client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": 700,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ── Parsing ────────────────────────────────────────────

    @staticmethod
    def _strip_fences(content: str) -> str:
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        return content

    def _parse_verdict(self, content: str) -> AIVerdict:
        content = self._strip_fences(content)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("AI returned invalid JSON: %s", content[:200])
            return AIVerdict(error="invalid_json", reasoning=content[:200])

        decision = parsed.get("decision", "").upper()
        confidence = int(parsed.get("confidence", 0))
        reasoning = parsed.get("reasoning", "")
        confirmed = decision == "CONFIRM" and confidence >= self.min_confidence

        # Per-trade parameter overrides
        sl = parsed.get("stop_loss")
        tp = parsed.get("take_profit")
        ps = parsed.get("position_size")

        return AIVerdict(
            confirmed=confirmed,
            confidence=confidence,
            reasoning=reasoning,
            stop_loss=float(sl) if sl is not None else None,
            take_profit=float(tp) if tp is not None else None,
            position_size=float(ps) if ps is not None else None,
        )

    def _parse_review(self, content: str) -> StrategyUpdate:
        content = self._strip_fences(content)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("AI review returned invalid JSON: %s", content[:200])
            return StrategyUpdate(error="invalid_json", reasoning=content[:200])

        changes = parsed.get("changes", {})
        reasoning = parsed.get("reasoning", "")

        # Validate changes against allowed ranges
        allowed = {
            "rsi_oversold": (20, 45),
            "rsi_overbought": (55, 80),
            "ema_fast": (5, 15),
            "ema_slow": (15, 50),
            "bb_period": (10, 30),
            "bb_std": (1.5, 3.0),
            "vol_threshold": (1.0, 3.0),
            "min_score": (1, 4),
            "stop_loss": (0.5, 3.0),
            "take_profit": (1.0, 5.0),
            "risk_per_trade": (1.0, 10.0),
        }

        validated = {}
        for key, value in changes.items():
            if key not in allowed:
                logger.warning("AI suggested unknown parameter: %s", key)
                continue
            lo, hi = allowed[key]
            try:
                val = float(value)
            except (TypeError, ValueError):
                logger.warning("AI suggested invalid value for %s: %s", key, value)
                continue
            if lo <= val <= hi:
                validated[key] = val
            else:
                logger.warning("AI suggested out-of-range %s=%s (allowed %s-%s)", key, val, lo, hi)

        return StrategyUpdate(changes=validated, reasoning=reasoning)


def extract_indicator_values(df: pd.DataFrame, config: dict) -> str:
    strat = config["strategy"]

    rsi = calculate_rsi(df, strat["rsi_period"])
    ema_fast, ema_slow = calculate_ema(df, strat["ema_fast"], strat["ema_slow"])
    macd_line, signal_line, macd_hist = calculate_macd(
        df, strat["macd_fast"], strat["macd_slow"], strat["macd_signal"]
    )
    upper, middle, lower = calculate_bollinger(
        df, strat.get("bb_period", 20), strat.get("bb_std", 2.0)
    )
    _, vol_ratio = calculate_volume_signal(df, strat.get("vol_period", 20))

    close = df["close"].iloc[-1]

    lines = [
        f"Цена: {close}",
        f"RSI({strat['rsi_period']}): {rsi.iloc[-1]:.1f}",
        f"EMA({strat['ema_fast']}): {ema_fast.iloc[-1]:.2f} | EMA({strat['ema_slow']}): {ema_slow.iloc[-1]:.2f}",
        f"MACD: {macd_line.iloc[-1]:.4f} | Signal: {signal_line.iloc[-1]:.4f} | Hist: {macd_hist.iloc[-1]:.4f}",
        f"BB Upper: {upper.iloc[-1]:.2f} | Mid: {middle.iloc[-1]:.2f} | Lower: {lower.iloc[-1]:.2f}",
        f"Volume ratio: {vol_ratio:.2f}x (порог: {strat.get('vol_threshold', 1.5)}x)",
    ]

    pat = detect_candlestick_patterns(df)
    if pat["patterns"]:
        pat_names = ", ".join(f"{k}({v:+d})" for k, v in pat["patterns"].items())
        lines.append(f"Свечные паттерны: {pat_names} (итого: {pat['score']:+d})")
    else:
        lines.append("Свечные паттерны: не обнаружены")

    return "\n".join(lines)


def format_risk_text(config: dict) -> str:
    risk = config["risk"]
    return (
        f"SL: {risk['stop_loss']}% | TP: {risk['take_profit']}%\n"
        f"Размер позиции: {risk['risk_per_trade']}% от баланса"
    )


def summarize_candles(df: pd.DataFrame, n: int = 20) -> str:
    tail = df.tail(n)
    lines = []
    for _, row in tail.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "strftime"):
            ts_str = ts.strftime("%H:%M")
        else:
            ts_str = str(ts)
        lines.append(
            f"{ts_str} O={row['open']:.2f} H={row['high']:.2f} "
            f"L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:.0f}"
        )
    return "\n".join(lines)
