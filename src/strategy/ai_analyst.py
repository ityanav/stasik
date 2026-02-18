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
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Ты — опытный криптотрейдер-аналитик. Твоя задача — подтвердить или отклонить торговый сигнал.

Ты получаешь:
- Направление сигнала (BUY/SELL) и оценку от технических индикаторов
- Текущие значения индикаторов (RSI, EMA, MACD, Bollinger Bands, Volume)
- Последние 20 свечей (OHLCV)

Правила:
- Будь консервативен. Лучше пропустить сделку, чем потерять деньги.
- Если только 1 индикатор за вход — скорее всего REJECT.
- Подтверждай (CONFIRM) только если видишь реальную конвергенцию сигналов.
- Учитывай паттерны свечей, тренд, объём.

Отвечай СТРОГО в формате JSON (без markdown, без ```):
{"decision": "CONFIRM" или "REJECT", "confidence": число от 1 до 10, "reasoning": "краткое объяснение на русском"}
"""


@dataclass
class AIVerdict:
    confirmed: bool = False
    confidence: int = 0
    reasoning: str = ""
    error: str | None = None


@dataclass
class AIAnalyst:
    api_key: str = ""
    model: str = "google/gemini-2.0-flash-001"
    min_confidence: int = 6
    timeout: int = 10
    enabled: bool = False
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
        )

    async def close(self):
        if self._client:
            await self._client.aclose()

    async def analyze(
        self,
        signal: str,
        score: int,
        details: dict,
        indicator_text: str,
        candles_text: str,
    ) -> AIVerdict:
        if not self.enabled or not self._client:
            return AIVerdict(error="AI disabled")

        user_prompt = (
            f"Сигнал: {signal} (score={score})\n"
            f"Детали индикаторов: {json.dumps(details)}\n\n"
            f"Текущие значения индикаторов:\n{indicator_text}\n\n"
            f"Последние 20 свечей (новые внизу):\n{candles_text}"
        )

        try:
            resp = await self._client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"].strip()
            return self._parse_response(content)

        except httpx.TimeoutException:
            logger.warning("AI analyst timeout after %ds", self.timeout)
            return AIVerdict(error="timeout")
        except Exception:
            logger.exception("AI analyst error")
            return AIVerdict(error="api_error")

    def _parse_response(self, content: str) -> AIVerdict:
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("AI returned invalid JSON: %s", content[:200])
            return AIVerdict(error="invalid_json", reasoning=content[:200])

        decision = parsed.get("decision", "").upper()
        confidence = int(parsed.get("confidence", 0))
        reasoning = parsed.get("reasoning", "")

        confirmed = decision == "CONFIRM" and confidence >= self.min_confidence

        return AIVerdict(
            confirmed=confirmed,
            confidence=confidence,
            reasoning=reasoning,
        )


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
    return "\n".join(lines)


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
