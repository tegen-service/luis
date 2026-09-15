"""
Double Calendar Scanner
========================

Busca candidatos para DOBLE CALENDARIO (calendar de calls arriba +
calendar de puts abajo, o un solo lado si el contexto lo justifica):
vender el vencimiento corto, comprar el vencimiento largo, en el mismo
strike, aprovechando:
  - IV Rank bajo (volatilidad barata para comprar la pata larga)
  - Term structure favorable (IV del corto > IV del largo, para que el
    decaimiento del corto sea más rápido)
  - Precio en rango/consolidación (NO tendencia — a diferencia del
    scanner de credit spreads)
  - Opcionalmente, earnings DENTRO de la ventana del vencimiento corto,
    para capturar el IV crush post-reporte

Reutiliza del scanner de credit spreads (mismo diseño, ya validado en
producción):
  - Gestión de sesión/streamer única, streamer abierto solo antes de
    pedir market data real (nunca durante el escaneo lento de yfinance)
  - fetch_metrics, verificación de earnings/dividendos con fallback
  - Universo (watchlists o S&P 500)
  - Walls/OI clustering (ya validados manualmente contra Dark Gamma
    en AKAM y CVX — mismo cálculo, misma confianza)

Nuevo en este script:
  - Screener de rango (ADX bajo + volatilidad realizada baja, en vez
    de tendencia)
  - Term structure de IV (corto vs. largo) por ticker
  - Expected move (derivado de IV) para ubicar los strikes
  - Modelo de valuación del calendario vía Black-Scholes: precio de la
    pata larga en el momento en que expira la corta, para distintos
    escenarios de precio del subyacente — no es un breakeven lineal
    simple como el credit spread

SIN Unusual Whales todavía (esa integración no está probada contra
datos reales — se evalúa sumarla recién cuando el token esté activo
y cada endpoint se haya validado por separado).

Requisitos
----------
    pip install "tastytrade>=11.0,<12.0" yfinance pandas numpy ta scipy python-dotenv

Variables de entorno (.env al lado del script)
----------------------------------------------
    TT_CLIENT_SECRET=...
    TT_REFRESH_TOKEN=...

Uso
---
    python calendar_scanner.py

DISCLAIMER: El modelo de valuación de la pata larga usa
Black-Scholes con la IV actual como estimador de la IV futura en la
fecha de expiración corta — esto es una simplificación conocida (en la
práctica la IV cambia con el tiempo, especialmente post-earnings); el
resultado es una aproximación para comparar candidatos entre sí, no
una predicción de P&L garantizado. Verificar siempre contra el broker
antes de operar con dinero real.
"""

from __future__ import annotations

import os
import sys
import time
import math
import asyncio
import logging
import urllib.parse
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

try:
    import httpx
except ImportError:
    httpx = None  # la integración UW se desactiva sola si falta la librería

from tastytrade import Session, DXLinkStreamer
from tastytrade.instruments import get_option_chain
from tastytrade.dxfeed import Quote, Greeks, Summary
from tastytrade.metrics import get_market_metrics

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # pantalla, igual que siempre
        logging.FileHandler("scanner_log.txt", mode="w", encoding="utf-8"),
        # ^ archivo de log automático, UTF-8 real (sin los problemas
        # de encoding de PowerShell al redirigir con Out-File/*>).
        # Se sobreescribe en cada corrida (mode="w"). Después de
        # correr el script, este archivo queda listo para filtrar
        # con Select-String sin pasos intermedios.
    ],
)
log = logging.getLogger("cal_scanner")
logging.getLogger("yfinance").setLevel(logging.WARNING)


# =====================================================================
# CONFIG
# =====================================================================

# --------- Cuenta y position sizing ---------
ACCOUNT_SIZE_USD     = 25_000
MAX_RISK_PCT         = 0.02
MAX_POSITION_PCT     = 0.05

# --------- Filtros universales (subyacente) ---------
MIN_PRICE            = 15.0
MIN_ADV_DOLLAR       = 50_000_000
MIN_MARKET_CAP       = 2_000_000_000
MIN_LIQUIDITY_RATING = 2
MAX_GAP_PCT          = 0.05

# --------- Filtro de RANGO (MODO NORMAL — sin earnings) ---------
MAX_ADX_RANGE        = 20.0    # ADX BAJO = sin tendencia fuerte, ideal para calendario
MAX_REALIZED_VOL_PCTL = 50.0   # volatilidad realizada reciente, percentil vs. su propio histórico

# --------- ATR% y Beta (filtros nuevos — piso ABSOLUTO, no relativo) ---------
# Motivo (caso real): realized_vol_percentile compara al ticker contra
# SU PROPIO histórico — un activo que siempre es movido puede pasar
# ese filtro con solo estar "tranquilo para lo que es él", y aun así
# moverse más de lo que la estrategia tolera (caso real: NEE, cerrado
# en -85% a los pocos días). ATR% pone un techo ABSOLUTO, comparable
# entre cualquier ticker del universo sin importar su historial propio.
MAX_ATR_PCT          = 3.0     # ATR/precio ≤ 3% — movimiento diario absoluto razonable
# Beta: YA se fetchea de tastytrade (mt["beta"]) pero antes solo se
# guardaba, nunca se usaba como filtro. Beta mide amplificación de
# movimientos del MERCADO general — complementa a ATR% (que mide
# movimiento propio, venga de donde venga: mercado o motivo propio).
MAX_BETA             = 1.3     # activos que amplifican mucho al mercado general, afuera

# --------- Vencimientos del calendario — MODO NORMAL (sin earnings) ---------
SHORT_DTE_TARGET     = 12
SHORT_DTE_RANGE      = 5        # 7-17 DTE para el corto
LONG_DTE_TARGET      = 45
LONG_DTE_RANGE       = 15       # 30-60 DTE para el largo

# --------- Selección de strikes — MODO NORMAL (vía expected move) ---------
STRIKE_EXPECTED_MOVE_MULT = 1.0   # ubicar los strikes a 1x expected move del vencimiento corto

# Piso de calidad mínima post-redondeo (hallazgo real de esta sesión,
# caso BKR: target simétrico a 1.0σ de cada lado, pero el strike
# disponible más cercano en la cadena real puede caer mucho más cerca
# del precio de lo previsto si el grid de strikes es ancho ($5, $10)
# relativo al expected move — find_closest_strike() no chequeaba esto,
# eligiendo "el más cercano" sin verificar que quedara razonablemente
# cerca del objetivo. Resultado real observado: Put a 0.96σ, Call a
# solo 0.16σ — la mitad "protegida" del calendario, la otra casi
# pegada al spot desde el momento mismo de la recomendación, no por
# movimiento posterior del precio. Se descarta el candidato si
# cualquiera de los dos lados queda por debajo de este piso, aunque el
# promedio de ambos lados "parezca" razonable.
MIN_STRIKE_COVERAGE_SIGMA = 0.5

# --------- Filtros de IV / term structure (ambos modos) ---------
MIN_IV_RANK_FOR_LONG   = 0.0    # el checklist pide IVR BAJO para comprar la pata larga
MAX_IV_RANK_FOR_LONG   = 40.0
# NOTA: el term structure ya NO tiene un umbral de filtro duro — se
# maneja como ponderación en rank_calendars() (pesos distintos según
# el modo, ver esa función). Se eliminó MIN_TERM_STRUCTURE_SLOPE
# porque estaba declarada pero nunca se aplicaba en ningún lado (bug
# de config muerta, detectado revisando por qué candidatos con term
# structure negativo seguían rankeando arriba).

# =====================================================================
# MODO EARNINGS — parámetros según la metodología real (video de
# Adri Garzón, Platícame Rafa): doble calendario pre-earnings.
#
#   - Vencimiento CORTO = primera expiración semanal DESPUÉS del
#     earnings (no antes — la posición se beneficia de la IV subiendo
#     hasta el evento, y se cierra manualmente antes de llegar ahí).
#   - Vencimiento LARGO = una semana después del corto.
#   - Entrada: ventana dura de 8-10 días antes del earnings.
#   - Radar: se muestra desde 15 días antes (para verlo venir), pero
#     solo entra al "recuadro verde" (candidato completo, con precios
#     y strikes) cuando cae en la ventana de entrada real.
#   - Selección de strikes: por DELTA de los contratos cortos (call y
#     put), objetivo 0.20-0.30, no por expected move directo — el
#     expected move solo acota el rango de strikes a mirar.
#   - Salida: manual, take profit 10-20% del débito, SIEMPRE antes
#     del earnings (nunca mantener la posición durante el evento).
# =====================================================================

ENTRY_WINDOW_MIN_DAYS = 8    # ventana de entrada: mínimo días antes del earnings
ENTRY_WINDOW_MAX_DAYS = 10   # ventana de entrada: máximo días antes del earnings
RADAR_MAX_DAYS        = 15   # mostrar en "Próximamente" desde esta cantidad de días

TARGET_SHORT_DELTA    = 0.25
DELTA_RANGE_MIN       = 0.20
DELTA_RANGE_MAX       = 0.30
DELTA_STRIKE_BAND_PCT = 0.30  # banda de strikes a redor del spot donde buscar el delta objetivo

TAKE_PROFIT_PCT_MIN   = 10.0  # % del débito — recordatorio en el reporte, no ejecuta nada solo
TAKE_PROFIT_PCT_MAX   = 20.0

MIN_SHORT_TO_LONG_GAP_DAYS = 5  # el largo debe ser al menos esto más lejos que el corto

# --------- Liquidez ---------
MAX_BID_ASK_PCT      = 0.20
MIN_OPEN_INTEREST    = 30

# --------- Automatizar la decisión sobre THEO (pedido explícito) ---------
# Antes: "THEO" era binario (¿al menos 1 de las 4 patas usó precio
# teórico? sí/no) y la única salida era "andá a verificar a mano" —
# lo cual, con THEO apareciendo en 80-90% de los candidatos reales,
# terminaba pidiendo revisar TODO a mano, exactamente lo opuesto de
# automatizar. Ahora el script decide por vos según CUÁNTAS de las 4
# patas fallaron:
#   - 0 patas teóricas: cotización 100% real, sin bandera.
#   - 1 pata teórica: la incertidumbre es acotada (3 de 4 precios son
#     reales) — se muestra la bandera como información, pero el
#     candidato sigue mostrándose y puede operarse con confianza
#     razonable sin re-verificar manualmente.
#   - 2+ patas teóricas: la incertidumbre se acumula demasiado como
#     para confiar en el crédito/débito mostrado — se DESCARTA
#     automáticamente, no llega ni siquiera a tu reporte. Esto es lo
#     que reemplaza al "andá a revisar a mano": en vez de pedirte que
#     vos filtres los malos, el script ya no te los muestra.
MAX_THEO_LEGS_ALLOWED = 1

# --------- Executable Debit (mejora V2 — validado con evidencia real HPQ) ---------
# HPQ mostró mid=$0.08 vs. natural=$0.60 en tastytrade — una diferencia de
# 7.5x. El "mid" que usábamos antes puede ser fantasía si el mercado real
# no tiene liquidez para llenar ahí. Executable Debit usa ASK de las patas
# largas (lo que realmente pagarías comprando) y BID de las cortas (lo que
# realmente recibirías vendiendo) — el escenario "peor caso realista".
SLIPPAGE_PENALTY_PCT   = 15.0   # por encima de esto, penaliza el Score
MAX_SLIPPAGE_PCT_DISCARD = 25.0 # por encima de esto, descarta directo

# --------- P/L Pre-Earnings (solo modo earnings) ---------
# LIMITACIÓN DOCUMENTADA: usa la IV ACTUAL de cada pata como constante en
# todos los snapshots temporales. En la práctica la IV de las opciones
# suele SUBIR a medida que se acerca el earnings (tal como describe Adri
# en el video: "el implied volatility va aumentando... porque se acercan
# earnings"). Esto significa que estos números de P/L pre-earnings
# probablemente SUBESTIMAN el resultado real — son un piso conservador,
# no una predicción exacta.
PRE_EARNINGS_SNAPSHOT_DAYS = [5, 3, 1]  # días antes del earnings a simular (solo modo earnings)

# --------- P/L por días DESDE LA ENTRADA (ambos modos) ---------
# Generalización pedida explícitamente: no esperar a un umbral fijo de
# ganancia ni a "un día antes del earnings" — simular el P&L en unos
# pocos días fijos desde que se abrió la posición, para poder cerrar
# apenas cubra gastos + algo de ganancia, sin importar el modo
# (Normal o Earnings). Validado contra OptionStrat real en COST: día
# 8 mostró +$228.52 (~7% del débito) con el precio prácticamente
# quieto — la ganancia vino del paso del tiempo, no de un movimiento
# fuerte, coherente con lo que predice el modelo.
DAYS_SINCE_ENTRY_SNAPSHOTS = [5, 8, 10]

# =====================================================================
# UW config — mismo patrón validado hoy en tasty_credit_spread_scanner.
# Umbral de tamaño de dark pool print, mismo criterio que el otro script.
# =====================================================================
DARKPOOL_MIN_NOTIONAL = 500_000

# --------- Modo after-hours ---------
ALLOW_THEORETICAL_FALLBACK = True
DEBUG_REJECT_REASONS       = True

# --------- Streaming ---------
STREAM_TIMEOUT_SEC   = 30.0
BATCH_SIZE           = 100
BATCH_PAUSE_SEC      = 1.0
SHUTDOWN_GRACE_SEC   = 1.0

# --------- Aislar modos para probar de a uno (recomendado para testing) ---------
# Ponelos en False/True según qué quieras correr y validar por separado.
# Ejemplo: para probar SOLO Modo Normal -> RUN_NORMAL_MODE=True, RUN_EARNINGS_MODE=False
RUN_NORMAL_MODE   = True
RUN_EARNINGS_MODE = True

# --------- Universo ---------
UNIVERSE_SOURCE = "watchlists"   # "watchlists" | "sp500"
INCLUDE_ALL_PRIVATE  = True
WATCHLISTS_PRIVATE   = []
WATCHLISTS_PUBLIC    = ["tasty IVR", "Liquid ETFs"]

# --------- Modelo ---------
RISK_FREE_RATE       = 0.045

# --------- Output ---------
OUTPUT_CSV           = "double_calendar_candidates.csv"
OUTPUT_HTML          = "double_calendar_report.html"
AUTO_OPEN_BROWSER    = True

# --------- Historial (para backtest futuro con UW) ---------
# Cada corrida guarda ADEMÁS una copia con la fecha del día en esta
# carpeta — así queda un registro de qué recomendó el script cada día,
# sin acumular duplicados si corrés varias veces el mismo día (pisa
# solo el archivo de HOY, nunca los de días anteriores). Esto es lo
# que después le vamos a poder cruzar contra datos históricos reales
# de Unusual Whales (/api/option-contract/{id}/historic) para
# comparar predicción vs. resultado real.
HISTORY_DIR           = "history"


# =====================================================================
# AUTH — UNA SOLA SESIÓN PARA TODA LA CORRIDA (idéntico al scanner de
# credit spreads — mismo fix del bug de sesiones huérfanas)
# =====================================================================

def make_session() -> Session:
    secret = os.environ.get("TT_CLIENT_SECRET")
    token  = os.environ.get("TT_REFRESH_TOKEN")
    if not secret or not token:
        sys.exit("ERROR: Falta TT_CLIENT_SECRET o TT_REFRESH_TOKEN en el .env")
    log.info("Autenticando contra TastyTrade (sesión única para toda la corrida)…")
    return Session(secret, token)


# =====================================================================
# STREAMER UNIFICADO — mismo patrón validado en el scanner de credit
# spreads: se abre UNA vez, justo antes de pedir market data real, y
# se cierra apenas termina. Nunca queda abierto durante el escaneo
# lento de yfinance.
# =====================================================================

class MarketDataHub:
    def __init__(self, streamer: DXLinkStreamer):
        self._streamer = streamer

    async def _collect_batch(self, syms: list[str], timeout: float,
                              want_quote: bool, want_greeks: bool,
                              want_summary: bool):
        quotes, greeks, summs = {}, {}, {}
        if not syms:
            return quotes, greeks, summs
        classes = []
        if want_quote:   classes.append((Quote, quotes))
        if want_greeks:  classes.append((Greeks, greeks))
        if want_summary: classes.append((Summary, summs))
        for cls, _ in classes:
            await self._streamer.subscribe(cls, syms)
        deadline = time.monotonic() + timeout
        needed = set(syms)
        try:
            while time.monotonic() < deadline:
                for cls, store in classes:
                    try:
                        ev = await asyncio.wait_for(
                            self._streamer.get_event(cls), timeout=0.20)
                        sym = (getattr(ev, "event_symbol", None)
                               or getattr(ev, "eventSymbol", None))
                        if sym:
                            store[sym] = ev
                    except asyncio.TimeoutError:
                        pass
                have_all = True
                if want_quote and not (needed <= quotes.keys()):
                    have_all = False
                if want_greeks and not (needed <= greeks.keys()):
                    have_all = False
                # FIX (hallazgo real, sesión OI por expiración exacta):
                # faltaba este chequeo. Sin él, un fetch que pide SOLO
                # Summary (want_quote=False, want_greeks=False, como el
                # de OI del chain corto) daba have_all=True desde el
                # primer ciclo — las otras dos condiciones se saltan
                # solas al estar sus flags en False — y el loop cortaba
                # antes de que llegara casi ningún evento de Summary.
                # Resultado real observado: todas las columnas de OI
                # (OIPutCallRatioShort y las demás) salían en None.
                if want_summary and not (needed <= summs.keys()):
                    have_all = False
                if have_all:
                    break
        finally:
            for cls, _ in classes:
                try:
                    await self._streamer.unsubscribe(cls, syms)
                except Exception:
                    pass
        return quotes, greeks, summs

    async def fetch(self, symbols: list[str], timeout: float = STREAM_TIMEOUT_SEC,
                     want_quote: bool = True, want_greeks: bool = True,
                     want_summary: bool = True,
                     batch_size: int = BATCH_SIZE,
                     batch_pause: float = BATCH_PAUSE_SEC):
        Q, G, S = {}, {}, {}
        if not symbols:
            return Q, G, S
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            log.info(f"  MarketDataHub batch {batch_num}/{total_batches} ({len(batch)} símbolos)")
            success = False
            for intento in range(3):
                try:
                    q, g, s = await self._collect_batch(
                        batch, timeout, want_quote, want_greeks, want_summary)
                    Q.update(q); G.update(g); S.update(s)
                    success = True
                    break
                except Exception as e:
                    log.warning(f"  batch {batch_num} intento {intento+1}/3 falló: {e}")
                    if intento < 2:
                        await asyncio.sleep(3)
            if not success:
                log.error(f"  batch {batch_num} abandonado tras 3 intentos")
            if batch_num < total_batches:
                await asyncio.sleep(batch_pause)
        return Q, G, S


class _MarketDataContext:
    def __init__(self, session: Session):
        self._session = session
        self._streamer: Optional[DXLinkStreamer] = None
        self.hub: Optional[MarketDataHub] = None

    async def __aenter__(self) -> MarketDataHub:
        self._streamer = DXLinkStreamer(self._session)
        await self._streamer.__aenter__()
        log.info("WebSocket DXLink abierto")
        self.hub = MarketDataHub(self._streamer)
        return self.hub

    async def __aexit__(self, exc_type, exc, tb):
        if self._streamer is not None:
            try:
                await self._streamer.__aexit__(exc_type, exc, tb)
                log.info("WebSocket DXLink cerrado correctamente")
            except Exception as e:
                log.warning(f"Error cerrando streamer (ignorado): {e}")
        try:
            await asyncio.sleep(SHUTDOWN_GRACE_SEC)
        except Exception:
            pass


def market_data_context(session: Session) -> _MarketDataContext:
    return _MarketDataContext(session)


# =====================================================================
# HELPERS
# =====================================================================

def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# =====================================================================
# UNIVERSO (idéntico al scanner de credit spreads)
# =====================================================================

from tastytrade.watchlists import PrivateWatchlist, PublicWatchlist


def _entries_symbols(entries) -> set[str]:
    """
    Extrae símbolos de las entradas de una watchlist. Robusto a dos
    formatos distintos que devuelve la API de tastytrade según la
    watchlist: entradas como dict ({"symbol": ..., "instrument-type":
    ...}) — el caso normal — o como string plano (solo el ticker) —
    detectado en la watchlist pública "Tasty IVR", que rompía con
    'str' object has no attribute 'get' antes de este fix.
    """
    out = set()
    for e in (entries or []):
        if isinstance(e, dict):
            if (e.get("instrument-type") or "").lower() == "equity":
                out.add(e["symbol"])
        elif isinstance(e, str):
            out.add(e)
        # cualquier otro tipo inesperado se ignora silenciosamente,
        # no debería romper la carga del resto de la watchlist
    return out


def load_symbols_watchlists(session: Session) -> list[str]:
    symbols: set[str] = set()
    try:
        priv = PrivateWatchlist.get(session)
        if not isinstance(priv, list):
            priv = [priv]
        if not INCLUDE_ALL_PRIVATE:
            priv = [w for w in priv if w.name in WATCHLISTS_PRIVATE]
        for wl in priv:
            got = _entries_symbols(wl.watchlist_entries)
            log.info(f"  Privada [{wl.name}]: {len(got)}")
            symbols |= got
    except Exception as e:
        log.warning(f"No pude leer privadas: {e}")
    for name in WATCHLISTS_PUBLIC:
        try:
            # FIX: la librería tastytrade no codifica espacios en el
            # nombre de la watchlist al armar la URL — bug confirmado
            # con traceback real (rompe en utils.py:validate_response,
            # 'str' object has no attribute 'get', porque la API
            # devuelve texto de error en vez de JSON con una URL mal
            # formada). Lo codificamos nosotros antes de llamar al SDK.
            encoded_name = urllib.parse.quote(name)
            wl = PublicWatchlist.get(session, encoded_name)
            got = _entries_symbols(wl.watchlist_entries)
            log.info(f"  Pública [{name}]: {len(got)}")
            symbols |= got
        except Exception as e:
            log.warning(f"  Pública [{name}]: {e}")
    return sorted(symbols)


def load_symbols_sp500() -> list[str]:
    url = ('https://raw.githubusercontent.com/datasets/'
           's-and-p-500-companies/master/data/constituents.csv')
    try:
        df = pd.read_csv(url)
        tickers = sorted(df['Symbol'].tolist())
        log.info(f"S&P 500 cargado desde GitHub: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.error(f"No se pudo cargar S&P 500: {e}")
        return []


def load_universe(session: Session) -> list[str]:
    if UNIVERSE_SOURCE == "sp500":
        out = load_symbols_sp500()
    else:
        out = load_symbols_watchlists(session)
    log.info(f"Universo total: {len(out)} símbolos (fuente: {UNIVERSE_SOURCE})")
    return out


# =====================================================================
# DATOS DE PRECIO Y SECTOR (idéntico)
# =====================================================================

def load_price_data(symbol: str) -> pd.DataFrame:
    try:
        # period="2y" (no "1y"): realized_vol_percentile necesita
        # lookback(252) + window(20) = 272 filas mínimo para calcular
        # el percentil. Un "1y" de yfinance devuelve ~250-252 filas —
        # insuficiente por diseño, causaba que el 100% de los tickers
        # fallara el gate de historial (bug real, detectado en la
        # primera corrida: normal_screen:insufficient_price_history
        # afectó a 610/610 tickers). "2y" da ~500 filas, margen de sobra.
        df = yf.download(symbol, period="2y", interval="1d",
                         auto_adjust=True, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


_SECTOR_CACHE: dict[str, str] = {}

def get_sector(symbol: str) -> str:
    if symbol in _SECTOR_CACHE:
        return _SECTOR_CACHE[symbol]
    try:
        info = yf.Ticker(symbol).info or {}
        s = info.get("sector") or info.get("category") or "Unknown"
    except Exception:
        s = "Unknown"
    _SECTOR_CACHE[symbol] = s
    return s


# =====================================================================
# INDICADORES TÉCNICOS — para RANGO, no tendencia
# =====================================================================
#
# A diferencia del scanner de credit spreads (que busca ADX ALTO y
# alineación de medias), acá buscamos lo opuesto: ADX BAJO (sin
# tendencia direccional fuerte) y volatilidad realizada baja respecto
# a su propio histórico — el escenario donde un calendario tiene
# sentido (el precio se queda "quieto" mientras la IV comprada decae
# más lento que la vendida).
# =====================================================================

from ta.trend import ADXIndicator

def get_adx(df: pd.DataFrame) -> float:
    try:
        adx = ADXIndicator(high=df["High"], low=df["Low"],
                           close=df["Close"], window=14).adx().iloc[-1]
        return float(adx) if not math.isnan(adx) else 0.0
    except Exception:
        return 0.0


def average_true_range(df: pd.DataFrame, window: int = 14) -> float:
    """
    ATR (Average True Range) en dólares — promedio del rango real de
    movimiento diario. True Range = el MAYOR entre:
      1. Máximo del día − Mínimo del día
      2. Máximo del día − Cierre del día anterior
      3. Mínimo del día − Cierre del día anterior
    A diferencia de un simple high-low, esto captura gaps overnight
    (si el activo salta al abrir por una noticia, se ve reflejado
    aunque el rango intradía haya sido chico).
    """
    if len(df) < window + 1:
        return 0.0
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.dropna().tail(window).mean()
    return float(atr) if not pd.isna(atr) else 0.0


def average_true_range_pct(df: pd.DataFrame, window: int = 14) -> float:
    """
    ATR normalizado como % del precio — necesario para comparar
    movimiento entre tickers de precio muy distinto (un ATR de $5 es
    enorme en una acción de $50, pero casi nada en una de $2000; sin
    normalizar, un solo umbral fijo no sirve para todo el universo).
    """
    if df.empty:
        return 0.0
    price = float(df["Close"].iloc[-1])
    if price <= 0:
        return 0.0
    atr = average_true_range(df, window)
    return float(atr / price * 100)


def realized_volatility(df: pd.DataFrame, window: int = 20) -> float:
    """Volatilidad realizada anualizada (desvío estándar de retornos
    log diarios × sqrt(252)), sobre la ventana más reciente."""
    if len(df) < window + 1:
        return 0.0
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent = returns.tail(window)
    if len(recent) < 2:
        return 0.0
    return float(recent.std() * math.sqrt(252))


def realized_vol_percentile(df: pd.DataFrame, window: int = 20,
                              lookback: int = 252) -> Optional[float]:
    """
    Percentil de la volatilidad realizada ACTUAL respecto a su propio
    histórico (rolling) — análogo al IV Rank pero con vol realizada.
    Bajo (<50) = el precio se está moviendo menos de lo habitual,
    coherente con un contexto de rango.
    """
    if len(df) < lookback + window:
        return None
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    rolling_vol = returns.rolling(window).std() * math.sqrt(252)
    rolling_vol = rolling_vol.dropna().tail(lookback)
    if len(rolling_vol) < 20:
        return None
    current = rolling_vol.iloc[-1]
    pct = (rolling_vol < current).sum() / len(rolling_vol) * 100
    return float(pct)


def average_dollar_volume(df: pd.DataFrame, window: int = 20) -> float:
    if "Volume" not in df.columns or len(df) < window:
        return 0.0
    dv = (df["Close"] * df["Volume"]).rolling(window).mean().iloc[-1]
    return float(dv) if not pd.isna(dv) else 0.0


def recent_gap_filter(df: pd.DataFrame, days: int = 10) -> bool:
    recent = df.tail(days + 1)
    closes = recent["Close"].values
    opens  = recent["Open"].values
    for i in range(1, len(recent)):
        if closes[i - 1] <= 0:
            continue
        if abs(opens[i] - closes[i - 1]) / closes[i - 1] > MAX_GAP_PCT:
            return False
    return True


def classify_range_context(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    True si el contexto favorece un calendario (rango/consolidación):
    ADX bajo, volatilidad realizada en percentil bajo respecto a su
    propio histórico, Y ATR% bajo un techo ABSOLUTO (no relativo —
    ver nota en MAX_ATR_PCT). Devuelve (es_rango, info).
    """
    if len(df) < 252:
        return False, {}
    price = float(df["Close"].iloc[-1])
    adx = get_adx(df)
    rv = realized_volatility(df, 20)
    rv_pctl = realized_vol_percentile(df, 20, 252)
    atr_pct = average_true_range_pct(df, 14)

    is_range = (adx < MAX_ADX_RANGE and
                rv_pctl is not None and rv_pctl < MAX_REALIZED_VOL_PCTL and
                atr_pct <= MAX_ATR_PCT)

    info = {
        "price": round(price, 2), "adx": round(adx, 1),
        "realized_vol": round(rv, 4),
        "realized_vol_percentile": round(rv_pctl, 1) if rv_pctl is not None else None,
        "atr_pct": round(atr_pct, 2),
    }
    return is_range, info


# =====================================================================
# EXPECTED MOVE Y TERM STRUCTURE
# =====================================================================

def expected_move(price: float, iv: float, dte: int) -> float:
    """Movimiento esperado (±1 desvío estándar) derivado de IV."""
    if price <= 0 or iv is None or iv <= 0 or dte <= 0:
        return 0.0
    return price * iv * math.sqrt(dte / 365.0)


def term_structure_slope(iv_short: Optional[float], iv_long: Optional[float]) -> Optional[float]:
    """
    IV_corto - IV_largo. Positivo = backwardation (corto más caro que
    largo) → favorable para calendario, porque el decaimiento del
    corto compensa mejor la pérdida de valor temporal del largo.
    """
    if iv_short is None or iv_long is None:
        return None
    return round(iv_short - iv_long, 4)


# =====================================================================
# INTEGRACIÓN UNUSUAL WHALES — pendiente resuelto de la sesión
# anterior. Portado desde tasty_credit_spread_scanner.py, donde estas
# mismas funciones ya se validaron hoy contra datos y documentación
# reales de UW — no se está portando "a ciegas": son las funciones que
# hoy mismo se probaron endpoint por endpoint.
#
# Alcance para calendarios (a diferencia del scanner de credit
# spreads): esto es para DECIDIR ENTRE candidatos ya filtrados por la
# lógica técnica existente, no para reemplazarla. El calendario sigue
# siendo un instrumento de "quietud" — GEX/dark pool acá se usan para
# preguntar "¿el strike largo elegido tiene respaldo real de mercado,
# o es un número puramente técnico?", no para vetar entradas.
#
# Si UW_API_TOKEN/UW_API_KEY no está seteado (o httpx no está
# instalado), todo esto devuelve estructuras vacías y el resto del
# script sigue funcionando exactamente igual que antes de este cambio.
# =====================================================================

UW_BASE_URL = "https://api.unusualwhales.com"
UW_CONCURRENCY = 5
UW_TIMEOUT_SEC = 15.0


def _uw_token() -> Optional[str]:
    return os.environ.get("UW_API_TOKEN") or os.environ.get("UW_API_KEY")


def _uw_enabled() -> bool:
    if httpx is None:
        return False
    return bool(_uw_token())


def _uw_headers() -> dict:
    return {"Authorization": f"Bearer {_uw_token()}"}


async def _uw_get(client: "httpx.AsyncClient", path: str,
                   params: Optional[dict] = None) -> Optional[list]:
    """
    GET genérico contra la API de UW. Nunca levanta excepción hacia
    arriba — un fallo puntual en un ticker no debe tumbar la corrida.
    """
    try:
        resp = await client.get(f"{UW_BASE_URL}{path}", params=params or {},
                                 headers=_uw_headers(), timeout=UW_TIMEOUT_SEC)
        if resp.status_code == 429:
            log.warning(f"  UW rate limit en {path} — esperando 5s y reintentando…")
            await asyncio.sleep(5)
            resp = await client.get(f"{UW_BASE_URL}{path}", params=params or {},
                                     headers=_uw_headers(), timeout=UW_TIMEOUT_SEC)
        if resp.status_code != 200:
            log.debug(f"  UW {path}: HTTP {resp.status_code}")
            return None
        # FIX (bug real encontrado hoy, confirmado con log real):
        # /api/stock/{ticker}/flow-per-strike es el ÚNICO de los 8
        # endpoints que devuelve una lista JSON directa en el nivel
        # superior, sin el envoltorio {"data": [...]} que sí usan
        # todos los demás. El código viejo asumía el envoltorio
        # siempre, y fallaba con "'list' object has no attribute
        # 'get'" en silencio (atrapado por el except de más abajo) —
        # exactamente por eso este endpoint venía dando 0% de
        # cobertura sin ningún error visible hasta subir el log a
        # WARNING. Ahora se soportan los dos formatos.
        body = resp.json()
        return body if isinstance(body, list) else body.get("data", [])
    except Exception as e:
        log.warning(f"  UW {path} falló: {type(e).__name__}: {e}")
        return None


async def calc_gex_uw(tickers: list[str]) -> dict:
    """
    GEX real por ticker, todas las expiraciones combinadas — idéntica
    a la versión validada hoy en el scanner de credit spreads, con las
    mismas 2 salvaguardas reales encontradas ahí (strikes/precios NaN
    descartados explícitamente; gamma flip descartado si el único
    cruce disponible está a más de 30% del spot, señal de ruido de
    strikes lejanos/ilíquidos en vez del nivel real).

    Para calendarios: el USO es distinto al de credit spreads — acá no
    se veta nada con esto, se usa para anotar en el reporte si el
    strike LARGO elegido (por expected move o por delta) coincide con
    un Call/Put Wall real, como dato de apoyo para decidir entre
    candidatos ya filtrados.
    """
    if not _uw_enabled():
        return {"timestamp": datetime.now().isoformat(), "by_ticker": {}, "source": "disabled"}

    log.info(f"UW: GEX/gamma flip para {len(tickers)} tickers…")
    sem = asyncio.Semaphore(UW_CONCURRENCY)

    async def fetch_one(client, ticker):
        async with sem:
            data = await _uw_get(client, f"/api/stock/{ticker}/spot-exposures/strike")
            return ticker, data or []

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch_one(client, t) for t in tickers])

    out: dict[str, dict] = {}
    for ticker, strikes in results:
        gex_by_strike: dict[float, float] = {}
        current_price = None
        for s in strikes:
            try:
                strike = float(s.get("strike", 0))
                call_gamma = float(s.get("call_gamma_oi", 0) or 0)
                put_gamma = float(s.get("put_gamma_oi", 0) or 0)
                if math.isnan(strike) or math.isnan(call_gamma) or math.isnan(put_gamma):
                    continue
                gex_by_strike[strike] = call_gamma - abs(put_gamma)
                if current_price is None and s.get("price") is not None:
                    price_candidato = float(s["price"])
                    if not math.isnan(price_candidato):
                        current_price = price_candidato
            except Exception:
                continue

        if not gex_by_strike:
            out[ticker] = {"gamma_flip": None, "call_wall": None, "put_wall": None, "net_gex": None}
            continue

        sorted_strikes = sorted(gex_by_strike.keys())
        cruces = []
        cumulative = 0.0
        for strike in sorted_strikes:
            prev = cumulative
            cumulative += gex_by_strike[strike]
            if (prev < 0 and cumulative >= 0) or (prev > 0 and cumulative <= 0):
                cruces.append(strike)

        gamma_flip = None
        if cruces:
            gamma_flip = (min(cruces, key=lambda k: abs(k - current_price))
                          if current_price is not None else cruces[0])
            if gamma_flip is not None and math.isnan(gamma_flip):
                gamma_flip = None
            elif current_price and current_price > 0:
                distancia_pct = abs(gamma_flip - current_price) / current_price
                if distancia_pct > 0.30:
                    gamma_flip = None

        call_wall = max(gex_by_strike, key=lambda k: gex_by_strike[k])
        put_wall = min(gex_by_strike, key=lambda k: gex_by_strike[k])
        net_gex = sum(gex_by_strike.values())

        out[ticker] = {
            "gamma_flip": gamma_flip, "call_wall": call_wall,
            "put_wall": put_wall, "net_gex": round(net_gex, 2),
        }

    return {"timestamp": datetime.now().isoformat(), "by_ticker": out, "source": "unusual_whales"}


async def calc_gex_uw_by_expiry(ticker_expiry_pairs: list[tuple[str, str]]) -> dict:
    """
    GEX real por ticker, PERO ACOTADO a la expiración CORTA exacta que
    se está evaluando en cada candidato — a diferencia de calc_gex_uw
    (todas las expiraciones combinadas, ver docstring de esa función).

    FIX (hallazgo real, pendiente #7 heredado del scanner de credit
    spreads): antes solo existía la versión ticker-wide, que mezcla el
    posicionamiento de TODOS los vencimientos — para juzgar si el
    vencimiento corto puntual de un calendario va a estar quieto, esa
    mezcla puede estar mirando parcialmente el dato equivocado.

    Usa el endpoint que reemplazó al deprecado
    /spot-exposures/{expiry}/strike (changelog UW 2025.02.19):
    /spot-exposures/expiry-strike?expirations[]=<fecha>. Confirmado
    con datos reales (2026-09-02, GOOG, expiry 2026-09-11): mismo
    schema de campos que el endpoint ticker-wide (strike,
    call_gamma_oi, put_gamma_oi, price) — cada fila ya viene filtrada
    a la expiración pedida, sin necesidad de filtrar del lado del
    script.

    ticker_expiry_pairs: lista de (ticker, exp_short_str) — un par por
    ticker, porque cada uno puede tener una expiración corta distinta
    (no se puede batchear en una sola llamada con expirations[]
    compartido entre tickers distintos).

    Por ahora es INFORMATIVO — no veta ni cambia el Score, mismo
    criterio conservador que el resto de la integración UW: mostrar
    antes de accionar, con datos reales acumulados durante la ventana
    de prueba, antes de convertir esto en un descarte.
    """
    if not _uw_enabled():
        return {"timestamp": datetime.now().isoformat(), "by_ticker": {}, "source": "disabled"}

    log.info(f"UW: GEX por expiración corta exacta para {len(ticker_expiry_pairs)} tickers…")
    sem = asyncio.Semaphore(UW_CONCURRENCY)

    async def fetch_one(client, ticker, expiry):
        async with sem:
            data = await _uw_get(client, f"/api/stock/{ticker}/spot-exposures/expiry-strike",
                                   params={"expirations[]": expiry})
            return ticker, data or []

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[fetch_one(client, t, e) for t, e in ticker_expiry_pairs])

    out: dict[str, dict] = {}
    for ticker, strikes in results:
        gex_by_strike: dict[float, float] = {}
        current_price = None
        for s in strikes:
            try:
                strike = float(s.get("strike", 0))
                call_gamma = float(s.get("call_gamma_oi", 0) or 0)
                put_gamma = float(s.get("put_gamma_oi", 0) or 0)
                if math.isnan(strike) or math.isnan(call_gamma) or math.isnan(put_gamma):
                    continue
                gex_by_strike[strike] = call_gamma - abs(put_gamma)
                if current_price is None and s.get("price") is not None:
                    price_candidato = float(s["price"])
                    if not math.isnan(price_candidato):
                        current_price = price_candidato
            except Exception:
                continue

        if not gex_by_strike:
            out[ticker] = {"gamma_flip": None, "call_wall": None, "put_wall": None, "net_gex": None}
            continue

        sorted_strikes = sorted(gex_by_strike.keys())
        cruces = []
        cumulative = 0.0
        for strike in sorted_strikes:
            prev = cumulative
            cumulative += gex_by_strike[strike]
            if (prev < 0 and cumulative >= 0) or (prev > 0 and cumulative <= 0):
                cruces.append(strike)

        gamma_flip = None
        if cruces:
            gamma_flip = (min(cruces, key=lambda k: abs(k - current_price))
                          if current_price is not None else cruces[0])
            if gamma_flip is not None and math.isnan(gamma_flip):
                gamma_flip = None
            elif current_price and current_price > 0:
                distancia_pct = abs(gamma_flip - current_price) / current_price
                if distancia_pct > 0.30:
                    gamma_flip = None

        call_wall = max(gex_by_strike, key=lambda k: gex_by_strike[k])
        put_wall = min(gex_by_strike, key=lambda k: gex_by_strike[k])
        net_gex = sum(gex_by_strike.values())

        out[ticker] = {
            "gamma_flip": gamma_flip, "call_wall": call_wall,
            "put_wall": put_wall, "net_gex": round(net_gex, 2),
        }

    return {"timestamp": datetime.now().isoformat(), "by_ticker": out,
            "source": "unusual_whales_expiry_exacta"}


async def calc_darkpool_levels(tickers: list[str], min_notional: float = DARKPOOL_MIN_NOTIONAL,
                                 top_n: int = 5) -> dict:
    """
    Dark pool prints reales de UW, agrupados por precio y con
    dirección (compra/venta agresiva vía nbbo_bid/nbbo_ask) — idéntica
    a la versión validada hoy en el scanner de credit spreads.

    Para calendarios: el uso natural es preguntar si hay actividad
    institucional real cerca del strike LARGO elegido — si un
    calendario "apunta" a $985 en COST por lectura técnica propia, y
    hay compras grandes de dark pool justo ahí, es una confirmación
    independiente; si no hay nada, no es necesariamente malo, pero es
    un dato menos a favor.
    """
    if not _uw_enabled():
        return {"timestamp": datetime.now().isoformat(), "by_ticker": {}, "source": "disabled"}

    log.info(f"UW: dark pool levels para {len(tickers)} tickers…")
    sem = asyncio.Semaphore(UW_CONCURRENCY)

    async def fetch_one(client, ticker):
        async with sem:
            data = await _uw_get(client, f"/api/darkpool/{ticker}")
            return ticker, data or []

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch_one(client, t) for t in tickers])

    out: dict[str, dict] = {}
    for ticker, prints in results:
        buckets: dict[float, float] = {}
        bullish_notional, bearish_notional = 0.0, 0.0
        for p in prints:
            try:
                price = float(p.get("price", 0))
                size = int(p.get("size", 0))
                notional = price * size
                if notional < min_notional or price <= 0:
                    continue
                level = round(price)
                buckets[level] = buckets.get(level, 0) + notional

                nbbo_bid = _to_float(p.get("nbbo_bid"))
                nbbo_ask = _to_float(p.get("nbbo_ask"))
                if nbbo_ask is not None and price >= nbbo_ask:
                    bullish_notional += notional
                elif nbbo_bid is not None and price <= nbbo_bid:
                    bearish_notional += notional
            except Exception:
                continue
        top_levels = sorted(buckets.items(), key=lambda x: x[1], reverse=True)[:top_n]
        out[ticker] = {
            "levels": [{"price": lvl, "notional": round(val)} for lvl, val in top_levels],
            "bullish_notional_M": round(bullish_notional / 1_000_000, 2),
            "bearish_notional_M": round(bearish_notional / 1_000_000, 2),
        }

    return {"timestamp": datetime.now().isoformat(), "by_ticker": out, "source": "unusual_whales"}


async def calc_oi_per_strike_uw(tickers: list[str]) -> dict:
    """
    OI por strike — /api/stock/{ticker}/oi-per-strike.

    RUTA SIN CONFIRMAR CON DATOS REALES TODAVÍA — a diferencia de
    calc_gex_uw y calc_darkpool_levels de arriba (esas sí se probaron
    hoy con datos reales), esta ruta solo se confirmó como EXISTENTE
    en la documentación oficial (apareció en la búsqueda de endpoints
    de "strike" del scanner de credit spreads), pero nunca se llamó
    con datos reales. Antes de confiar en el resultado, correr el
    mismo tipo de diagnóstico que se usó para los demás endpoints
    (ver diagnostico_unusual_whales.py) apuntándolo acá.

    Objetivo (pedido original, punto 1 del TODO de esta sección): para
    cada candidato, ver si el mercado tiene OI real concentrado cerca
    del strike LARGO elegido — no solo la lectura técnica propia del
    scanner. Se resume como el OI total en un rango de ±3% alrededor
    del strike de interés (se pasa aparte, no acá — este cálculo trae
    la lista completa por strike, sin opinar sobre qué strike importa).
    """
    if not _uw_enabled():
        return {"timestamp": datetime.now().isoformat(), "by_ticker": {}, "source": "disabled"}

    log.info(f"UW: OI por strike (SIN CONFIRMAR) para {len(tickers)} tickers…")
    sem = asyncio.Semaphore(UW_CONCURRENCY)

    async def fetch_one(client, ticker):
        async with sem:
            data = await _uw_get(client, f"/api/stock/{ticker}/oi-per-strike")
            return ticker, data or []

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch_one(client, t) for t in tickers])

    out: dict[str, dict] = {}
    for ticker, strikes in results:
        by_strike: dict[float, float] = {}
        for s in strikes:
            try:
                strike = float(s.get("strike", 0))
                oi = float(s.get("oi", s.get("open_interest", 0)) or 0)
                if math.isnan(strike) or math.isnan(oi):
                    continue
                by_strike[strike] = by_strike.get(strike, 0) + oi
            except Exception:
                continue
        out[ticker] = {"oi_by_strike": by_strike}

    return {"timestamp": datetime.now().isoformat(), "by_ticker": out, "source": "unusual_whales"}


async def calc_unusual_flow_uw(tickers: list[str]) -> dict:
    """
    Flow alertas reales — /api/option-trades/flow-alerts. Portado
    desde el scanner de credit spreads, ya validado ahí (bug de
    dirección compra/venta corregido). Misma lógica ask/bid: compra
    agresiva de calls o venta agresiva de puts = alcista; viceversa
    bajista.

    NOTA para calendarios: acá no hay una "tesis" alcista o bajista
    que confirmar (el calendario apuesta a quietud, no a dirección) —
    el valor de este dato es distinto: convicción direccional GRANDE
    de cualquier lado es más bien una señal de ALERTA para un
    calendario (sugiere que el mercado espera movimiento, lo opuesto
    de lo que la estrategia necesita), no una confirmación positiva.
    Queda informativo — la interpretación queda en manos de quien lea
    el reporte, no se traduce en un ajuste de Score automático.
    """
    if not _uw_enabled():
        return {"timestamp": datetime.now().isoformat(), "flow_by_ticker": {}, "source": "disabled"}

    log.info(f"UW: flow-alerts para {len(tickers)} tickers…")
    sem = asyncio.Semaphore(UW_CONCURRENCY)

    async def fetch_one(client, ticker):
        async with sem:
            data = await _uw_get(client, "/api/option-trades/flow-alerts", params={
                "ticker_symbol": ticker, "min_premium": 50_000, "limit": 200,
            })
            return ticker, data or []

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch_one(client, t) for t in tickers])

    PESO_VENTA_AMBIGUA = 0.5
    # Peso extra por SWEEP — mismo criterio que en scanner.py, portado
    # hoy: para un calendario no confirma ninguna tesis direccional
    # (no hay una), pero hace más preciso el nivel de "convicción
    # grande de cualquier lado" que ya usamos como ALERTA (ver
    # docstring de esta función más arriba) — un sweep marcado pesa
    # más porque es más probable que sea urgencia real, no ruido.
    PESO_SWEEP = 1.15
    flow_by_ticker: dict[str, dict] = {}
    for ticker, alerts in results:
        bullish_p, bearish_p, n = 0.0, 0.0, 0
        for a in alerts:
            try:
                ask_prem = float(a.get("total_ask_side_prem", 0) or 0)
                bid_prem = float(a.get("total_bid_side_prem", 0) or 0)
                is_call = a.get("type", "").lower() == "call"
                peso_apertura = 1.0 if a.get("all_opening_trades") else 0.6
                peso_sweep = PESO_SWEEP if a.get("has_sweep") else 1.0
                peso_total = peso_apertura * peso_sweep
                if is_call:
                    bullish_p += ask_prem * peso_total
                    bearish_p += bid_prem * PESO_VENTA_AMBIGUA * peso_total
                else:
                    bearish_p += ask_prem * peso_total
                    bullish_p += bid_prem * peso_total
                n += 1
            except Exception:
                continue
        flow_by_ticker[ticker] = {
            "call_premium_m": round(bullish_p / 1_000_000, 2),
            "put_premium_m": round(bearish_p / 1_000_000, 2),
            "total_premium_m": round((bullish_p + bearish_p) / 1_000_000, 2),
            "n_signals": n,
        }

    return {"timestamp": datetime.now().isoformat(),
            "flow_by_ticker": flow_by_ticker, "source": "unusual_whales"}


async def calc_net_premium_uw(tickers: list[str]) -> dict:
    """
    Net Premium Ticks — /api/stock/{ticker}/net-prem-ticks. Portado
    desde el scanner de credit spreads (ruta confirmada). A diferencia
    de flow-alerts (solo sweeps grandes), esto cubre TODA la actividad
    del día, tick a tick — complementario, no reemplazo.
    """
    if not _uw_enabled():
        return {"timestamp": datetime.now().isoformat(), "flow_by_ticker": {}, "source": "disabled"}

    log.info(f"UW: net premium ticks para {len(tickers)} tickers…")
    sem = asyncio.Semaphore(UW_CONCURRENCY)

    async def fetch_one(client, ticker):
        async with sem:
            data = await _uw_get(client, f"/api/stock/{ticker}/net-prem-ticks")
            return ticker, data or []

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch_one(client, t) for t in tickers])

    flow_by_ticker: dict[str, dict] = {}
    for ticker, ticks in results:
        if not ticks:
            flow_by_ticker[ticker] = {"net_call_premium_m": 0.0, "net_put_premium_m": 0.0,
                                        "net_delta_total": 0.0, "n_ticks": 0}
            continue
        ultima_fecha = max(t.get("date") for t in ticks if t.get("date"))
        ticks_hoy = [t for t in ticks if t.get("date") == ultima_fecha]
        net_call = sum(float(t.get("net_call_premium", 0) or 0) for t in ticks_hoy)
        net_put = sum(float(t.get("net_put_premium", 0) or 0) for t in ticks_hoy)
        net_delta = sum(float(t.get("net_delta", 0) or 0) for t in ticks_hoy)
        flow_by_ticker[ticker] = {
            "net_call_premium_m": round(net_call / 1_000_000, 3),
            "net_put_premium_m": round(net_put / 1_000_000, 3),
            "net_delta_total": round(net_delta, 1),
            "n_ticks": len(ticks_hoy), "date": ultima_fecha,
        }

    return {"timestamp": datetime.now().isoformat(),
            "flow_by_ticker": flow_by_ticker, "source": "unusual_whales"}


async def calc_oi_change_uw(tickers: list[str]) -> dict:
    """
    OI Change — /api/stock/{ticker}/oi-change. Portado desde el
    scanner de credit spreads, ya con la corrección de dirección
    (ponderado por ask/bid, no el conteo crudo — "OI Change sin
    dirección de trade es posicionamiento, no dirección").
    """
    if not _uw_enabled():
        return {"timestamp": datetime.now().isoformat(), "by_ticker": {}, "source": "disabled"}

    log.info(f"UW: OI change para {len(tickers)} tickers…")
    sem = asyncio.Semaphore(UW_CONCURRENCY)

    async def fetch_one(client, ticker):
        async with sem:
            data = await _uw_get(client, f"/api/stock/{ticker}/oi-change")
            return ticker, data or []

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch_one(client, t) for t in tickers])

    PESO_VENTA_AMBIGUA = 0.5
    out: dict[str, dict] = {}
    for ticker, contratos in results:
        if not contratos:
            out[ticker] = {"call_oi_new": 0, "put_oi_new": 0,
                           "bullish_oi_new": 0, "bearish_oi_new": 0, "top_contract": None}
            continue
        call_oi_new, put_oi_new = 0, 0
        bullish_oi_new, bearish_oi_new = 0.0, 0.0
        for c in contratos:
            sym = c.get("option_symbol", "")
            es_call = "C" in sym[-9:-8] if len(sym) >= 9 else None
            diff = c.get("oi_diff_plain", 0) or 0
            try:
                diff = int(diff)
            except (TypeError, ValueError):
                diff = 0
            if es_call is True:
                call_oi_new += diff
            elif es_call is False:
                put_oi_new += diff
            if diff <= 0 or es_call is None:
                continue
            try:
                ask_vol = float(c.get("prev_ask_volume", 0) or 0)
                bid_vol = float(c.get("prev_bid_volume", 0) or 0)
            except (TypeError, ValueError):
                continue
            total_vol = ask_vol + bid_vol
            if total_vol <= 0:
                continue
            ask_frac, bid_frac = ask_vol / total_vol, bid_vol / total_vol
            # Peso extra por acumulación sostenida — mismo criterio
            # portado de scanner.py: útil acá también, sin importar
            # tesis direccional, porque un salto de OI de un solo día
            # es más ruidoso que 5 días seguidos acumulando.
            dias_acumulando = c.get("days_of_oi_increases", 0) or 0
            peso_acumulacion = 1.0 + min(dias_acumulando, 5) * 0.05
            if es_call:
                bullish_oi_new += diff * ask_frac * peso_acumulacion
                bearish_oi_new += diff * bid_frac * PESO_VENTA_AMBIGUA * peso_acumulacion
            else:
                bearish_oi_new += diff * ask_frac * peso_acumulacion
                bullish_oi_new += diff * bid_frac * PESO_VENTA_AMBIGUA * peso_acumulacion

        top = min(contratos, key=lambda c: c.get("rnk", 999999))
        out[ticker] = {
            "call_oi_new": call_oi_new, "put_oi_new": put_oi_new,
            "bullish_oi_new": round(bullish_oi_new, 1), "bearish_oi_new": round(bearish_oi_new, 1),
            "top_contract": top.get("option_symbol"),
            "top_contract_oi_change_pct": top.get("oi_change"),
            "top_contract_days_increasing": top.get("days_of_oi_increases"),
        }

    return {"timestamp": datetime.now().isoformat(), "by_ticker": out, "source": "unusual_whales"}


async def calc_flow_per_strike_uw(tickers: list[str]) -> dict:
    """
    Flow por strike — /api/stock/{ticker}/flow-per-strike. Portado
    desde el scanner de credit spreads.

    Para calendarios: útil para comparar contra los strikes CORTOS
    (no largos como GEX/dark pool) — si hay convicción real fuerte
    concentrada justo en el strike corto elegido, es una señal de
    alerta (el mercado podría moverse justo hacia ahí antes del
    vencimiento corto, arruinando la ganancia por paso del tiempo).
    """
    if not _uw_enabled():
        return {"timestamp": datetime.now().isoformat(), "by_ticker": {}, "source": "disabled"}

    log.info(f"UW: flow por strike para {len(tickers)} tickers…")
    sem = asyncio.Semaphore(UW_CONCURRENCY)

    async def fetch_one(client, ticker):
        async with sem:
            data = await _uw_get(client, f"/api/stock/{ticker}/flow-per-strike")
            return ticker, data or []

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch_one(client, t) for t in tickers])

    PESO_VENTA_AMBIGUA = 0.5
    out: dict[str, dict] = {}
    debug_count = 0
    for ticker, registros in results:
        # DIAGNÓSTICO puntual — FIX: el chequeo anterior ("if
        # registros:") nunca se disparó en la corrida real, lo que en
        # sí mismo es información: sugiere que "registros" viene VACÍO
        # desde el origen para todos los tickers, no que el problema
        # esté más abajo (fecha o parseo). Ahora se imprime SIEMPRE
        # para los primeros 5 tickers, vacíos o no, para confirmarlo
        # directamente en vez de inferirlo por ausencia de logs.
        if debug_count < 5:
            log.info(f"  [DEBUG flow-per-strike {ticker}] registros crudos="
                      f"{len(registros) if registros else 0}"
                      + (f" · primer registro={registros[0]}" if registros else " · VACÍO"))
            debug_count += 1
        if not registros:
            out[ticker] = {"top_bullish_strike": None, "top_bearish_strike": None}
            continue
        ultima_fecha = max((r.get("date") for r in registros if r.get("date")), default=None)
        registros_hoy = [r for r in registros if r.get("date") == ultima_fecha]
        if ticker and registros and not registros_hoy:
            log.warning(f"  [DEBUG flow-per-strike {ticker}] {len(registros)} registros pero "
                         f"0 coinciden con ultima_fecha={ultima_fecha!r} — revisar formato de fecha")
        bullish_by_strike: dict[float, float] = {}
        bearish_by_strike: dict[float, float] = {}
        errores_parseo = 0
        for r in registros_hoy:
            try:
                strike = float(r.get("strike", 0))
                call_ask = float(r.get("call_premium_ask_side", 0) or 0)
                call_bid = float(r.get("call_premium_bid_side", 0) or 0)
                put_ask = float(r.get("put_premium_ask_side", 0) or 0)
                put_bid = float(r.get("put_premium_bid_side", 0) or 0)
            except (TypeError, ValueError):
                errores_parseo += 1
                continue
            bullish = call_ask + put_bid * PESO_VENTA_AMBIGUA
            bearish = put_ask + call_bid * PESO_VENTA_AMBIGUA
            bullish_by_strike[strike] = bullish_by_strike.get(strike, 0) + bullish
            bearish_by_strike[strike] = bearish_by_strike.get(strike, 0) + bearish

        if registros_hoy and errores_parseo == len(registros_hoy):
            log.warning(f"  [DEBUG flow-per-strike {ticker}] los {errores_parseo} registros de "
                         f"hoy fallaron TODOS al parsear — revisar nombres de campo esperados")

        top_bullish = max(bullish_by_strike, key=bullish_by_strike.get) if bullish_by_strike else None
        top_bearish = max(bearish_by_strike, key=bearish_by_strike.get) if bearish_by_strike else None
        out[ticker] = {
            "top_bullish_strike": top_bullish,
            "top_bullish_premium": round(bullish_by_strike.get(top_bullish, 0), 0) if top_bullish else None,
            "top_bearish_strike": top_bearish,
            "top_bearish_premium": round(bearish_by_strike.get(top_bearish, 0), 0) if top_bearish else None,
        }

    return {"timestamp": datetime.now().isoformat(), "by_ticker": out, "source": "unusual_whales"}


def oi_near_strike(oi_by_strike: dict[float, float], target_strike: float,
                     range_pct: float = 0.03) -> float:
    """Suma el OI real en un rango de ±range_pct alrededor de un strike de interés."""
    if not oi_by_strike or target_strike <= 0:
        return 0.0
    lo, hi = target_strike * (1 - range_pct), target_strike * (1 + range_pct)
    return sum(oi for k, oi in oi_by_strike.items() if lo <= k <= hi)


# =====================================================================
# TASTYTRADE METRICS (IVR, earnings) + fallback yfinance
# (idéntico al scanner de credit spreads — misma lógica probada)
# =====================================================================

def fetch_metrics(session: Session, symbols: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    chunk = 50
    for i in range(0, len(symbols), chunk):
        sub = symbols[i:i + chunk]
        try:
            metrics = get_market_metrics(session, sub)
        except Exception as e:
            log.warning(f"  metrics chunk fail: {e}")
            continue
        for m in metrics:
            ivr = _to_float(m.implied_volatility_index_rank)
            ivp = _to_float(m.implied_volatility_percentile)
            if ivr is not None and ivr <= 1.5: ivr *= 100
            if ivp is not None and ivp <= 1.5: ivp *= 100
            edate = None
            estatus = "unknown"
            if m.earnings is not None:
                edate = getattr(m.earnings, "expected_report_date", None)
                if edate is not None:
                    estatus = "tt_verified"
            mcap = _to_float(m.market_cap)
            lr   = m.liquidity_rating
            beta = _to_float(m.beta)
            out[m.symbol] = {
                "ivr": ivr, "ivp": ivp,
                "earnings_date": edate, "earnings_status": estatus,
                "market_cap": mcap,
                "liquidity_rating": int(lr) if lr is not None else None,
                "beta": beta,
            }

    missing_er = [sym for sym, d in out.items() if d.get("earnings_date") is None]
    if missing_er:
        log.info(f"Verificando earnings via yfinance para {len(missing_er)} ticker(s)...")
        for sym in missing_er:
            yf_date, yf_status = _yfinance_earnings_info(sym)
            if yf_date is not None:
                out[sym]["earnings_date"] = yf_date
                out[sym]["earnings_status"] = yf_status
            elif yf_status == "no_earnings_applicable":
                out[sym]["earnings_status"] = "no_earnings_applicable"
            else:
                out[sym]["earnings_status"] = "unknown"
    return out


def days_to_earnings(edate) -> Optional[int]:
    if edate is None: return None
    try:
        d = pd.Timestamp(edate).normalize()
        return int((d - pd.Timestamp.today().normalize()).days)
    except Exception:
        return None


def _yfinance_earnings_info(symbol: str) -> tuple[Optional[pd.Timestamp], str]:
    try:
        tk = yf.Ticker(symbol)
        today = pd.Timestamp.today().normalize()
        try:
            cal = tk.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if isinstance(ed, (list, tuple)) and len(ed) > 0:
                        d = pd.Timestamp(ed[0]).normalize()
                        if d >= today: return d, "yf_verified"
                    elif ed is not None:
                        d = pd.Timestamp(ed).normalize()
                        if d >= today: return d, "yf_verified"
                elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    val = cal["Earnings Date"].iloc[0]
                    d = pd.Timestamp(val[0] if isinstance(val, (list, tuple)) else val).normalize()
                    if d >= today: return d, "yf_verified"
        except Exception:
            pass
        try:
            ed_df = tk.earnings_dates
            if ed_df is not None and len(ed_df) > 0:
                future = [pd.Timestamp(idx).normalize() for idx in ed_df.index
                          if pd.Timestamp(idx).normalize() >= today]
                if future: return min(future), "yf_verified"
        except Exception:
            pass
        try:
            info = tk.info or {}
            qt = str(info.get("quoteType", "")).upper()
            if qt in ("ETF", "INDEX", "MUTUALFUND", "CURRENCY", "CRYPTOCURRENCY", "FUTURE"):
                return None, "no_earnings_applicable"
        except Exception:
            pass
    except Exception:
        pass
    return None, "unknown"


# =====================================================================
# SCREENER DE SUBYACENTE — para calendario (rango + IVR bajo)
# =====================================================================

def screen_underlying_calendar(symbol: str, df: pd.DataFrame, mt: dict) -> Optional[dict]:
    """MODO NORMAL — rango técnico + IVR bajo, sin depender de earnings."""
    if df.empty or len(df) < 252:
        return {"_rejected": "insufficient_price_history"}
    price = float(df["Close"].iloc[-1])
    if price < MIN_PRICE:
        return {"_rejected": "price_too_low"}
    adv = average_dollar_volume(df, 20)
    if adv < MIN_ADV_DOLLAR:
        return {"_rejected": "adv_too_low"}
    if not recent_gap_filter(df):
        return {"_rejected": "recent_gap"}

    is_range, range_info = classify_range_context(df)
    if not is_range:
        # Reason específica, no genérica — así se puede diagnosticar
        # cuál de las condiciones es el verdadero cuello de botella.
        adx_val = range_info.get("adx")
        rv_pctl = range_info.get("realized_vol_percentile")
        atr_pct = range_info.get("atr_pct")
        adx_fails = adx_val is not None and adx_val >= MAX_ADX_RANGE
        rv_fails = rv_pctl is None or rv_pctl >= MAX_REALIZED_VOL_PCTL
        atr_fails = atr_pct is not None and atr_pct > MAX_ATR_PCT
        failed = []
        if adx_fails: failed.append("adx")
        if rv_fails: failed.append("vol_pctl")
        if atr_fails: failed.append("atr")
        reason = f"not_range({'_and_'.join(failed) if failed else 'unknown'})"
        return {"_rejected": reason, "_info": range_info}

    mcap = mt.get("market_cap") or 0
    if mcap < MIN_MARKET_CAP:
        return {"_rejected": "low_market_cap"}
    lr = mt.get("liquidity_rating")
    if lr is not None and lr < MIN_LIQUIDITY_RATING:
        return {"_rejected": "low_liquidity_rating"}

    # Beta (de tastytrade, ya se fetchea, hoy activado como filtro real
    # por primera vez): complementa a ATR% — ATR% mide movimiento YA
    # ocurrido (venga de donde venga), Beta mide amplificación
    # POTENCIAL si el mercado en general se mueve, aunque el activo
    # esté tranquilo ahora mismo. Si falta el dato, no bloqueamos (no
    # queremos que "sin dato" se comporte como "dato malo" — mismo
    # principio aplicado en el otro scanner con el fix de SPY).
    beta = mt.get("beta")
    if beta is not None and beta > MAX_BETA:
        return {"_rejected": "beta_too_high"}

    ivr = mt.get("ivr")
    if ivr is None:
        return {"_rejected": "ivr_missing"}
    if not (MIN_IV_RANK_FOR_LONG <= ivr <= MAX_IV_RANK_FOR_LONG):
        return {"_rejected": "ivr_too_high"}  # con MIN=0, solo puede fallar por alto

    edate = mt.get("earnings_date")
    dte_e = days_to_earnings(edate)

    return {
        "symbol": symbol, "price": round(price, 2), "mode": "normal",
        "adx": range_info["adx"], "realized_vol": range_info["realized_vol"],
        "realized_vol_percentile": range_info["realized_vol_percentile"],
        "atr_pct": range_info["atr_pct"],
        "adv_$M": round(adv / 1e6, 1), "ivr": ivr, "ivp": mt.get("ivp"),
        "market_cap_$B": round(mcap / 1e9, 2) if mcap else None,
        "beta": beta, "earnings_date": edate,
        "earnings_status": mt.get("earnings_status", "unknown"),
        "days_to_earnings": dte_e,
        "sector": get_sector(symbol),
    }


def screen_underlying_earnings(symbol: str, df: pd.DataFrame, mt: dict) -> Optional[dict]:
    """
    MODO EARNINGS — según la metodología real (video de Adri Garzón):
    NO exige rango/ADX bajo (el edge acá viene de la expansión de IV
    pre-earnings, no de que el activo esté quieto hoy). Exige earnings
    dentro de RADAR_MAX_DAYS (15) para aparecer en el radar; el
    candidato "completo" (recuadro verde) se arma después, cuando cae
    en la ventana real de entrada (8-10 días).
    """
    if df.empty or len(df) < 60:
        return None
    price = float(df["Close"].iloc[-1])
    if price < MIN_PRICE: return None
    adv = average_dollar_volume(df, 20)
    if adv < MIN_ADV_DOLLAR: return None

    mcap = mt.get("market_cap") or 0
    if mcap < MIN_MARKET_CAP:
        return {"_rejected": "low_market_cap"}
    lr = mt.get("liquidity_rating")
    if lr is not None and lr < MIN_LIQUIDITY_RATING:
        return {"_rejected": "low_liquidity_rating"}

    edate = mt.get("earnings_date")
    dte_e = days_to_earnings(edate)
    if dte_e is None or not (0 <= dte_e <= RADAR_MAX_DAYS):
        return {"_rejected": "earnings_out_of_radar_window"}

    in_entry_window = ENTRY_WINDOW_MIN_DAYS <= dte_e <= ENTRY_WINDOW_MAX_DAYS

    # FIX (detectado en la primera corrida real): distinguir "todavía
    # falta para la ventana" de "la ventana ya pasó". Antes, ambos
    # casos (dte_e=3, ya pasada, y dte_e=9, en ventana) podían dar
    # days_until_entry_window=0, mostrando en el radar tickers cuya
    # oportunidad de entrada YA SE PERDIÓ como si estuvieran "por
    # llegar". Si ya pasó y no estamos en ventana, se rechaza
    # explícitamente en vez de mandarlo al radar con un número
    # engañoso.
    if not in_entry_window and dte_e < ENTRY_WINDOW_MIN_DAYS:
        return {"_rejected": "entry_window_already_passed"}

    days_until_entry = max(0, dte_e - ENTRY_WINDOW_MAX_DAYS) if not in_entry_window else 0

    return {
        "symbol": symbol, "price": round(price, 2), "mode": "earnings",
        "adv_$M": round(adv / 1e6, 1), "ivr": mt.get("ivr"), "ivp": mt.get("ivp"),
        "market_cap_$B": round(mcap / 1e9, 2) if mcap else None,
        "beta": mt.get("beta"), "earnings_date": edate,
        "earnings_status": mt.get("earnings_status", "unknown"),
        "days_to_earnings": dte_e,
        "in_entry_window": in_entry_window,
        "days_until_entry_window": days_until_entry,
        "sector": get_sector(symbol),
        # placeholders para consistencia con el modo normal (no aplican acá)
        "adx": None, "realized_vol": None, "realized_vol_percentile": None,
        "atr_pct": None,
    }


# =====================================================================
# SELECCIÓN DE VENCIMIENTOS Y STRIKES
# =====================================================================

def get_strike_increment(chain_for_exp: list, price: float, band_pct: float = 0.15) -> Optional[float]:
    """
    Detecta el incremento típico de strikes (ej. $1, $2.5, $5) que usa
    un vencimiento específico, mirando solo los strikes cerca del
    precio actual (±band_pct) — la zona relevante para nuestros
    strikes, no toda la cadena completa.
    """
    try:
        lo, hi = price * (1 - band_pct), price * (1 + band_pct)
        strikes = sorted({float(o.strike_price) for o in chain_for_exp
                           if lo <= float(o.strike_price) <= hi})
        if len(strikes) < 2:
            return None
        diffs = [round(strikes[i+1] - strikes[i], 4) for i in range(len(strikes) - 1)]
        diffs.sort()
        return diffs[len(diffs) // 2]  # mediana — robusto a algún hueco raro puntual
    except Exception:
        return None


def pick_expiration(chain: dict[date, list], target_dte: int, dte_range: int):
    today = pd.Timestamp.today().normalize()
    best, best_diff = None, None
    for exp in chain.keys():
        d = (pd.Timestamp(exp) - today).days
        if abs(d - target_dte) > dte_range: continue
        diff = abs(d - target_dte)
        if best_diff is None or diff < best_diff:
            best_diff = diff; best = exp
    return best


def pick_two_expirations(chain: dict[date, list], price: Optional[float] = None
                           ) -> tuple[Optional[date], Optional[date]]:
    """
    MODO NORMAL — por DTE fijo (target_dte de config).

    FIX (bug real, confirmado con evidencia: 6 de 6 candidatos de una
    corrida real salieron con strikes de patas desalineados —
    "DIAGONAL" — porque el vencimiento largo elegido por pura cercanía
    de días (ej. 48d) resultó tener incrementos de strike más anchos
    que el corto, sin que el código lo supiera. Antes se tomaba
    ciegamente el vencimiento MÁS CERCANO en días al target, sin mirar
    nunca si había otra opción dentro de la misma tolerancia (ej. 41d,
    que el propio usuario encontró a mano y sí tenía buena
    granularidad).

    Ahora: si se pasa `price`, se evalúan TODOS los vencimientos largos
    dentro de la tolerancia y se prefiere el que tenga incremento de
    strike IGUAL o MÁS FINO que el del corto (garantiza que un
    calendario "puro" sea posible). Si ninguno cumple eso, cae de
    vuelta al comportamiento anterior (el más cercano en días) — este
    fix solo puede MEJORAR el resultado, nunca empeorarlo.
    """
    short_exp = pick_expiration(chain, SHORT_DTE_TARGET, SHORT_DTE_RANGE)
    if short_exp is None:
        return None, None

    if price is None:
        # Sin precio no podemos comparar incrementos — comportamiento
        # anterior, sin el fix (mejor que romper).
        long_exp = pick_expiration(chain, LONG_DTE_TARGET, LONG_DTE_RANGE)
        if long_exp is not None and short_exp >= long_exp:
            return None, None
        return short_exp, long_exp

    short_increment = get_strike_increment(chain[short_exp], price)

    today = pd.Timestamp.today().normalize()
    candidates = []
    for exp in chain.keys():
        d = (pd.Timestamp(exp) - today).days
        if abs(d - LONG_DTE_TARGET) > LONG_DTE_RANGE: continue
        if exp <= short_exp: continue
        dte_diff = abs(d - LONG_DTE_TARGET)
        long_increment = get_strike_increment(chain[exp], price)
        candidates.append((exp, dte_diff, long_increment))

    if not candidates:
        return short_exp, None

    # Preferencia 1: vencimientos con incremento IGUAL o MÁS FINO que
    # el del corto (garantiza calendario puro posible) — entre esos,
    # el más cercano en días al target.
    compatible = [c for c in candidates
                  if c[2] is not None and short_increment is not None
                  and c[2] <= short_increment]
    if compatible:
        compatible.sort(key=lambda c: c[1])  # por cercanía de DTE
        long_exp = compatible[0][0]
        if compatible[0][2] != short_increment:
            log.info(f"  Vencimiento largo elegido por compatibilidad de strikes "
                      f"(incremento {compatible[0][2]:g} vs. corto {short_increment:g}), "
                      f"no el más cercano en días")
    else:
        # Ninguno compatible — comportamiento anterior (más cercano en
        # días), pero ahora al menos queda logueado que es un diagonal
        # inevitable, no un descuido.
        candidates.sort(key=lambda c: c[1])
        long_exp = candidates[0][0]
        log.info(f"  Ningún vencimiento largo tiene strikes tan finos como el corto "
                  f"(corto={short_increment}) — diagonal inevitable, se usa el más "
                  f"cercano en días ({long_exp})")

    return short_exp, long_exp


def pick_earnings_expirations(chain: dict[date, list], earnings_date
                                ) -> tuple[Optional[date], Optional[date]]:
    """
    MODO EARNINGS — corto = primera expiración disponible EN o DESPUÉS
    del earnings; largo = próxima expiración al menos
    MIN_SHORT_TO_LONG_GAP_DAYS después del corto (típicamente la
    semana siguiente, si el ticker tiene semanales).
    """
    try:
        er = pd.Timestamp(earnings_date).normalize()
    except Exception:
        return None, None

    exps_sorted = sorted(chain.keys())
    short_exp = None
    for e in exps_sorted:
        if pd.Timestamp(e).normalize() >= er:
            short_exp = e
            break
    if short_exp is None:
        return None, None

    long_exp = None
    for e in exps_sorted:
        if e <= short_exp:
            continue
        gap = (pd.Timestamp(e) - pd.Timestamp(short_exp)).days
        if gap >= MIN_SHORT_TO_LONG_GAP_DAYS:
            long_exp = e
            break

    return short_exp, long_exp


def is_put(opt) -> bool:
    t = str(opt.option_type).lower()
    return t.endswith("put") or t == "p"


def calc_oi_balance_short_expiry(chain_short: list, summs: dict) -> dict:
    """
    FIX (pendiente #7, hallazgo real de hoy): el balance put/call de
    Open Interest de la expiración corta EXACTA — no del ticker
    completo. Se calcula con datos de tastytrade (Summary, ya
    fetcheados vía streamer para todo el chain corto), no de Unusual
    Whales: se confirmó hoy con datos reales
    (test_uw_oi_per_strike.py, GOOG) que /api/stock/{t}/oi-per-strike
    de UW es agregado de TODAS las expiraciones y el parámetro
    expirations[] no filtra nada (107 filas idénticas con y sin
    filtro) — así que la única fuente confiable de este dato por
    expiración puntual es el propio chain del broker.

    Devuelve, además del total put/call OI de esta expiración, el
    strike con más OI de cada lado (un "muro por OI crudo" — más
    simple que el GEX, útil como comparación/chequeo cruzado, no como
    reemplazo).

    Informativo por ahora — no descarta ni cambia el Score, mismo
    criterio conservador que el resto de esta integración: mostrar
    con datos reales antes de decidir un umbral.
    """
    call_oi_total, put_oi_total = 0, 0
    call_oi_by_strike: dict[float, int] = {}
    put_oi_by_strike: dict[float, int] = {}
    for opt in chain_short:
        s = summs.get(opt.streamer_symbol)
        if s is None:
            continue
        oi = getattr(s, "open_interest", None)
        if oi is None or oi <= 0:
            continue
        try:
            strike = float(opt.strike_price)
        except (TypeError, ValueError):
            continue
        oi = int(oi)
        if is_put(opt):
            put_oi_total += oi
            put_oi_by_strike[strike] = put_oi_by_strike.get(strike, 0) + oi
        else:
            call_oi_total += oi
            call_oi_by_strike[strike] = call_oi_by_strike.get(strike, 0) + oi

    if call_oi_total == 0 and put_oi_total == 0:
        return {"call_oi_total": None, "put_oi_total": None, "oi_put_call_ratio": None,
                "oi_call_wall_strike": None, "oi_put_wall_strike": None}

    ratio = (put_oi_total / call_oi_total) if call_oi_total > 0 else None
    oi_call_wall = max(call_oi_by_strike, key=call_oi_by_strike.get) if call_oi_by_strike else None
    oi_put_wall = max(put_oi_by_strike, key=put_oi_by_strike.get) if put_oi_by_strike else None

    return {
        "call_oi_total": call_oi_total, "put_oi_total": put_oi_total,
        "oi_put_call_ratio": round(ratio, 3) if ratio is not None else None,
        "oi_call_wall_strike": oi_call_wall, "oi_put_wall_strike": oi_put_wall,
    }


def find_closest_strike(chain_for_exp: list, target_strike: float,
                          option_type_is_put: bool) -> Optional[object]:
    candidates = [o for o in chain_for_exp if is_put(o) == option_type_is_put]
    if not candidates:
        return None
    return min(candidates, key=lambda o: abs(float(o.strike_price) - target_strike))


def restrict_to_common_strikes(chain_a: list, chain_b: list) -> list:
    """
    FIX (hallazgo real, caso GOOG): filtra chain_a a las opciones
    (mismo tipo put/call, mismo strike en dólares) que TAMBIÉN existen
    en chain_b. Se usa antes de elegir cualquier strike para un
    Double Calendar, así el strike corto elegido está garantizado a
    existir también en la expiración larga — evita el diagonal por
    redondeo silencioso que salía cuando se elegía el strike corto
    primero y recién después se buscaba "el más parecido" en el chain
    largo (podían no coincidir si el grid de strikes difiere entre
    las dos fechas, ej. la corta tiene strikes cada $2.5 y la larga
    cada $5).
    """
    keys_b = {(is_put(o), round(float(o.strike_price), 2)) for o in chain_b}
    return [o for o in chain_a
            if (is_put(o), round(float(o.strike_price), 2)) in keys_b]


def select_calendar_strikes(chain_short: list, chain_long: list, ctx: dict,
                              iv_for_move: float) -> dict:
    """
    MODO NORMAL — ubica los strikes a STRIKE_EXPECTED_MOVE_MULT ×
    expected move (proxy: volatilidad realizada), strike disponible
    más cercano.

    FIX (hallazgo real, caso GOOG): antes se elegía el strike corto
    mirando solo chain_short, y recién después se buscaba "el más
    parecido" en chain_long — si el grid de strikes difería entre
    las dos fechas, el resultado podía ser un diagonal por accidente
    (long ≠ short) sin que nada lo señalara ni lo evitara. Ahora se
    restringe la búsqueda del corto a los strikes que YA existen en
    ambas expiraciones, así el "más cercano en el largo" que se busca
    después siempre da un match exacto — Double Calendar real
    garantizado, o descarte explícito si no hay ningún strike común
    razonablemente cerca del precio.
    """
    price = ctx["price"]
    em = expected_move(price, iv_for_move, SHORT_DTE_TARGET)
    target_call_strike = price + em * STRIKE_EXPECTED_MOVE_MULT
    target_put_strike = price - em * STRIKE_EXPECTED_MOVE_MULT

    common_short = restrict_to_common_strikes(chain_short, chain_long)
    call_opt = find_closest_strike(common_short, target_call_strike, option_type_is_put=False)
    put_opt = find_closest_strike(common_short, target_put_strike, option_type_is_put=True)

    return {
        "expected_move": round(em, 2),
        "target_call_strike": round(target_call_strike, 2),
        "target_put_strike": round(target_put_strike, 2),
        "call_strike_opt": call_opt,
        "put_strike_opt": put_opt,
    }


def gather_delta_candidate_strikes(chain_short: list, price: float) -> list:
    """
    MODO EARNINGS — a diferencia del modo normal, acá NO elegimos el
    strike directamente: juntamos TODOS los strikes de calls y puts
    dentro de una banda razonable alrededor del spot
    (±DELTA_STRIKE_BAND_PCT) para pedir sus Greeks vía streamer, y
    RECIÉN con el delta real de cada uno elegimos el más cercano a
    TARGET_SHORT_DELTA. Es un primer fetch "exploratorio" — el script
    hace dos rondas de streamer.fetch() dentro de la misma apertura
    del streamer (nunca abre uno nuevo).
    """
    low = price * (1 - DELTA_STRIKE_BAND_PCT)
    high = price * (1 + DELTA_STRIKE_BAND_PCT)
    out = []
    for opt in chain_short:
        try:
            sk = float(opt.strike_price)
        except (TypeError, ValueError):
            continue
        if low <= sk <= high:
            out.append(opt)
    return out


def select_strike_by_delta(candidates: list, greeks: dict,
                             option_type_is_put: bool,
                             target_delta: float = TARGET_SHORT_DELTA,
                             delta_min: float = DELTA_RANGE_MIN,
                             delta_max: float = DELTA_RANGE_MAX) -> Optional[object]:
    """
    Entre los candidatos (mismo tipo, call o put) con Greeks ya
    fetcheados, elige el de |delta| más cercano a target_delta,
    PRIORIZANDO los que caen dentro de [delta_min, delta_max]. Si
    ninguno cae en el rango, devuelve el más cercano al rango de
    todas formas (mejor esfuerzo, igual que el resto del script
    prefiere seguir con la mejor aproximación disponible antes que
    descartar todo el ticker).
    """
    same_type = [o for o in candidates if is_put(o) == option_type_is_put]
    scored = []
    for opt in same_type:
        g = greeks.get(opt.streamer_symbol)
        if g is None:
            continue
        d = _to_float(getattr(g, "delta", None))
        if d is None:
            continue
        scored.append((opt, abs(d)))
    if not scored:
        return None

    in_range = [(o, d) for o, d in scored if delta_min <= d <= delta_max]
    pool = in_range if in_range else scored
    return min(pool, key=lambda x: abs(x[1] - target_delta))[0]


# =====================================================================
# MODELO DE VALUACIÓN DEL CALENDARIO (Black-Scholes)
# =====================================================================
#
# A diferencia de un credit spread (payoff lineal, breakeven único,
# POP calculable con una sola fórmula cerrada), el calendario tiene un
# payoff en forma de "colina": gana si el precio queda cerca del
# strike a la fecha de expiración del vencimiento CORTO, pierde si se
# aleja mucho en cualquier dirección.
#
# Metodología: al momento en que expira la pata corta —
#   - Las patas CORTAS valen su intrínseco puro (ya expiraron).
#   - Las patas LARGAS todavía tienen tiempo restante (long_dte -
#     short_dte días) y se valúan con Black-Scholes, usando la IV
#     ACTUAL de esas opciones como estimador de su IV futura.
#
# LIMITACIÓN DOCUMENTADA (ya la marcamos en el docstring del archivo):
# asumir que la IV de la pata larga se mantiene igual a la de hoy es
# una simplificación. En la práctica, sobre todo si hay earnings en el
# medio del vencimiento corto, la IV de la pata larga también puede
# moverse. Esto es un ESTIMADOR para comparar candidatos entre sí, no
# una predicción de P&L garantizado — igual que el aviso que ya tiene
# el resto del script sobre precios teóricos.
# =====================================================================

def bs_price(S: float, K: float, T: float, sigma: float,
             r: float = RISK_FREE_RATE, is_call: bool = True) -> float:
    """Precio Black-Scholes de una opción europea. T en años."""
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma is None or sigma <= 0:
        # Ya expiró (o sin tiempo/vol): vale su intrínseco puro
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if is_call:
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    except Exception:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)


def calendar_pnl_at_short_expiry(
    S: float, K_call_short: float, K_put_short: float,
    K_call_long: float, K_put_long: float,
    T_long_remaining_years: float,
    iv_long_call: float, iv_long_put: float,
    net_debit: float, r: float = RISK_FREE_RATE,
) -> float:
    """
    P&L del calendario/diagonal para un precio S del subyacente,
    evaluado en la fecha de expiración del vencimiento CORTO.

    FIX (bug real detectado comparando contra tastytrade en vivo): las
    patas larga y corta pueden tener strikes DISTINTOS si la cadena
    del vencimiento largo no lista el mismo strike que la corta (los
    vencimientos más lejanos suelen tener incrementos de strike más
    anchos) — eso convierte la estructura en un "diagonal", no un
    "calendario" puro. Antes este código usaba el strike de la pata
    CORTA para valuar también la pata LARGA, lo cual es incorrecto en
    esos casos. Ahora cada pata se valúa con SU PROPIO strike real.
    """
    short_call_intrinsic = max(S - K_call_short, 0.0)
    short_put_intrinsic = max(K_put_short - S, 0.0)

    long_call_value = bs_price(S, K_call_long, T_long_remaining_years, iv_long_call,
                                 r, is_call=True)
    long_put_value = bs_price(S, K_put_long, T_long_remaining_years, iv_long_put,
                                r, is_call=False)

    position_value = (long_call_value + long_put_value
                       - short_call_intrinsic - short_put_intrinsic)
    return position_value - net_debit


def build_calendar_pnl_curve(
    price: float, K_call_short: float, K_put_short: float,
    K_call_long: float, K_put_long: float,
    short_dte: int, long_dte: int,
    iv_long_call: float, iv_long_put: float,
    net_debit: float, n_points: int = 61,
) -> list[tuple[float, float]]:
    """Curva de P&L en un rango de +/-40% del precio actual."""
    T_remaining = max((long_dte - short_dte) / 365.0, 1 / 365.0)
    s_min, s_max = price * 0.6, price * 1.4
    curve = []
    for i in range(n_points):
        S = s_min + (s_max - s_min) * i / (n_points - 1)
        pnl = calendar_pnl_at_short_expiry(
            S, K_call_short, K_put_short, K_call_long, K_put_long,
            T_remaining, iv_long_call, iv_long_put, net_debit)
        curve.append((round(S, 2), round(pnl, 4)))
    return curve


def analyze_pnl_curve(curve: list[tuple[float, float]], current_price: float,
                        net_debit: float) -> dict:
    """
    Extrae métricas de la curva: P&L en el precio actual, ganancia
    máxima estimada, breakevens (por interpolación lineal donde la
    curva cruza cero), y el rango de precios rentable.
    """
    prices = [p for p, _ in curve]
    pnls = [v for _, v in curve]

    max_pnl = max(pnls)
    max_pnl_price = prices[pnls.index(max_pnl)]

    # P&L en el precio actual (interpolación entre los dos puntos más cercanos)
    pnl_at_current = None
    for i in range(len(prices) - 1):
        if prices[i] <= current_price <= prices[i + 1]:
            p0, p1 = prices[i], prices[i + 1]
            v0, v1 = pnls[i], pnls[i + 1]
            frac = (current_price - p0) / (p1 - p0) if p1 != p0 else 0
            pnl_at_current = v0 + (v1 - v0) * frac
            break
    if pnl_at_current is None:
        pnl_at_current = pnls[0] if current_price < prices[0] else pnls[-1]

    # Breakevens: cruces de cero por interpolación lineal
    breakevens = []
    for i in range(len(prices) - 1):
        v0, v1 = pnls[i], pnls[i + 1]
        if (v0 < 0 <= v1) or (v0 >= 0 > v1):
            p0, p1 = prices[i], prices[i + 1]
            if v1 != v0:
                frac = -v0 / (v1 - v0)
                # float(...) explícito: p0/p1/v0/v1 vienen de bs_price,
                # que usa scipy.stats.norm.cdf y devuelve numpy.float64,
                # no float nativo. Sin este cast, la lista queda con
                # numpy.float64 adentro, y cuando pandas la guarda en el
                # CSV escribe "[np.float64(932.13), ...]" en vez de
                # "[932.13, ...]" — el parseo de la plataforma no podía
                # leer ese formato y el breakeven quedaba vacío en
                # silencio (bug real detectado en producción).
                breakevens.append(float(round(p0 + (p1 - p0) * frac, 2)))

    profit_zone_width = None
    has_losing_valley = False
    min_pnl_in_range = None
    if len(breakevens) >= 2:
        profit_zone_width = round(breakevens[-1] - breakevens[0], 2)
        # FIX (detectado comparando contra OptionStrat en HOOD/AVGO):
        # con 4 breakevens (dos "jorobas" separadas por un valle en el
        # medio), profit_zone_width = breakevens[-1]-breakevens[0]
        # asume que TODO ese rango es rentable — falso si el valle del
        # medio cruza por debajo de cero. Chequeamos explícitamente el
        # mínimo P&L estrictamente entre el primer y el último
        # breakeven para detectar y exponer ese caso, en vez de
        # esconder que hay un tramo perdedor adentro del "ancho" que
        # reportamos.
        inner_pnls = [v for p, v in zip(prices, pnls)
                      if breakevens[0] < p < breakevens[-1]]
        if inner_pnls:
            min_pnl_in_range = round(min(inner_pnls), 2)
            has_losing_valley = min_pnl_in_range < 0

    return {
        "max_profit_estimate": round(max_pnl, 2),
        "max_profit_price": round(max_pnl_price, 2),
        "pnl_at_current_price": round(pnl_at_current, 2),
        "breakevens": breakevens,
        "profit_zone_width": profit_zone_width,
        "has_losing_valley": has_losing_valley,
        "min_pnl_in_range": min_pnl_in_range,
        "return_on_debit_pct": (round(max_pnl / net_debit * 100, 1)
                                  if net_debit > 0 else None),
    }


# =====================================================================
# REJECT TRACKING + EVALUACIÓN DEL CALENDARIO
# =====================================================================

_REJECTS: dict[str, int] = {}

def _reject(reason: str):
    if DEBUG_REJECT_REASONS:
        _REJECTS[reason] = _REJECTS.get(reason, 0) + 1
    return None


def _mark_price(bid, ask, theo, max_spread_pct: float):
    quote_ok = (bid is not None and ask is not None and bid > 0 and ask > 0
                and (ask - bid) / ask <= max_spread_pct)
    if quote_ok:
        return (bid + ask) / 2, False
    if ALLOW_THEORETICAL_FALLBACK and theo is not None and theo > 0:
        return float(theo), True
    return None, False


def _leg_mid(opt, quotes, greeks) -> tuple[Optional[float], Optional[float], bool]:
    """Devuelve (mid_price, iv, is_theoretical) de una pata."""
    sym = opt.streamer_symbol
    q, g = quotes.get(sym), greeks.get(sym)
    if q is None or g is None:
        return None, None, False
    bid, ask = _to_float(q.bid_price), _to_float(q.ask_price)
    theo = _to_float(getattr(g, "price", None))
    mid, is_theo = _mark_price(bid, ask, theo, MAX_BID_ASK_PCT)
    iv = _to_float(getattr(g, "volatility", None))
    return mid, iv, is_theo


def _leg_bid_ask(opt, quotes) -> tuple[Optional[float], Optional[float]]:
    """Devuelve (bid, ask) crudos de una pata, sin fallback teórico —
    para Executable Debit necesitamos precios de mercado reales, no
    estimaciones de modelo."""
    q = quotes.get(opt.streamer_symbol)
    if q is None:
        return None, None
    return _to_float(q.bid_price), _to_float(q.ask_price)


def _leg_greeks(opt, greeks) -> dict:
    """Devuelve delta/gamma/vega/theta de una pata (0.0 si falta el
    dato, para que la suma neta no explote por un None)."""
    g = greeks.get(opt.streamer_symbol)
    if g is None:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    return {
        "delta": _to_float(getattr(g, "delta", None)) or 0.0,
        "gamma": _to_float(getattr(g, "gamma", None)) or 0.0,
        "vega": _to_float(getattr(g, "vega", None)) or 0.0,
        "theta": _to_float(getattr(g, "theta", None)) or 0.0,
    }


def calc_executable_debit(call_short_opt, put_short_opt, call_long_opt, put_long_opt,
                            quotes: dict, mid_debit: float) -> dict:
    """
    Executable Debit = ASK de las patas largas (lo que realmente
    pagarías) - BID de las patas cortas (lo que realmente recibirías).
    Escenario "peor caso realista" de ejecución, a diferencia del mid
    que puede ser inalcanzable en mercados poco líquidos — validado
    con evidencia real: HPQ mostró mid=$0.08 vs. natural=$0.60 en
    tastytrade (7.5x de diferencia).
    """
    sc_bid, _ = _leg_bid_ask(call_short_opt, quotes)
    sp_bid, _ = _leg_bid_ask(put_short_opt, quotes)
    _, lc_ask = _leg_bid_ask(call_long_opt, quotes)
    _, lp_ask = _leg_bid_ask(put_long_opt, quotes)

    missing = [v is None for v in (sc_bid, sp_bid, lc_ask, lp_ask)]
    if any(missing):
        # Sin datos de bid/ask reales para alguna pata — no podemos
        # calcular el ejecutable real. Devolvemos None explícito en
        # vez de inventar un número; el llamador decide qué hacer.
        return {"executable_debit": None, "slippage_pct": None, "data_complete": False}

    executable_debit = (lc_ask + lp_ask) - (sc_bid + sp_bid)
    slippage_pct = ((executable_debit - mid_debit) / mid_debit * 100
                     if mid_debit > 0 else None)

    return {
        "executable_debit": round(executable_debit, 2),
        "slippage_pct": round(slippage_pct, 1) if slippage_pct is not None else None,
        "data_complete": True,
    }


def calc_net_greeks(call_short_opt, put_short_opt, call_long_opt, put_long_opt,
                      greeks: dict) -> dict:
    """
    Greeks netos de la posición completa (largas - cortas), en
    términos POR CONTRATO (×100, convención estándar de mercado de
    opciones — un contrato = 100 acciones).
    """
    sc = _leg_greeks(call_short_opt, greeks)
    sp = _leg_greeks(put_short_opt, greeks)
    lc = _leg_greeks(call_long_opt, greeks)
    lp = _leg_greeks(put_long_opt, greeks)

    net_gamma = (lc["gamma"] + lp["gamma"]) - (sc["gamma"] + sp["gamma"])
    net_vega = (lc["vega"] + lp["vega"]) - (sc["vega"] + sp["vega"])
    net_theta = (lc["theta"] + lp["theta"]) - (sc["theta"] + sp["theta"])

    return {
        "net_gamma": round(net_gamma * 100, 4),
        "net_vega": round(net_vega * 100, 4),
        "net_theta": round(net_theta * 100, 4),
    }


def calc_leg_greeks_table(call_short_opt, put_short_opt, call_long_opt, put_long_opt,
                            greeks: dict) -> dict:
    """
    Greeks de CADA una de las 4 patas por separado (no netos) — para
    la tabla de "Legs" de la plataforma, con el mismo detalle que
    muestra la referencia (delta/gamma/theta/vega por pata, no solo
    la suma de la posición completa).

    Por contrato (×100), consistente con calc_net_greeks — misma
    convención que el resto de la plataforma (evitamos mezclar
    per-share y per-contrato entre funciones parecidas).
    """
    legs = {
        "short_call": _leg_greeks(call_short_opt, greeks),
        "short_put": _leg_greeks(put_short_opt, greeks),
        "long_call": _leg_greeks(call_long_opt, greeks),
        "long_put": _leg_greeks(put_long_opt, greeks),
    }
    out = {}
    for name, g in legs.items():
        out[f"{name}_delta"] = round(g["delta"] * 100, 2)
        out[f"{name}_gamma"] = round(g["gamma"] * 100, 4)
        out[f"{name}_theta"] = round(g["theta"] * 100, 3)
        out[f"{name}_vega"] = round(g["vega"] * 100, 3)
    return out


def calc_expected_move_coverage(price: float, put_strike: float, call_strike: float,
                                  iv_short_avg: float, short_dte: int) -> dict:
    """
    Expected move calculado con la IV REAL del vencimiento corto (no
    el proxy de volatilidad realizada que usa la selección de strikes
    en modo normal) — más preciso porque usa lo que el mercado está
    pricing hoy mismo. Strike coverage expresado en desvíos estándar
    (σ): cuántos "expected moves" de distancia tiene cada strike.
    """
    em = expected_move(price, iv_short_avg, short_dte)
    if em <= 0:
        return {"expected_move_pct": None, "put_coverage_sigma": None,
                "call_coverage_sigma": None}
    em_pct = em / price * 100
    put_sigma = (price - put_strike) / em
    call_sigma = (call_strike - price) / em
    return {
        "expected_move_pct": round(em_pct, 2),
        "put_coverage_sigma": round(put_sigma, 2),
        "call_coverage_sigma": round(call_sigma, 2),
    }


def calc_pnl_preearnings_snapshots(
    price: float, K_call_short: float, K_put_short: float,
    K_call_long: float, K_put_long: float,
    exp_short: date, exp_long: date, earnings_date,
    iv_short_call: float, iv_short_put: float,
    iv_long_call: float, iv_long_put: float,
    net_debit: float, offsets: list[int] = None,
) -> dict:
    """
    P/L en distintos snapshots ANTES del earnings — la simulación más
    relevante en la práctica, porque la estrategia real (según Adri)
    es cerrar manualmente antes del evento, nunca llegar al
    vencimiento del corto. A diferencia de calendar_pnl_at_short_expiry
    (que asume que el corto YA EXPIRÓ), acá AMBAS patas siguen vivas
    en cada snapshot y se valúan con Black-Scholes usando su tiempo
    restante real a esa fecha.

    LIMITACIÓN: usa la IV actual como constante — ver nota en config
    (PRE_EARNINGS_SNAPSHOT_DAYS). Es un piso conservador, no una
    predicción exacta, porque en la realidad la IV suele subir a
    medida que se acerca el evento.

    FIX: cada pata usa SU PROPIO strike (ver nota en
    calendar_pnl_at_short_expiry — misma corrección de fondo).
    """
    if offsets is None:
        offsets = PRE_EARNINGS_SNAPSHOT_DAYS

    try:
        er = pd.Timestamp(earnings_date).normalize()
        exp_s = pd.Timestamp(exp_short).normalize()
        exp_l = pd.Timestamp(exp_long).normalize()
    except Exception:
        return {}

    results = {}
    for offset in offsets:
        snapshot = er - timedelta(days=offset)
        short_remaining = (exp_s - snapshot).days
        long_remaining = (exp_l - snapshot).days
        if short_remaining <= 0 or long_remaining <= 0:
            results[offset] = None  # snapshot inválido (ya venció alguna pata)
            continue

        T_short = short_remaining / 365.0
        T_long = long_remaining / 365.0

        val_short_call = bs_price(price, K_call_short, T_short, iv_short_call, is_call=True)
        val_short_put = bs_price(price, K_put_short, T_short, iv_short_put, is_call=False)
        val_long_call = bs_price(price, K_call_long, T_long, iv_long_call, is_call=True)
        val_long_put = bs_price(price, K_put_long, T_long, iv_long_put, is_call=False)

        position_value = (val_long_call + val_long_put) - (val_short_call + val_short_put)
        results[offset] = round(position_value - net_debit, 2)

    return results


def calc_pnl_since_entry_snapshots(
    price: float, K_call_short: float, K_put_short: float,
    K_call_long: float, K_put_long: float,
    short_dte: int, long_dte: int,
    iv_short_call: float, iv_short_put: float,
    iv_long_call: float, iv_long_put: float,
    net_debit: float, days_list: list[int] = None,
) -> dict:
    """
    P&L simulado a N días DESDE LA ENTRADA (no desde el earnings) —
    generalización pedida explícitamente para poder cerrar apenas la
    posición cubra gastos + algo de ganancia, sin importar el modo
    (Normal o Earnings) ni esperar a un umbral fijo de % ni a "un día
    antes del earnings". Ambas patas siguen vivas en cada snapshot
    (igual metodología que calc_pnl_preearnings_snapshots — Black-
    Scholes con el tiempo restante real de cada pata a esa fecha).

    Aplica a AMBOS modos — a diferencia de calc_pnl_preearnings_snapshots,
    que solo tiene sentido en modo earnings porque está anclado a la
    fecha del evento.

    Misma limitación de IV estática documentada en el resto del
    script: es un piso conservador, no una predicción exacta.
    """
    if days_list is None:
        days_list = DAYS_SINCE_ENTRY_SNAPSHOTS

    results = {}
    for day in days_list:
        short_remaining = short_dte - day
        long_remaining = long_dte - day
        if short_remaining <= 0 or long_remaining <= 0:
            results[day] = None  # ya venció alguna pata a ese día
            continue

        T_short = short_remaining / 365.0
        T_long = long_remaining / 365.0

        val_short_call = bs_price(price, K_call_short, T_short, iv_short_call, is_call=True)
        val_short_put = bs_price(price, K_put_short, T_short, iv_short_put, is_call=False)
        val_long_call = bs_price(price, K_call_long, T_long, iv_long_call, is_call=True)
        val_long_put = bs_price(price, K_put_long, T_long, iv_long_put, is_call=False)

        position_value = (val_long_call + val_long_put) - (val_short_call + val_short_put)
        results[day] = round(position_value - net_debit, 2)

    return results


def evaluate_double_calendar(ctx: dict, call_short_opt, put_short_opt,
                               call_long_opt, put_long_opt,
                               exp_short: date, exp_long: date,
                               quotes: dict, greeks: dict, summs: dict,
                               uw_gex_data: Optional[dict] = None,
                               uw_gex_short_data: Optional[dict] = None,
                               oi_balance_short_data: Optional[dict] = None,
                               uw_darkpool_data: Optional[dict] = None,
                               uw_oi_data: Optional[dict] = None,
                               uw_flow_data: Optional[dict] = None,
                               uw_net_premium_data: Optional[dict] = None,
                               uw_oi_change_data: Optional[dict] = None,
                               uw_flow_per_strike_data: Optional[dict] = None) -> Optional[dict]:
    """
    Evalúa un doble calendario completo (4 patas: short call, short
    put, long call, long put) usando quotes/greeks ya pre-cargados
    (mismo patrón que evaluate_spread() del scanner de credit spreads
    — recibe dicts, no toca el streamer directamente).
    """
    sc_mid, sc_iv, sc_theo = _leg_mid(call_short_opt, quotes, greeks)
    sp_mid, sp_iv, sp_theo = _leg_mid(put_short_opt, quotes, greeks)
    lc_mid, lc_iv, lc_theo = _leg_mid(call_long_opt, quotes, greeks)
    lp_mid, lp_iv, lp_theo = _leg_mid(put_long_opt, quotes, greeks)

    if any(v is None for v in [sc_mid, sp_mid, lc_mid, lp_mid]):
        missing = []
        if sc_mid is None: missing.append(f"short_call({call_short_opt.streamer_symbol})")
        if sp_mid is None: missing.append(f"short_put({put_short_opt.streamer_symbol})")
        if lc_mid is None: missing.append(f"long_call({call_long_opt.streamer_symbol})")
        if lp_mid is None: missing.append(f"long_put({put_long_opt.streamer_symbol})")
        log.info(f"  {ctx['symbol']}: sin datos de mercado en: {', '.join(missing)}")
        return _reject("no_market_data")

    n_theo_legs = sum([bool(sc_theo), bool(sp_theo), bool(lc_theo), bool(lp_theo)])
    if n_theo_legs > MAX_THEO_LEGS_ALLOWED:
        return _reject(f"too_many_theo_legs({n_theo_legs}/4)")
    is_theo = n_theo_legs > 0

    # OI de las patas cortas (las más importantes para liquidez de entrada/salida)
    sc_summ = summs.get(call_short_opt.streamer_symbol)
    sp_summ = summs.get(put_short_opt.streamer_symbol)
    sc_oi = int(getattr(sc_summ, "open_interest", 0) or 0) if sc_summ else 0
    sp_oi = int(getattr(sp_summ, "open_interest", 0) or 0) if sp_summ else 0
    if sc_oi < MIN_OPEN_INTEREST or sp_oi < MIN_OPEN_INTEREST:
        log.info(f"  {ctx['symbol']}: OI insuficiente — "
                  f"short_call_oi={sc_oi} short_put_oi={sp_oi} (mínimo {MIN_OPEN_INTEREST})")
        return _reject("low_oi_short_legs")

    net_debit = (lc_mid + lp_mid) - (sc_mid + sp_mid)
    if net_debit <= 0:
        log.info(f"  {ctx['symbol']}: débito neto <= 0 (${net_debit:.2f})")
        return _reject("zero_or_negative_debit")

    today = pd.Timestamp.today().normalize()
    short_dte = int((pd.Timestamp(exp_short) - today).days)
    long_dte = int((pd.Timestamp(exp_long) - today).days)
    if short_dte <= 0 or long_dte <= short_dte:
        log.info(f"  {ctx['symbol']}: combinación de DTE inválida "
                  f"(short={short_dte} long={long_dte})")
        return _reject("invalid_dte_combo")

    # IV de las patas LARGAS — son las que se valúan a futuro con B-S
    if lc_iv is None or lp_iv is None or lc_iv <= 0 or lp_iv <= 0:
        log.info(f"  {ctx['symbol']}: IV de patas largas faltante o inválida "
                  f"(long_call_iv={lc_iv} long_put_iv={lp_iv})")
        return _reject("missing_long_leg_iv")

    K_call_short = float(call_short_opt.strike_price)
    K_put_short = float(put_short_opt.strike_price)
    K_call_long = float(call_long_opt.strike_price)
    K_put_long = float(put_long_opt.strike_price)
    is_diagonal = (K_call_short != K_call_long) or (K_put_short != K_put_long)
    if is_diagonal:
        log.info(f"  {ctx['symbol']}: es un DIAGONAL, no calendario puro — "
                  f"call corto={K_call_short} vs largo={K_call_long} · "
                  f"put corto={K_put_short} vs largo={K_put_long}")
    price = ctx["price"]

    curve = build_calendar_pnl_curve(
        price=price, K_call_short=K_call_short, K_put_short=K_put_short,
        K_call_long=K_call_long, K_put_long=K_put_long,
        short_dte=short_dte, long_dte=long_dte,
        iv_long_call=lc_iv, iv_long_put=lp_iv,
        net_debit=net_debit,
    )
    analysis = analyze_pnl_curve(curve, price, net_debit)

    # Term structure (IV corto vs. IV largo) — usamos el promedio call/put
    # de cada lado para tener un solo número comparable
    iv_short_avg = (sc_iv + sp_iv) / 2 if sc_iv and sp_iv else None
    iv_long_avg = (lc_iv + lp_iv) / 2
    ts_slope = term_structure_slope(iv_short_avg, iv_long_avg)

    # ================================================================
    # MEJORAS V2 (según propuesta revisada — Prioridad 1)
    # ================================================================

    # 1. Executable Debit — validado con evidencia real HPQ (mid $0.08
    #    vs. natural $0.60 en tastytrade). Descarta si el slippage
    #    estimado supera MAX_SLIPPAGE_PCT_DISCARD.
    exec_info = calc_executable_debit(call_short_opt, put_short_opt,
                                        call_long_opt, put_long_opt,
                                        quotes, net_debit)
    if exec_info["data_complete"] and exec_info["slippage_pct"] is not None:
        if exec_info["slippage_pct"] > MAX_SLIPPAGE_PCT_DISCARD:
            log.info(f"  {ctx['symbol']}: slippage {exec_info['slippage_pct']:.1f}% "
                      f"supera el máximo ({MAX_SLIPPAGE_PCT_DISCARD}%) — "
                      f"mid=${net_debit:.2f} ejecutable=${exec_info['executable_debit']:.2f}")
            return _reject("slippage_too_high")

    # 2. Greeks netos de la posición (dato ya fetcheado, casi gratis)
    net_greeks = calc_net_greeks(call_short_opt, put_short_opt,
                                   call_long_opt, put_long_opt, greeks)

    # 2b. Greeks por pata individual — para la tabla de "Legs" de la
    #     plataforma (pedido del usuario, mismo dato ya fetcheado).
    leg_greeks = calc_leg_greeks_table(call_short_opt, put_short_opt,
                                         call_long_opt, put_long_opt, greeks)

    # 3. Expected move con IV REAL del corto (no el proxy de vol
    #    realizada que se usa solo para ubicar strikes en modo normal)
    # Usa los strikes CORTOS (K_put_short/K_call_short) — es la
    # cobertura relevante para el vencimiento que expira primero.
    move_info = calc_expected_move_coverage(
        price, K_put_short, K_call_short, iv_short_avg or iv_long_avg, short_dte)

    # FIX (hallazgo real, caso BKR): rechazar si el redondeo al strike
    # disponible más cercano dejó a CUALQUIERA de los dos lados con
    # muy poco colchón real, aunque el objetivo original fuera
    # simétrico. Solo aplica a modo "normal" — en modo "earnings" los
    # strikes se eligen por delta real, no por este mismo mecanismo de
    # expected move + redondeo, así que no está expuesto al mismo
    # problema.
    if ctx.get("mode") != "earnings":
        put_cov = move_info.get("put_coverage_sigma")
        call_cov = move_info.get("call_coverage_sigma")
        if (put_cov is not None and put_cov < MIN_STRIKE_COVERAGE_SIGMA) or \
           (call_cov is not None and call_cov < MIN_STRIKE_COVERAGE_SIGMA):
            return _reject(f"strike_rounding_too_asymmetric(put={put_cov}σ,call={call_cov}σ)")

    # 4. P/L pre-earnings — solo tiene sentido en modo earnings
    pnl_pre = {}
    if ctx.get("mode") == "earnings" and ctx.get("earnings_date") is not None:
        pnl_pre = calc_pnl_preearnings_snapshots(
            price, K_call_short, K_put_short, K_call_long, K_put_long,
            exp_short, exp_long, ctx["earnings_date"],
            sc_iv or iv_long_avg, sp_iv or iv_long_avg, lc_iv, lp_iv, net_debit)

    # 5. P/L por días desde la entrada — AMBOS modos. Validado contra
    #    OptionStrat en COST: día 8 real mostró +$228.52 con precio
    #    prácticamente quieto. Permite decidir cerrar por ganancia
    #    temprana sin esperar un umbral fijo ni acercarse al earnings.
    pnl_since_entry = calc_pnl_since_entry_snapshots(
        price, K_call_short, K_put_short, K_call_long, K_put_long,
        short_dte, long_dte,
        sc_iv or iv_long_avg, sp_iv or iv_long_avg, lc_iv, lp_iv, net_debit)

    # Position sizing: usamos la pérdida MÁS NEGATIVA de toda la curva
    # (no solo el net_debit) porque, como confirmamos en el test, el
    # riesgo real puede superar levemente el débito en movimientos
    # extremos (descuento de valor temporal en puts europeos profundos
    # ITM). Es una estimación más conservadora y correcta.
    worst_case_loss_per_share = abs(min(v for _, v in curve))
    max_loss_per_contract = worst_case_loss_per_share * 100
    risk_budget = ACCOUNT_SIZE_USD * MAX_RISK_PCT
    pos_cap = ACCOUNT_SIZE_USD * MAX_POSITION_PCT
    contracts_by_risk = math.floor(risk_budget / max_loss_per_contract) if max_loss_per_contract > 0 else 0
    contracts_by_cap = math.floor(pos_cap / max_loss_per_contract) if max_loss_per_contract > 0 else 0
    contracts = max(0, min(contracts_by_risk, contracts_by_cap))

    # --------- UW: confirmación de mercado para el strike LARGO ---------
    # Pedido original (pendiente de la sesión anterior): "¿el mercado
    # también mira el strike largo elegido, o es una lectura puramente
    # técnica nuestra?" — usa GEX (¿actúa como imán o techo?), dark
    # pool (¿hay compras institucionales grandes cerca?), y OI real
    # (sin confirmar todavía, ver docstring de calc_oi_per_strike_uw).
    # Es informativo — no cambia el Score ni descarta nada, mismo
    # criterio conservador que se usó al integrar esto por primera vez
    # en el scanner de credit spreads: mostrar antes de accionar.
    sym = ctx["symbol"]
    gex_info = (uw_gex_data or {}).get("by_ticker", {}).get(sym, {})
    dp_info = (uw_darkpool_data or {}).get("by_ticker", {}).get(sym, {})
    oi_info = (uw_oi_data or {}).get("by_ticker", {}).get(sym, {})

    uw_call_wall = gex_info.get("call_wall")
    uw_put_wall = gex_info.get("put_wall")
    uw_gamma_flip = gex_info.get("gamma_flip")

    # NUEVO: mismo dato pero acotado a la expiración CORTA exacta
    # (calc_gex_uw_by_expiry) — esto sí responde la pregunta relevante
    # para la tesis de quietud del calendario: "¿el mercado de
    # opciones de ESTE vencimiento puntual empuja a un pin o a un
    # escape?". Se compara contra los strikes CORTOS (no los largos),
    # a diferencia del chequeo de arriba, porque son los que vencen
    # primero y a los que este GEX corresponde. Informativo por ahora
    # — no descarta ni cambia el Score (ver docstring de la función).
    gex_short_info = (uw_gex_short_data or {}).get("by_ticker", {}).get(sym, {})
    uw_call_wall_short = gex_short_info.get("call_wall")
    uw_put_wall_short = gex_short_info.get("put_wall")
    uw_gamma_flip_short = gex_short_info.get("gamma_flip")
    uw_net_gex_short = gex_short_info.get("net_gex")

    short_call_beyond_wall = (uw_call_wall_short is not None and
                                K_call_short >= uw_call_wall_short)
    short_put_beyond_wall = (uw_put_wall_short is not None and
                               K_put_short <= uw_put_wall_short)

    # NUEVO (pendiente #7): balance real de OI put/call de ESTA
    # expiración corta exacta (fuente: tastytrade, no UW — ver
    # calc_oi_balance_short_expiry). Informativo, no descarta.
    oi_balance_info = (oi_balance_short_data or {}).get(sym, {})
    # ¿El call/put largo (el "límite" del calendario) coincide con un
    # muro real? Tolerancia del 2%, mismo criterio que FlowStrikeAlignment
    # en el scanner de credit spreads.
    def _cerca(a, b, tol_pct=2.0):
        if a is None or b is None or a <= 0:
            return False
        return abs(a - b) <= a * (tol_pct / 100.0)

    long_call_near_wall = _cerca(K_call_long, uw_call_wall)
    long_put_near_wall = _cerca(K_put_long, uw_put_wall)

    dp_levels = dp_info.get("levels", [])
    dp_near_call_long = any(_cerca(K_call_long, lvl["price"]) for lvl in dp_levels)
    dp_near_put_long = any(_cerca(K_put_long, lvl["price"]) for lvl in dp_levels)

    oi_by_strike = oi_info.get("oi_by_strike", {})
    oi_near_call_long = oi_near_strike(oi_by_strike, K_call_long) if oi_by_strike else None
    oi_near_put_long = oi_near_strike(oi_by_strike, K_put_long) if oi_by_strike else None

    # Flow, net premium y OI change — a nivel TICKER (no por strike),
    # informativos. Flow por strike sí es por strike — se compara
    # contra los strikes CORTOS (no largos), ver docstring de
    # calc_flow_per_strike_uw más arriba.
    flow_info = (uw_flow_data or {}).get("flow_by_ticker", {}).get(sym, {})
    net_prem_info = (uw_net_premium_data or {}).get("flow_by_ticker", {}).get(sym, {})
    oi_change_info = (uw_oi_change_data or {}).get("by_ticker", {}).get(sym, {})
    flow_strike_info = (uw_flow_per_strike_data or {}).get("by_ticker", {}).get(sym, {})

    top_bullish_strike = flow_strike_info.get("top_bullish_strike")
    top_bearish_strike = flow_strike_info.get("top_bearish_strike")
    flow_near_call_short = _cerca(K_call_short, top_bullish_strike) or _cerca(K_call_short, top_bearish_strike)
    flow_near_put_short = _cerca(K_put_short, top_bullish_strike) or _cerca(K_put_short, top_bearish_strike)

    return {
        "Symbol": ctx["symbol"], "Sector": ctx["sector"], "Price": price,
        "IVR": round(ctx["ivr"], 1) if ctx["ivr"] else None,
        "ADX": ctx["adx"], "RealizedVolPctl": ctx["realized_vol_percentile"],
        "ATR_Pct": ctx.get("atr_pct"),
        "ADV_$M": ctx["adv_$M"], "MCap_$B": ctx["market_cap_$B"],
        "Beta": round(ctx["beta"], 2) if ctx["beta"] else None,
        "DaysToER": ctx["days_to_earnings"],
        "EarningsDate": (pd.Timestamp(ctx["earnings_date"]).date().isoformat()
                         if ctx.get("earnings_date") is not None else None),
        "Mode": ctx.get("mode", "normal"),
        "SuggestedExitDate": (
            (pd.Timestamp(ctx["earnings_date"]).date() - timedelta(days=1)).isoformat()
            if ctx.get("mode") == "earnings" and ctx.get("earnings_date") is not None
            else None
        ),
        "TakeProfitRange": f"${net_debit*TAKE_PROFIT_PCT_MIN/100:.2f}-${net_debit*TAKE_PROFIT_PCT_MAX/100:.2f}"
                            if ctx.get("mode") == "earnings" else None,
        "ShortDTE": short_dte, "LongDTE": long_dte,
        "ShortExp": exp_short.isoformat() if hasattr(exp_short, "isoformat") else str(exp_short),
        "LongExp": exp_long.isoformat() if hasattr(exp_long, "isoformat") else str(exp_long),
        "CallStrike": K_call_short, "PutStrike": K_put_short,
        "LongCallStrike": K_call_long, "LongPutStrike": K_put_long,
        "IsDiagonal": is_diagonal,
        "ShortCallMid": round(sc_mid, 2), "ShortPutMid": round(sp_mid, 2),
        "LongCallMid": round(lc_mid, 2), "LongPutMid": round(lp_mid, 2),
        "NetDebit": round(net_debit, 2),
        "ExecutableDebit": exec_info["executable_debit"],
        "SlippagePct": exec_info["slippage_pct"],
        "ShortCallOI": sc_oi, "ShortPutOI": sp_oi,
        "IVShortAvg": round(iv_short_avg, 3) if iv_short_avg else None,
        "IVLongAvg": round(iv_long_avg, 3),
        # IV individual por pata (además del promedio) — necesarias para
        # reproducir la forma real de "doble joroba" del payoff fuera de
        # este script (ej. en la plataforma interactiva). Promediar
        # call+put aplana la curva cuando hay skew entre ambas, que es
        # justo el bug que encontramos comparando contra el reporte.
        "IVShortCall": round(sc_iv, 3) if sc_iv else None,
        "IVShortPut": round(sp_iv, 3) if sp_iv else None,
        "IVLongCall": round(lc_iv, 3) if lc_iv else None,
        "IVLongPut": round(lp_iv, 3) if lp_iv else None,
        "TermStructureSlope": ts_slope,
        "NetGamma": net_greeks["net_gamma"],
        "NetVega": net_greeks["net_vega"],
        "NetTheta": net_greeks["net_theta"],
        # Greeks por pata individual, para la tabla de Legs de la
        # plataforma — 16 campos (delta/gamma/theta/vega × 4 patas).
        "ShortCallDelta": leg_greeks["short_call_delta"],
        "ShortCallGamma": leg_greeks["short_call_gamma"],
        "ShortCallTheta": leg_greeks["short_call_theta"],
        "ShortCallVega": leg_greeks["short_call_vega"],
        "ShortPutDelta": leg_greeks["short_put_delta"],
        "ShortPutGamma": leg_greeks["short_put_gamma"],
        "ShortPutTheta": leg_greeks["short_put_theta"],
        "ShortPutVega": leg_greeks["short_put_vega"],
        "LongCallDelta": leg_greeks["long_call_delta"],
        "LongCallGamma": leg_greeks["long_call_gamma"],
        "LongCallTheta": leg_greeks["long_call_theta"],
        "LongCallVega": leg_greeks["long_call_vega"],
        "LongPutDelta": leg_greeks["long_put_delta"],
        "LongPutGamma": leg_greeks["long_put_gamma"],
        "LongPutTheta": leg_greeks["long_put_theta"],
        "LongPutVega": leg_greeks["long_put_vega"],
        "ExpectedMovePct": move_info["expected_move_pct"],
        "PutCoverageSigma": move_info["put_coverage_sigma"],
        "CallCoverageSigma": move_info["call_coverage_sigma"],
        "PnLPreEarnings": pnl_pre,   # dict {offset_dias: pnl}
        "PnLSinceEntry": pnl_since_entry,   # dict {dia: pnl} — ambos modos
        "MaxProfitEstimate": analysis["max_profit_estimate"],
        "MaxProfitPrice": analysis["max_profit_price"],
        "PnLAtCurrentPrice": analysis["pnl_at_current_price"],
        "Breakevens": analysis["breakevens"],
        "ProfitZoneWidth": analysis["profit_zone_width"],
        "HasLosingValley": analysis["has_losing_valley"],
        "MinPnLInRange": analysis["min_pnl_in_range"],
        "ReturnOnDebitPct": analysis["return_on_debit_pct"],
        "WorstCaseLoss_$": round(max_loss_per_contract, 2),
        "SuggContracts": contracts,
        "MaxRisk_$": round(max_loss_per_contract * contracts, 2),
        "NetDebit_$": round(net_debit * 100 * contracts, 2),
        "PriceSource": "theoretical" if is_theo else "live_quote",
        "TheoLegsCount": n_theo_legs,
        # Flag para que puedas ver el grupo "0 teóricas" (más estricto)
        # separado del "1 teórica" (ya aceptado) SIN correr de nuevo ni
        # adivinar qué umbral es el correcto — ordená/filtrá por esta
        # columna en la tabla del reporte para comparar ambos grupos
        # con datos reales, en vez de discutir el número en abstracto.
        "AllLegsLive": n_theo_legs == 0,
        # ---- UW: confirmación de mercado para el strike largo (nuevo) ----
        "UW_CallWall": uw_call_wall, "UW_PutWall": uw_put_wall,
        "UW_GammaFlip": uw_gamma_flip,
        # ---- UW: mismo dato acotado a la expiración corta exacta
        # (nuevo, ver calc_gex_uw_by_expiry) — informativo, no descarta.
        "UW_CallWallShort": uw_call_wall_short, "UW_PutWallShort": uw_put_wall_short,
        "UW_GammaFlipShort": uw_gamma_flip_short, "UW_NetGexShort": uw_net_gex_short,
        "ShortCallBeyondWall": short_call_beyond_wall,
        "ShortPutBeyondWall": short_put_beyond_wall,
        # ---- OI real por expiración corta exacta (nuevo, tastytrade) ----
        "OIPutCallRatioShort": oi_balance_info.get("oi_put_call_ratio"),
        "OICallOITotalShort": oi_balance_info.get("call_oi_total"),
        "OIPutOITotalShort": oi_balance_info.get("put_oi_total"),
        "OICallWallStrikeShort": oi_balance_info.get("oi_call_wall_strike"),
        "OIPutWallStrikeShort": oi_balance_info.get("oi_put_wall_strike"),
        "LongCallNearWall": long_call_near_wall, "LongPutNearWall": long_put_near_wall,
        "DarkpoolNearCallLong": dp_near_call_long, "DarkpoolNearPutLong": dp_near_put_long,
        "DarkpoolBullish_M": dp_info.get("bullish_notional_M"),
        "DarkpoolBearish_M": dp_info.get("bearish_notional_M"),
        "OI_NearCallLong": oi_near_call_long, "OI_NearPutLong": oi_near_put_long,
        "FlowCallPremium_M": flow_info.get("call_premium_m"),
        "FlowPutPremium_M": flow_info.get("put_premium_m"),
        "NetCallPremium_M": net_prem_info.get("net_call_premium_m"),
        "NetPutPremium_M": net_prem_info.get("net_put_premium_m"),
        "BullishOI_New": oi_change_info.get("bullish_oi_new"),
        "BearishOI_New": oi_change_info.get("bearish_oi_new"),
        "TopBullishStrike": top_bullish_strike, "TopBearishStrike": top_bearish_strike,
        "FlowNearCallShort": flow_near_call_short, "FlowNearPutShort": flow_near_put_short,
        "PnLCurve": curve,
    }


# =====================================================================
# RANKING
# =====================================================================
#
# A diferencia del scanner de credit spreads (POP + Credit/Width),
# acá el score combina:
#   - Return on debit (cuánto podés ganar relativo a lo que arriesgás)
#   - Ancho de la zona de ganancia relativo al expected move (una zona
#     más ancha que el movimiento esperado da más margen de error)
#   - Term structure slope (positivo = favorable, corto más caro que
#     largo)
#   - Cercanía del precio actual a la zona de mayor ganancia (que el
#     spot ya esté cerca de donde el calendario gana más)
# =====================================================================

def rank_calendars(rows: list[dict], mode: str = "normal") -> pd.DataFrame:
    """
    mode: "normal" | "earnings" — el term structure pesa distinto
    según el modo. En Earnings es el mecanismo central de la
    estrategia (según la metodología de Adri: el edge viene de que
    el corto esté más caro que el largo); en Normal es un plus, no
    el motor de la tesis (que es rango + IV barata). Todavía sin
    backtest que valide estos pesos — ajustar cuando haya datos
    reales de resultados.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Return on debit normalizado (cap a 200% para que un outlier no domine)
    rod = df["ReturnOnDebitPct"].fillna(0).clip(lower=0, upper=200) / 200.0

    # Ancho de zona de ganancia vs. distancia entre strikes (proxy de margen)
    strike_width = (df["CallStrike"] - df["PutStrike"]).clip(lower=1)
    zone_ratio = (df["ProfitZoneWidth"].fillna(0) / strike_width).clip(upper=2.0) / 2.0

    # FIX (detectado comparando HOOD/AVGO contra OptionStrat): cuando
    # hay "valle perdedor" en el medio (4 breakevens en vez de 2, el
    # ancho entre el primero y el último NO es una zona continua de
    # ganancia), penalizamos zone_ratio — no es tan bueno como
    # aparenta el ancho crudo.
    if "HasLosingValley" in df.columns:
        valley_penalty = df["HasLosingValley"].fillna(False).map({True: 0.6, False: 1.0})
        zone_ratio = zone_ratio * valley_penalty

    # Term structure: positivo es mejor. Penalización más agresiva en
    # modo earnings (rango de normalización más angosto -> un mismo
    # valor negativo castiga más el Score que en modo normal).
    ts = df["TermStructureSlope"].fillna(0)
    if mode == "earnings":
        ts_clipped = ts.clip(lower=-0.03, upper=0.08)   # rango más angosto = más sensible
        ts_norm = (ts_clipped + 0.03) / 0.11
        ts_weight, rod_weight, zone_weight, pnl_weight = 0.30, 0.30, 0.20, 0.20
    else:
        ts_clipped = ts.clip(lower=-0.05, upper=0.10)
        ts_norm = (ts_clipped + 0.05) / 0.15
        ts_weight, rod_weight, zone_weight, pnl_weight = 0.20, 0.35, 0.25, 0.20

    # P&L en precio actual, normalizado por el débito neto (cuánto ya
    # estás "ganando" si el precio no se mueve nada)
    pnl_now_norm = (df["PnLAtCurrentPrice"] / df["NetDebit"].replace(0, np.nan)
                    ).clip(lower=-1, upper=1).fillna(0)
    pnl_now_norm = (pnl_now_norm + 1) / 2  # a [0,1]

    df["Score"] = (
        rod * rod_weight +
        zone_ratio * zone_weight +
        ts_norm * ts_weight +
        pnl_now_norm * pnl_weight
    ).round(3)

    # Penalización por slippage (mejora V2 — validado con HPQ). Entre
    # SLIPPAGE_PENALTY_PCT (15%) y MAX_SLIPPAGE_PCT_DISCARD (25%, ya
    # descartado antes en evaluate_double_calendar) se aplica una
    # penalización lineal al Score — un candidato con Return/Debit
    # espectacular pero mal ejecutable ya no debería rankear arriba
    # solo por el número teórico.
    if "SlippagePct" in df.columns:
        slippage = df["SlippagePct"].fillna(0).clip(lower=0)
        penalty = ((slippage - SLIPPAGE_PENALTY_PCT) / 
                   (MAX_SLIPPAGE_PCT_DISCARD - SLIPPAGE_PENALTY_PCT)
                   ).clip(lower=0, upper=1)
        slippage_mult = 1.0 - (penalty * 0.5)  # hasta -50% de Score en el peor caso no descartado
        df["Score"] = (df["Score"] * slippage_mult).round(3)

    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


# =====================================================================
# GRÁFICO DE LA CURVA DE P&L (SVG) — reemplaza el diagrama de payoff
# lineal del scanner de credit spreads, porque acá la forma es una
# "colina doble", no una línea recta con un solo breakeven.
# =====================================================================

def make_calendar_pnl_svg(curve: list[tuple[float, float]], current_price: float,
                            call_strike: float, put_strike: float,
                            breakevens: list[float],
                            w: int = 420, h: int = 180) -> str:
    prices = [p for p, _ in curve]
    pnls = [v for _, v in curve]
    x_min, x_max = min(prices), max(prices)
    y_min, y_max = min(pnls) * 1.15, max(pnls) * 1.15
    if y_min == y_max:
        y_min, y_max = -1, 1
    pad_l, pad_r, pad_t, pad_b = 55, 15, 15, 28
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    def X(x): return pad_l + (x - x_min) / (x_max - x_min) * plot_w
    def Y(y): return pad_t + (y_max - y) / (y_max - y_min) * plot_h

    pts_px = [(X(p), Y(v)) for p, v in curve]
    line_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts_px)
    z = Y(0)

    # Área bajo la curva, coloreada por signo (simplificado: dos paths,
    # uno para zona positiva aproximada con clip visual vía polígono)
    fill_pts = pts_px + [(pts_px[-1][0], z), (pts_px[0][0], z)]
    fill_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in fill_pts) + " Z"

    def vline(x, color, label, dash=True):
        d = ' stroke-dasharray="3,3"' if dash else ''
        return (f'<line x1="{X(x):.1f}" y1="{pad_t}" x2="{X(x):.1f}" y2="{pad_t+plot_h}" '
                f'stroke="{color}" stroke-width="1.2"{d}/>'
                f'<text x="{X(x):.1f}" y="{pad_t+plot_h+14}" fill="{color}" font-size="9" '
                f'text-anchor="middle" font-family="monospace">{label}</text>')

    svg = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#0f1422" stroke="#2d3748"/>')
    svg.append(f'<line x1="{pad_l}" y1="{z:.1f}" x2="{pad_l+plot_w}" y2="{z:.1f}" stroke="#475569" stroke-width="0.7" stroke-dasharray="2,2"/>')
    svg.append(f'<text x="{pad_l-6}" y="{z+3:.1f}" fill="#94a3b8" font-size="9" text-anchor="end" font-family="monospace">0</text>')
    svg.append(f'<path d="{fill_d}" fill="#10b981" fill-opacity="0.12"/>')
    svg.append(f'<path d="{line_d}" stroke="#e2e8f0" stroke-width="1.8" fill="none"/>')
    svg.append(vline(put_strike, "#ef4444", f"P{put_strike:g}"))
    svg.append(vline(call_strike, "#f59e0b", f"C{call_strike:g}"))
    svg.append(vline(current_price, "#a78bfa", f"${current_price:.0f}", dash=True))
    for i, be in enumerate(breakevens[:2]):
        svg.append(vline(be, "#3b82f6", f"BE{i+1}"))
    svg.append('</svg>')
    return "".join(svg)


# =====================================================================
# HTML REPORT
# =====================================================================

CSS = """
body { background:#0a0e1a; color:#e2e8f0; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
       margin:0; padding:24px; line-height:1.5; }
h1,h2,h3 { margin:0 0 8px 0; font-weight:600; }
h1 { font-size:24px; color:#f8fafc; }
h2 { font-size:18px; color:#f1f5f9; margin-top:24px; }
.small { color:#94a3b8; font-size:12px; }
.panel { background:#1a1f2e; border:1px solid #2d3748; border-radius:10px; padding:16px; margin:12px 0; }
table { width:100%; border-collapse:collapse; margin-top:12px; font-size:12px; }
th { background:#0f1422; padding:8px 10px; text-align:right; color:#94a3b8; font-weight:500;
     border-bottom:1px solid #2d3748; cursor:pointer; user-select:none; position:sticky; top:0; }
th:first-child, td:first-child { text-align:left; }
th:hover { color:#e2e8f0; }
td { padding:8px 10px; text-align:right; border-bottom:1px solid #1e293b; font-family:monospace; }
tr:hover td { background:#0f1422; }
.candidate-card { display:grid; grid-template-columns:auto 1fr; gap:18px; padding:14px;
                  border-bottom:1px solid #2d3748; align-items:center; }
.candidate-meta { font-size:12px; }
.candidate-meta b { color:#f1f5f9; font-size:14px; }
.metric { display:inline-block; margin-right:14px; color:#94a3b8; font-family:monospace; }
.metric b { color:#e2e8f0; }
.theo { background:#7c2d12; color:#fdba74; font-size:9px; padding:1px 5px; border-radius:3px; margin-left:6px; }
.events-line { margin-top:8px; padding:6px 10px; background:#0f1422; border-radius:6px; font-size:12px;
               font-family:monospace; border:1px solid #1e293b; }
.event-ok { color:#86efac; }
.event-na { color:#94a3b8; }
.event-warn { color:#fca5a5; font-weight:600; }
.summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-top:8px; }
.summary-grid .item { background:#0f1422; padding:10px 12px; border-radius:6px; border:1px solid #2d3748; }
.summary-grid .item .label { color:#94a3b8; font-size:11px; text-transform:uppercase; }
.summary-grid .item .val { color:#f8fafc; font-size:18px; font-weight:600; font-family:monospace; margin-top:3px; }
.reject-list { font-family:monospace; font-size:12px; color:#94a3b8; }
.reject-list li { padding:3px 0; }
"""

JS_SORT = """
function sortTable(table, col) {
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const dir = table.dataset.sortDir === 'asc' && table.dataset.sortCol == col ? 'desc' : 'asc';
  rows.sort((a,b) => {
    let av = a.cells[col].dataset.v ?? a.cells[col].textContent.trim();
    let bv = b.cells[col].dataset.v ?? b.cells[col].textContent.trim();
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return dir==='asc' ? an-bn : bn-an;
    return dir==='asc' ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(r => tbody.appendChild(r));
  table.dataset.sortDir = dir;
  table.dataset.sortCol = col;
}
document.querySelectorAll('table.sortable').forEach(t => {
  t.querySelectorAll('th').forEach((th,i) => th.addEventListener('click',()=>sortTable(t,i)));
});
"""


def _render_candidate_card(row: dict) -> str:
    svg = make_calendar_pnl_svg(
        row["PnLCurve"], row["Price"], row["CallStrike"], row["PutStrike"],
        row["Breakevens"] or [])
    n_theo = row.get("TheoLegsCount", 0)
    theo_tag = (f'<span class="theo" title="1 de 4 patas usó precio calculado, no cotización real — '
                f'ya pasó el filtro automático (máximo {MAX_THEO_LEGS_ALLOWED} permitida), no hace '
                f'falta verificar a mano">THEO (1/4)</span>' if n_theo == 1 else "")
    diagonal_tag = ('<span class="theo" style="background:#7c3aed;color:#ede9fe">DIAGONAL '
                    f'(largo: {row.get("LongPutStrike","?"):g}P/{row.get("LongCallStrike","?"):g}C)</span>'
                    if row.get("IsDiagonal") else "")

    er_date = row.get("EarningsDate")
    dte_er = row.get("DaysToER")
    if er_date and dte_er is not None and 0 <= dte_er <= row["ShortDTE"]:
        er_line = f'<span class="event-ok">📅 Earnings {er_date} DENTRO del corto ({dte_er}d) ✓ crush objetivo</span>'
    elif er_date:
        er_line = f'<span class="event-na">📅 Earnings {er_date} (fuera del corto, {dte_er}d)</span>'
    else:
        er_line = '<span class="event-na">📅 Sin earnings verificado</span>'

    ts = row.get("TermStructureSlope")
    ts_line = (f'<span class="event-ok">📐 Term structure: +{ts:.3f} (corto más caro, favorable)</span>'
               if ts is not None and ts > 0 else
               f'<span class="event-warn">📐 Term structure: {ts:.3f} (largo más caro, desfavorable)</span>'
               if ts is not None else
               '<span class="event-na">📐 Term structure: sin datos</span>')

    # --- Executable Debit (mejora V2) ---
    exec_debit = row.get("ExecutableDebit")
    slippage = row.get("SlippagePct")
    if exec_debit is not None and slippage is not None:
        cls = "event-ok" if slippage <= SLIPPAGE_PENALTY_PCT else "event-warn"
        exec_line = (f'<span class="{cls}">💸 Mid ${row["NetDebit"]:.2f} → '
                     f'Ejecutable ${exec_debit:.2f} (slippage {slippage:+.0f}%)</span>')
    else:
        exec_line = '<span class="event-na">💸 Executable debit: sin datos bid/ask completos</span>'

    # --- Greeks netos (mejora V2) ---
    ng, nv, nt = row.get("NetGamma"), row.get("NetVega"), row.get("NetTheta")
    greeks_line = (f'<span class="event-na">Γ {ng:+.2f} · V {nv:+.2f} · Θ {nt:+.2f} (por contrato)</span>'
                   if ng is not None else '')

    # --- Expected move / strike coverage (mejora V2) ---
    em_pct = row.get("ExpectedMovePct")
    put_sig, call_sig = row.get("PutCoverageSigma"), row.get("CallCoverageSigma")
    move_line = ""
    if em_pct is not None:
        move_line = (f'<span class="event-na">📏 Expected move: ±{em_pct:.1f}% · '
                     f'Put a {put_sig:.2f}σ · Call a {call_sig:.2f}σ</span>')

    # --- P/L pre-earnings (mejora V2, solo modo earnings) ---
    pnl_pre_line = ""
    pnl_pre = row.get("PnLPreEarnings")
    if pnl_pre:
        parts = []
        for offset in sorted(pnl_pre.keys(), reverse=True):
            v = pnl_pre[offset]
            if v is not None:
                cls = "event-ok" if v >= 0 else "event-warn"
                parts.append(f'<span class="{cls}">E-{offset}d: ${v:+.2f}</span>')
        if parts:
            pnl_pre_line = ('<div class="events-line" style="margin-top:6px">'
                            '📈 P/L pre-earnings (IV estática, piso conservador): '
                            + " · ".join(parts) + '</div>')

    # --- P/L por días desde la entrada (AMBOS modos) ---
    pnl_since_line = ""
    pnl_since = row.get("PnLSinceEntry")
    if pnl_since:
        parts = []
        for day in sorted(pnl_since.keys()):
            v = pnl_since[day]
            if v is not None:
                cls = "event-ok" if v >= 0 else "event-warn"
                parts.append(f'<span class="{cls}">Día {day}: ${v:+.2f}</span>')
        if parts:
            net_debit_val = row.get("NetDebit", 0)
            pnl_since_line = ('<div class="events-line" style="margin-top:6px">'
                              f'⏱️ P/L desde entrada (sobre débito ${net_debit_val:.2f}, '
                              'IV estática, piso conservador): '
                              + " · ".join(parts) + '</div>')

    be_str = " / ".join(f"${b:g}" for b in (row["Breakevens"] or []))
    valley_warning = ""
    if row.get("HasLosingValley"):
        valley_warning = (f'<div class="events-line" style="margin-top:6px;background:#3f1d1d">'
                          f'<span class="event-warn">⚠️ Zona de ganancia NO continua — hay un '
                          f'valle perdedor entre los breakevens internos (mínimo P&L en el rango: '
                          f'${row.get("MinPnLInRange", 0):.2f}). El ancho mostrado NO es todo '
                          f'zona segura.</span></div>')

    # --- UW: confirmación de mercado para el strike LARGO (nuevo) ---
    # pd.notna() en vez de chequeos simples "if x:" o "if x is not
    # None:" a propósito — un None se vuelve NaN al pasar por un
    # DataFrame de pandas (rank_calendars), y ninguno de esos dos
    # chequeos simples atrapa NaN (lección real de varios bugs
    # encontrados hoy en el scanner de credit spreads).
    uw_lines = []
    call_wall, put_wall = row.get("UW_CallWall"), row.get("UW_PutWall")
    if pd.notna(call_wall) or pd.notna(put_wall):
        cw_str = f"${call_wall:g}" if pd.notna(call_wall) else "N/A"
        pw_str = f"${put_wall:g}" if pd.notna(put_wall) else "N/A"
        call_match = " · call largo COINCIDE con Call Wall" if row.get("LongCallNearWall") else ""
        put_match = " · put largo COINCIDE con Put Wall" if row.get("LongPutNearWall") else ""
        uw_lines.append(f'<span class="event-na">🧱 [UW] Muros: CW={cw_str} · '
                         f'PW={pw_str}{call_match}{put_match}</span>')
    dp_bull, dp_bear = row.get("DarkpoolBullish_M"), row.get("DarkpoolBearish_M")
    if pd.notna(dp_bull) or pd.notna(dp_bear):
        dp_match = ""
        if row.get("DarkpoolNearCallLong"):
            dp_match += " · dark pool cerca del call largo"
        if row.get("DarkpoolNearPutLong"):
            dp_match += " · dark pool cerca del put largo"
        uw_lines.append(f'<span class="event-na">🐋 [UW] Dark pool: bullish ${dp_bull or 0:.1f}M · '
                         f'bearish ${dp_bear or 0:.1f}M{dp_match}</span>')
    uw_line = ('<div class="events-line" style="margin-top:6px">' + " ".join(uw_lines) + '</div>'
               if uw_lines else "")

    # --- NUEVO (hoy): GEX y OI, pero de la expiración CORTA exacta —
    # a diferencia del bloque de arriba (uw_lines), que es ticker-wide
    # (todas las expiraciones mezcladas) y compara contra el strike
    # LARGO. Esto compara contra los strikes CORTOS, que son los que
    # importan para la tesis de quietud del vencimiento que expira
    # primero. Antes de este cambio, estos 11 campos existían en los
    # datos pero solo se veían en la tabla completa, no acá en la card.
    short_lines = []
    cw_s, pw_s = row.get("UW_CallWallShort"), row.get("UW_PutWallShort")
    if pd.notna(cw_s) or pd.notna(pw_s):
        cw_s_str = f"${cw_s:g}" if pd.notna(cw_s) else "N/A"
        pw_s_str = f"${pw_s:g}" if pd.notna(pw_s) else "N/A"
        alerta_call = " · ⚠️ call corto YA PASÓ el Call Wall" if row.get("ShortCallBeyondWall") else ""
        alerta_put = " · ⚠️ put corto YA PASÓ el Put Wall" if row.get("ShortPutBeyondWall") else ""
        cls_short_gex = "event-warn" if (row.get("ShortCallBeyondWall") or row.get("ShortPutBeyondWall")) else "event-na"
        short_lines.append(f'<span class="{cls_short_gex}">🧱⏱️ [UW] Muros (expiración corta exacta): '
                            f'CW={cw_s_str} · PW={pw_s_str}{alerta_call}{alerta_put}</span>')
    oi_ratio = row.get("OIPutCallRatioShort")
    oi_call_tot, oi_put_tot = row.get("OICallOITotalShort"), row.get("OIPutOITotalShort")
    if pd.notna(oi_ratio) or pd.notna(oi_call_tot) or pd.notna(oi_put_tot):
        ratio_str = f"{oi_ratio:.2f}" if pd.notna(oi_ratio) else "N/A"
        call_tot_str = f"{oi_call_tot:,.0f}" if pd.notna(oi_call_tot) else "N/A"
        put_tot_str = f"{oi_put_tot:,.0f}" if pd.notna(oi_put_tot) else "N/A"
        short_lines.append(f'<span class="event-na">📊⏱️ [tastytrade] OI expiración corta: '
                            f'ratio put/call={ratio_str} · call OI={call_tot_str} · '
                            f'put OI={put_tot_str}</span>')
    short_exp_line = ('<div class="events-line" style="margin-top:6px">' +
                       " ".join(short_lines) + '</div>' if short_lines else "")

    # --- UW: flow/net premium/OI change (nivel ticker) + flow por
    # strike (nivel strike CORTO) — agregado hoy ---
    uw_flow_lines = []
    flow_call, flow_put = row.get("FlowCallPremium_M"), row.get("FlowPutPremium_M")
    net_call, net_put = row.get("NetCallPremium_M"), row.get("NetPutPremium_M")
    if pd.notna(flow_call) or pd.notna(flow_put):
        uw_flow_lines.append(f'<span class="event-na">📊 [UW] Flow alertas: alcista '
                              f'${flow_call or 0:.2f}M · bajista ${flow_put or 0:.2f}M</span>')
    if pd.notna(net_call) or pd.notna(net_put):
        uw_flow_lines.append(f'<span class="event-na">💹 [UW] Net premium: call '
                              f'${net_call or 0:.2f}M · put ${net_put or 0:.2f}M</span>')
    bull_oi, bear_oi = row.get("BullishOI_New"), row.get("BearishOI_New")
    if pd.notna(bull_oi) or pd.notna(bear_oi):
        uw_flow_lines.append(f'<span class="event-na">📈 [UW] OI Change: alcista '
                              f'{bull_oi or 0:.0f} · bajista {bear_oi or 0:.0f} contratos</span>')
    top_bull_s, top_bear_s = row.get("TopBullishStrike"), row.get("TopBearishStrike")
    if pd.notna(top_bull_s) or pd.notna(top_bear_s):
        alerta = ""
        if row.get("FlowNearCallShort"):
            alerta += " · ⚠️ convicción cerca del CALL CORTO"
        if row.get("FlowNearPutShort"):
            alerta += " · ⚠️ convicción cerca del PUT CORTO"
        uw_flow_lines.append(f'<span class="event-na">🎯 [UW] Flow por strike: bullish='
                              f'{f"${top_bull_s:g}" if pd.notna(top_bull_s) else "N/A"} · bearish='
                              f'{f"${top_bear_s:g}" if pd.notna(top_bear_s) else "N/A"}{alerta}</span>')
    uw_flow_line = ('<div class="events-line" style="margin-top:6px">' +
                     " ".join(uw_flow_lines) + '</div>' if uw_flow_lines else "")

    metrics = (
        f'<span class="metric">Short DTE <b>{row["ShortDTE"]}</b></span>'
        f'<span class="metric">Long DTE <b>{row["LongDTE"]}</b></span>'
        f'<span class="metric">Strikes P/C <b>{row["PutStrike"]:g}/{row["CallStrike"]:g}</b></span>'
        f'<span class="metric">Débito (mid) <b>${row["NetDebit"]:.2f}</b></span>'
        f'<span class="metric">Max Profit (est.) <b>${row["MaxProfitEstimate"]:.2f}</b></span>'
        f'<span class="metric">Return/Debit <b>{row["ReturnOnDebitPct"]:.0f}%</b></span>'
        f'<span class="metric">BE <b>{be_str}</b></span>'
        f'<span class="metric">IVR <b>{row["IVR"]:.0f}</b></span>'
        f'<span class="metric">Contratos <b>{row["SuggContracts"]}</b></span>'
        f'<span class="metric">Worst Case <b>${row["WorstCaseLoss_$"]:.0f}</b></span>'
        f'<span class="metric">Score <b>{row.get("Score", 0):.3f}</b></span>'
    )
    return (f'<div class="candidate-card">{svg}<div class="candidate-meta">'
            f'<b>{row["Symbol"]}</b> <span class="small">[{row["Sector"]}] '
            f'{row["ShortExp"]} / {row["LongExp"]}</span>{theo_tag}{diagonal_tag}'
            f'<div class="events-line">{er_line}</div>'
            f'<div class="events-line" style="margin-top:6px">{ts_line}</div>'
            f'<div class="events-line" style="margin-top:6px">{exec_line}</div>'
            f'<div class="events-line" style="margin-top:6px">{greeks_line}{move_line}</div>'
            f'{pnl_pre_line}{pnl_since_line}{valley_warning}{uw_line}{short_exp_line}{uw_flow_line}'
            f'<div style="margin-top:8px">{metrics}</div></div></div>')


def make_html_report(earnings_ready_df: pd.DataFrame, radar_list: list[dict],
                     normal_df: pd.DataFrame, ctx_count: int, n_universe: int,
                     rejects: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_ready = len(earnings_ready_df) if not earnings_ready_df.empty else 0
    n_radar = len(radar_list)
    n_normal = len(normal_df) if not normal_df.empty else 0

    # ---- Recuadro verde: listos para entrar HOY (modo earnings, en ventana) ----
    green_html = ""
    if not earnings_ready_df.empty:
        cards = "\n".join(_render_candidate_card(r.to_dict()) for _, r in earnings_ready_df.iterrows())
        green_html = (
            '<div class="green-box">'
            '<h2 style="color:#10b981;margin-top:0">🟢 LISTOS PARA ENTRAR HOY '
            f'<span class="small" style="color:#a7f3d0">({n_ready} candidato(s), '
            f'earnings en ventana {ENTRY_WINDOW_MIN_DAYS}-{ENTRY_WINDOW_MAX_DAYS} días)</span></h2>'
            f'{cards}</div>'
        )
    else:
        green_html = ('<div class="panel"><span class="small">'
                      '🟢 Nada en ventana de entrada hoy.</span></div>')

    # ---- Radar: próximamente (aún no entran en ventana) ----
    radar_html = ""
    if radar_list:
        items = "".join(
            f'<div class="radar-item">● <b>{r["symbol"]}</b> — '
            f'earnings en {r["days_to_earnings"]}d '
            f'(entra en ventana en {r["days_until_entry_window"]}d)</div>'
            for r in radar_list
        )
        radar_html = (f'<div class="panel"><h3>🟡 PRÓXIMAMENTE '
                      f'<span class="small">({n_radar} en radar, aún no es el día)</span></h3>'
                      f'<div class="radar-list">{items}</div></div>')
    else:
        radar_html = '<div class="panel"><span class="small">🟡 Nada en radar próximo.</span></div>'

    # ---- Modo normal (rango, sin earnings) ----
    normal_html = ""
    if not normal_df.empty:
        cards = "\n".join(_render_candidate_card(r.to_dict()) for _, r in normal_df.iterrows())
        normal_html = (f'<h2 style="margin-top:32px">🔄 MODO NORMAL (rango, sin earnings) '
                       f'<span class="small">({n_normal} candidato(s))</span></h2>'
                       f'<div class="panel">{cards}</div>')
    else:
        normal_html = ('<h2 style="margin-top:32px">🔄 MODO NORMAL</h2>'
                       '<div class="panel"><span class="small">Sin candidatos.</span></div>')

    # ---- Tabla combinada (ambos modos, todo lo evaluado) ----
    all_df = pd.concat([earnings_ready_df, normal_df], ignore_index=True) if (
        not earnings_ready_df.empty or not normal_df.empty) else pd.DataFrame()
    table_html = ""
    if not all_df.empty:
        cols = ["Symbol", "Mode", "Sector", "ShortDTE", "LongDTE",
                "PutStrike", "CallStrike", "LongPutStrike", "LongCallStrike", "IsDiagonal",
                "NetDebit", "ExecutableDebit", "SlippagePct",
                "MaxProfitEstimate", "ReturnOnDebitPct", "ProfitZoneWidth",
                "TermStructureSlope", "NetGamma", "NetVega", "NetTheta",
                "ExpectedMovePct", "PutCoverageSigma", "CallCoverageSigma",
                "UW_CallWallShort", "UW_PutWallShort", "UW_GammaFlipShort", "UW_NetGexShort",
                "ShortCallBeyondWall", "ShortPutBeyondWall",
                "OIPutCallRatioShort", "OICallOITotalShort", "OIPutOITotalShort",
                "OICallWallStrikeShort", "OIPutWallStrikeShort",
                "IVR", "EarningsDate", "SuggestedExitDate",
                "TakeProfitRange", "SuggContracts", "AllLegsLive", "TheoLegsCount", "Score"]
        table_html = '<table class="sortable"><thead><tr>' + "".join(f'<th>{c}</th>' for c in cols) + '</tr></thead><tbody>'
        for _, r in all_df.iterrows():
            table_html += "<tr>"
            for c in cols:
                v = r.get(c)
                if pd.isna(v) if not isinstance(v, list) else False:
                    vstr = ""
                elif isinstance(v, float):
                    vstr = f"{v:.3f}".rstrip("0").rstrip(".") or "0"
                elif isinstance(v, list):
                    vstr = ", ".join(str(x) for x in v)
                else:
                    vstr = str(v)
                table_html += f'<td data-v="{v if not isinstance(v, list) else 0}">{vstr}</td>'
            table_html += "</tr>"
        table_html += "</tbody></table>"

    summary = (
        '<div class="summary-grid">'
        f'<div class="item"><div class="label">Universo</div><div class="val">{n_universe}</div></div>'
        f'<div class="item"><div class="label">Pasaron screener</div><div class="val">{ctx_count}</div></div>'
        f'<div class="item"><div class="label">🟢 Listos hoy</div><div class="val">{n_ready}</div></div>'
        f'<div class="item"><div class="label">🟡 En radar</div><div class="val">{n_radar}</div></div>'
        f'<div class="item"><div class="label">🔄 Modo normal</div><div class="val">{n_normal}</div></div>'
        '</div>'
    )

    reject_html = ""
    if rejects:
        items = sorted(rejects.items(), key=lambda x: -x[1])
        reject_html = ('<div class="panel"><h3>Motivos de descarte</h3><ul class="reject-list">' +
                       "".join(f'<li>{r}: <b>{c}</b></li>' for r, c in items) + '</ul></div>')

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<title>Double Calendar Report — {now}</title>
<style>{CSS}
.green-box {{ background:rgba(16,185,129,0.08); border:2px solid #10b981; border-radius:12px;
              padding:18px; margin:16px 0; }}
.radar-list {{ font-family:monospace; font-size:13px; }}
.radar-item {{ padding:6px 0; color:#fde68a; border-bottom:1px solid #1e293b; }}
.radar-item:last-child {{ border-bottom:none; }}
</style></head><body>
<h1>Double Calendar Scanner</h1>
<div class="small">Generado: {now} · {n_universe} símbolos analizados</div>
{summary}
{green_html}
{radar_html}
{normal_html}
<h2 style="margin-top:32px">Tabla completa (todos los modos)</h2>
<div class="panel">{table_html}</div>
{reject_html}
<script>{JS_SORT}</script>
</body></html>"""


# =====================================================================
# MAIN — UNA Session, streamer abierto en DOS RONDAS dentro de la
# MISMA apertura (nunca dos streamers, nunca fuera de una sola sesión):
#   Ronda 1 (solo modo earnings): fetch de Greeks de una banda amplia
#     de strikes por ticker, para poder elegir el strike real por
#     DELTA (0.20-0.30) — no se puede saber el delta sin streamear.
#   Ronda 2 (ambos modos): fetch de las 4 patas finales ya elegidas
#     (short call/put + long call/put), para precios y IV reales.
# Todo el escaneo lento con yfinance (Fase 1) corre ANTES de abrir el
# streamer — mismo principio ya validado en el scanner de credit
# spreads: nunca dejar el streamer abierto e inactivo durante fases
# lentas basadas en yfinance.
# =====================================================================

async def run():
    session = make_session()

    vix_now = None
    try:
        vix_df = yf.download("^VIX", period="5d", interval="1d",
                              progress=False, auto_adjust=False, threads=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        if not vix_df.empty:
            vix_now = float(vix_df["Close"].iloc[-1])
            log.info(f"VIX actual: {vix_now:.2f}")
    except Exception:
        pass

    symbols = load_universe(session)
    if not symbols:
        log.error("Sin símbolos. Revisa universo/watchlists."); return

    log.info("Fase 0: bulk metrics TastyTrade…")
    metrics_by_sym = fetch_metrics(session, symbols)
    log.info(f"  Metrics OK: {len(metrics_by_sym)}/{len(symbols)}")

    log.info(f"Fase 1: screeners (yfinance, streamer CERRADO)… "
              f"[Normal={'ON' if RUN_NORMAL_MODE else 'OFF'} · "
              f"Earnings={'ON' if RUN_EARNINGS_MODE else 'OFF'}]")
    contexts_normal: dict[str, dict] = {}
    contexts_earnings_ready: dict[str, dict] = {}   # en ventana de entrada HOY
    radar_list: list[dict] = []                      # próximamente, aún sin evaluar completo

    for i, sym in enumerate(symbols, 1):
        try:
            mt = metrics_by_sym.get(sym, {})
            df = load_price_data(sym)

            # --- Modo normal ---
            if RUN_NORMAL_MODE:
                ctx_n = screen_underlying_calendar(sym, df, mt)
                if ctx_n is not None and "_rejected" not in ctx_n:
                    contexts_normal[sym] = ctx_n
                elif ctx_n is not None and ctx_n.get("_rejected"):
                    _reject(f"normal_screen:{ctx_n['_rejected']}")

            # --- Modo earnings ---
            if RUN_EARNINGS_MODE:
                ctx_e = screen_underlying_earnings(sym, df, mt)
                if ctx_e is not None and "_rejected" not in ctx_e:
                    if ctx_e["in_entry_window"]:
                        contexts_earnings_ready[sym] = ctx_e
                        log.info(f"  {sym}: entró en ventana de entrada "
                                  f"(earnings en {ctx_e['days_to_earnings']}d)")
                    else:
                        radar_list.append({
                            "symbol": sym,
                            "days_to_earnings": ctx_e["days_to_earnings"],
                            "days_until_entry_window": ctx_e["days_until_entry_window"],
                        })
                elif ctx_e is not None and ctx_e.get("_rejected"):
                    _reject(f"earnings_screen:{ctx_e['_rejected']}")

            if i % 50 == 0:
                log.info(f"  [{i}/{len(symbols)}] procesados — "
                          f"normal={len(contexts_normal)} "
                          f"earnings_ready={len(contexts_earnings_ready)} "
                          f"radar={len(radar_list)}")
        except Exception as e:
            log.debug(f"  {sym} error: {e}")

    radar_list.sort(key=lambda r: r["days_until_entry_window"])

    log.info(f"Screener completo: normal={len(contexts_normal)}  "
              f"earnings_ready={len(contexts_earnings_ready)}  radar={len(radar_list)}")

    total_ctx = len(contexts_normal) + len(contexts_earnings_ready)
    if total_ctx == 0:
        log.warning("Ningún subyacente pasó filtros. Generando HTML mínimo…")
        html = make_html_report(pd.DataFrame(), radar_list, pd.DataFrame(),
                                 0, len(symbols), _REJECTS)
        Path(OUTPUT_HTML).write_text(html, encoding="utf-8")
        return

    log.info("Fase 2: chains + selección de vencimientos (REST, streamer CERRADO)…")

    # ---- Modo normal: selección de strikes por expected move (no necesita delta real) ----
    normal_candidates = []   # (ctx, call_short, put_short, call_long, put_long, exp_short, exp_long)
    # NUEVO (pendiente #7, OI por expiración exacta): se guarda el
    # chain CORTO completo de cada ticker que llega a candidato, para
    # pedir su Open Interest real (vía tastytrade, no UW — UW
    # confirmado hoy que /oi-per-strike es agregado y NO filtra por
    # expiración, ver test_uw_oi_per_strike.py) y calcular el balance
    # put/call de ESE vencimiento puntual, no del ticker completo.
    chain_short_by_ticker: dict[str, list] = {}
    for sym, ctx in contexts_normal.items():
        try:
            chain = get_option_chain(session, sym)
        except Exception as e:
            log.warning(f"  {sym}: chain fetch falló ({e}) — descartado sin evaluar")
            _reject("chain_fetch_failed(normal)")
            continue
        exp_short, exp_long = pick_two_expirations(chain, ctx.get("price"))
        if exp_short is None or exp_long is None:
            _reject("no_valid_expiration_pair(normal)")
            continue
        chain_short, chain_long = chain[exp_short], chain[exp_long]
        strikes_info = select_calendar_strikes(chain_short, chain_long, ctx, ctx["realized_vol"])
        call_short, put_short = strikes_info["call_strike_opt"], strikes_info["put_strike_opt"]
        if call_short is None or put_short is None:
            # No hay ningún strike común entre las dos expiraciones
            # razonablemente cerca del precio — no se puede armar un
            # Double Calendar real (mismo strike) para este ticker.
            _reject("no_common_strikes_found(normal)")
            continue
        call_long = find_closest_strike(chain_long, float(call_short.strike_price), False)
        put_long = find_closest_strike(chain_long, float(put_short.strike_price), True)
        if call_long is None or put_long is None:
            _reject("no_matching_strikes_long(normal)")
            continue
        normal_candidates.append((ctx, call_short, put_short, call_long, put_long, exp_short, exp_long))
        chain_short_by_ticker[sym] = chain_short

    # ---- Modo earnings: vencimientos por earnings + banda amplia de strikes (delta pendiente) ----
    earnings_prep = []  # (ctx, chain_short_band, chain_long, exp_short, exp_long)
    delta_fetch_syms = set()
    for sym, ctx in contexts_earnings_ready.items():
        try:
            chain = get_option_chain(session, sym)
        except Exception as e:
            log.warning(f"  {sym}: chain fetch falló ({e}) — descartado sin evaluar")
            _reject("chain_fetch_failed(earnings)")
            continue
        exp_short, exp_long = pick_earnings_expirations(chain, ctx["earnings_date"])
        if exp_short is None or exp_long is None:
            log.info(f"  {sym}: sin par de vencimientos válido (earnings {ctx['earnings_date']})")
            _reject("no_valid_expiration_pair(earnings)")
            continue
        chain_short = chain[exp_short]
        chain_long = chain[exp_long]
        # Mismo motivo que en modo normal: guardar el chain corto
        # completo (no solo la banda por delta) para poder pedir su
        # Open Interest completo más abajo (Fase 3d) y calcular el
        # balance put/call real de ESTA expiración exacta también acá.
        # oi_short_syms y oi_balance_short_by_ticker ya son genéricos
        # a ambos modos — con esta línea alcanza, no hace falta tocar
        # nada más del pipeline de OI.
        chain_short_by_ticker[sym] = chain_short
        band = gather_delta_candidate_strikes(chain_short, ctx["price"])
        # FIX (mismo hallazgo que en modo normal, caso GOOG): se
        # restringe la banda a strikes que también existen en la
        # expiración larga ANTES de pedir Greeks — evita gastar
        # llamadas de datos en strikes que igual no podrían usarse
        # para armar un calendario real, y evita el mismo diagonal
        # por redondeo silencioso al elegir el largo más abajo.
        band = restrict_to_common_strikes(band, chain_long)
        if not band:
            log.info(f"  {sym}: sin strikes en la banda alrededor del spot (${ctx['price']})")
            _reject("no_strikes_in_band(earnings)")
            continue
        earnings_prep.append((ctx, band, chain_long, exp_short, exp_long))
        for opt in band:
            delta_fetch_syms.add(opt.streamer_symbol)

    log.info(f"  Candidatos modo normal: {len(normal_candidates)}  |  "
              f"Candidatos modo earnings (pendiente delta): {len(earnings_prep)}")

    # ---- UW: GEX, dark pool, OI por strike, flow alertas, net premium,
    # OI change y flow por strike — se llama UNA vez para todos los
    # tickers candidatos de ambos modos, antes de evaluar. Si no hay
    # token configurado, las 7 funciones devuelven vacío y el resto
    # sigue exactamente igual que antes de este cambio.
    uw_gex_data, uw_darkpool_data, uw_oi_data = {}, {}, {}
    uw_flow_data, uw_net_premium_data, uw_oi_change_data, uw_flow_per_strike_data = {}, {}, {}, {}
    uw_gex_short_data = {}
    if _uw_enabled():
        all_uw_tickers = sorted(set(ctx["symbol"] for ctx, *_ in normal_candidates) |
                                  set(ctx["symbol"] for ctx, *_ in earnings_prep))
        if all_uw_tickers:
            log.info(f"Fase 3.5: UW (GEX, dark pool, OI, flow, net premium, OI change, "
                      f"flow/strike) para {len(all_uw_tickers)} tickers…")
            uw_gex_data = await calc_gex_uw(all_uw_tickers)
            uw_darkpool_data = await calc_darkpool_levels(all_uw_tickers)
            uw_oi_data = await calc_oi_per_strike_uw(all_uw_tickers)
            uw_flow_data = await calc_unusual_flow_uw(all_uw_tickers)
            uw_net_premium_data = await calc_net_premium_uw(all_uw_tickers)
            uw_oi_change_data = await calc_oi_change_uw(all_uw_tickers)
            uw_flow_per_strike_data = await calc_flow_per_strike_uw(all_uw_tickers)

            # NUEVO: GEX acotado a la expiración corta exacta de cada
            # ticker (ver docstring de calc_gex_uw_by_expiry) — a
            # diferencia de uw_gex_data de arriba, que mezcla todas
            # las expiraciones. exp_short ya se conoce acá (viene de
            # normal_candidates/earnings_prep), así que no hace falta
            # reordenar nada del flujo. Un ticker puede aparecer en
            # los dos modos con distinto exp_short — se prioriza el
            # de modo normal si pasa eso (caso raro).
            ticker_to_exp_short: dict[str, date] = {}
            for ctx, band, chain_long, exp_short, exp_long in earnings_prep:
                ticker_to_exp_short[ctx["symbol"]] = exp_short
            for ctx, call_short, put_short, call_long, put_long, exp_short, exp_long in normal_candidates:
                ticker_to_exp_short[ctx["symbol"]] = exp_short
            pairs = [(t, e.isoformat()) for t, e in ticker_to_exp_short.items()]
            uw_gex_short_data = await calc_gex_uw_by_expiry(pairs)

            # NUEVO: resumen de cobertura — cuántos tickers trajeron
            # dato REAL (no vacío) por cada función. Sin esto, un fallo
            # silencioso puntual (rate limit, timeout) en una sola
            # función pasa desapercibido salvo que se tenga la consola
            # abierta en el momento exacto de la corrida — pasó hoy
            # mismo con flow-per-strike, sin poder confirmar la causa
            # después porque la consola ya se había cerrado.
            def _cobertura(d, get_fn):
                by_t = d.get("by_ticker", d.get("flow_by_ticker", {}))
                return sum(1 for t in all_uw_tickers if get_fn(by_t.get(t, {})))

            cobertura = {
                "GEX (muros)": _cobertura(uw_gex_data, lambda v: v.get("call_wall") is not None),
                "Dark pool": _cobertura(uw_darkpool_data, lambda v: v.get("levels")),
                "OI por strike": _cobertura(uw_oi_data, lambda v: v.get("oi_by_strike")),
                "Flow alertas": _cobertura(uw_flow_data, lambda v: v.get("n_signals", 0) > 0),
                "Net premium": _cobertura(uw_net_premium_data, lambda v: v.get("n_ticks", 0) > 0),
                "OI Change": _cobertura(uw_oi_change_data, lambda v: v.get("top_contract")),
                "Flow por strike": _cobertura(uw_flow_per_strike_data,
                                                lambda v: v.get("top_bullish_strike") or v.get("top_bearish_strike")),
            }
            log.info(f"  Cobertura UW real (de {len(all_uw_tickers)} tickers consultados):")
            for nombre, n in cobertura.items():
                pct = n / len(all_uw_tickers) * 100
                alerta = "  <-- revisar, cobertura muy baja" if pct < 10 else ""
                log.info(f"    {nombre}: {n}/{len(all_uw_tickers)} ({pct:.0f}%){alerta}")
    else:
        log.info("Sin UW_API_TOKEN/UW_API_KEY — se omite toda la integración UW "
                  "(el resto del script funciona igual, sin este dato adicional)")

    # ---- Streamer: se abre UNA vez, con dos rondas de fetch adentro ----
    quotes: dict = {}
    greeks: dict = {}
    summs: dict = {}
    oi_short_summs: dict = {}
    earnings_candidates = []  # (ctx, call_short, put_short, call_long, put_long, exp_short, exp_long)

    normal_syms = set()
    for _, cs, ps, cl, pl, _, _ in normal_candidates:
        for opt in (cs, ps, cl, pl):
            normal_syms.add(opt.streamer_symbol)

    # NUEVO (pendiente #7, OI por expiración exacta): símbolos de TODO
    # el chain corto de cada candidato normal, solo para pedir su
    # Open Interest (Summary) — no Quote ni Greeks, es el fetch más
    # barato posible. Esto es lo que permite calcular el balance
    # put/call real de la expiración corta puntual, en vez del
    # agregado por ticker que da el endpoint de UW (confirmado hoy que
    # no filtra por expiración).
    oi_short_syms = set()
    for opt_list in chain_short_by_ticker.values():
        for opt in opt_list:
            oi_short_syms.add(opt.streamer_symbol)

    if delta_fetch_syms or normal_syms:
        async with market_data_context(session) as hub:
            # Ronda 1: Greeks de la banda amplia (modo earnings) para elegir por delta
            if delta_fetch_syms:
                log.info(f"Fase 3a: fetch de Greeks para selección por delta "
                          f"({len(delta_fetch_syms)} contratos)…")
                _, greeks_round1, _ = await hub.fetch(sorted(delta_fetch_syms),
                                                         want_quote=False, want_summary=False)
                greeks.update(greeks_round1)

                # Elegir strikes reales por delta, y armar la lista final de patas largas a pedir
                long_leg_syms = set()
                for ctx, band, chain_long, exp_short, exp_long in earnings_prep:
                    call_short = select_strike_by_delta(band, greeks, option_type_is_put=False)
                    put_short = select_strike_by_delta(band, greeks, option_type_is_put=True)
                    if call_short is None or put_short is None:
                        log.info(f"  {ctx['symbol']}: sin match de delta en la banda de strikes")
                        _reject("no_delta_match(earnings)")
                        continue
                    call_long = find_closest_strike(chain_long, float(call_short.strike_price), False)
                    put_long = find_closest_strike(chain_long, float(put_short.strike_price), True)
                    if call_long is None or put_long is None:
                        log.info(f"  {ctx['symbol']}: sin strike equivalente en la cadena larga")
                        _reject("no_matching_strikes_long(earnings)")
                        continue
                    earnings_candidates.append((ctx, call_short, put_short, call_long, put_long,
                                                  exp_short, exp_long))
                    for opt in (call_short, put_short, call_long, put_long):
                        long_leg_syms.add(opt.streamer_symbol)

                # Ronda 2a: patas finales del modo earnings (quotes+greeks+summary completos)
                if long_leg_syms:
                    log.info(f"Fase 3b: fetch final patas modo earnings "
                              f"({len(long_leg_syms)} contratos)…")
                    q2, g2, s2 = await hub.fetch(sorted(long_leg_syms))
                    quotes.update(q2); greeks.update(g2); summs.update(s2)

            # Ronda 2b: patas del modo normal (independiente, no necesitaba selección por delta)
            if normal_syms:
                log.info(f"Fase 3c: fetch patas modo normal ({len(normal_syms)} contratos)…")
                q3, g3, s3 = await hub.fetch(sorted(normal_syms))
                quotes.update(q3); greeks.update(g3); summs.update(s3)

            # Ronda 3: OI real (Summary solamente, sin Quote ni Greeks —
            # el fetch más barato) del chain corto COMPLETO de cada
            # candidato normal, para el balance put/call por expiración
            # exacta (pendiente #7). Se guarda aparte de `summs` (que
            # solo tiene las 4 patas elegidas) para no mezclar cosas.
            if oi_short_syms:
                log.info(f"Fase 3d: fetch OI del chain corto completo para balance "
                          f"put/call por expiración exacta ({len(oi_short_syms)} contratos)…")
                _, _, oi_short_summs = await hub.fetch(sorted(oi_short_syms),
                                                          want_quote=False, want_greeks=False)
        # streamer cerrado acá — todo lo que sigue (evaluación, ranking, reporte) no lo necesita más

    # NUEVO (pendiente #7): balance OI put/call de la expiración corta
    # exacta, por ticker — usando el chain completo + Summary recién
    # fetcheados. Ver calc_oi_balance_short_expiry.
    oi_balance_short_by_ticker: dict[str, dict] = {}
    for sym, chain_short_opt_list in chain_short_by_ticker.items():
        oi_balance_short_by_ticker[sym] = calc_oi_balance_short_expiry(
            chain_short_opt_list, oi_short_summs)

    log.info("Fase 4: evaluando calendarios (ambos modos)…")
    rows_normal, rows_earnings = [], []
    for ctx, cs, ps, cl, pl, exp_s, exp_l in normal_candidates:
        row = evaluate_double_calendar(ctx, cs, ps, cl, pl, exp_s, exp_l, quotes, greeks, summs,
                                         uw_gex_data, uw_gex_short_data, oi_balance_short_by_ticker,
                                         uw_darkpool_data, uw_oi_data,
                                         uw_flow_data, uw_net_premium_data,
                                         uw_oi_change_data, uw_flow_per_strike_data)
        if row is not None:
            rows_normal.append(row)
    for ctx, cs, ps, cl, pl, exp_s, exp_l in earnings_candidates:
        row = evaluate_double_calendar(ctx, cs, ps, cl, pl, exp_s, exp_l, quotes, greeks, summs,
                                         uw_gex_data, uw_gex_short_data, oi_balance_short_by_ticker,
                                         uw_darkpool_data, uw_oi_data,
                                         uw_flow_data, uw_net_premium_data,
                                         uw_oi_change_data, uw_flow_per_strike_data)
        if row is not None:
            rows_earnings.append(row)

    if DEBUG_REJECT_REASONS and _REJECTS:
        log.info("Motivos de descarte:")
        for reason, count in sorted(_REJECTS.items(), key=lambda x: -x[1]):
            log.info(f"   {reason}: {count}")

    normal_df = rank_calendars(rows_normal, mode="normal") if rows_normal else pd.DataFrame()
    earnings_df = rank_calendars(rows_earnings, mode="earnings") if rows_earnings else pd.DataFrame()

    combined_for_csv = pd.concat([earnings_df, normal_df], ignore_index=True) if (
        not earnings_df.empty or not normal_df.empty) else pd.DataFrame()
    if not combined_for_csv.empty:
        csv_df = combined_for_csv.drop(columns=["PnLCurve"], errors="ignore")
        csv_df.to_csv(OUTPUT_CSV, index=False)
        log.info(f"OK  CSV: {OUTPUT_CSV}  ({len(combined_for_csv)} candidatos)")

        # Archivo con fecha — para el historial/backtest futuro. Si
        # ya corriste el script hoy, esto PISA el archivo de hoy (no
        # acumula duplicados dentro del mismo día), pero nunca toca
        # los archivos de días anteriores.
        try:
            Path(HISTORY_DIR).mkdir(exist_ok=True)
            today_str = date.today().isoformat()
            history_path = Path(HISTORY_DIR) / f"double_calendar_candidates_{today_str}.csv"
            csv_df.to_csv(history_path, index=False)
            log.info(f"OK  Historial: {history_path}")
        except Exception as e:
            log.warning(f"  No pude guardar el historial (no afecta el resto): {e}")
    else:
        log.warning("Sin candidatos finales en ningún modo.")

    html = make_html_report(earnings_df, radar_list, normal_df,
                             total_ctx, len(symbols), _REJECTS)
    Path(OUTPUT_HTML).write_text(html, encoding="utf-8")
    log.info(f"OK  HTML: {OUTPUT_HTML}")

    if AUTO_OPEN_BROWSER:
        try:
            import webbrowser
            webbrowser.open(Path(OUTPUT_HTML).resolve().as_uri())
        except Exception:
            pass

    log.info("Corrida completa. Sesión y streamer cerrados correctamente.")


# =====================================================================
# BACKTEST — cierra el círculo del historial. Hasta ahora se guardaba
# un CSV con fecha por cada corrida (HISTORY_DIR), pero nada comparaba
# esas recomendaciones pasadas contra lo que realmente pasó después —
# se guardaba qué dijo el script, nunca se verificaba si acertó.
#
# LIMITACIÓN HONESTA: no se puede re-obtener el precio histórico
# exacto de las opciones en la fecha pasada (yfinance no da eso, y
# tastytrade tampoco expone cotizaciones históricas de opciones vía
# esta integración) — así que esto NO recalcula el P&L real de cada
# calendario. Lo que SÍ se puede medir con datos reales accesibles
# (precio histórico del SUBYACENTE, que yfinance sí da bien) es la
# pregunta central de la tesis: "¿el precio se quedó QUIETO, dentro
# del rango entre los strikes, como predijo el screener, o se escapó
# más allá de lo esperado?" — que es la variable que más determina si
# un calendario gana o pierde, aunque no dé el dólar exacto.
# =====================================================================

def compare_history_to_actual(days_back: int = 8) -> pd.DataFrame:
    """
    Toma el CSV de historial de hace `days_back` días (mismo horizonte
    que ya usás para mirar P&L desde la entrada, DAYS_SINCE_ENTRY_SNAPSHOTS)
    y compara, para cada candidato recomendado ese día, dónde terminó
    el precio real del subyacente HOY contra los strikes que eligió el
    screener en su momento.

    Uso (fuera del flujo normal de run(), se llama aparte):
        python -c "from calendar_scanner import compare_history_to_actual; \\
                   df = compare_history_to_actual(8); \\
                   print(df.to_string())"

    Columnas del resultado:
      - Symbol, FechaRecomendacion
      - PrecioEntrada, PrecioActual, MovimientoPct
      - PutStrike, CallStrike (los strikes CORTOS elegidos en su momento)
      - SeQuedoEnRango: True si el precio actual sigue entre ambos
        strikes cortos — la señal más directa de "la tesis de quietud
        se cumplió", sin necesitar reconstruir el P&L exacto.
      - ExpectedMovePct (el que el script había estimado ese día) vs.
        MovimientoPct real — si el real superó bastante al esperado,
        el modelo de IV/expected move subestimó el movimiento real.
    """
    target_date = (date.today() - timedelta(days=days_back)).isoformat()
    history_path = Path(HISTORY_DIR) / f"double_calendar_candidates_{target_date}.csv"
    if not history_path.exists():
        log.warning(f"No existe historial para {target_date} (buscado en {history_path}). "
                     f"Archivos disponibles: {list(Path(HISTORY_DIR).glob('*.csv')) if Path(HISTORY_DIR).exists() else 'ninguno'}")
        return pd.DataFrame()

    hist_df = pd.read_csv(history_path)
    if hist_df.empty:
        log.warning(f"Historial de {target_date} está vacío (sin candidatos ese día).")
        return pd.DataFrame()

    resultados = []
    for _, row in hist_df.iterrows():
        symbol = row.get("Symbol")
        precio_entrada = row.get("Price")
        put_strike = row.get("PutStrike")
        call_strike = row.get("CallStrike")
        expected_move_pct = row.get("ExpectedMovePct")
        if symbol is None or pd.isna(precio_entrada):
            continue
        try:
            df_precio = load_price_data(symbol)
            if df_precio is None or df_precio.empty:
                continue
            precio_actual = float(df_precio["Close"].iloc[-1])
        except Exception as e:
            log.debug(f"  {symbol}: no pude traer precio actual para comparar ({e})")
            continue

        movimiento_pct = (precio_actual - precio_entrada) / precio_entrada * 100
        se_quedo_en_rango = (pd.notna(put_strike) and pd.notna(call_strike) and
                              put_strike <= precio_actual <= call_strike)

        resultados.append({
            "Symbol": symbol, "FechaRecomendacion": target_date,
            "PrecioEntrada": round(precio_entrada, 2), "PrecioActual": round(precio_actual, 2),
            "MovimientoPct": round(movimiento_pct, 2),
            "PutStrike": put_strike, "CallStrike": call_strike,
            "SeQuedoEnRango": se_quedo_en_rango,
            "ExpectedMovePct": expected_move_pct,
            "SuperoExpectedMove": (pd.notna(expected_move_pct) and
                                    abs(movimiento_pct) > expected_move_pct),
            "Mode": row.get("Mode"), "Score": row.get("Score"),
        })

    result_df = pd.DataFrame(resultados)
    if not result_df.empty:
        n_en_rango = result_df["SeQuedoEnRango"].sum()
        n_total = len(result_df)
        log.info(f"Backtest {target_date} → hoy: {n_en_rango}/{n_total} "
                  f"({n_en_rango/n_total*100:.0f}%) se quedaron dentro del rango de strikes.")
    return result_df


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()


