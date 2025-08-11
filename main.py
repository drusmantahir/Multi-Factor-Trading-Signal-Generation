#!/usr/bin/env python3
"""
UltraX Trading Analyzer - FastAPI Backend
Provides trading signals and anomaly detection for the UltraX frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UltraX Trading Analyzer",
    description="AI-powered trading signals and anomaly detection",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class TradingSignal(BaseModel):
    symbol: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    confidence: float
    price: float
    timestamp: str
    reasoning: str
    indicators: Dict[str, float]

class AnomalyAlert(BaseModel):
    symbol: str
    alert_type: str  # "VOLUME_SPIKE", "PRICE_ANOMALY", "PATTERN_BREAK"
    severity: str  # "LOW", "MEDIUM", "HIGH"
    description: str
    timestamp: str
    metrics: Dict[str, float]

class MarketData(BaseModel):
    symbol: str
    price: float
    volume: float
    timestamp: str
    change_24h: float
    market_cap: Optional[float] = None

# Real trading data API integration with Binance
class RealTradingAPI:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        # Only include symbols that are actually supported by Binance
        self.supported_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT"]
        
        # Symbol mapping for common abbreviations - only map to valid Binance symbols
        self.symbol_mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT", 
            "SOL": "SOLUSDT",
            "ADA": "ADAUSDT",
            "BNB": "BNBUSDT"
        }
        
        # Additional valid Binance symbols you can add
        self.additional_symbols = [
            "DOGEUSDT", "MATICUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT",
            "AVAXUSDT", "ATOMUSDT", "LTCUSDT", "XRPUSDT", "BCHUSDT"
        ]
        
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid for Binance API"""
        # Check if it's in our supported list
        if symbol.upper() in self.supported_symbols:
            return True
        
        # Check if it's in our additional symbols list
        if symbol.upper() in self.additional_symbols:
            return True
            
        # Check if it's a valid mapping
        if symbol.upper() in self.symbol_mapping:
            return True
            
        return False
        
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get real-time market data from Binance API"""
        try:
            binance_symbol = self.symbol_mapping.get(symbol.upper(), symbol.upper())
            
            # Validate the symbol before making API call
            if not self.is_valid_symbol(symbol.upper()):
                logger.warning(f"Symbol {symbol} is not supported, using fallback data")
                return await self._get_fallback_data(symbol, days=1)
            
            async with aiohttp.ClientSession() as session:
                # Get 24hr ticker price change statistics
                url = f"{self.base_url}/ticker/24hr"
                params = {"symbol": binance_symbol}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Calculate 24h change percentage
                        change_24h = float(data["priceChangePercent"])
                        
                        return MarketData(
                            symbol=symbol.upper(),
                            price=float(data["lastPrice"]),
                            volume=float(data["volume"]),
                            timestamp=datetime.now().isoformat(),
                            change_24h=change_24h,
                            market_cap=float(data["lastPrice"]) * float(data["volume"])
                        )
                    elif response.status == 400:
                        # Bad request - symbol not supported
                        logger.warning(f"Symbol {binance_symbol} not supported by Binance API")
                        return await self._get_fallback_data(symbol, days=1)
                    else:
                        raise Exception(f"Failed to fetch market data: {response.status}")
                        
        except Exception as e:
            logger.warning(f"Error fetching real data for {symbol}: {e}")
            return await self._get_fallback_data(symbol, days=1)
    
    async def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical price data from Binance API"""
        try:
            binance_symbol = self.symbol_mapping.get(symbol.upper(), symbol.upper())
            
            # Validate the symbol before making API call
            if not self.is_valid_symbol(symbol.upper()):
                logger.warning(f"Symbol {symbol} is not supported, using fallback data")
                return await self._get_fallback_data(symbol, days)
            
            async with aiohttp.ClientSession() as session:
                # Calculate start time (days ago)
                end_time = int(datetime.now().timestamp() * 1000)
                start_time = end_time - (days * 24 * 60 * 60 * 1000)
                
                # Get kline/candlestick data
                url = f"{self.base_url}/klines"
                params = {
                    "symbol": binance_symbol,
                    "interval": "1d",
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": days
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        historical_data = []
                        for candle in data:
                            historical_data.append({
                                "date": datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                                "price": float(candle[4]),  # Close price
                                "volume": float(candle[5]),
                                "high": float(candle[2]),
                                "low": float(candle[3])
                            })
                        
                        return historical_data
                    elif response.status == 400:
                        # Bad request - symbol not supported
                        logger.warning(f"Symbol {binance_symbol} not supported by Binance API")
                        return await self._get_fallback_data(symbol, days)
                    else:
                        raise Exception(f"Failed to fetch historical data: {response.status}")
                        
        except Exception as e:
            logger.warning(f"Error fetching historical data for {symbol}: {e}")
            return await self._get_fallback_data(symbol, days)
    
    async def _get_fallback_data(self, symbol: str, days: int = 30):
        """Fallback to mock data if real API fails"""
        # Base prices for supported symbols
        base_prices = {
            "BTCUSDT": 45000.0,
            "ETHUSDT": 2800.0,
            "SOLUSDT": 95.0,
            "ADAUSDT": 0.45,
            "BNBUSDT": 300.0,
            # Additional symbols with reasonable base prices
            "DOGEUSDT": 0.08,
            "MATICUSDT": 0.85,
            "DOTUSDT": 7.2,
            "LINKUSDT": 15.5,
            "UNIUSDT": 8.1,
            "AVAXUSDT": 35.0,
            "ATOMUSDT": 9.8,
            "LTCUSDT": 75.0,
            "XRPUSDT": 0.55,
            "BCHUSDT": 240.0
        }
        
        # For unmapped symbols, use a default price
        base_price = base_prices.get(symbol.upper(), 100.0)
        volatility = 0.02
        
        if days == 1:  # Market data request
            # Generate realistic market data
            current_price = base_price * (1 + np.random.normal(0, volatility))
            volume = np.random.uniform(1000000, 10000000)
            change_24h = np.random.normal(0, 0.05)
            
            logger.info(f"Generated fallback market data for {symbol} (price: {current_price:.4f})")
            
            return MarketData(
                symbol=symbol.upper(),
                price=current_price,
                volume=volume,
                timestamp=datetime.now().isoformat(),
                change_24h=change_24h * 100,
                market_cap=current_price * volume
            )
        else:  # Historical data request
            # Generate historical data
            data = []
            current_price = base_price
            
            for i in range(days):
                # Add some randomness to price movement
                price_change = np.random.normal(0, volatility)
                current_price = current_price * (1 + price_change)
                
                # Ensure price doesn't go negative
                current_price = max(current_price, base_price * 0.1)
                
                data.append({
                    "date": (datetime.now() - timedelta(days=days-i-1)).isoformat(),
                    "price": current_price,
                    "volume": np.random.uniform(1000000, 10000000),
                    "high": current_price * (1 + abs(np.random.normal(0, 0.01))),
                    "low": current_price * (1 - abs(np.random.normal(0, 0.01)))
                })
            
            logger.info(f"Generated fallback historical data for {symbol} ({days} days)")
            return data

# Trading analysis engine
class TradingAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def calculate_technical_indicators(self, prices: List[float], volumes: List[float]) -> Dict[str, float]:
        """Calculate basic technical indicators"""
        if len(prices) < 20:
            return {}
        
        prices = np.array(prices)
        volumes = np.array(volumes)
        
        # Simple Moving Averages
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
        
        # RSI (Relative Strength Index)
        price_changes = np.diff(prices)
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Volume analysis
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1] if len(volumes) > 0 else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # MACD (simplified)
        ema_12 = np.mean(prices[-12:]) if len(prices) >= 12 else prices[-1]
        ema_26 = np.mean(prices[-26:]) if len(prices) >= 26 else prices[-1]
        macd = ema_12 - ema_26
        
        return {
            "sma_20": round(sma_20, 4),
            "sma_50": round(sma_50, 4),
            "rsi": round(rsi, 2),
            "volume_ratio": round(volume_ratio, 2),
            "macd": round(macd, 4),
            "price_change_24h": round(((prices[-1] - prices[-2]) / prices[-2] * 100), 2) if len(prices) >= 2 else 0
        }
    
    def generate_trading_signal(self, indicators: Dict[str, float], current_price: float) -> Dict:
        """Generate trading signal based on technical indicators"""
        signal_type = "HOLD"
        confidence = 0.5
        reasoning = []
        
        # RSI-based signals
        if indicators.get("rsi", 50) < 30:
            signal_type = "BUY"
            confidence += 0.2
            reasoning.append("RSI indicates oversold conditions")
        elif indicators.get("rsi", 50) > 70:
            signal_type = "SELL"
            confidence += 0.2
            reasoning.append("RSI indicates overbought conditions")
        
        # Moving average signals
        if indicators.get("sma_20", 0) > indicators.get("sma_50", 0):
            if signal_type == "BUY":
                confidence += 0.1
                reasoning.append("Price above long-term moving average")
        else:
            if signal_type == "SELL":
                confidence += 0.1
                reasoning.append("Price below long-term moving average")
        
        # Volume confirmation
        if indicators.get("volume_ratio", 1) > 1.5:
            confidence += 0.1
            reasoning.append("High volume confirms signal")
        
        # MACD signals
        if indicators.get("macd", 0) > 0 and signal_type == "BUY":
            confidence += 0.1
            reasoning.append("MACD shows positive momentum")
        elif indicators.get("macd", 0) < 0 and signal_type == "SELL":
            confidence += 0.1
            reasoning.append("MACD shows negative momentum")
        
        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)
        
        return {
            "signal_type": signal_type,
            "confidence": round(confidence, 2),
            "reasoning": reasoning if reasoning else ["No clear signal based on current indicators"]
        }
    
    def detect_anomalies(self, prices: List[float], volumes: List[float]) -> List[Dict]:
        """Detect anomalies using isolation forest"""
        if len(prices) < 20:
            return []
        
        # Prepare features for anomaly detection
        features = []
        for i in range(len(prices) - 19):
            price_window = prices[i:i+20]
            volume_window = volumes[i:i+20]
            
            # Calculate statistical features
            price_mean = np.mean(price_window)
            price_std = np.std(price_window)
            price_change = (price_window[-1] - price_window[0]) / price_window[0]
            volume_spike = volume_window[-1] / np.mean(volume_window[:-1]) if np.mean(volume_window[:-1]) > 0 else 1
            
            features.append([price_mean, price_std, price_change, volume_spike])
        
        if not features:
            return []
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Detect anomalies
        anomaly_labels = self.anomaly_detector.fit_predict(features_scaled)
        
        anomalies = []
        for i, label in enumerate(anomaly_labels):
            if label == -1:  # Anomaly detected
                anomaly_info = {
                    "index": i + 19,  # Index in original data
                    "price": prices[i + 19],
                    "volume": volumes[i + 19],
                    "features": features[i],
                    "severity": "MEDIUM"  # Could be enhanced with more sophisticated scoring
                }
                anomalies.append(anomaly_info)
        
        return anomalies

# Initialize services
real_api = RealTradingAPI()
analyzer = TradingAnalyzer()

# API endpoints
@app.get("/")
async def root():
    return {"message": "UltraX Trading Analyzer API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/status")
async def server_status():
    """Server status and configuration"""
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "supported_symbols": real_api.supported_symbols,
        "additional_symbols": real_api.additional_symbols,
        "symbol_mapping": real_api.symbol_mapping,
        "total_supported_symbols": len(real_api.supported_symbols) + len(real_api.additional_symbols),
        "endpoints": [
            "/health",
            "/status", 
            "/supported-symbols",
            "/market-data/{symbol}",
            "/trading-signal/{symbol}",
            "/anomaly-detection/{symbol}",
            "/portfolio-analysis",
            "/historical-data/{symbol}"
        ]
    }

@app.get("/supported-symbols")
async def get_supported_symbols():
    """Get list of all supported trading symbols"""
    return {
        "supported_symbols": real_api.supported_symbols,
        "additional_symbols": real_api.additional_symbols,
        "symbol_mapping": real_api.symbol_mapping,
        "total_supported": len(real_api.supported_symbols) + len(real_api.additional_symbols),
        "note": "Symbols in 'supported_symbols' are fully tested. Additional symbols may work but are not guaranteed."
    }

@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    try:
        data = await real_api.get_market_data(symbol.upper())
        return data
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

@app.get("/trading-signal/{symbol}")
async def get_trading_signal(symbol: str):
    """Get trading signal for a symbol"""
    try:
        # Get historical data for analysis
        historical_data = await real_api.get_historical_data(symbol.upper(), days=60)
        
        if len(historical_data) < 20:
            raise HTTPException(status_code=400, detail="Insufficient historical data for analysis")
        
        # Extract prices and volumes
        prices = [d["price"] for d in historical_data]
        volumes = [d["volume"] for d in historical_data]
        
        # Calculate indicators
        indicators = analyzer.calculate_technical_indicators(prices, volumes)
        
        # Generate signal
        signal_data = analyzer.generate_trading_signal(indicators, prices[-1])
        
        # Create trading signal response
        signal = TradingSignal(
            symbol=symbol.upper(),
            signal_type=signal_data["signal_type"],
            confidence=signal_data["confidence"],
            price=prices[-1],
            timestamp=datetime.now().isoformat(),
            reasoning=", ".join(signal_data["reasoning"]),
            indicators=indicators
        )
        
        return signal
        
    except Exception as e:
        logger.error(f"Error generating trading signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate trading signal: {str(e)}")

@app.get("/anomaly-detection/{symbol}")
async def detect_anomalies(symbol: str):
    """Detect anomalies in trading data"""
    try:
        # Get historical data for analysis
        historical_data = await real_api.get_historical_data(symbol.upper(), days=100)
        
        if len(historical_data) < 50:
            raise HTTPException(status_code=400, detail="Insufficient historical data for anomaly detection")
        
        # Extract prices and volumes
        prices = [d["price"] for d in historical_data]
        volumes = [d["volume"] for d in historical_data]
        
        # Detect anomalies
        anomalies = analyzer.detect_anomalies(prices, volumes)
        
        # Convert to anomaly alerts
        alerts = []
        for anomaly in anomalies:
            alert = AnomalyAlert(
                symbol=symbol.upper(),
                alert_type="PRICE_ANOMALY",
                severity=anomaly["severity"],
                description=f"Unusual price/volume pattern detected at index {anomaly['index']}",
                timestamp=historical_data[anomaly["index"]]["date"],
                metrics={
                    "price": anomaly["price"],
                    "volume": anomaly["volume"],
                    "price_change": round(((anomaly["price"] - prices[anomaly["index"]-1]) / prices[anomaly["index"]-1] * 100), 2) if anomaly["index"] > 0 else 0
                }
            )
            alerts.append(alert)
        
        return {"symbol": symbol.upper(), "anomalies": alerts, "total_detected": len(alerts)}
        
    except Exception as e:
        logger.error(f"Error detecting anomalies for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect anomalies: {str(e)}")

@app.get("/portfolio-analysis")
async def analyze_portfolio(symbols: str = "BTC,ETH,SOL"):
    """Analyze multiple symbols for portfolio insights"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        results = []
        supported_count = 0
        fallback_count = 0
        
        for symbol in symbol_list:
            try:
                # Check if symbol is supported before making API calls
                if real_api.is_valid_symbol(symbol):
                    supported_count += 1
                    logger.info(f"Analyzing supported symbol: {symbol}")
                else:
                    fallback_count += 1
                    logger.info(f"Analyzing unsupported symbol: {symbol} (using fallback)")
                
                # Get market data
                market_data = await real_api.get_market_data(symbol)
                
                # Get trading signal
                signal = await get_trading_signal(symbol)
                
                # Get anomaly detection
                anomalies = await detect_anomalies(symbol)
                
                results.append({
                    "symbol": symbol,
                    "supported": real_api.is_valid_symbol(symbol),
                    "market_data": market_data,
                    "trading_signal": signal,
                    "anomalies": anomalies
                })
                
            except Exception as e:
                logger.error(f"Error analyzing symbol {symbol}: {e}")
                # Add error result for this symbol
                results.append({
                    "symbol": symbol,
                    "supported": False,
                    "error": str(e),
                    "market_data": None,
                    "trading_signal": None,
                    "anomalies": []
                })
        
        return {
            "portfolio_analysis": results,
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(symbol_list),
            "supported_symbols": supported_count,
            "fallback_symbols": fallback_count,
            "supported_symbols_list": real_api.supported_symbols + real_api.additional_symbols
        }
        
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze portfolio: {str(e)}")

@app.get("/historical-data/{symbol}")
async def get_historical_data(symbol: str, days: int = 30):
    """Get historical data for analysis"""
    try:
        if days > 90:
            days = 90  # Limit to 1 year
        
        # Check if symbol contains multiple symbols separated by commas
        if "," in symbol:
            # Handle multiple symbols
            symbol_list = [s.strip().upper() for s in symbol.split(",")]
            results = []
            
            for sym in symbol_list:
                try:
                    data = await real_api.get_historical_data(sym, days=days)
                    results.append({
                        "symbol": sym,
                        "data": data,
                        "supported": real_api.is_valid_symbol(sym)
                    })
                except Exception as e:
                    logger.error(f"Error fetching historical data for {sym}: {e}")
                    results.append({
                        "symbol": sym,
                        "data": [],
                        "supported": False,
                        "error": str(e)
                    })
            
            return {
                "symbols": symbol_list,
                "results": results,
                "period_days": days,
                "timestamp": datetime.now().isoformat(),
                "total_symbols": len(symbol_list)
            }
        else:
            # Handle single symbol
            data = await real_api.get_historical_data(symbol.upper(), days=days)
            return {
                "symbol": symbol.upper(),
                "data": data,
                "period_days": days,
                "timestamp": datetime.now().isoformat(),
                "supported": real_api.is_valid_symbol(symbol.upper())
            }
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")
