#!/usr/bin/env python3
"""
Comprehensive test suite for UltraX Trading Analyzer Backend
Tests all functions, classes, and API endpoints
"""

import pytest
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the current directory to Python path to import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the classes and functions from main.py
from main import (
    RealTradingAPI, 
    TradingAnalyzer, 
    TradingSignal, 
    AnomalyAlert, 
    MarketData,
    app
)

# Test configuration
TEST_SYMBOLS = ["BTC", "ETH", "SOL", "ADA", "BNB"]
INVALID_SYMBOLS = ["INVALID", "FAKE", "TEST123"]

class TestRealTradingAPI:
    """Test the RealTradingAPI class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.api = RealTradingAPI()
    
    def test_init(self):
        """Test API initialization"""
        assert self.api.base_url == "https://api.binance.com/api/v3"
        assert len(self.api.supported_symbols) > 0
        assert len(self.api.symbol_mapping) > 0
        assert len(self.api.additional_symbols) > 0
    
    def test_is_valid_symbol(self):
        """Test symbol validation"""
        # Test valid symbols
        for symbol in TEST_SYMBOLS:
            assert self.api.is_valid_symbol(symbol) == True
        
        # Test invalid symbols
        for symbol in INVALID_SYMBOLS:
            assert self.api.is_valid_symbol(symbol) == False
        
        # Test USDT pairs
        assert self.api.is_valid_symbol("BTCUSDT") == True
        assert self.api.is_valid_symbol("ETHUSDT") == True
    
    def test_symbol_mapping(self):
        """Test symbol mapping functionality"""
        assert self.api.symbol_mapping["BTC"] == "BTCUSDT"
        assert self.api.symbol_mapping["ETH"] == "ETHUSDT"
        assert self.api.symbol_mapping["SOL"] == "SOLUSDT"
    
    @pytest.mark.asyncio
    async def test_get_market_data_fallback(self):
        """Test fallback market data generation"""
        # Test with invalid symbol to trigger fallback
        data = await self.api.get_market_data("INVALID")
        
        assert isinstance(data, MarketData)
        assert data.symbol == "INVALID"
        assert data.price > 0
        assert data.volume > 0
        assert data.timestamp is not None
        assert data.change_24h is not None
    
    @pytest.mark.asyncio
    async def test_get_historical_data_fallback(self):
        """Test fallback historical data generation"""
        # Test with invalid symbol to trigger fallback
        data = await self.api.get_historical_data("INVALID", days=30)
        
        assert isinstance(data, list)
        assert len(data) == 30
        
        for entry in data:
            assert "date" in entry
            assert "price" in entry
            assert "volume" in entry
            assert "high" in entry
            assert "low" in entry
            assert entry["price"] > 0
    
    @pytest.mark.asyncio
    async def test_get_historical_data_fallback_single_day(self):
        """Test fallback data for single day request"""
        data = await self.api._get_fallback_data("BTC", days=1)
        
        assert isinstance(data, MarketData)
        assert data.symbol == "BTC"
        assert data.price > 0
        assert data.volume > 0

class TestTradingAnalyzer:
    """Test the TradingAnalyzer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = TradingAnalyzer()
    
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculations"""
        # Generate test data
        np.random.seed(42)
        prices = [100 + np.random.normal(0, 2) for _ in range(60)]
        volumes = [1000000 + np.random.normal(0, 100000) for _ in range(60)]
        
        indicators = self.analyzer.calculate_technical_indicators(prices, volumes)
        
        # Check that all expected indicators are present
        expected_keys = ["sma_20", "sma_50", "rsi", "volume_ratio", "macd", "price_change_24h"]
        for key in expected_keys:
            assert key in indicators
        
        # Check value ranges
        assert 0 <= indicators["rsi"] <= 100
        assert indicators["volume_ratio"] > 0
        assert indicators["sma_20"] > 0
        assert indicators["sma_50"] > 0
    
    def test_calculate_technical_indicators_insufficient_data(self):
        """Test indicators with insufficient data"""
        prices = [100, 101, 102]  # Less than 20 data points
        volumes = [1000000, 1000001, 1000002]
        
        indicators = self.analyzer.calculate_technical_indicators(prices, volumes)
        assert indicators == {}
    
    def test_generate_trading_signal(self):
        """Test trading signal generation"""
        # Test oversold condition (RSI < 30)
        indicators = {
            "rsi": 25,
            "sma_20": 100,
            "sma_50": 95,
            "volume_ratio": 1.2,
            "macd": 0.5,
            "price_change_24h": -2.0
        }
        
        signal = self.analyzer.generate_trading_signal(indicators, 98.0)
        
        assert signal["signal_type"] == "BUY"
        assert signal["confidence"] > 0.5
        assert len(signal["reasoning"]) > 0
        assert "RSI indicates oversold conditions" in signal["reasoning"]
    
    def test_generate_trading_signal_overbought(self):
        """Test trading signal for overbought condition"""
        # Test overbought condition (RSI > 70)
        indicators = {
            "rsi": 75,
            "sma_20": 100,
            "sma_50": 105,
            "volume_ratio": 1.8,
            "macd": -0.3,
            "price_change_24h": 3.0
        }
        
        signal = self.analyzer.generate_trading_signal(indicators, 102.0)
        
        assert signal["signal_type"] == "SELL"
        assert signal["confidence"] > 0.5
        assert "RSI indicates overbought conditions" in signal["reasoning"]
    
    def test_generate_trading_signal_hold(self):
        """Test trading signal for hold condition"""
        # Test neutral condition
        indicators = {
            "rsi": 50,
            "sma_20": 100,
            "sma_50": 100,
            "volume_ratio": 1.0,
            "macd": 0.0,
            "price_change_24h": 0.5
        }
        
        signal = self.analyzer.generate_trading_signal(indicators, 100.0)
        
        assert signal["signal_type"] == "HOLD"
        assert signal["confidence"] == 0.5
        assert "No clear signal based on current indicators" in signal["reasoning"]
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        # Generate test data with some anomalies
        np.random.seed(42)
        prices = [100 + np.random.normal(0, 1) for _ in range(50)]
        volumes = [1000000 + np.random.normal(0, 50000) for _ in range(50)]
        
        # Add some obvious anomalies
        prices[25] = 150  # Price spike
        volumes[30] = 5000000  # Volume spike
        
        anomalies = self.analyzer.detect_anomalies(prices, volumes)
        
        # Should detect some anomalies
        assert isinstance(anomalies, list)
        assert len(anomalies) > 0
        
        for anomaly in anomalies:
            assert "index" in anomaly
            assert "price" in anomaly
            assert "volume" in anomaly
            assert "features" in anomaly
            assert "severity" in anomaly
            assert anomaly["severity"] == "MEDIUM"
    
    def test_detect_anomalies_insufficient_data(self):
        """Test anomaly detection with insufficient data"""
        prices = [100, 101, 102]  # Less than 20 data points
        volumes = [1000000, 1000001, 1000002]
        
        anomalies = self.analyzer.detect_anomalies(prices, volumes)
        assert anomalies == []

class TestDataModels:
    """Test the Pydantic data models"""
    
    def test_trading_signal_model(self):
        """Test TradingSignal model"""
        signal = TradingSignal(
            symbol="BTC",
            signal_type="BUY",
            confidence=0.8,
            price=45000.0,
            timestamp="2024-01-01T00:00:00",
            reasoning="Strong buy signal",
            indicators={"rsi": 30, "sma_20": 44000}
        )
        
        assert signal.symbol == "BTC"
        assert signal.signal_type == "BUY"
        assert signal.confidence == 0.8
        assert signal.price == 45000.0
        assert signal.reasoning == "Strong buy signal"
        assert len(signal.indicators) == 2
    
    def test_anomaly_alert_model(self):
        """Test AnomalyAlert model"""
        alert = AnomalyAlert(
            symbol="ETH",
            alert_type="VOLUME_SPIKE",
            severity="HIGH",
            description="Unusual volume detected",
            timestamp="2024-01-01T00:00:00",
            metrics={"volume": 5000000, "price": 2800}
        )
        
        assert alert.symbol == "ETH"
        assert alert.alert_type == "VOLUME_SPIKE"
        assert alert.severity == "HIGH"
        assert alert.description == "Unusual volume detected"
        assert len(alert.metrics) == 2
    
    def test_market_data_model(self):
        """Test MarketData model"""
        data = MarketData(
            symbol="SOL",
            price=95.0,
            volume=2000000,
            timestamp="2024-01-01T00:00:00",
            change_24h=5.2,
            market_cap=190000000
        )
        
        assert data.symbol == "SOL"
        assert data.price == 95.0
        assert data.volume == 2000000
        assert data.change_24h == 5.2
        assert data.market_cap == 190000000

class TestAPIEndpoints:
    """Test the FastAPI endpoints"""
    
    def setup_method(self):
        """Set up test client"""
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "UltraX Trading Analyzer API"
        assert data["status"] == "running"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_status_endpoint(self):
        """Test server status endpoint"""
        response = self.client.get("/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "running"
        assert data["version"] == "1.0.0"
        assert "supported_symbols" in data
        assert "endpoints" in data
        assert len(data["endpoints"]) > 0
    
    def test_supported_symbols_endpoint(self):
        """Test supported symbols endpoint"""
        response = self.client.get("/supported-symbols")
        assert response.status_code == 200
        
        data = response.json()
        assert "supported_symbols" in data
        assert "additional_symbols" in data
        assert "symbol_mapping" in data
        assert "total_supported" in data
        assert data["total_supported"] > 0
    
    def test_market_data_endpoint(self):
        """Test market data endpoint"""
        response = self.client.get("/market-data/BTC")
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "BTC"
        assert "price" in data
        assert "volume" in data
        assert "timestamp" in data
        assert "change_24h" in data
    
    def test_market_data_invalid_symbol(self):
        """Test market data endpoint with invalid symbol"""
        response = self.client.get("/market-data/INVALID")
        # Should still work due to fallback
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "INVALID"
        assert data["price"] > 0
    
    def test_trading_signal_endpoint(self):
        """Test trading signal endpoint"""
        response = self.client.get("/trading-signal/BTC")
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "BTC"
        assert data["signal_type"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= data["confidence"] <= 1
        assert "price" in data
        assert "reasoning" in data
        assert "indicators" in data
    
    def test_anomaly_detection_endpoint(self):
        """Test anomaly detection endpoint"""
        response = self.client.get("/anomaly-detection/BTC")
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "BTC"
        assert "anomalies" in data
        assert "total_detected" in data
        assert isinstance(data["anomalies"], list)
    
    def test_portfolio_analysis_endpoint(self):
        """Test portfolio analysis endpoint"""
        response = self.client.get("/portfolio-analysis?symbols=BTC,ETH,SOL")
        assert response.status_code == 200
        
        data = response.json()
        assert "portfolio_analysis" in data
        assert "timestamp" in data
        assert data["total_symbols"] == 3
        assert "supported_symbols" in data
        assert "fallback_symbols" in data
        
        # Check individual symbol analysis
        for analysis in data["portfolio_analysis"]:
            assert "symbol" in analysis
            assert "supported" in analysis
            assert "market_data" in analysis
            assert "trading_signal" in analysis
            assert "anomalies" in analysis
    
    def test_historical_data_endpoint_single(self):
        """Test historical data endpoint for single symbol"""
        response = self.client.get("/historical-data/BTC?days=30")
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "BTC"
        assert "data" in data
        assert data["period_days"] == 30
        assert len(data["data"]) == 30
        
        # Check data structure
        for entry in data["data"]:
            assert "date" in entry
            assert "price" in entry
            assert "volume" in entry
            assert "high" in entry
            assert "low" in entry
    
    def test_historical_data_endpoint_multiple(self):
        """Test historical data endpoint for multiple symbols"""
        response = self.client.get("/historical-data/BTC,ETH?days=7")
        assert response.status_code == 200
        
        data = response.json()
        assert "symbols" in data
        assert "results" in data
        assert data["total_symbols"] == 2
        assert data["period_days"] == 7
        
        # Check results for each symbol
        for result in data["results"]:
            assert "symbol" in result
            assert "data" in result
            assert "supported" in result
            assert len(result["data"]) == 7
    
    def test_historical_data_endpoint_invalid_symbol(self):
        """Test historical data endpoint with invalid symbol"""
        response = self.client.get("/historical-data/INVALID?days=30")
        # Should still work due to fallback
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "INVALID"
        assert "data" in data
        assert len(data["data"]) == 30

class TestIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.api = RealTradingAPI()
        self.analyzer = TradingAnalyzer()
    
    @pytest.mark.asyncio
    async def test_complete_trading_analysis_workflow(self):
        """Test complete workflow from market data to trading signal"""
        # Get market data
        market_data = await self.api.get_market_data("BTC")
        assert isinstance(market_data, MarketData)
        
        # Get historical data
        historical_data = await self.api.get_historical_data("BTC", days=60)
        assert len(historical_data) == 60
        
        # Extract prices and volumes
        prices = [d["price"] for d in historical_data]
        volumes = [d["volume"] for d in historical_data]
        
        # Calculate indicators
        indicators = self.analyzer.calculate_technical_indicators(prices, volumes)
        assert len(indicators) > 0
        
        # Generate trading signal
        signal = self.analyzer.generate_trading_signal(indicators, prices[-1])
        assert "signal_type" in signal
        assert "confidence" in signal
        assert "reasoning" in signal
        
        # Detect anomalies
        anomalies = self.analyzer.detect_anomalies(prices, volumes)
        assert isinstance(anomalies, list)
    
    @pytest.mark.asyncio
    async def test_multiple_symbols_analysis(self):
        """Test analysis of multiple symbols"""
        symbols = ["BTC", "ETH", "SOL"]
        results = []
        
        for symbol in symbols:
            try:
                # Get market data
                market_data = await self.api.get_market_data(symbol)
                assert isinstance(market_data, MarketData)
                
                # Get historical data
                historical_data = await self.api.get_historical_data(symbol, days=30)
                assert len(historical_data) == 30
                
                # Calculate indicators
                prices = [d["price"] for d in historical_data]
                volumes = [d["volume"] for d in historical_data]
                indicators = self.analyzer.calculate_technical_indicators(prices, volumes)
                
                # Generate signal
                signal = self.analyzer.generate_trading_signal(indicators, prices[-1])
                
                results.append({
                    "symbol": symbol,
                    "market_data": market_data,
                    "indicators": indicators,
                    "signal": signal
                })
                
            except Exception as e:
                pytest.fail(f"Failed to analyze {symbol}: {e}")
        
        assert len(results) == 3
        for result in results:
            assert "symbol" in result
            assert "market_data" in result
            assert "indicators" in result
            assert "signal" in result

def run_tests():
    """Run all tests"""
    print("Running UltraX Trading Analyzer Backend Tests...")
    print("=" * 50)
    
    # Test data models
    print("\n1. Testing Data Models...")
    test_models = TestDataModels()
    test_models.test_trading_signal_model()
    test_models.test_anomaly_alert_model()
    test_models.test_market_data_model()
    print("✓ Data models tests passed")
    
    # Test RealTradingAPI
    print("\n2. Testing RealTradingAPI...")
    test_api = TestRealTradingAPI()
    test_api.setup_method()  # Manually call setup
    test_api.test_init()
    test_api.test_is_valid_symbol()
    test_api.test_symbol_mapping()
    print("✓ RealTradingAPI tests passed")
    
    # Test TradingAnalyzer
    print("\n3. Testing TradingAnalyzer...")
    test_analyzer = TestTradingAnalyzer()
    test_analyzer.setup_method()  # Manually call setup
    test_analyzer.test_calculate_technical_indicators()
    test_analyzer.test_calculate_technical_indicators_insufficient_data()
    test_analyzer.test_generate_trading_signal()
    test_analyzer.test_generate_trading_signal_overbought()
    test_analyzer.test_generate_trading_signal_hold()
    test_analyzer.test_detect_anomalies()
    test_analyzer.test_detect_anomalies_insufficient_data()
    print("✓ TradingAnalyzer tests passed")
    
    # Test API endpoints
    print("\n4. Testing API Endpoints...")
    test_endpoints = TestAPIEndpoints()
    test_endpoints.setup_method()  # Manually call setup
    test_endpoints.test_root_endpoint()
    test_endpoints.test_health_endpoint()
    test_endpoints.test_status_endpoint()
    test_endpoints.test_supported_symbols_endpoint()
    test_endpoints.test_market_data_endpoint()
    test_endpoints.test_market_data_invalid_symbol()
    test_endpoints.test_trading_signal_endpoint()
    test_endpoints.test_anomaly_detection_endpoint()
    test_endpoints.test_portfolio_analysis_endpoint()
    test_endpoints.test_historical_data_endpoint_single()
    test_endpoints.test_historical_data_endpoint_multiple()
    test_endpoints.test_historical_data_endpoint_invalid_symbol()
    print("✓ API endpoints tests passed")
    
    # Test fallback functionality
    print("\n5. Testing Fallback Functionality...")
    async def test_fallbacks():
        test_api = TestRealTradingAPI()
        test_api.setup_method()  # Manually call setup
        await test_api.test_get_market_data_fallback()
        await test_api.test_get_historical_data_fallback()
        await test_api.test_get_historical_data_fallback_single_day()
    
    asyncio.run(test_fallbacks())
    print("✓ Fallback functionality tests passed")
    
    # Test integration
    print("\n6. Testing Integration...")
    async def test_integration():
        test_integration = TestIntegration()
        test_integration.setup_method()  # Manually call setup
        await test_integration.test_complete_trading_analysis_workflow()
        await test_integration.test_multiple_symbols_analysis()
    
    asyncio.run(test_integration())
    print("✓ Integration tests passed")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed successfully!")
    print("The UltraX Trading Analyzer Backend is working correctly.")

if __name__ == "__main__":
    run_tests() 