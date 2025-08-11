#!/usr/bin/env python3
"""
Simple test runner for UltraX Trading Analyzer Backend
Runs all tests without requiring pytest installation
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_basic_tests():
    """Run basic functionality tests"""
    print("Running Basic Functionality Tests...")
    print("=" * 50)
    
    try:
        # Test imports
        from main import RealTradingAPI, TradingAnalyzer, TradingSignal, AnomalyAlert, MarketData, app
        print("✓ All imports successful")
        
        # Test API initialization
        api = RealTradingAPI()
        print(f"✓ RealTradingAPI initialized with {len(api.supported_symbols)} supported symbols")
        
        # Test analyzer initialization
        analyzer = TradingAnalyzer()
        print("✓ TradingAnalyzer initialized successfully")
        
        # Test data models
        signal = TradingSignal(
            symbol="BTC",
            signal_type="BUY",
            confidence=0.8,
            price=45000.0,
            timestamp=datetime.now().isoformat(),
            reasoning="Test signal",
            indicators={"rsi": 30}
        )
        print("✓ TradingSignal model works")
        
        alert = AnomalyAlert(
            symbol="ETH",
            alert_type="VOLUME_SPIKE",
            severity="HIGH",
            description="Test alert",
            timestamp=datetime.now().isoformat(),
            metrics={"volume": 5000000}
        )
        print("✓ AnomalyAlert model works")
        
        data = MarketData(
            symbol="SOL",
            price=95.0,
            volume=2000000,
            timestamp=datetime.now().isoformat(),
            change_24h=5.2
        )
        print("✓ MarketData model works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic tests failed: {e}")
        return False

def run_symbol_validation_tests():
    """Test symbol validation functionality"""
    print("\nRunning Symbol Validation Tests...")
    print("=" * 50)
    
    try:
        from main import RealTradingAPI
        
        api = RealTradingAPI()
        
        # Test valid symbols
        valid_symbols = ["BTC", "ETH", "SOL", "ADA", "BNB"]
        for symbol in valid_symbols:
            assert api.is_valid_symbol(symbol) == True
        print("✓ Valid symbol validation works")
        
        # Test invalid symbols
        invalid_symbols = ["INVALID", "FAKE", "TEST123"]
        for symbol in invalid_symbols:
            assert api.is_valid_symbol(symbol) == False
        print("✓ Invalid symbol validation works")
        
        # Test USDT pairs
        assert api.is_valid_symbol("BTCUSDT") == True
        assert api.is_valid_symbol("ETHUSDT") == True
        print("✓ USDT pair validation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Symbol validation tests failed: {e}")
        return False

def run_technical_analysis_tests():
    """Test technical analysis functionality"""
    print("\nRunning Technical Analysis Tests...")
    print("=" * 50)
    
    try:
        from main import TradingAnalyzer
        import numpy as np
        
        analyzer = TradingAnalyzer()
        
        # Generate test data
        np.random.seed(42)
        prices = [100 + np.random.normal(0, 2) for _ in range(60)]
        volumes = [1000000 + np.random.normal(0, 100000) for _ in range(60)]
        
        # Test technical indicators
        indicators = analyzer.calculate_technical_indicators(prices, volumes)
        assert len(indicators) > 0
        assert "rsi" in indicators
        assert "sma_20" in indicators
        print("✓ Technical indicators calculation works")
        
        # Test trading signal generation
        signal = analyzer.generate_trading_signal(indicators, prices[-1])
        assert "signal_type" in signal
        assert "confidence" in signal
        assert "reasoning" in signal
        print("✓ Trading signal generation works")
        
        # Test anomaly detection
        anomalies = analyzer.detect_anomalies(prices, volumes)
        assert isinstance(anomalies, list)
        print("✓ Anomaly detection works")
        
        return True
        
    except Exception as e:
        print(f"✗ Technical analysis tests failed: {e}")
        return False

async def run_api_tests():
    """Test API functionality"""
    print("\nRunning API Functionality Tests...")
    print("=" * 50)
    
    try:
        from main import RealTradingAPI, MarketData
        
        api = RealTradingAPI()
        
        # Test fallback market data
        data = await api.get_market_data("INVALID")
        print(f"   Debug: data type = {type(data)}")
        print(f"   Debug: data content = {data}")
        assert isinstance(data, MarketData)
        assert data.symbol == "INVALID"
        assert data.price > 0
        print("✓ Fallback market data works")
        
        # Test fallback historical data
        historical = await api.get_historical_data("INVALID", days=30)
        assert isinstance(historical, list)
        assert len(historical) == 30
        print("✓ Fallback historical data works")
        
        return True
        
    except Exception as e:
        print(f"✗ API tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_endpoint_tests():
    """Test FastAPI endpoints"""
    print("\nRunning API Endpoint Tests...")
    print("=" * 50)
    
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "UltraX Trading Analyzer API"
        print("✓ Root endpoint works")
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✓ Health endpoint works")
        
        # Test status endpoint
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        print("✓ Status endpoint works")
        
        # Test supported symbols endpoint
        response = client.get("/supported-symbols")
        assert response.status_code == 200
        data = response.json()
        assert "supported_symbols" in data
        print("✓ Supported symbols endpoint works")
        
        return True
        
    except Exception as e:
        print(f"✗ Endpoint tests failed: {e}")
        return False

async def run_integration_tests():
    """Test complete workflow integration"""
    print("\nRunning Integration Tests...")
    print("=" * 50)
    
    try:
        from main import RealTradingAPI, TradingAnalyzer
        
        api = RealTradingAPI()
        analyzer = TradingAnalyzer()
        
        # Test complete workflow for a symbol
        symbol = "BTC"
        
        # Get market data
        market_data = await api.get_market_data(symbol)
        from main import MarketData
        assert isinstance(market_data, MarketData)
        print("✓ Market data retrieval works")
        
        # Get historical data
        historical_data = await api.get_historical_data(symbol, days=60)
        assert len(historical_data) == 60
        print("✓ Historical data retrieval works")
        
        # Extract prices and volumes
        prices = [d["price"] for d in historical_data]
        volumes = [d["volume"] for d in historical_data]
        
        # Calculate indicators
        indicators = analyzer.calculate_technical_indicators(prices, volumes)
        assert len(indicators) > 0
        print("✓ Technical analysis integration works")
        
        # Generate trading signal
        signal = analyzer.generate_trading_signal(indicators, prices[-1])
        assert "signal_type" in signal
        print("✓ Trading signal integration works")
        
        # Detect anomalies
        anomalies = analyzer.detect_anomalies(prices, volumes)
        assert isinstance(anomalies, list)
        print("✓ Anomaly detection integration works")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration tests failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("UltraX Trading Analyzer Backend - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all test suites
    test_results.append(("Basic Functionality", run_basic_tests()))
    test_results.append(("Symbol Validation", run_symbol_validation_tests()))
    test_results.append(("Technical Analysis", run_technical_analysis_tests()))
    test_results.append(("API Functionality", await run_api_tests()))
    test_results.append(("API Endpoints", run_endpoint_tests()))
    test_results.append(("Integration", await run_integration_tests()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed successfully!")
        print("The UltraX Trading Analyzer Backend is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        sys.exit(1) 