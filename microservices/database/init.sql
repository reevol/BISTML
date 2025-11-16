-- BIST AI Trading System Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Stocks table
CREATE TABLE IF NOT EXISTS stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(200),
    sector VARCHAR(100),
    industry VARCHAR(100),
    is_bist_100 BOOLEAN DEFAULT FALSE,
    is_bist_30 BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_stocks_symbol ON stocks(symbol);
CREATE INDEX idx_stocks_sector ON stocks(sector);

-- OHLCV data tables (separated by timeframe)
CREATE TABLE IF NOT EXISTS ohlcv_1d (
    id BIGSERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(12, 4),
    high DECIMAL(12, 4),
    low DECIMAL(12, 4),
    close DECIMAL(12, 4),
    volume BIGINT,
    adj_close DECIMAL(12, 4),
    UNIQUE(stock_id, timestamp)
);

CREATE INDEX idx_ohlcv_1d_stock_time ON ohlcv_1d(stock_id, timestamp DESC);

-- Trading signals
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    timestamp TIMESTAMP NOT NULL,
    signal_type VARCHAR(20),  -- STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    confidence_score DECIMAL(5, 2),
    target_price DECIMAL(12, 4),
    stop_loss DECIMAL(12, 4),
    take_profit DECIMAL(12, 4),
    wai_score DECIMAL(5, 2),
    sentiment_score DECIMAL(5, 2),
    model_predictions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signals_stock_time ON signals(stock_id, timestamp DESC);
CREATE INDEX idx_signals_confidence ON signals(confidence_score DESC);

-- News articles
CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    source VARCHAR(100),
    url TEXT,
    published_date TIMESTAMP,
    sentiment_score DECIMAL(3, 2),
    impact_score DECIMAL(3, 2),
    stock_codes TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_published ON news_articles(published_date DESC);
CREATE INDEX idx_news_stocks ON news_articles USING GIN(stock_codes);

-- Portfolios
CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    user_id VARCHAR(100),
    initial_cash DECIMAL(15, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio positions
CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id),
    stock_id INTEGER REFERENCES stocks(id),
    shares DECIMAL(12, 4),
    avg_cost DECIMAL(12, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id),
    stock_id INTEGER REFERENCES stocks(id),
    transaction_type VARCHAR(10),  -- BUY, SELL
    shares DECIMAL(12, 4),
    price DECIMAL(12, 4),
    commission DECIMAL(10, 2),
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_transactions_portfolio ON transactions(portfolio_id, timestamp DESC);

-- Insert sample BIST 100 stocks
INSERT INTO stocks (symbol, name, sector, is_bist_100) VALUES
('THYAO', 'Türk Hava Yolları', 'Transportation', TRUE),
('GARAN', 'Garanti Bankası', 'Banking', TRUE),
('AKBNK', 'Akbank', 'Banking', TRUE),
('ISCTR', 'İş Bankası', 'Banking', TRUE),
('EREGL', 'Ereğli Demir Çelik', 'Steel', TRUE),
('SISE', 'Şişe Cam', 'Glass', TRUE),
('TUPRS', 'Tüpraş', 'Energy', TRUE),
('KCHOL', 'Koç Holding', 'Holding', TRUE),
('SAHOL', 'Sabancı Holding', 'Holding', TRUE),
('VAKBN', 'Vakıfbank', 'Banking', TRUE)
ON CONFLICT (symbol) DO NOTHING;

-- Create view for latest signals
CREATE OR REPLACE VIEW latest_signals AS
SELECT
    s.symbol,
    sig.timestamp,
    sig.signal_type,
    sig.confidence_score,
    sig.target_price,
    sig.wai_score,
    sig.sentiment_score
FROM signals sig
JOIN stocks s ON sig.stock_id = s.id
WHERE sig.timestamp > NOW() - INTERVAL '1 day'
ORDER BY sig.confidence_score DESC;
