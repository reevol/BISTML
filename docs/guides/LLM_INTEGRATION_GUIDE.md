# LLM Integration Guide for BIST News Analysis

## Overview

The LLM Integration module (`src/models/nlp/llm_integration.py`) provides sophisticated AI-powered analysis of Turkish financial news articles, synthesizing multiple news sources into nuanced impact scores for BIST stocks.

## Features

### Core Capabilities

1. **Multi-Article Synthesis**: Analyzes multiple news articles simultaneously to provide comprehensive impact assessment
2. **Turkish Financial Context**: Specialized prompt engineering for Turkish market dynamics
3. **Multiple LLM Support**:
   - OpenAI GPT models (GPT-3.5-turbo, GPT-4, GPT-4-turbo)
   - Local LLMs (Llama 2, Mistral, Gemma)
4. **Nuanced Scoring**:
   - Impact score from -1.0 (very negative) to +1.0 (very positive)
   - Confidence levels
   - Risk assessment
5. **Comprehensive Analysis**:
   - Short-term outlook (1-7 days)
   - Medium-term outlook (1-3 months)
   - Sector impact analysis
   - Macro factor identification
6. **Performance Optimization**:
   - Result caching with configurable TTL
   - Batch processing support
   - Rate limiting for API calls

## Installation

### Basic Requirements

```bash
# Install OpenAI package
pip install openai>=1.0.0

# For local LLM support
pip install transformers torch
```

### Optional Dependencies

```bash
# For GPU acceleration with local LLMs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For specific model support
pip install accelerate bitsandbytes  # For efficient model loading
```

## Configuration

### Environment Setup

Create a `.env` file or set environment variables:

```bash
# OpenAI Configuration
export OPENAI_API_KEY="sk-your-api-key-here"

# Optional: Model preferences
export LLM_PROVIDER="openai"  # or "llama", "mistral", "gemma"
export LLM_MODEL_NAME="gpt-4-turbo-preview"
export LLM_TEMPERATURE="0.3"
```

### Local LLM Setup

For local LLMs, you'll need to download the models first:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Example: Download Llama 2 model
model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

## Usage

### Basic Usage

```python
from src.models.nlp.llm_integration import create_analyzer

# Create analyzer with OpenAI
analyzer = create_analyzer(
    provider="openai",
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    language="turkish"
)

# Sample news articles
news_articles = [
    {
        'title': 'Akbank\'ın 9 aylık karı beklentileri aştı',
        'source': 'Bloomberg HT',
        'published_date': '2024-11-15',
        'content': 'Akbank güçlü finansal sonuçlar açıkladı...',
    },
    # ... more articles
]

# Analyze
result = analyzer.analyze_news_impact("AKBNK", news_articles)

# Access results
print(f"Impact Score: {result.impact_score}")
print(f"Category: {result.impact_category.value}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

### Integration with News Collector

```python
from src.data.collectors.news_collector import NewsCollectorManager
from src.models.nlp.llm_integration import create_analyzer

# Collect news
news_manager = NewsCollectorManager()
articles = news_manager.collect_for_stock("THYAO", days_back=7)

# Convert to dicts for analysis
article_dicts = [article.to_dict() for article in articles]

# Analyze with LLM
analyzer = create_analyzer(provider="openai")
result = analyzer.analyze_news_impact("THYAO", article_dicts)

print(f"Analyzed {result.analyzed_articles} articles")
print(f"Impact: {result.impact_score:+.2f}")
print(f"Outlook: {result.short_term_outlook}")
```

### Batch Analysis

```python
# Analyze multiple stocks at once
stock_news_map = {
    'AKBNK': akbank_news,
    'THYAO': thy_news,
    'GARAN': garanti_news,
}

results = analyzer.batch_analyze(stock_news_map)

for stock_code, result in results.items():
    print(f"{stock_code}: {result.impact_score:+.2f} ({result.impact_category.value})")
```

### Using Local LLMs

```python
# Create analyzer with local Llama model
analyzer = create_analyzer(
    provider="llama",
    model_name="meta-llama/Llama-2-13b-chat-hf",
    temperature=0.3
)

# Use the same way as OpenAI
result = analyzer.analyze_news_impact("EREGL", news_articles)
```

## Result Structure

The `NewsImpactScore` object contains:

```python
@dataclass
class NewsImpactScore:
    stock_code: str              # BIST stock code
    impact_category: SentimentImpact  # Categorical assessment
    impact_score: float          # -1.0 to +1.0
    confidence: float            # 0.0 to 1.0
    reasoning: str               # Detailed explanation
    key_factors: List[str]       # Main factors influencing score
    analyzed_articles: int       # Number of articles analyzed
    analysis_timestamp: datetime # When analysis was performed
    short_term_outlook: str      # 1-7 day outlook
    medium_term_outlook: str     # 1-3 month outlook (optional)
    sector_impact: float         # Sector-wide impact (optional)
    macro_factors: List[str]     # Macroeconomic factors (optional)
    risk_level: str              # "low", "medium", "high"
```

### Impact Categories

```python
class SentimentImpact(Enum):
    VERY_POSITIVE = "very_positive"       # Impact > +0.7
    POSITIVE = "positive"                  # Impact > +0.3
    SLIGHTLY_POSITIVE = "slightly_positive" # Impact > 0
    NEUTRAL = "neutral"                    # Impact ≈ 0
    SLIGHTLY_NEGATIVE = "slightly_negative" # Impact < 0
    NEGATIVE = "negative"                  # Impact < -0.3
    VERY_NEGATIVE = "very_negative"       # Impact < -0.7
```

## Advanced Features

### Caching

```python
# Enable caching (default: enabled with 1-hour TTL)
analyzer = create_analyzer(
    provider="openai",
    cache_enabled=True,
    cache_ttl=3600  # seconds
)

# Clear cache when needed
analyzer.clear_cache()
```

### Custom Temperature

```python
# Lower temperature for more deterministic results
analyzer_conservative = create_analyzer(temperature=0.1)

# Higher temperature for more creative analysis
analyzer_creative = create_analyzer(temperature=0.7)
```

### Language Selection

```python
# Turkish analysis (default)
analyzer_tr = create_analyzer(language="turkish")

# English analysis
analyzer_en = create_analyzer(language="english")
```

## Prompt Engineering

The module uses specialized prompts for Turkish financial analysis:

### Turkish System Prompt
- BIST market expertise
- Turkish financial terminology
- Context awareness (macro factors, regulations, etc.)
- Objective, data-driven analysis

### Analysis Considerations
1. **Company-specific news**: Earnings, management, projects
2. **Sector dynamics**: Competition, trends
3. **Macro factors**: Interest rates, inflation, TRY exchange rate
4. **Regulatory changes**: BDDK, SPK regulations
5. **Foreign investment**: Capital flows, sentiment
6. **Technical levels**: Support/resistance

## Performance Considerations

### OpenAI API

- **Rate Limits**: Implement delays between calls (0.5s default)
- **Cost**: GPT-4 is more expensive but provides better analysis
- **Latency**: Typically 2-5 seconds per analysis

### Local LLMs

- **GPU Required**: For acceptable performance with 7B+ models
- **Memory**: 13B model requires ~26GB VRAM
- **Speed**: Slower than API calls but no per-request cost
- **Quality**: May be lower than GPT-4 for Turkish financial content

### Optimization Tips

1. **Batch Processing**: Analyze multiple stocks together
2. **Caching**: Reuse results for same news sets
3. **Content Truncation**: Limit article content to 2000 chars
4. **Model Selection**: Use GPT-3.5-turbo for faster/cheaper analysis

## Examples

See `examples/llm_news_analysis_example.py` for comprehensive examples:

1. **Basic Analysis**: Simple single-stock analysis
2. **Real News Collection**: Integration with news collectors
3. **Batch Analysis**: Multiple stocks simultaneously
4. **Model Comparison**: Compare different LLM models
5. **Trading Signals**: Generate trading signals from news

Run examples:

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run examples
python examples/llm_news_analysis_example.py
```

## Integration with Trading System

### Combining with Technical Analysis

```python
from src.features.technical.momentum import calculate_rsi
from src.models.nlp.llm_integration import create_analyzer

# Get technical score
technical_score = calculate_rsi(price_data)

# Get news sentiment score
news_result = analyzer.analyze_news_impact(stock_code, news)
sentiment_score = news_result.impact_score

# Combine scores
combined_score = (0.6 * technical_score + 0.4 * sentiment_score)

# Generate signal
if combined_score > 0.7 and news_result.confidence > 0.8:
    signal = "STRONG BUY"
```

### Portfolio Alert System

```python
# Analyze holdings
for stock in portfolio.holdings:
    recent_news = collect_news(stock.code, days_back=3)
    result = analyzer.analyze_news_impact(stock.code, recent_news)

    if result.impact_score < -0.5 and result.confidence > 0.7:
        send_alert(f"SELL WARNING: {stock.code} - {result.reasoning}")
```

## Troubleshooting

### Common Issues

#### OpenAI API Errors

```python
# Handle rate limits
try:
    result = analyzer.analyze_news_impact(stock_code, news)
except Exception as e:
    if "rate_limit" in str(e):
        time.sleep(60)  # Wait and retry
        result = analyzer.analyze_news_impact(stock_code, news)
```

#### Memory Issues with Local LLMs

```python
# Use quantization for lower memory usage
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

#### Turkish Character Encoding

```python
# Ensure UTF-8 encoding
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

## Best Practices

1. **Always Check Confidence**: Don't act on low-confidence scores
2. **Combine Multiple Signals**: Use news analysis with technical/fundamental
3. **Monitor Cache**: Clear cache for time-sensitive analysis
4. **Rate Limiting**: Respect API limits to avoid blocks
5. **Error Handling**: Always wrap API calls in try-except
6. **Validate Results**: Sanity-check LLM outputs before using

## API Reference

### Main Classes

- `LLMNewsAnalyzer`: Main analysis class
- `NewsImpactScore`: Result data class
- `TurkishFinancialPrompts`: Prompt templates
- `LLMProvider`: Supported providers enum
- `SentimentImpact`: Impact categories enum

### Key Methods

- `analyze_news_impact()`: Analyze news for single stock
- `batch_analyze()`: Analyze news for multiple stocks
- `clear_cache()`: Clear analysis cache
- `create_analyzer()`: Convenience function to create analyzer

## Further Development

Potential enhancements:

1. **Fine-tuned Models**: Train Turkish financial sentiment models
2. **Multi-modal Analysis**: Include charts, images from news
3. **Real-time Streaming**: WebSocket integration for live news
4. **Ensemble Scoring**: Combine multiple LLM outputs
5. **Explanation Generation**: Detailed trade rationale
6. **Risk Quantification**: More sophisticated risk models

## Support

For issues or questions:
- Check examples in `examples/llm_news_analysis_example.py`
- Review module documentation in `src/models/nlp/llm_integration.py`
- See project documentation in `project.md`

## License

Part of the BIST AI Trading System project.
