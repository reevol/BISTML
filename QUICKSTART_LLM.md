# Quick Start: LLM News Analysis

Get started with LLM-powered news analysis in 5 minutes!

## 1. Install Dependencies

```bash
pip install openai>=1.0.0 tiktoken>=0.5.0
```

## 2. Set API Key

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

Or create a `.env` file:
```env
OPENAI_API_KEY=sk-your-api-key-here
```

## 3. Basic Usage

```python
from src.models.nlp.llm_integration import create_analyzer

# Create analyzer
analyzer = create_analyzer(
    provider="openai",
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    language="turkish"
)

# Sample news
news = [
    {
        'title': 'Akbank kar açıkladı',
        'content': 'Beklentileri aşan kar...',
        'source': 'Bloomberg HT',
        'published_date': '2024-11-15'
    }
]

# Analyze
result = analyzer.analyze_news_impact("AKBNK", news)

# Results
print(f"Impact Score: {result.impact_score:+.2f}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Reasoning: {result.reasoning}")
```

## 4. With News Collector

```python
from src.data.collectors.news_collector import NewsCollectorManager
from src.models.nlp.llm_integration import create_analyzer

# Collect news
manager = NewsCollectorManager()
articles = manager.collect_for_stock("THYAO", days_back=7)

# Analyze
analyzer = create_analyzer(provider="openai")
result = analyzer.analyze_news_impact(
    "THYAO",
    [a.to_dict() for a in articles]
)

print(f"Analyzed {result.analyzed_articles} articles")
print(f"Impact: {result.impact_category.value}")
print(f"Outlook: {result.short_term_outlook}")
```

## 5. Run Examples

```bash
python examples/llm_news_analysis_example.py
```

## Cost Estimation

### OpenAI GPT-3.5-turbo
- ~$0.002 per analysis (3 articles)
- 100 analyses = ~$0.20

### OpenAI GPT-4
- ~$0.03 per analysis (3 articles)
- 100 analyses = ~$3.00

## Tips

1. **Use GPT-3.5-turbo** for cost-effective analysis
2. **Enable caching** to avoid re-analyzing same news
3. **Batch process** multiple stocks to save time
4. **Check confidence** before acting on signals
5. **Combine with technical** analysis for best results

## Troubleshooting

### No API Key
```
ValueError: OpenAI API key not provided
```
**Solution**: Set `OPENAI_API_KEY` environment variable

### Rate Limit Error
```
openai.RateLimitError: Rate limit exceeded
```
**Solution**: Reduce request rate or upgrade API plan

### JSON Parse Error
```
Error parsing LLM response
```
**Solution**: Use higher temperature (0.3-0.5) or try GPT-4

## Next Steps

- See [LLM_INTEGRATION_GUIDE.md](LLM_INTEGRATION_GUIDE.md) for full documentation
- Check [examples/llm_news_analysis_example.py](examples/llm_news_analysis_example.py) for advanced usage
- Run tests: `python tests/test_llm_integration.py`

## Support

For detailed documentation, see:
- `LLM_INTEGRATION_GUIDE.md` - Complete guide
- `src/models/nlp/llm_integration.py` - Source code with docstrings
- `examples/llm_news_analysis_example.py` - Working examples
