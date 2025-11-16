#!/bin/bash
# Verification script for Macro Collector setup

echo "=========================================="
echo "Macro Collector Verification"
echo "=========================================="
echo ""

# Check files exist
echo "1. Checking files..."
files=(
    "src/data/collectors/macro_collector.py"
    "configs/data_sources.yaml"
    "examples/macro_data_example.py"
    "tests/test_data/test_macro_collector.py"
    "docs/MACRO_COLLECTOR_QUICKSTART.md"
    "src/data/collectors/README_MACRO.md"
    ".env.example"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
    fi
done

echo ""
echo "2. Checking Python syntax..."
python3 -m py_compile src/data/collectors/macro_collector.py 2>&1 && echo "  ✓ macro_collector.py syntax OK" || echo "  ✗ Syntax error"
python3 -m py_compile tests/test_data/test_macro_collector.py 2>&1 && echo "  ✓ test_macro_collector.py syntax OK" || echo "  ✗ Syntax error"
python3 -m py_compile examples/macro_data_example.py 2>&1 && echo "  ✓ macro_data_example.py syntax OK" || echo "  ✗ Syntax error"

echo ""
echo "3. Checking dependencies..."
packages=("pandas" "numpy" "fredapi" "evds" "yfinance")
for pkg in "${packages[@]}"; do
    python3 -c "import $pkg" 2>/dev/null && echo "  ✓ $pkg installed" || echo "  ⚠ $pkg not installed (run: pip install $pkg)"
done

echo ""
echo "4. Checking API keys..."
if [ -n "$FRED_API_KEY" ]; then
    echo "  ✓ FRED_API_KEY is set"
else
    echo "  ⚠ FRED_API_KEY not set (export FRED_API_KEY='your_key')"
fi

if [ -n "$EVDS_API_KEY" ]; then
    echo "  ✓ EVDS_API_KEY is set"
else
    echo "  ⚠ EVDS_API_KEY not set (export EVDS_API_KEY='your_key')"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "2. Set up API keys:"
echo "   - Get FRED key: https://fred.stlouisfed.org/docs/api/api_key.html"
echo "   - Get EVDS key: https://evds2.tcmb.gov.tr/"
echo ""
echo "3. Set environment variables:"
echo "   export FRED_API_KEY='your_key'"
echo "   export EVDS_API_KEY='your_key'"
echo ""
echo "4. Run examples:"
echo "   python examples/macro_data_example.py"
echo ""
echo "5. Run tests:"
echo "   pytest tests/test_data/test_macro_collector.py -v"
echo ""
echo "=========================================="

