#!/bin/sh
# Re-run the two prototype demos (Section III of the paper). Requires: git, python3, pip install gensim matplotlib scipy
set -e
cd "$(mktemp -d)"
git clone -q https://github.com/Haidriyam/statutory-semantic-intertia
git clone -q https://github.com/Haidriyam/diachronic-semantic-drift
python3 -c "import gensim;print('gensim',gensim.__version__)"
echo "== statutory-semantic-intertia =="; (cd statutory-semantic-intertia && python3 src/legal_inertia_engine.py --output /tmp/s.png 2>/dev/null)
echo "== diachronic-semantic-drift ==";  (cd diachronic-semantic-drift && python3 src/fossilization.py 2>/dev/null)
