# MECE Cart Segmentation

A Python tool for segmenting recent cart abandoners using **MECE (Mutually Exclusive, Collectively Exhaustive)** rules with scoring and size constraints.  
Outputs ranked segment summaries in CSV or JSON format.

## 🔍 Features
- Rule-based segmentation using quantiles/thresholds or custom logic  
- Mutually exclusive and collectively exhaustive segments  
- Enforces min/max segment sizes; merges tiny ones and can split oversized ones  
- Computes segment metrics and overall weighted scores  
- Exports CSV / JSON summaries for marketing retention workflows

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+ recommended
- (Optional) a virtual environment via `venv` or `conda`

### Installation
```bash
# Clone the repo
git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName

# Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# Install dependencies (if you add a requirements.txt later)
# pip install -r requirements.txt
```

### Usage
```bash
python mece_cart_segmentation.py --input sample_cart_abandoners.csv --output segments_output.csv
```
> Replace file names/paths with yours. Add any extra CLI flags here once finalized.

## 🧩 Example Output (illustrative)
| Segment | Size | Score | Notes |
|---|---:|---:|---|
| High Intenders | 100 | 0.95 | Many recent sessions, high AOV |
| Medium Interest | 250 | 0.60 | Moderate engagement |
| Low Engagement | 500 | 0.20 | Least likely to convert |

## 🤝 Contributing
1. Fork the repo  
2. Create a feature branch: `git checkout -b feature/new-rule`  
3. Commit changes: `git commit -m "Add new segmentation rule"`  
4. Push: `git push origin feature/new-rule`  
5. Open a Pull Request

Please follow the existing style and update docs/examples when relevant.

## 📄 License
MIT (or change to your preferred license).

## 🗂️ Topics
`mece` `segmentation` `cart-abandonment` `marketing-analytics` `python`
