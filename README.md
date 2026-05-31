# Fashion Retail Shelf Optimizer

This project helps a fashion retail store decide which products to place on shelves, where exactly to put them, and how many units to display. Instead of doing this manually, we use an algorithm called NSGA-II to figure out the best layout automatically.

It looks at 10 products and 12 shelf locations across 4 racks and tries to maximize profit, visibility, cross-selling, and shelf space usage all at the same time.

---

## What is inside this project

```
fashion_retail_optimizer/
├── data/
│   └── retail_data.xlsx       ← the Excel file with all product and shelf data
├── src/
│   ├── data_loader.py         ← reads and cleans the Excel data
│   ├── problem.py             ← defines the optimization problem for the algorithm
│   ├── optimizer.py           ← runs the NSGA-II algorithm and picks the best layout
│   └── visualizer.py         ← creates charts from the results
├── app.py                     ← the web dashboard (run this to use the project)
└── requirements.txt           ← list of Python packages needed
```

---

## How to set it up

First clone the repo and go into the folder

```bash
git clone https://github.com/dark-mind-x/fashion_retail_optimizer.git
cd fashion_retail_optimizer
```

Create a virtual environment so packages don't mix with your system Python

```bash
python3 -m venv venv
source venv/bin/activate
```

Install all the required packages

```bash
pip install -r requirements.txt
```

---

## How to run it

To open the dashboard in your browser

```bash
streamlit run app.py
```

Then go to `http://localhost:8501` in your browser. You will see the dashboard. Use the sidebar to adjust settings and click the **Run Optimizer** button. It takes around 30 to 60 seconds to finish.

---

## If you just want to test individual files

Run only the data loader to check if the Excel file is being read correctly

```bash
python src/data_loader.py
```

Run only the optimizer without the dashboard

```bash
python src/optimizer.py
```

Run only the visualizer to generate and save the charts

```bash
python src/visualizer.py
```

---

## What the dashboard shows

- The optimized shelf layout as a grid (which product goes on which rack and shelf)
- Profit improvement compared to the minimum facing baseline
- A Pareto front chart showing all the trade-off solutions the algorithm found
- A before vs after chart comparing old facings and new facings per product
- A download button to export the result as a CSV file

---

## Requirements

- Python 3.8 or above
- The packages in requirements.txt (pymoo, pandas, streamlit, matplotlib, etc.)
- The retail_data.xlsx file in the data folder

---

## Built with

- Python 3
- pymoo — for the NSGA-II algorithm
- pandas and numpy — for data processing
- matplotlib — for the charts
- Streamlit — for the web dashboard
