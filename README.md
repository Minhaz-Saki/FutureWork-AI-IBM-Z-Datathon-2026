# FutureWork AI

**A career-exploration tool built for the IBM Z Datathon 2026.**

FutureWork AI helps people explore other careers using their existing skills. It compares 910 O*NET occupations and displays Victorian employment projections when Australian occupation-mapping information is available.

## Run it on your computer

You need **Python 3** and an internet connection the first time to install the required packages.

**Windows:** Download the repository, extract it, and double-click **`START_WINDOWS.bat`**.

**Mac or Linux:** Open a terminal in the extracted project folder and run:

```bash
bash START_MAC_LINUX.sh
```

Then open **http://localhost:8899** in a browser. Leave the terminal window open while using the app.

If the launcher doesn't work, run these commands from the project folder:

```bash
python -m pip install -r requirements-ui.txt
python app.py --port 8899
```

## What it does

- Search 910 O*NET occupations and find similar career options.
- Compare transferable skills and potential skill gaps.
- Explore Victorian employment projections for Australian occupation groups.
- Label automatically suggested occupation mappings as **unverified**.

## Important limitations

Career similarity is **not** a hiring probability. Automatic US-to-Australian occupation mappings are **unverified** and can be incorrect. Victorian projections apply to broader occupation groups, **not individual job vacancies**. Qualifications and licensing requirements must be checked separately.

## Technology and sources

Python, pandas, NumPy, scikit-learn, and a lightweight browser interface. Developed and demonstrated on **IBM LinuxONE**; it can also run locally.

Data sources: [O*NET](https://www.onetcenter.org/database.html), [Victorian Skills Authority](https://www.vic.gov.au/employment-projections-dashboard), and [Jobs and Skills Australia](https://www.jobsandskills.gov.au/). Follow the original providers' licensing and attribution terms.

## Team

IBM Z Datathon 2026 — *Tech for Good*. Add team-member names here before submitting.
