"""keep_awake.py - visit the app every couple of hours so Streamlit Community
Cloud counts it as in use and doesn't put it to sleep (it sleeps after about
12 hours with no visitors; the daily data commits don't count as visits).

A slow page load is not an error here: the visit itself is what keeps the app
awake, so this always finishes quietly and just logs what it saw."""

from playwright.sync_api import sync_playwright

URL = "https://massachusetts-peak-electricity-demand-forecasting-jlapsnfvdheu.streamlit.app/"

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(90_000)  # stay connected so it counts as a visit
        print("Visited the app.")
        browser.close()
except Exception as e:
    print(f"Visit did not complete this time ({str(e)[:150]}); the next one is in 2 hours.")
