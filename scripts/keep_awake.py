"""keep_awake.py - open the app in a real browser so Streamlit Community Cloud
doesn't put it to sleep (it sleeps after about 12 hours with no visitors;
the daily data commits don't count as visits). If it is already asleep,
click the wake-up button and wait until the app is running again."""

import sys

from playwright.sync_api import sync_playwright

URL = "https://massachusetts-peak-electricity-demand-forecasting-jlapsnfvdheu.streamlit.app/"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(10_000)
    wake = page.get_by_role("button", name="get this app back up")
    if wake.count() > 0:
        print("App was asleep; waking it up.")
        wake.first.click()
        page.wait_for_timeout(5_000)
    # Wait for the app itself (it runs inside an iframe) to finish loading.
    ok = False
    for _ in range(36):  # up to 6 minutes
        for frame in page.frames:
            try:
                if frame.locator('[data-testid="stApp"]').count() > 0:
                    ok = True
                    break
            except Exception:
                pass
        if ok:
            break
        page.wait_for_timeout(10_000)
    page.wait_for_timeout(20_000)  # stay connected briefly so it counts as a visit
    browser.close()
    if not ok:
        print("App did not finish loading.")
        sys.exit(1)
    print("App is awake.")
