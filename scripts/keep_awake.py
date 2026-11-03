"""keep_awake.py - open the app in a real browser so Streamlit Community Cloud
doesn't put it to sleep (it sleeps after about 12 hours with no visitors;
the daily data commits don't count as visits). If it is asleep, click the
wake-up button and wait until the app is running again. Tries three times
(reloading in between) before giving up, and on failure prints what the page
showed and saves a screenshot to the run's artifacts."""

import re
import sys

from playwright.sync_api import sync_playwright

URL = "https://massachusetts-peak-electricity-demand-forecasting-jlapsnfvdheu.streamlit.app/"
WAKE = re.compile(r"back up|wake|get this app", re.I)


def app_loaded(page):
    for frame in page.frames:
        try:
            if frame.locator('[data-testid="stApp"]').count() > 0 and \
               frame.locator('[data-testid="stMarkdownContainer"]').count() > 0:
                return True
        except Exception:
            pass
    return False


def try_once(page, attempt):
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
    for _ in range(48):  # up to 8 minutes; waking a sleeping app can take a few
        if app_loaded(page):
            return True
        for frame in page.frames:
            try:
                btn = frame.get_by_role("button", name=WAKE)
                if btn.count() > 0:
                    print(f"Attempt {attempt}: app was asleep; clicking the wake-up button.")
                    btn.first.click()
                    break
            except Exception:
                pass
        page.wait_for_timeout(10_000)
    return False


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    ok = False
    for attempt in (1, 2, 3):
        try:
            ok = try_once(page, attempt)
        except Exception as e:
            print(f"Attempt {attempt}: {str(e)[:200]}")
        if ok:
            break
        print(f"Attempt {attempt}: app did not load; trying again.")
    if ok:
        page.wait_for_timeout(30_000)  # stay connected so it counts as a visit
        print("App is awake.")
    else:
        try:
            page.screenshot(path="keep_awake_failure.png", full_page=True)
            print("Page text:", page.inner_text("body")[:1500])
            for f in page.frames:
                print("Frame:", f.url)
        except Exception:
            pass
    browser.close()
    sys.exit(0 if ok else 1)
