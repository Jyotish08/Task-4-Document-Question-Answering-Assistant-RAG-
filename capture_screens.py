"""Capture verified application screenshots of the running Streamlit app."""

import time
from playwright.sync_api import sync_playwright

def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 950})
        page.goto("http://localhost:8501")
        page.wait_for_selector("text=Document Question-Answering Assistant", timeout=15000)
        time.sleep(2)
        page.screenshot(path="reports/screenshots/app_home.png", full_page=True)
        print("1. Saved app_home.png")

        # 1. Click Supported Query
        page.get_by_text("✅ Supported Query").click()
        time.sleep(1)
        page.get_by_role("button", name="Ask Assistant").click()
        page.wait_for_selector("text=STATUS: SUPPORTED BY EVIDENCE", timeout=15000)
        time.sleep(1)
        page.screenshot(path="reports/screenshots/supported_answer.png", full_page=True)
        print("2. Saved supported_answer.png")

        # 2. Click Unsupported Query
        page.get_by_text("🚫 Unsupported Query").click()
        time.sleep(1)
        page.get_by_role("button", name="Ask Assistant").click()
        page.wait_for_selector("text=STATUS: INFORMATION NOT FOUND", timeout=15000)
        time.sleep(1)
        page.screenshot(path="reports/screenshots/unsupported_abstention.png", full_page=True)
        print("3. Saved unsupported_abstention.png")

        # 3. Click Partial Query
        page.get_by_text("⚠️ Partial Query").click()
        time.sleep(1)
        page.get_by_role("button", name="Ask Assistant").click()
        page.wait_for_selector("text=STATUS: PARTIAL / UNCERTAIN EVIDENCE", timeout=15000)
        time.sleep(1)
        page.screenshot(path="reports/screenshots/partial_evidence.png", full_page=True)
        print("4. Saved partial_evidence.png")

        browser.close()
        print("All 4 verified screenshots captured successfully!")

if __name__ == "__main__":
    capture_all()
