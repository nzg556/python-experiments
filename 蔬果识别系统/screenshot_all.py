# -*- coding: utf-8 -*-
import os, time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
PAGES = [
    ("http://127.0.0.1:5000/", "home.png"),
    ("http://127.0.0.1:5000/static", "static_page.png"),
    ("http://127.0.0.1:5000/dynamic", "dynamic_page.png"),
    ("http://127.0.0.1:5000/evaluation", "evaluation_page.png"),
]

opts = webdriver.EdgeOptions()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1280,900")
driver = webdriver.Edge(service=Service(os.path.join(BASE, "driver", "msedgedriver.exe")), options=opts)

try:
    for url, name in PAGES:
        driver.get(url)
        time.sleep(2)
        driver.save_screenshot(os.path.join(OUT, name))
        print(name, "saved")
finally:
    driver.quit()
print("done")
