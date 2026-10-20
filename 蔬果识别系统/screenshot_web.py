# -*- coding: utf-8 -*-
"""自动截图：Web 首页与识别结果"""
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")

options = webdriver.EdgeOptions()
options.add_argument("--headless=new")
options.add_argument("--window-size=1280,900")
driver = webdriver.Edge(
    service=Service(os.path.join(BASE, "driver", "msedgedriver.exe")),
    options=options)

try:
    driver.get("http://127.0.0.1:5000/")
    time.sleep(2)
    driver.save_screenshot(os.path.join(OUT, "web_home.png"))
    print("home ok")

    # 静态检测页：上传图片识别（文件上传控件在 /static 页面）
    driver.get("http://127.0.0.1:5000/static")
    time.sleep(2)
    for name in ["banana", "eggplant", "kiwi"]:
        file_input = driver.find_element(By.ID, "file")
        file_input.send_keys(os.path.join(BASE, "test_images", f"{name}.jpg"))
        time.sleep(1)
        driver.find_element(By.ID, "predBtn").click()
        time.sleep(4)
        driver.save_screenshot(os.path.join(OUT, f"web_result_{name}.png"))
        res = driver.find_element(By.ID, "result").text.replace("\n", " | ")
        print(name, "->", res[:150])
finally:
    driver.quit()
print("done")
