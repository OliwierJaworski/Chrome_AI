from os import access
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

options = Options()
options.binary_location = "/home/oliwier-desktop/Downloads/firefox/firefox"
options.set_preference("browser.download.folderList",2)
options.set_preference("browser.download.manager.showWhenStarting", False)
options.set_preference("browser.download.dir","/Data")
options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream,application/vnd.ms-excel")
driver = webdriver.Firefox(options=options)

driver.get("https://chromedino.com/")
assert "T-Rex Dinosaur Game" in driver.title

#elem = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME,"fc-button-label")))   
elem = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME,"fc-button")))    
elem = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME,"fc-button")))    
print("element is clickable")

action = ActionChains(driver)

consent_btn = driver.find_element(By.CLASS_NAME, "fc-button")

action.move_to_element(consent_btn)
action.click(consent_btn)
action.perform()

runner = driver.find_element(By.CLASS_NAME, "runner-container")

#action.move_to_element(runner)
action.send_keys(" ")
action.perform()
