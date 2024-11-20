# pip install serenium
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Path to your ChromeDriver
CHROME_DRIVER_PATH = 'C:\\Program Files\\chrome-win64\\chrome-win64\\chrome.exe'

# Start WebDriver
service = Service(CHROME_DRIVER_PATH)
options = webdriver.ChromeOptions()
#options.add_argument('--headless')
#options.add_argument("--disable-gpu")
#options.add_argument("--no-sandbox")
#options.add_argument("--disable-software-rasterizer")
options.add_argument("--mute-audio")
options.add_argument("--window-size=800,600")
options.add_argument("--disable-popup-blocking")
#options.add_argument("--disable-extensions")

# Open Chrome browser and access Dino game
browser = webdriver.Chrome(options=options)

try:
    print("Debug: Opening Browser...")
    browser.get('https://chromedino.com')
    # browser.get('chrome://dino/') --> Does not work

    print("debug: Navigated to the Dino game.")
    time.sleep(2)

    
    a = ActionChains(service)

    m = service.find_element_by_link_text("Consent")
    print("debug: Label found")

    #hover over element
    a.move_to_element(m).click().perform()

    print("debug: Label clicked")



    print("Debug: Attempting to find canvas...")
    # Find the game canvas element
    canvas = browser.find_element(By.TAG_NAME, 'body')
    print("Debug: Canvas found, send key to start game...")

    # Start the game by sending the SPACE key
    canvas.send_keys(Keys.SPACE)

    print("Debug: Starting game")


    # Game play loop
    try:
        while True:
            canvas.send_keys(Keys.SPACE)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Game stopped.")
    finally:
        # Close the browser
        browser.quit()

except Exception as e:
    # Catch and print any errors that occur
    print(f"Debug: An error occurred: {e}")
finally:
    print("Debug: Closing the browser.")
    browser.quit()