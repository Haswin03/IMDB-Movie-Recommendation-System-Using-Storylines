import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def scrape_imdb_continuous(start_movie=1501, target_clicks=49):
    """
    Scrapes IMDb movies continuously in batches.
    start_movie: The index to start scraping from (1, 1501, 3001).
    target_clicks: How many times to click 'Load More' (29 clicks ≈ 1500 movies).
    """
    print(f"Starting Scraper at movie #{start_movie}...")
    
    # Setup Chrome
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-notifications")
    # Uncomment the line below if you want the scraper to run invisibly in the background
    # options.add_argument("--headless") 
    
    driver = webdriver.Chrome(options=options)
    
    # URL with the start parameter
    url = f"https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&start={start_movie}"
    driver.get(url)
    driver.maximize_window()
    time.sleep(3) # Let initial page load
    
    # --- PHASE 1: CLICKING 'LOAD MORE' ---
    print(f"Preparing to click 'Load More' {target_clicks} times...")
    for i in range(target_clicks):
        try:
            # Scroll down to trigger lazy loading and reveal the button
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1500);")
            time.sleep(2)
            
            # Find and click the '50 more' button
            load_more_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), '50 more')] | //button[contains(@class, 'ipc-see-more__button')]"))
            )
            driver.execute_script("arguments[0].click();", load_more_btn)
            print(f"Clicked Load More ({i+1}/{target_clicks})")
            time.sleep(2.5) # Wait for DOM to render new movies
            
        except TimeoutException:
            print("'Load More' button not found. Moving to extraction...")
            break
        except Exception as e:
            print(f"Interruption during clicking: {e}")
            break

    # --- PHASE 2: DATA EXTRACTION ---
    print("Extracting movie data from the page...")
    movie_data = []
    
    # Target all movie containers currently loaded on the screen
    movie_elements = driver.find_elements(By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item")
    print(f"Found {len(movie_elements)} movies on this page. Processing...")
    
    for element in movie_elements:
        try:
            # Extract Title
            raw_title = element.find_element(By.CSS_SELECTOR, "h3.ipc-title__text").text
            title = raw_title.split(". ", 1)[1] if ". " in raw_title else raw_title
            
            # Extract Storyline
            try:
                plot = element.find_element(By.CSS_SELECTOR, "div.ipc-html-content-inner-div").text
            except NoSuchElementException:
                plot = "Plot not available."
                
            # Only keep movies that actually have a storyline
            if plot != "Plot not available.":
                movie_data.append({
                    "Movie_Name": title,
                    "Storyline": plot
                })
        except Exception:
            continue # Skip corrupted HTML elements silently

    driver.quit()
    
# --- PHASE 3: SAVING & CLEANING DATA ---
    if not movie_data:
        print("No data extracted. Check your internet or IMDb layout changes.")
        return

    df_new = pd.DataFrame(movie_data)
    
    # Define the directory and file path relative to src/
    data_dir = "../data"
    csv_path = os.path.join(data_dir, "movies_2024.csv")
    
    # CORRECTED: Create the 'data' folder at the root level, not inside 'src'
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if we need to write headers
    file_exists = os.path.isfile(csv_path)
    
    # Append the new batch
    df_new.to_csv(csv_path, mode='a', index=False, header=not file_exists)
    
    # Reload the full file to clean up overlaps/duplicates
    df_full = pd.read_csv(csv_path)
    initial_count = len(df_full)
    df_full.drop_duplicates(subset=['Movie_Name'], inplace=True)
    final_count = len(df_full)
    
    # Save the cleaned, unique dataset
    df_full.to_csv(csv_path, index=False)
    
    print(f"Batch complete! Saved {len(df_new)} new movies.")
    print(f"Removed {initial_count - final_count} duplicates.")
    print(f"TOTAL MOVIES IN DATASET NOW: {final_count}")

if __name__ == "__main__":
    # --- YOUR EXECUTION PLAN ---
    # Change 'start_movie' for each batch. 
    # 'target_clicks' is set to 29 (1 page + 29 clicks * 50 = ~1500 movies)
    
    # RUN 1: (Uncomment the line below for your first run)
    scrape_imdb_continuous(start_movie=1501, target_clicks=49)
    
    # RUN 2: (When ready for the next batch, comment RUN 1, and uncomment below)
    # scrape_imdb_continuous(start_movie=1501, target_clicks=29)
    
    # RUN 3:
    # scrape_imdb_continuous(start_movie=3001, target_clicks=29)