# Author: Will Blanton
import re

import pandas as pd
import time

from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver


# URL of the page
CONNECTIONS_HISTORY_URL = "https://tryhardguides.com/nyt-connections-answers/"
DATE_PATTERN = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(st|nd|rd|th),?\s+\d{4}'

def retrieve_date(string):
    date_str = re.search(DATE_PATTERN, string)

    if date_str is None:

        date = None

    else:
        date_str = date_str.group(0)

        cleaned_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

        date = datetime.strptime(cleaned_date_str, "%B %d, %Y")

    return date


def create_connections_dataset(save_file):
    """Collect all previous official NYT Connections answers to produce a dataset in csv format.

    Retrieves info from: 
    "https://tryhardguides.com/nyt-connections-answers"

    Each row in the output csv ("data/connections.csv) corresponds to a category at a specific date.
    """
    # request each page of the connections history
    driver = webdriver.Chrome()

    try:
        connections = []
        page = 1

        while True:
            page_url = CONNECTIONS_HISTORY_URL if page == 1 else f"{CONNECTIONS_HISTORY_URL}{page}/"
            driver.get(page_url)
            time.sleep(5)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            entry_content = soup.find('div', class_='entry-content')
            first_ul = entry_content.find('ul') if entry_content else None

            if first_ul is None:
                break

            page_connections = []

            for li in first_ul.children:

                date = retrieve_date(li.text)

                for c in li.find_all('li'):

                    # category is always in the strong tag (assuming correct structure)
                    strong = c.find('strong')
                    if strong is None:
                        continue
                    category = strong.text.strip()

                    after_strong = c.text.split(strong.text, 1)[-1]
                    words = after_strong.split('-', 1)[-1]

                    df = pd.DataFrame({
                        'date': [date],
                        'category': [category.strip().lower()],
                        'connections': [words.strip().lower().split(", ")]
                    })

                    page_connections.append(df)

            if not page_connections:
                break
            connections.extend(page_connections)
            page += 1

        connections_df = pd.concat(connections)

        # reverse the order of the connections
        connections_df = connections_df[::-1].reset_index(drop=True)

        connections_df.to_csv(save_file, index=True)

    except Exception as e:
        print(f"An error occurred while retrieving the connections history.: {e}")
    finally:
        # close the browser
        driver.quit()


if __name__ == "__main__":
    create_connections_dataset("data/connections.csv")
