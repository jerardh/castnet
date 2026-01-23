import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

BASE_WIKI_URL = "https://en.wikipedia.org"
START_YEAR = 2010
END_YEAR = 2024

data_rows = []

headers = {
    "User-Agent": "Mozilla/5.0"
}

# ---------------------------
# Fetch HTML
# ---------------------------
def get_soup(url):
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.text, "html.parser")

# ---------------------------
# Extract movies for a year
# ---------------------------
def get_movies_for_year(year):
    url = f"{BASE_WIKI_URL}/wiki/List_of_Malayalam_films_of_{year}"
    soup = get_soup(url)

    movies = []

    tables = soup.find_all("table", class_="wikitable")
    for table in tables:
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if cols:
                title_cell = cols[0]
                title = title_cell.get_text(strip=True)
                link_tag = title_cell.find("a")
                if link_tag and link_tag.get("href"):
                    link = BASE_WIKI_URL + link_tag["href"]
                    movies.append((title, link))
    return movies

# ---------------------------
# Extract plot
# ---------------------------
def extract_plot(soup):
    plot_span = soup.find("span", id="Plot")
    if plot_span:
        paragraphs = []
        for sib in plot_span.parent.find_next_siblings():
            if sib.name == "p":
                paragraphs.append(sib.get_text())
            elif sib.name == "h2":
                break
        return " ".join(paragraphs)
    return ""

# ---------------------------
# Extract cast
# ---------------------------
def extract_cast(soup):
    cast_pairs = []

    for ul in soup.find_all("ul"):
        for li in ul.find_all("li"):
            text = li.get_text()
            if " as " in text:
                parts = text.split(" as ", 1)
                actor = parts[0].strip()
                character = parts[1].strip()
                cast_pairs.append((actor, character))

    return cast_pairs

# ---------------------------
# Generate character description
# ---------------------------
def generate_character_description(plot, character_name):
    sentences = sent_tokenize(plot)
    relevant = [
        s for s in sentences
        if character_name.split()[0].lower() in s.lower()
    ]
    return " ".join(relevant[:3])

# ---------------------------
# Main scraping loop
# ---------------------------
for year in range(START_YEAR, END_YEAR + 1):
    print(f"Scraping movies for year {year}...")
    movies = get_movies_for_year(year)

    for title, link in movies:
        try:
            soup = get_soup(link)
            plot = extract_plot(soup)
            cast = extract_cast(soup)

            for actor, character in cast:
                char_desc = generate_character_description(plot, character)

                data_rows.append({
                    "movie_name": title,
                    "year": year,
                    "plot": plot,
                    "actor_name": actor,
                    "character_name": character,
                    "character_description": char_desc
                })

            time.sleep(1)

        except Exception as e:
            print(f"Failed for {title}: {e}")

# ---------------------------
# Save dataset
# ---------------------------
df = pd.DataFrame(data_rows)
df.to_csv("malayalam_movie_cast_dataset.csv", index=False)

print("Scraping completed. Dataset saved as malayalam_movie_cast_dataset.csv")
