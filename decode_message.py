from bs4 import BeautifulSoup
import requests

def decode_secret_message(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table")

    cells = [td.get_text(strip=True) for td in table.find_all("td")]

    # Skip header row
    cells = cells[3:]

    points = []

    for i in range(0, len(cells), 3):
        try:
            x = int(cells[i])
            char = cells[i + 1]
            y = int(cells[i + 2])
            points.append((x, y, char))
        except:
            continue

    max_x = max(x for x, _, _ in points)
    max_y = max(y for _, y, _ in points)

    grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    for x, y, char in points:
        grid[y][x] = char

    for row in grid:
        print("".join(row))


url = "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"

decode_secret_message(url)