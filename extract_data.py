from bs4 import BeautifulSoup

# Load the HTML content
with open('college_page.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

soup = BeautifulSoup(html_content)

# Extract college cards
college_cards = soup.find_all('div', class_='card_block')

data = []

print('html_content: ', html_content)

for card in college_cards:
    college_data = {}

    # College Name
    name_tag = card.find('h3', class_='college_name')
    college_data['Name'] = name_tag.get_text(strip=True) if name_tag else "N/A"

    # College URL
    college_data['URL'] = name_tag.find('a')['href'] if name_tag and name_tag.find('a') else "N/A"

    # NIRF Rank
    rank_block = card.find('div', class_='tupple_top_block_left')
    college_data['NIRF Rank 2024'] = rank_block.find('strong').text.strip() if rank_block and rank_block.find('strong') else "N/A"

    # Rating
    rating_block = card.find('div', class_='content_block')
    if rating_block:
        rating_text = rating_block.find(string=lambda s: 'Rating' in s)
        if rating_text:
            college_data['Rating'] = rating_text.find_next('strong').text.strip()
        else:
            college_data['Rating'] = "N/A"
    else:
        college_data['Rating'] = "N/A"

    # Ownership
    ownership_tag = card.find('strong', class_='strong_ownership')
    college_data['Ownership'] = ownership_tag.text.strip() if ownership_tag else "N/A"

    # Review Score
    star_text = card.find('span', class_='star_text')
    college_data['Review Score'] = star_text.get_text(strip=True) if star_text else "N/A"

    # Number of Reviews
    reviews_tag = card.find('span', class_='review_text')
    college_data['Reviews'] = reviews_tag.get_text(strip=True) if reviews_tag else "N/A"

    # Courses & Fees
    courses_info = card.find_all('ul', class_='snippet_list')
    course_fees = []
    for ul in courses_info:
        items = ul.find_all('li')
        course_name = items[0].get_text(strip=True)
        fee = items[1].get_text(strip=True) if len(items) > 1 else "N/A"
        course_fees.append(f"{course_name} | {fee}")
    college_data['Courses'] = course_fees

    data.append(college_data)

# Save to .txt file
with open('college_data.txt', 'w', encoding='utf-8') as f:
    for entry in data:
        f.write(f"College Name: {entry['Name']}\n")
        f.write(f"URL: {entry['URL']}\n")
        f.write(f"NIRF Rank 2024: {entry['NIRF Rank 2024']}\n")
        f.write(f"Rating: {entry['Rating']}\n")
        f.write(f"Ownership: {entry['Ownership']}\n")
        f.write(f"Review Score: {entry['Review Score']}\n")
        f.write(f"Reviews: {entry['Reviews']}\n")
        f.write("Courses and Fees:\n")
        for course in entry['Courses']:
            f.write(f"  - {course}\n")
        f.write("-" * 50 + "\n")

print("Data extracted and saved to college_data.txt")