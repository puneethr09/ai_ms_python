import requests as req

def get_data(url):
    response = req.get(url)
    return response.json()

def fetch_details(ids):
    stories = []
    for i in ids:
        story = req.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json")
        stories.append(story.json())
    return stories    

def print_stories(stories):
    for s in stories:
        title = s.get("title")
        url = s.get("url", "No Link")
        if(url != "No Link"):
            print(f"Title: {title} \n Link: {url}\n")
        else:
            print(f"Title: {title} \n No Link \n")

def main():
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    data = get_data(url)
    initial10 = data[:10]
    stories = fetch_details(initial10)
    print_stories(stories)

if __name__ == "__main__":
    main()