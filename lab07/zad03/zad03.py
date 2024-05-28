from ntscraper import Nitter

scraper = Nitter()

tweets = scraper.get_tweets("eurovision", mode = "user", number=5)