from nltk.sentiment.vader import SentimentIntensityAnalyzer

with open("bad.txt", 'r', encoding='utf-8') as file:
    bad_review = file.read()

with open("good.txt", 'r', encoding='utf-8') as file:
    good_review = file.read()

sid = SentimentIntensityAnalyzer()

bad_sentiment = sid.polarity_scores(bad_review)
print("Negatywna opinia:")
print(f"Pozytywny (pos): {bad_sentiment['pos']}")
print(f"Negatywny (neg): {bad_sentiment['neg']}")
print(f"Zagregowany wynik (compound): {bad_sentiment['compound']}")

good_sentiment = sid.polarity_scores(good_review)
print("\nPozytywna opinia:")
print(f"Pozytywny (pos): {good_sentiment['pos']}")
print(f"Negatywny (neg): {good_sentiment['neg']}")
print(f"Zagregowany wynik (compound): {good_sentiment['compound']}")


overall_sentiment = {
    'pos': (bad_sentiment['pos'] + good_sentiment['pos']) / 2,
    'neg': (bad_sentiment['neg'] + good_sentiment['neg']) / 2,
    'compound': (bad_sentiment['compound'] + good_sentiment['compound']) / 2
}
print("\nZagregowany wynik wszystkich opinii:")
print(f"Pozytywny (pos): {overall_sentiment['pos']}")
print(f"Negatywny (neg): {overall_sentiment['neg']}")
print(f"Zagregowany wynik (compound): {overall_sentiment['compound']}")
# skala od -1 do 1, 

# Negatywna opinia:
# Pozytywny (pos): 0.0
# Negatywny (neg): 0.245
# Zagregowany wynik (compound): -0.9709

# Pozytywna opinia:
# Pozytywny (pos): 0.402
# Negatywny (neg): 0.0
# Zagregowany wynik (compound): 0.9747

# Zagregowany wynik wszystkich opinii:
# Pozytywny (pos): 0.201
# Negatywny (neg): 0.1225
# Zagregowany wynik (compound): 0.0019000000000000128