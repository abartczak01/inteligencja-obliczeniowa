import text2emotion as te


with open("bad.txt", 'r', encoding='utf-8') as file:
    bad_review = file.read()

with open("good.txt", 'r', encoding='utf-8') as file:
    good_review = file.read()

bad_emotions = te.get_emotion(bad_review)
print("Emocje w negatywnej opinii:")
for emotion, score in bad_emotions.items():
    print(f"{emotion}: {score}")

good_emotions = te.get_emotion(good_review)
print("\nEmocje w pozytywnej opinii:")
for emotion, score in good_emotions.items():
    print(f"{emotion}: {score}")

overall_emotions = {
    emotion: (bad_emotions[emotion] + good_emotions[emotion]) / 2
    for emotion in bad_emotions
}
print("\nZagregowane emocje wszystkich opinii:")
for emotion, score in overall_emotions.items():
    print(f"{emotion}: {score}")
