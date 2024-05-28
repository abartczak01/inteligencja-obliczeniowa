import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# nltk.download('all')

with open("tekst.txt", 'r', encoding='utf-8') as plik:
    text = plik.read()

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text.lower())
print("Liczba słów po tokenizacji:", len(words))

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]
print("Liczba słów po usunięciu stop words:", len(filtered_words))

stop_words.add("new")
filtered_words = [word for word in filtered_words if word not in stop_words]
print("Liczba słów po usunięciu 'new':", len(filtered_words))

# zamienia słowa do podstawowej formy, np. running na run
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("Liczba słów po lematyzacji:", len(lemmatized_words))

fd = nltk.FreqDist(lemmatized_words)
most_common_words = fd.most_common(10)
print(most_common_words)

words, counts = zip(*most_common_words)
plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.xlabel('Słowa')
plt.ylabel('Liczba wystąpień')
plt.title('10 najczęściej występujących słów')
plt.savefig('most_common.png')

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fd)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Chmura tagów')
plt.savefig('word_cloud.png')
