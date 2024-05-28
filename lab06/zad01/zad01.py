import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

df = pd.read_csv("./titanic1.csv")

age_avg = round(df['age'].mean())

df['age'] = df['age'].fillna(age_avg)
df['survived'] = df['survived'].replace({0: False, 1: True})

def assign_age_group(age):
    if age <= 10:
        return 'child'
    elif age <= 19:
        return 'teenager'
    elif age <= 35:
        return 'young adult'
    elif age <= 50:
        return 'adult'
    else:
        return 'senior'

df['age'] = df['age'].apply(assign_age_group)

print(df)

y = df['survived']
X = df.drop('survived', axis=1)

items = set()
for col in df:
    items.update(df[col].unique())

itemset = set(items)
encoded_vals = []
for index, row in df.iterrows():
    rowset = set(row) 
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]

ohe_df = pd.DataFrame(encoded_vals)

freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
print(freq_items.head(10))

rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
sorted_rules = rules.sort_values(by='confidence', ascending=False)
filtered_rules = sorted_rules.query('confidence >= 0.8')

print(filtered_rules)
# print(sorted_rules)

def plot1():
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
    plt.xlabel('support')
    plt.ylabel('confidence')
    plt.title('Support vs Confidence')
    plt.savefig('plot1.png')
    plt.close()

def plot2():
    plt.scatter(rules['support'], rules['lift'], alpha=0.5)
    plt.xlabel('support')
    plt.ylabel('lift')
    plt.title('Support vs Lift')
    plt.savefig('plot2.png')
    plt.close()

def plot3():
    fit = np.polyfit(rules['lift'], rules['confidence'], 1)
    fit_fn = np.poly1d(fit)
    plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
    fit_fn(rules['lift']))
    plt.savefig('plot3.png')
    plt.close()

#supprt i confidence, im bardziej na prawo i wyżej tym lepiej
plot1()
# support i lift
plot2()

plot3()

# antescedents - przyczny, consequents - skutki
# antecedent support - jak czesto zbiór przyczyn wystepuje w zbiorze danych
# consequent support - jak czesto zbiór skutków wystepuje w zbiorze danych
# support - jak czesto wystepuje reguła
# confidence - jak pewna jest reguła
# lift -  jak bardzo zależność między cechami przyczynowymi i skutkami jest większa niż 
# gdyby były one niezależne; Wartości liftu większe niż 1 oznaczają, że zdarzenie jest bardziej prawdopodobne niż gdyby cechy były niezależne.
# leverage: mierzy, jak bardzo często występuje reguła asocjacyjna w stosunku do tego, co można by oczekiwać, gdyby cechy były niezależne
# conviction: mierzy stopień, w jakim reguła jest niezależna od odwrotnej reguły.
# zhangs_metric: Metryka Zhang's, która jest kombinacją innych miar 