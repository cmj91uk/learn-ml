from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pandas import read_csv, DataFrame
from numpy import array

raw_data = read_csv('Pokemon.csv')
# print(raw_data.shape)
# (800, 13)
# 800 rows of data, 13 data points per row

# Drop additional columns that aren't required
raw_data.drop(columns=['#', 'Name', 'Type 1', 'Type 2'], inplace=True)

# print(raw_data.head())
# (800, 10)

attributes = raw_data.iloc[:, :7]
labels = raw_data.iloc[:, 8]
# print(attributes.head())
# print(labels.head())

# random_state seeds the splitting, specifying gives a reproducible split
# Use 25% of the data for testing
attributes_train, attributes_test, labels_train, labels_test = train_test_split(
    attributes,
    labels,
    test_size=0.25,
    random_state=42
)
#
# print(attributes_train.head())
# print(attributes_test.head())
# print(labels_train.head())
# print(labels_test.head())

lr = LogisticRegression(solver='lbfgs', multi_class='ovr')
lr.fit(attributes_train, labels_train)

predictions = lr.predict(attributes_test)

tester_data = {
    'Total': [600],
    'HP': [92],
    'Attack': [105],
    'Defense': [90],
    'Sp. Atk': [125],
    'Sp. Def': [90],
    'Speed': [117] # 117 turns prediction to be True instead of False
}
df = DataFrame(data=tester_data)

print('Predicting whether Pokemon is Legendary for: ')
print(df.iloc[:1, :])
result = lr.predict_proba(df)
# print(result)
print('Prediction results are: ')
print('[Probability of false, Probability of true]')
print(result)
# print(attributes_test.head())
# print('fifth value')
# test_input = attributes_test.iloc[4, :]
# print(test_input.reshape)
# print(test_input)
# print('fifth result')
# result = labels_test.iloc[4:]
# print(result)



# expected = labels_test[5]
# print(expected)
