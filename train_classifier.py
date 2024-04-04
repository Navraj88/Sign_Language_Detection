import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
labels = np.asarray(data_dict['labels'])
data = np.asarray(data_dict['data'])

# Assuming 'data' contains sequences of varying lengths
# Find the maximumquence length
max_length = max(len(seq) for seq in data)

# Pad sequences to the maximum length
data_padded = []
for seq in data:
    seq_length = len(seq)
    if seq_length < max_length:
        padding = max_length - seq_length
        seq = np.pad(seq, (0, padding), mode='constant', constant_values=0)
    data_padded.append(seq)

# Convert to NumPy array
data_array = np.array(data_padded)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data_array, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
