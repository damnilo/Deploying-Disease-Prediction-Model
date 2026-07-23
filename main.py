import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from neural_network import SymptomClassifier

def main():
    df = pd.read_csv('Dataset/dataset.csv')

    symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]

    for col in symptom_cols:
        df[col] = df[col].str.strip()

    df_unique = df.drop_duplicates(subset=symptom_cols + ['Disease']).copy()

    df_unique['symptom_set'] = df_unique[symptom_cols].apply(
        lambda row: [s for s in row if pd.notna(s)], axis=1
    )

    mlb = MultiLabelBinarizer()
    le = LabelEncoder()
    x_unique = mlb.fit_transform(df_unique['symptom_set'])
    y_unique = le.fit_transform(df_unique['Disease'])

    x_train, x_test, y_train, y_test = train_test_split(
        x_unique, y_unique, test_size=0.2, stratify=y_unique, random_state=42
    )

    model = SymptomClassifier(input=x_unique.shape[1], output=len(le.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    n_epochs = 150
    best_acc = 0
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        train_pred = torch.argmax(y_pred, dim=1)
        train_acc = (train_pred == y_train).float().mean().item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                output = model(x_test)
                test_loss = criterion(output, y_test)
                test_pred = torch.argmax(output, dim=1)
                test_acc = (test_pred == y_test).float().mean().item()

            print(f"Epoch {epoch+1}/{n_epochs} | Train loss: {loss.item():.4f} | Train acc: {train_acc:.4f}"
                  f"| Test acc: {test_acc:.4f} | Test loss: {test_loss:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'app/model.pt')
                joblib.dump(mlb, 'app/mlb.joblib')
                joblib.dump(le, 'app/le.joblib')

if __name__ == '__main__':
    main()