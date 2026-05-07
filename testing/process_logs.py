
import re
import os
import pandas as pd

files_phases = {
    'full.txt': 'full',
    'semi.txt': 'semi',
    'lines.txt': 'lines'
}

data = []

regex = re.compile(r'Época\s+(\d+)\s*\|\s*Train Loss:\s*([\d.]+)\s*\|\s*Val CER:\s*([\d.]+)%\s*\|\s*Val WER:\s*([\d.]+)%')

for file_name, phase in files_phases.items():
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                match = regex.search(line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2)) # Capturamos el loss
                    cer = float(match.group(3))
                    wer = float(match.group(4))
                    data.append({
                        'Fase': phase, 
                        'Epoca': epoch, 
                        'Train Loss': train_loss, 
                        'CER (%)': cer, 
                        'WER (%)': wer
                    })

df = pd.DataFrame(data)
csv_filename = 'metricas_entrenamiento.csv'
df.to_csv(csv_filename, index=False)
print(f"Generated CSV with {len(df)} rows.")
print(df.head())
print(df['Fase'].value_counts())