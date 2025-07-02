import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# === PARAMÈTRES ===
INPUT_CSV = "SEANCE_TIR_3/tir900plaque500.csv"  # Chemin vers ton fichier
OUTPUT_CSV = "SEANCE_TIR_3/tir900_plaque500_extracted.csv"  # Nouveau fichier avec les données extraites
OUTPUT_PNG = "SEANCE_TIR_3/tir900_plaque500_extracted.png"
PRE_WINDOW_SEC = 0.001  # Temps à prendre avant le pic
POST_WINDOW_SEC = 0.002  # Temps à prendre après le pic

# === CHARGEMENT DES DONNÉES ===
def load_csv(csv_path):
    times = []
    values = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                times.append(float(row["time_s"]))
                values.append(int(row["adc_value"]))
            except:
                continue
    return np.array(times), np.array(values)

# === EXTRACTION DE LA FENÊTRE ===
def extract_window(times, values, pre_sec, post_sec):
    peak_idx = np.argmax(values)
    peak_time = times[peak_idx]

    start_time = peak_time - pre_sec
    end_time = peak_time + post_sec

    mask = (times >= start_time) & (times <= end_time)
    return times[mask], values[mask]

# === SAUVEGARDE DU NOUVEAU CSV ===
def save_csv(times, values, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "adc_value"])
        writer.writerows(zip(times, values))
    print(f"✅ Nouveau CSV sauvegardé : {output_path}")

# === PLOT POUR VÉRIFICATION ===
def plot_signal(times, values, title="Signal extrait"):
    plt.figure(figsize=(10, 4))
    plt.plot(times, values)
    plt.title(title)
    plt.xlabel("Temps (s)")
    plt.ylabel("Valeur ADC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(OUTPUT_PNG)

# === MAIN ===
if __name__ == "__main__":
    times, values = load_csv(INPUT_CSV)
    print(f"📊 Données chargées : {len(values)} points")

    new_times, new_values = extract_window(times, values, PRE_WINDOW_SEC, POST_WINDOW_SEC)
    print(f"📦 Fenêtre extraite : {len(new_values)} points")

    save_csv(new_times, new_values, OUTPUT_CSV)
    plot_signal(new_times, new_values)
