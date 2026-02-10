# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv

input_file_dir = "raw_dataset.txt"
output_file_dir = "csv_dataset.csv"

def corrigir_mojibake(texto):
    try:
        return texto.encode("latin1").decode("utf-8")
    except UnicodeError:
        return texto


with open(input_file_dir, "r", encoding="utf-8", errors="ignore") as txt_file, \
     open(output_file_dir, "w", newline="", encoding="utf-8-sig") as csv_file:

    writer = csv.writer(csv_file)
    writer.writerow(["frase"])

    for linha in txt_file:
        linha = linha.strip()
        if linha:
            linha_corrigida = corrigir_mojibake(linha)
            writer.writerow([linha_corrigida])

print("Texto corrigido e CSV criado 👌")