# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv

arquivo_txt = "gpt.txt"
arquivo_csv = "gpt.csv"

with open(arquivo_txt, "r", encoding="utf-8") as txt_file, \
     open(arquivo_csv, "w", newline="", encoding="utf-8") as csv_file:
    
    writer = csv.writer(csv_file)
    
    # cabeçalho do CSV
    writer.writerow(["frase"])
    
    for linha in txt_file:
        frase = linha.strip()
        if frase:  # ignora linhas vazias
            writer.writerow([frase])

print("CSV criado com sucesso!")