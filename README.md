# Classify

**BERTimbau-based multi-label classifier** for Portuguese text: predicts **empathy**, **assertiveness**, and **tone** from short sentences. Built with PyTorch and Hugging Face Transformers (`neuralmind/bert-base-portuguese-cased`), with an 80/20 train–test split and metrics (accuracy, F1 weighted/macro, Cohen's kappa, confusion matrix, classification report). Suited for small datasets and reproducible experiments (fixed seed, 3 epochs, configurable batch size and max length).

Bonus: a web app, cause i'm a web engineer lol

Purpose: train the baseline not in Excel for god's sake


<div align="center" >
   
<img width="820" height="640" alt="avaliação por rúbrica sendo feita através de botões para os tom, empatia e asseritividade, visanndo facilitar a classificação textual de acordo com a rúbrlica" src="https://github.com/user-attachments/assets/45e1dacc-70c9-4dbc-b99f-f33bdeab71a2" />

</div>

</br>


<div align="center">

   
<img width="820" height="640"  alt="avaliação por rúbrica sendo feita através de botões para os tom, empatia e asseritividade, visanndo facilitar a classificação textual de acordo com a rúbrlica" src="https://github.com/user-attachments/assets/13fb3e6e-17f9-4235-a50a-dedc86f59b64" />

</div>


<hr>


## Python — BERTimbau classifier

Pipeline to run the empathy / assertiveness / tone classifier (from the `scripts` folder).

1. **Create the virtual environment** (once):

   ```
   cd scripts
   python3 -m venv .venv
   ```

2. **Activate the virtual environment**

   - **Windows (cmd):**  
     `.venv\Scripts\activate.bat`
   - **Windows (PowerShell):**  
     `.venv\Scripts\Activate.ps1`
   - **Linux / macOS:**  
     `source .venv/bin/activate`

   When active, your prompt usually shows `(.venv)`.

3. **Install the requirements**

   ```
   pip install -r requirements.txt
   ```

4. **Run the classifier**

   ```
   python classify_bertimbau.py
   ```

---

## JavaScript

To install dependencies for frontend:

```
bun install
bun run dev
```

To run:

```
bun run index.ts
```

This project was created using `bun init` in bun v1.3.5. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.
