# classify

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
