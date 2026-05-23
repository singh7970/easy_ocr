# OCR Document Analyzer & Diary

Offline and web OCR for **PDF**, **PNG**, and **JPEG** using [EasyOCR](https://github.com/JaidedAI/EasyOCR). The application digitizes your physical documents and archives them securely into a personalized, interactive, and fully searchable **Database-backed Digital Diary**.

## Prerequisites
- **Python 3.10+** (3.12 works perfectly)
- **MySQL Server** (for archiving your entries)
- Enough disk space and RAM for PyTorch + EasyOCR models 

---

## 1. Install & Configure MySQL

Since this application stores all extracted OCR documents in a persistent diary, you need to set up a local MySQL database.

### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install mysql-server
sudo systemctl start mysql
sudo systemctl enable mysql
```

### Windows:
1. Download the **MySQL Installer** from the [official website](https://dev.mysql.com/downloads/installer/).
2. Run the installer and choose **"Developer Default"** setup type → click **Execute**.
3. During configuration, set a **root password** (remember it!) and keep the default port **3306**.
4. Make sure **"Start MySQL Server at System Startup"** is checked → click **Execute** → **Finish**.

### macOS (using Homebrew):
```bash
brew install mysql
brew services start mysql
```

### Database Setup

**On Linux**, log into MySQL as root:
```bash
sudo mysql
```

**On Windows**, open **Command Prompt** and run:
```cmd
mysql -u root -p
```
Then enter your root password when prompted.

Execute the following SQL commands to create the database, the user, and immediately grant privileges:
```sql
CREATE DATABASE rawdat;
CREATE USER 'rawdat_user'@'localhost' IDENTIFIED BY 'rawdata123';
GRANT ALL PRIVILEGES ON rawdat.* TO 'rawdat_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

---

## 2. Set up the Python Environment

From the project folder, create and activate a virtual environment, then install all required dependencies:

```bash
python3 -m venv env
source env/bin/activate
# Windows: env\Scripts\activate

pip install -r requirements.txt
```
*(Note: If you encounter `cryptography` or MySQL driver issues on Linux during `pip install`, you may need to run `sudo apt install python3-dev libmysqlclient-dev pkg-config` first).*

---

## 3. Run the Web Server

Start the application from your project directory. 

```bash
source env/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Once running, open **http://127.0.0.1:8000** in your web browser.

- **Main Upload Page (`/`)**: Upload a PDF or image, visually edit the extracted OCR text side-by-side, assign an explicit document date, and securely save it directly to your MySQL database.
- **The Diary (`/diary/`)**: Browse a gorgeous, beautifully animated page-flipping book containing your entire archive. You can dynamically search for names, dates, precise combinations, or even explicit phrases pulled directly from the OCR extraction body.

### First Run Note
The very first time EasyOCR runs, it will download necessary AI weights over the internet. Later runs will be fully offline!

---

## 4. Run from the command line (No database required)

If you only want quick terminal extractions:
```bash
source env/bin/activate
python main.py path/to/scan.png
python main.py ./images/ ./doc.pdf --preset accurate
python main.py a.png b.png --out ~/ocr_combined.txt
```

## Troubleshooting

- **Port 8000 in use** — If uvicorn crashes with `[Errno 98] address already in use`, it means the server is accidentally running in the background. If you previously suspended the process with `CTRL+Z`, type `fg` then `CTRL+C` to terminate it correctly. Alternatively, use `fuser -k 8000/tcp` to forcefully end it.
- **MySQL Connection Refused** — Verify that your database username and password exactly match `rawdat_user` and `rawdata123`.

## Project layout

| File | Role |
|------|------|
| `api.py` | FastAPI application, database router, backend endpoints |
| `database.py` | SQLAlchemy ORM database definitions |
| `templates/index.html` | Responsive Upload and Side-by-Side Editor UI |
| `templates/diary.html` | Animated Flipbook Diary Search Interface |
| `static/uploads/` | Image persistence storage directory |
