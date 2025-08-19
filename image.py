#!/usr/bin/env python3
"""
Image Crawler

Vereint:
- Playwright-basiertes Crawlen (JS-rendering, lazy-load, network capture)  
- Recursion auf interne Links (bis max_depth)
- Bruteforce für typische Upload-Verzeichnisse (mit Wortliste)
- Speichert alle gefundenen Bild-URLs in 'bilder_alles.txt'

NUTZUNG: Nur auf Seiten einsetzen, für die du die Erlaubnis hast!
"""

import asyncio
import re
import time
import os
import hashlib
import json
from urllib.parse import urljoin, urlparse, urldefrag
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np

from playwright.async_api import async_playwright

# Import für Metadaten-Verwaltung
try:
    from image_metadata_utils import add_image_metadata, get_metadata_stats
except ImportError:
    # Fallback wenn utils nicht verfügbar
    def add_image_metadata(*args, **kwargs):
        pass
    def get_metadata_stats():
        return {}

# ------------- KONFIG -------------
BASE_URL = "https://schachklub-toeging.de/"   # <-- anpassen
MAX_DEPTH = float('inf')  # Keine Begrenzung - crawle alle Seiten!
CONCURRENT_PAGES = 3
NAV_TIMEOUT = 20000  # ms
OUTPUT_FILE = "bilder_alles.txt"

# Download-Konfiguration
DOWNLOAD_DIR = "data/scraped"  # Direkt in static/images
MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024  # 50MB max pro Bild
CONCURRENT_DOWNLOADS = 5
METADATA_FILE = "image_metadata.json"  # Speichert URL-Mapping

# Globale Listen
found_images = set()
visited = set()
errors = []
image_metadata = {}  # Speichert URL -> lokaler Pfad Mapping
queue = []  # Globale Queue für Worker
saved_urls = set()  # Track bereits gespeicherte URLs

# Wortliste für Upload-Verzeichnisse
UPLOAD_KEYWORDS = [
    "uploads", "upload", "media", "files", "images", "img", "pics", "pictures", 
    "photos", "gallery", "assets", "content", "wp-content", "resources", "static",
    "data", "storage", "documents", "docs", "downloads", "public", "shared",
    "user", "users", "member", "members", "profile", "profiles", "avatar", "avatars",
    "news", "articles", "blog", "post", "posts", "thumb", "thumbnails", "cache",
    "temp", "tmp", "archive", "backup", "old", "new", "current", "latest"
]

def is_internal(url):
    """Prüft, ob URL zur gleichen Domain gehört"""
    return urlparse(url).netloc in ['', urlparse(BASE_URL).netloc]

def normalize(url):
    """Normalisiert URL (entfernt Fragment)"""
    return urldefrag(url)[0]

def is_allowed_image(url):
    """Prüft, ob URL ein erlaubtes Bildformat hat"""
    return url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'))

async def crawl_playwright():
    """Hauptcrawl-Funktion mit Playwright"""
    global queue
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # Semaphore für gleichzeitige Seiten
        sem = asyncio.Semaphore(CONCURRENT_PAGES)
        
        # Startseite zur globalen queue hinzufügen
        queue = [(BASE_URL, 0)]
        workers = [worker(browser, sem) for _ in range(CONCURRENT_PAGES)]
        
        await asyncio.gather(*workers)
        await browser.close()

async def worker(browser, sem):
    """Worker für parallele Verarbeitung"""
    global queue
    
    while True:
        try:
            url, depth = queue.pop(0)
        except IndexError:
            break
        
        if url in visited:
            continue
            
        await process_page(browser, url, depth)

async def process_page(browser, url, depth):
    """Verarbeitet eine einzelne Seite"""
    global found_images, visited, errors, queue
    
    if url in visited:
        return
    
    print(f"[{depth}] {url}")
    visited.add(url)
    
    def on_response(response):
        """Interceptor für alle Netzwerk-Requests"""
        resp_url = response.url
        if is_allowed_image(resp_url) and is_internal(resp_url):
            if resp_url not in found_images:  # Noch nicht in dieser Session gefunden
                found_images.add(resp_url)
                if save_image_url_live(resp_url):  # Nur ausgeben wenn wirklich neu gespeichert
                    print(f"    🖼️  {resp_url}")
                else:
                    print(f"    🔄 {resp_url} (bereits vorhanden)")
    
    try:
        page = await browser.new_page()
        page.on("response", on_response)
        
        # Seite laden
        await page.goto(url, timeout=NAV_TIMEOUT, wait_until='networkidle')
        
        # Auto-scroll für Lazy Loading
        await auto_scroll(page)
        
        # HTML parsen
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Alle img-Tags sammeln
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if src:
                full_url = urljoin(url, src)
                if is_allowed_image(full_url) and is_internal(full_url):
                    if full_url not in found_images:  # Noch nicht in dieser Session gefunden
                        found_images.add(full_url)
                        if save_image_url_live(full_url):  # Nur ausgeben wenn wirklich neu gespeichert
                            print(f"    🖼️  {full_url}")
                        else:
                            print(f"    🔄 {full_url} (bereits vorhanden)")
        
        # Links für weitere Crawling sammeln (alle Tiefen erlaubt)
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = normalize(urljoin(url, href))
            
            if is_internal(full_url) and full_url not in visited:
                queue.append((full_url, depth + 1))
        
        await page.close()
        
    except Exception as e:
        errors.append(f"{url}: {e}")
        print(f"    ❌ Fehler: {e}")

async def auto_scroll(page, pause=0.2, max_scrolls=30):
    """Automatisches Scrollen für Lazy Loading"""
    for i in range(max_scrolls):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(pause)

def brute_force_directories():
    """Brute-Force für typische Upload-Verzeichnisse"""
    global found_images
    
    print("[*] Starte Bruteforce-Scan...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    for keyword in UPLOAD_KEYWORDS:
        test_urls = [
            f"{base_clean}/{keyword}/",
            f"{base_clean}/wp-content/{keyword}/",
            f"{base_clean}/assets/{keyword}/",
            f"{base_clean}/static/{keyword}/",
            f"{base_clean}/media/{keyword}/",
            f"{base_clean}/content/{keyword}/",
            f"{base_clean}/files/{keyword}/",
            f"{base_clean}/public/{keyword}/",
            f"{base_clean}/resources/{keyword}/",
            f"{base_clean}/data/{keyword}/",
        ]
        
        for test_url in test_urls:
            try:
                response = requests.get(test_url, timeout=10, allow_redirects=False)
                if response.status_code == 200:
                    print(f"    ✅ Gefunden: {test_url}")
                    
                    # Parse Directory Listing
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if is_allowed_image(href):
                            full_url = urljoin(test_url, href)
                            if full_url not in found_images:  # Noch nicht in dieser Session gefunden
                                found_images.add(full_url)
                                if save_image_url_live(full_url):  # Nur ausgeben wenn wirklich neu gespeichert
                                    print(f"        🖼️  {full_url}")
                                else:
                                    print(f"        🔄 {full_url} (bereits vorhanden)")
                            
            except Exception:
                pass  # Stille Fehler für Bruteforce

def load_existing_urls():
    """Lädt bereits gespeicherte URLs aus der Datei."""
    global saved_urls
    
    if Path(OUTPUT_FILE).exists():
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_urls = {line.strip() for line in f if line.strip()}
                saved_urls.update(existing_urls)
                print(f"📄 {len(existing_urls)} bereits gespeicherte URLs geladen")
        except Exception as e:
            print(f"⚠️ Fehler beim Laden existierender URLs: {e}")
            saved_urls = set()
    else:
        saved_urls = set()

def save_image_url_live(url):
    """Speichert gefundene URL sofort in die Datei (nur wenn noch nicht vorhanden)."""
    global saved_urls
    
    if url in saved_urls:
        return False  # Bereits gespeichert
    
    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{url}\n")
        saved_urls.add(url)  # Track dass URL gespeichert wurde
        return True  # Erfolgreich gespeichert
    except Exception as e:
        print(f"    ⚠️ Fehler beim Speichern von {url}: {e}")
        return False

def initialize_output_file():
    """Initialisiert Output-Datei und lädt existierende URLs."""
    global saved_urls
    
    # Lade existierende URLs falls Datei existiert
    if Path(OUTPUT_FILE).exists():
        load_existing_urls()
        print(f"📄 Erweitere existierende Datei: {OUTPUT_FILE}")
    else:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("")  # Leere Datei erstellen
            saved_urls = set()
            print(f"📄 Neue Output-Datei erstellt: {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ Fehler beim Erstellen der Output-Datei: {e}")

def save_results():
    """Zeigt finale Statistiken (Datei wird bereits live aktualisiert)."""
    new_urls = len(found_images - saved_urls) if hasattr(save_results, '_initial_count') else len(found_images)
    print(f"\n✅ {len(saved_urls)} URLs total in '{OUTPUT_FILE}' gespeichert")
    if new_urls > 0:
        print(f"🆕 {new_urls} neue URLs in dieser Session gefunden")

def get_filename_from_url(url, website_name):
    """Extrahiert Dateinamen aus URL mit Website-Prefix"""
    path = urlparse(url).path
    filename = os.path.basename(path)
    
    if not filename or '.' not in filename:
        # Fallback: Hash der URL verwenden
        hash_obj = hashlib.md5(url.encode())
        filename = f"image_{hash_obj.hexdigest()[:8]}.jpg"
    
    # Prefix mit Website-Name hinzufügen
    name, ext = os.path.splitext(filename)
    return f"{website_name}_{name}{ext}"

async def download_image(session, url, download_dir, website_name):
    """Lädt ein einzelnes Bild herunter und speichert Metadaten"""
    global image_metadata
    
    try:
        filename = get_filename_from_url(url, website_name)
        filepath = Path(download_dir) / website_name / filename
        
        # Erstelle Website-spezifisches Verzeichnis
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip wenn bereits existiert
        if filepath.exists():
            # Metadaten trotzdem speichern falls noch nicht vorhanden
            if str(filepath) not in image_metadata:
                image_metadata[str(filepath)] = {
                    'source_url': url,
                    'website': website_name,
                    'download_date': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            return filepath
        
        async with session.get(url) as response:
            if response.status == 200:
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
                    print(f"[!] Überspringe {url} (zu groß: {content_length} bytes)")
                    return None
                
                content = await response.read()
                
                with open(filepath, 'wb') as f:
                    f.write(content)
                
                # Metadaten mit Utilities speichern
                add_image_metadata(
                    local_path=str(filepath),
                    source_url=url,
                    website=website_name,
                    download_date=time.strftime('%Y-%m-%d %H:%M:%S'),
                    file_size=len(content)
                )
                
                # Auch in globale Variable für Kompatibilität
                image_metadata[str(filepath)] = {
                    'source_url': url,
                    'website': website_name,
                    'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'file_size': len(content)
                }
                
                print(f"[✓] {website_name}/{filename}")
                return filepath
            else:
                print(f"[!] HTTP {response.status}: {url}")
                return None
                
    except Exception as e:
        print(f"[!] Download-Fehler {url}: {e}")
        return None

async def download_and_filter_images():
    """Lädt alle Bilder herunter und speichert Metadaten"""
    global image_metadata
    
    # Erstelle Verzeichnisse
    Path(DOWNLOAD_DIR).mkdir(exist_ok=True)
    
    # Lade URLs aus Datei
    if not Path(OUTPUT_FILE).exists():
        print(f"[!] {OUTPUT_FILE} nicht gefunden!")
        return
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        print("[!] Keine URLs in der Datei gefunden!")
        return
    
    # Website-Name aus BASE_URL extrahieren
    website_name = urlparse(BASE_URL).netloc.replace('.', '_').replace('-', '_')
    
    # Lade existierende Metadaten falls vorhanden
    metadata_path = Path(DOWNLOAD_DIR) / METADATA_FILE
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)
            print(f"[*] {len(image_metadata)} existierende Metadaten geladen")
        except Exception as e:
            print(f"[!] Fehler beim Laden der Metadaten: {e}")
            image_metadata = {}
    
    print(f"[*] Starte Download von {len(urls)} Bildern in {DOWNLOAD_DIR}/{website_name}/...")
    
    # Asynchroner Download mit Semaphore
    import aiohttp
    
    connector = aiohttp.TCPConnector(limit=CONCURRENT_DOWNLOADS)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
        
        async def download_with_semaphore(url):
            async with sem:
                return await download_image(session, url, DOWNLOAD_DIR, website_name)
        
        download_tasks = [download_with_semaphore(url) for url in urls]
        downloaded_files = await asyncio.gather(*download_tasks, return_exceptions=True)
    
    # Speichere Metadaten
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(image_metadata, f, indent=2, ensure_ascii=False)
        print(f"[*] Metadaten gespeichert: {metadata_path}")
    except Exception as e:
        print(f"[!] Fehler beim Speichern der Metadaten: {e}")
    
    # Filtere nach erfolgreichen Downloads
    successful_downloads = [f for f in downloaded_files if isinstance(f, Path) and f and f.exists()]
    print(f"[*] {len(successful_downloads)} Bilder erfolgreich heruntergeladen")
    
    if successful_downloads:
        print("\n" + "="*50)
        print("DOWNLOAD ABGESCHLOSSEN")
        print("="*50)
        print(f"Heruntergeladene Bilder: {len(successful_downloads)}")
        print(f"Verzeichnis: {DOWNLOAD_DIR}/{website_name}/")
        print(f"Metadaten: {metadata_path}")
        print("="*50)
    else:
        print("[!] Keine Bilder heruntergeladen.")
    
    # Leeres Download-Verzeichnis aufräumen (falls leer)
    try:
        if Path(DOWNLOAD_DIR).exists() and not any(Path(DOWNLOAD_DIR).iterdir()):
            Path(DOWNLOAD_DIR).rmdir()
            print(f"[*] Leeres Verzeichnis '{DOWNLOAD_DIR}' aufgeräumt")
    except:
        pass

async def main():
    start = time.time()
    
    # Prüfe ob bereits URLs vorhanden sind
    if Path(OUTPUT_FILE).exists():
        print(f"[*] {OUTPUT_FILE} gefunden.")
        choice = input("Möchten Sie die URLs herunterladen? (j/n): ").lower().strip()
        if choice in ['j', 'ja', 'y', 'yes', '']:
            print("[*] Starte Download...")
            await download_and_filter_images()
            print(f"Fertig in {time.time()-start:.1f}s")
            return
    
    # Initialisiere Output-Datei für Live-Updates
    initialize_output_file()
    
    print("[*] Playwright-Crawl startet...")
    await crawl_playwright()
    print("[*] Playwright-Crawl fertig.")
    print("[*] Bruteforce-Scan startet...")
    brute_force_directories()
    print("[*] Bruteforce fertig.")
    save_results()  # Zeigt nur Statistiken
    
    # Automatisch Download starten wenn URLs gefunden
    if found_images:
        print(f"\n[*] {len(found_images)} Bild-URLs gefunden.")
        choice = input("Möchten Sie die Bilder jetzt herunterladen? (j/n): ").lower().strip()
        if choice in ['j', 'ja', 'y', 'yes', '']:
            await download_and_filter_images()
    
    print(f"Fertig in {time.time()-start:.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
