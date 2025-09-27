#!/usr/bin/env python3
"""
Image Crawler

Vereint:
- Playwright-basiertes Crawlen (JS-rendering, lazy-load, network capture)  
- Recursion auf interne Links (bis max_depth)
- Bruteforce für typische Upload-Verzeichnisse (mit         # Auch direkte URLs im Text suchen (für AWS S3 Links etc.)
        text_content = soup.get_text()
        aws_pattern = r'(https?://[^\s<>"\']*\.s3[^\s<>"\']*\.(jpg|jpeg|png|gif|webp|bmp))'
        aws_matches = re.findall(aws_pattern, text_content, re.I)
        for match in aws_matches:
            full_url = match[0] if isinstance(match, tuple) else match
            if is_related_image_url(full_url):
                if full_url not in found_images:
                    found_images.add(full_url)
                    # Track which page this image was found on
                    image_page_mapping[full_url] = url
                    if save_image_url_live(full_url):
                        print(f"    📄 {full_url} (Text-Link)")
                    else:
                        print(f"    🔄 {full_url} (bereits vorhanden)") Speichert alle gefundenen Bild-URLs in 'bilder_alles.txt'

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
BASE_URL = "https://oxfordhigh.gdst.net/"   # <-- anpassen
MAX_DEPTH = float('inf')  # Keine Begrenzung - crawle alle Seiten!
CONCURRENT_PAGES = 3
NAV_TIMEOUT = 20000  # ms
OUTPUT_FILE = "bilder_alles.txt"

# Download-Konfiguration
DOWNLOAD_DIR = "data/images"  # Direkt in static/images
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
image_page_mapping = {}  # Mapping von image_url -> source_page_url

# Erweiterte Wortliste für Upload-Verzeichnisse (ALLE möglichen Begriffe)
UPLOAD_KEYWORDS = [
    "uploads", "upload", "media", "files", "images", "img", "pics", "pictures", 
    "photos", "gallery", "assets", "content", "wp-content", "resources", "static",
    "data", "storage", "documents", "docs", "downloads", "public", "shared",
    "user", "users", "member", "members", "profile", "profiles", "avatar", "avatars",
    "news", "articles", "blog", "post", "posts", "thumb", "thumbnails", "cache",
    "temp", "tmp", "archive", "backup", "old", "new", "current", "latest",
    # WordPress spezifische Verzeichnisse
    "wp-uploads", "wp-includes", "wp-admin", "themes", "plugins", "mu-plugins",
    # Weitere typische CMS-Verzeichnisse
    "administrator", "admin", "manager", "cms", "system", "lib", "libraries",
    # ALLE Jahre (2000-2030)
    "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009",
    "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
    "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029", "2030",
    # Monats-basierte Strukturen  
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
    "january", "february", "march", "april", "may", "june", 
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # Event-basierte Verzeichnisse
    "events", "galleries", "slideshow", "slider", "carousel", "lightbox", "portfolio",
    "exhibitions", "awards", "concerts", "sports", "trips", "activities", "clubs",
    "academic", "curriculum", "boarding", "sixth-form", "prep", "senior",
    # Größen-Varianten
    "thumb", "thumbnail", "small", "medium", "large", "xl", "original",
    "150x150", "300x200", "1024x768", "scaled", "resized", "cropped",
    # Historische/Legacy Begriffe
    "legacy", "archive", "historical", "oldsite", "backup", "migration"
]

# Zusätzliche WordPress-spezifische Pfade
WORDPRESS_PATHS = [
    "wp-content/uploads/",
    "wp-content/themes/",
    "wp-content/plugins/", 
    "wp-includes/images/",
    "wp-admin/images/",
    "uploads/",
    "files/",
    "media/",
    "assets/img/",
    "assets/images/",
    "images/",
    "img/",
    "pictures/",
    "photos/",
    "gallery/"
]

# Datei-Endungen für aggressivere Suche
IMAGE_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.tiff', '.tif',
    '.ico', '.avif', '.heic', '.heif', '.raw', '.dng', '.cr2', '.nef', '.arw'
]

def is_internal(url):
    """Prüft, ob URL zur gleichen Domain gehört"""
    return urlparse(url).netloc in ['', urlparse(BASE_URL).netloc]

def is_related_image_url(url):
    """Prüft, ob URL ein verwandtes Bild ist (auch externe CDNs/S3)"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    
    # Interne URLs sind immer OK
    if domain in ['', urlparse(BASE_URL).netloc]:
        return True
    
    # AWS S3 URLs für diese Website (spezifisch)
    if 'twk-media-offload.s3.eu-west-1.amazonaws.com' in domain:
        return True
    
    # Allgemeine AWS S3 Patterns für diese Domain
    if 's3' in domain and any(pattern in domain for pattern in [
        'oxfordhigh', 'gdst', 'twk-media'
    ]):
        return True
    
    # CloudFront und andere AWS CDNs
    if '.cloudfront.net' in domain:
        return True
    
    # Weitere CDNs die zur Website gehören könnten
    base_domain = urlparse(BASE_URL).netloc.replace('www.', '')
    if base_domain in domain or domain in base_domain:
        return True
    
    # Typische CDN-Pattern für diese Website
    if any(pattern in domain for pattern in [
        'oxfordhigh', 'gdst', 'twk-media', 'cdn', 'assets', 'media', 'static'
    ]):
        return True
    
    # WordPress spezifische Patterns
    if 'wp-content' in path or 'wp-uploads' in path or 'uploads' in path:
        return True
    
    print(f"    ⚠️  Externe URL übersprungen: {domain}")
    return False

def normalize(url):
    """Normalisiert URL (entfernt Fragment)"""
    return urldefrag(url)[0]

def is_allowed_image(url):
    """Prüft, ob URL ein erlaubtes Bildformat hat - erweiterte Version"""
    url_lower = url.lower()
    # Erweiterte Liste von Bild-Formaten
    return any(url_lower.endswith(ext) for ext in IMAGE_EXTENSIONS)

def get_potential_image_urls(base_url):
    """Generiert potentielle Bild-URLs basierend auf typischen Strukturen"""
    base_parsed = urlparse(base_url)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    urls = set()
    
    # WordPress-spezifische Pfade
    for wp_path in WORDPRESS_PATHS:
        urls.add(f"{base_clean}/{wp_path}")
    
    # Jahres-basierte WordPress-Strukturen (erweitert auf 30 Jahre)
    for year in range(2000, 2031):  # 31 Jahre: 2000-2030
        for month in range(1, 13):
            month_str = f"{month:02d}"
            urls.add(f"{base_clean}/wp-content/uploads/{year}/{month_str}/")
            urls.add(f"{base_clean}/uploads/{year}/{month_str}/")
            urls.add(f"{base_clean}/media/{year}/{month_str}/")
            # Auch S3-Strukturen für alle Jahre
            urls.add(f"https://twk-media-offload.s3.eu-west-1.amazonaws.com/oxfordhigh.gdst.net/wp-uploads/{year}/{month_str}/")
    
    # Kombinationen von Keywords
    for keyword in UPLOAD_KEYWORDS:
        urls.add(f"{base_clean}/{keyword}/")
        urls.add(f"{base_clean}/wp-content/{keyword}/")
        urls.add(f"{base_clean}/assets/{keyword}/")
        urls.add(f"{base_clean}/static/{keyword}/")
        urls.add(f"{base_clean}/media/{keyword}/")
        urls.add(f"{base_clean}/content/{keyword}/")
        urls.add(f"{base_clean}/files/{keyword}/")
    
    return urls

def search_alternative_domains():
    """Sucht nach Bildern auf alternativen/historischen Domains"""
    global found_images
    
    print("[*] 🔍 Suche nach alternativen Domains und Subdomains...")
    
    # Alternative Domains/Subdomains die versucht werden sollen
    base_domain = urlparse(BASE_URL).netloc.replace('www.', '')
    alternative_domains = [
        f"https://www.{base_domain}",
        f"https://{base_domain}",
        f"https://old.{base_domain}",
        f"https://archive.{base_domain}",
        f"https://legacy.{base_domain}",
        f"https://backup.{base_domain}",
        f"https://media.{base_domain}",
        f"https://images.{base_domain}",
        f"https://files.{base_domain}",
        f"https://assets.{base_domain}",
        f"https://cdn.{base_domain}",
        f"https://static.{base_domain}",
        # Auch ohne HTTPS probieren
        f"http://www.{base_domain}",
        f"http://{base_domain}",
        f"http://old.{base_domain}",
        f"http://archive.{base_domain}",
    ]
    
    for alt_domain in alternative_domains:
        try:
            # Teste ob Domain erreichbar ist
            response = requests.get(alt_domain, timeout=5, allow_redirects=True)
            if response.status_code == 200:
                print(f"    ✅ Alternative Domain gefunden: {alt_domain}")
                
                # Suche typische Upload-Verzeichnisse auf dieser Domain
                for wp_path in WORDPRESS_PATHS:
                    test_url = f"{alt_domain}/{wp_path}"
                    try:
                        test_response = requests.get(test_url, timeout=3)
                        if test_response.status_code == 200:
                            print(f"        📁 {test_url}")
                            # Parse für Bilder
                            soup = BeautifulSoup(test_response.text, 'html.parser')
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                if is_allowed_image(href):
                                    full_url = urljoin(test_url, href)
                                    if full_url not in found_images:
                                        found_images.add(full_url)
                                        if save_image_url_live(full_url):
                                            print(f"            🖼️  {full_url}")
                    except:
                        continue
                        
        except:
            continue

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
    global found_images, visited, errors, queue, image_page_mapping
    
    if url in visited:
        return
    
    print(f"[{depth}] {url}")
    visited.add(url)
    
    def on_response(response):
        """Interceptor für alle Netzwerk-Requests"""
        resp_url = response.url
        if is_allowed_image(resp_url) and is_related_image_url(resp_url):
            if resp_url not in found_images:  # Noch nicht in dieser Session gefunden
                found_images.add(resp_url)
                # Track which page this image was found on
                image_page_mapping[resp_url] = url
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
        
        # Alle img-Tags sammeln (inkl. verschiedene lazy-loading Attribute)
        for img in soup.find_all('img'):
            # Verschiedene mögliche Attribute für Bild-URLs
            src_attrs = ['src', 'data-src', 'data-lazy-src', 'data-original', 'data-srcset', 'data-url']
            for attr in src_attrs:
                src = img.get(attr)
                if src:
                    # Mehrere URLs in srcset behandeln
                    urls = [s.split()[0] for s in src.split(',')]
                    for url_part in urls:
                        full_url = urljoin(url, url_part.strip())
                        if is_allowed_image(full_url) and is_related_image_url(full_url):
                            if full_url not in found_images:
                                found_images.add(full_url)
                                # Track which page this image was found on
                                image_page_mapping[full_url] = url
                                if save_image_url_live(full_url):
                                    print(f"    🖼️  {full_url}")
                                else:
                                    print(f"    🔄 {full_url} (bereits vorhanden)")
        
        # CSS background-images suchen
        for element in soup.find_all(style=True):
            style = element.get('style', '')
            # Regex für background-image URLs
            bg_matches = re.findall(r'background-image:\s*url\(["\']?([^"\')\s]+)["\']?\)', style, re.I)
            for bg_url in bg_matches:
                full_url = urljoin(url, bg_url)
                if is_allowed_image(full_url) and is_related_image_url(full_url):
                    if full_url not in found_images:
                        found_images.add(full_url)
                        # Track which page this image was found on
                        image_page_mapping[full_url] = url
                        if save_image_url_live(full_url):
                            print(f"    🎨 {full_url} (CSS background)")
                        else:
                            print(f"    🔄 {full_url} (bereits vorhanden)")
        
        # Auch direkte URLs im Text suchen (für AWS S3 Links etc.)
        text_content = soup.get_text()
        aws_matches = re.findall(r'https?://[^\s<>"\']*\.s3[^\s<>"\']*\.(jpg|jpeg|png|gif|webp|bmp)', text_content, re.I)
        for match in aws_matches:
            full_url = match[0] if isinstance(match, tuple) else match
            if is_related_image_url(full_url):
                if full_url not in found_images:
                    found_images.add(full_url)
                    if save_image_url_live(full_url):
                        print(f"    � {full_url} (Text-Link)")
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
    """Erweiterte Brute-Force für ALLE möglichen Upload-Verzeichnisse"""
    global found_images
    
    print("[*] 🚀 Starte ERWEITERTEN Bruteforce-Scan...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    # 1. Generiere alle potentiellen URLs
    potential_urls = get_potential_image_urls(BASE_URL)
    print(f"[*] 📋 Prüfe {len(potential_urls)} potentielle Verzeichnisse...")
    
    scan_count = 0
    found_dirs = 0
    
    for test_url in potential_urls:
        scan_count += 1
        if scan_count % 50 == 0:
            print(f"[*] 📊 Fortschritt: {scan_count}/{len(potential_urls)} - {found_dirs} Verzeichnisse gefunden")
        
        try:
            response = requests.get(test_url, timeout=5, allow_redirects=False)
            if response.status_code == 200:
                found_dirs += 1
                print(f"    ✅ Directory: {test_url}")
                
                # Parse Directory Listing
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if is_allowed_image(href):
                        full_url = urljoin(test_url, href)
                        if is_related_image_url(full_url) and full_url not in found_images:
                            found_images.add(full_url)
                            if save_image_url_live(full_url):
                                print(f"        🖼️  {full_url}")
                            else:
                                print(f"        🔄 {full_url} (bereits vorhanden)")
                        
        except Exception:
            continue  # Weiter mit nächstem Verzeichnis
    
    print(f"[*] ✅ Bruteforce abgeschlossen: {found_dirs} zugängliche Verzeichnisse gefunden")

def search_wordpress_api():
    """Durchsucht WordPress REST API nach ALLEN Medien (mit Pagination)"""
    global found_images
    
    print("[*] 🔍 Durchsuche WordPress REST API (ALLE Medien)...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    total_media_found = 0
    page = 1
    per_page = 100  # Maximum pro Seite
    
    while True:
        print(f"    📄 API Seite {page} wird durchsucht...")
        
        # WordPress REST API Endpunkte mit Pagination
        api_endpoints = [
            f"{base_clean}/wp-json/wp/v2/media?per_page={per_page}&page={page}",
            f"{base_clean}/wp-json/wp/v2/media?per_page={per_page}&page={page}&status=publish",
            f"{base_clean}/?rest_route=/wp/v2/media&per_page={per_page}&page={page}",
        ]
        
        media_found_this_page = 0
        
        for api_url in api_endpoints:
            try:
                response = requests.get(api_url, timeout=15)
                if response.status_code == 200:
                    try:
                        media_data = response.json()
                        if isinstance(media_data, list) and len(media_data) > 0:
                            media_found_this_page = len(media_data)
                            print(f"        ✅ Seite {page}: {media_found_this_page} Medien-Objekte gefunden")
                            
                            for item in media_data:
                                # Haupt-URL
                                if 'source_url' in item:
                                    img_url = item['source_url']
                                    if is_related_image_url(img_url) and img_url not in found_images:
                                        found_images.add(img_url)
                                        if save_image_url_live(img_url):
                                            print(f"        📡 {img_url}")
                                
                                # Alle Größen-Varianten
                                if 'media_details' in item and 'sizes' in item['media_details']:
                                    sizes = item['media_details']['sizes']
                                    for size_name, size_data in sizes.items():
                                        if 'source_url' in size_data:
                                            img_url = size_data['source_url']
                                            if is_related_image_url(img_url) and img_url not in found_images:
                                                found_images.add(img_url)
                                                if save_image_url_live(img_url):
                                                    print(f"        📐 {img_url} ({size_name})")
                            
                            total_media_found += media_found_this_page
                            break  # Erfolg mit diesem Endpoint, nicht andere probieren
                        else:
                            # Keine Daten mehr - Ende erreicht
                            break
                    except json.JSONDecodeError:
                        continue
                elif response.status_code == 400:
                    # Bad Request - wahrscheinlich keine Seiten mehr
                    print(f"    ℹ️  Seite {page}: Keine weiteren Seiten verfügbar")
                    break
            except Exception as e:
                continue
        
        # Wenn keine Medien auf dieser Seite gefunden wurden, sind wir fertig
        if media_found_this_page == 0:
            break
        
        page += 1
        
        # Sicherheits-Stop (falls API endlos läuft) - ERWEITERT für ALLE Bilder
        if page > 5000:  # Max 500.000 Medien (100 * 5000) - für ALLE Bilder
            print("    ⚠️  Sicherheits-Stop bei Seite 5000 erreicht (500.000+ Medien geprüft)")
            break
    
    print(f"    🎉 WordPress API Abschluss: {total_media_found} Medien-Objekte über {page-1} Seiten durchsucht!")

def search_sitemaps():
    """Durchsucht XML-Sitemaps nach Bild-URLs"""
    global found_images
    
    print("[*] 🗺️  Durchsuche Sitemaps...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    # Typische Sitemap-URLs
    sitemap_urls = [
        f"{base_clean}/sitemap.xml",
        f"{base_clean}/sitemap_index.xml", 
        f"{base_clean}/wp-sitemap.xml",
        f"{base_clean}/sitemap-images.xml",
        f"{base_clean}/image-sitemap.xml",
        f"{base_clean}/robots.txt",  # Kann Sitemap-Links enthalten
    ]
    
    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                print(f"    ✅ Sitemap gefunden: {sitemap_url}")
                
                # XML-Sitemaps
                if sitemap_url.endswith('.xml'):
                    soup = BeautifulSoup(content, 'xml')
                    # Standard-Sitemaps
                    for loc in soup.find_all('loc'):
                        url = loc.get_text()
                        if is_allowed_image(url) and is_related_image_url(url):
                            if url not in found_images:
                                found_images.add(url)
                                if save_image_url_live(url):
                                    print(f"        🗺️  {url}")
                    
                    # Google Bilder-Sitemaps
                    for img in soup.find_all('image:loc'):
                        url = img.get_text()
                        if is_related_image_url(url) and url not in found_images:
                            found_images.add(url)
                            if save_image_url_live(url):
                                print(f"        🖼️  {url}")
                
                # robots.txt
                if sitemap_url.endswith('robots.txt'):
                    for line in content.split('\n'):
                        if line.startswith('Sitemap:'):
                            nested_sitemap = line.split(':', 1)[1].strip()
                            sitemap_urls.append(nested_sitemap)
                            
        except Exception as e:
            continue

def search_historical_uploads():
    """Durchsucht systematisch historische WordPress Upload-Strukturen"""
    global found_images
    
    print("[*] 📚 Durchsuche HISTORISCHE Upload-Verzeichnisse...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    # Erweiterte historische Suche: 2000-2030 (30 Jahre!)
    years_to_check = list(range(2000, 2031))  # 31 Jahre!
    months_to_check = list(range(1, 13))      # Alle Monate
    
    total_dirs_checked = 0
    found_dirs = 0
    
    print(f"    📅 Prüfe {len(years_to_check)} Jahre × {len(months_to_check)} Monate = {len(years_to_check) * len(months_to_check)} Verzeichnisse...")
    
    for year in years_to_check:
        year_found = 0
        for month in months_to_check:
            month_str = f"{month:02d}"
            total_dirs_checked += 1
            
            # Verschiedene WordPress Upload-Strukturen
            upload_patterns = [
                f"{base_clean}/wp-content/uploads/{year}/{month_str}/",
                f"{base_clean}/wp-uploads/{year}/{month_str}/",
                f"{base_clean}/uploads/{year}/{month_str}/",
                f"{base_clean}/media/{year}/{month_str}/",
                f"{base_clean}/files/{year}/{month_str}/",
                # Auch S3/CDN Strukturen
                f"https://twk-media-offload.s3.eu-west-1.amazonaws.com/oxfordhigh.gdst.net/wp-uploads/{year}/{month_str}/",
            ]
            
            for upload_url in upload_patterns:
                try:
                    response = requests.get(upload_url, timeout=3, allow_redirects=False)
                    if response.status_code == 200:
                        found_dirs += 1
                        year_found += 1
                        print(f"        ✅ {year}/{month_str}: {upload_url}")
                        
                        # Parse Directory Listing
                        soup = BeautifulSoup(response.text, 'html.parser')
                        images_in_dir = 0
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            if is_allowed_image(href):
                                full_url = urljoin(upload_url, href)
                                if is_related_image_url(full_url) and full_url not in found_images:
                                    found_images.add(full_url)
                                    images_in_dir += 1
                                    if save_image_url_live(full_url):
                                        print(f"            📸 {full_url}")
                        
                        if images_in_dir > 0:
                            print(f"            → {images_in_dir} Bilder in {year}/{month_str}")
                        break  # Erstes gefundenes Pattern reicht
                        
                except Exception:
                    continue
        
        if year_found > 0:
            print(f"    📈 Jahr {year}: {year_found} Verzeichnisse mit Inhalten gefunden")
        
        # Fortschritt alle 5 Jahre anzeigen (statt 2)
        if year % 5 == 0:
            progress_pct = ((year - 2000) / (2030 - 2000)) * 100
            print(f"    🔄 Historische Suche: {progress_pct:.0f}% abgeschlossen...")
    
    print(f"    🎯 Historische Suche abgeschlossen:")
    print(f"        - {total_dirs_checked} Verzeichnisse geprüft")
    print(f"        - {found_dirs} zugängliche Verzeichnisse gefunden")
    print(f"        - Zeitraum: 2000-2030 (31 Jahre)")

def search_additional_wordpress_endpoints():
    """Sucht zusätzliche WordPress-Endpunkte nach Medien"""
    global found_images
    
    print("[*] 🔧 Durchsuche zusätzliche WordPress-Endpunkte...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    # Zusätzliche WordPress-Endpunkte
    additional_endpoints = [
        # Posts mit eingebetteten Medien
        f"{base_clean}/wp-json/wp/v2/posts?per_page=100&_embed",
        f"{base_clean}/wp-json/wp/v2/pages?per_page=100&_embed",
        # Galleries
        f"{base_clean}/wp-json/wp/v2/media?media_type=image&per_page=100",
        # Attachments
        f"{base_clean}/wp-json/wp/v2/media?post_status=inherit&per_page=100",
        # Custom fields
        f"{base_clean}/wp-json/acf/v3/posts",
    ]
    
    for endpoint in additional_endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        print(f"    ✅ Endpoint: {len(data)} Objekte gefunden")
                        
                        for item in data:
                            # Verschiedene Felder durchsuchen
                            fields_to_check = [
                                'featured_media', 'content', 'excerpt', 
                                'meta', 'acf', '_embedded'
                            ]
                            
                            # Rekursive Suche nach Bild-URLs in den Daten
                            urls = extract_urls_from_data(item)
                            for url in urls:
                                if is_allowed_image(url) and is_related_image_url(url):
                                    if url not in found_images:
                                        found_images.add(url)
                                        if save_image_url_live(url):
                                            print(f"        🔗 {url}")
                except json.JSONDecodeError:
                    continue
        except Exception:
            continue

def extract_urls_from_data(data, max_depth=3):
    """Extrahiert rekursiv alle URLs aus JSON-Daten"""
    urls = set()
    
    if max_depth <= 0:
        return urls
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and value.startswith(('http://', 'https://')):
                urls.add(value)
            elif isinstance(value, (dict, list)):
                urls.update(extract_urls_from_data(value, max_depth - 1))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.startswith(('http://', 'https://')):
                urls.add(item)
            elif isinstance(item, (dict, list)):
                urls.update(extract_urls_from_data(item, max_depth - 1))
    
    return urls  # Stille Fehler für Bruteforce

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
    global image_metadata, image_page_mapping
    
    try:
        filename = get_filename_from_url(url, website_name)
        filepath = Path(download_dir) / website_name / filename
        
        # Erstelle Website-spezifisches Verzeichnis
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip wenn bereits existiert
        if filepath.exists():
            # Metadaten trotzdem speichern falls noch nicht vorhanden
            if str(filepath) not in image_metadata:
                source_page_url = image_page_mapping.get(url)
                image_metadata[str(filepath)] = {
                    'source_url': url,
                    'website': website_name,
                    'download_date': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                # Save enhanced metadata
                add_image_metadata(
                    local_path=str(filepath),
                    image_url=url,
                    source_page_url=source_page_url,
                    website=website_name,
                    download_date=time.strftime('%Y-%m-%d %H:%M:%S')
                )
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
                
                # Get source page URL from mapping
                source_page_url = image_page_mapping.get(url)
                
                # Enhanced metadata with both URLs
                add_image_metadata(
                    local_path=str(filepath),
                    image_url=url,
                    source_page_url=source_page_url,
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
    
    # Automatische Duplikat-Bereinigung nach erfolgreichem Download
    if successful_downloads:
        try:
            from image_quality_manager import clean_image_duplicates
            print("\n" + "="*50)
            print("🧹 AUTOMATISCHE DUPLIKAT-BEREINIGUNG")
            print("="*50)
            
            # Bereinige Duplikate
            results = clean_image_duplicates(DOWNLOAD_DIR, dry_run=False)
            stats = results['removal_stats']
            
            if stats['files_to_remove'] > 0:
                mb_freed = stats['bytes_to_free'] / (1024 * 1024)
                print(f"✅ {stats['files_to_remove']} doppelte Bilder entfernt")
                print(f"💾 {mb_freed:.2f} MB Speicherplatz freigegeben")
                print(f"📁 {stats['files_to_keep']} beste Qualitäts-Versionen beibehalten")
                
                # Aktualisierte Statistiken
                remaining_files = len(successful_downloads) - stats['files_to_remove']
                print(f"\n📊 FINALE STATISTIKEN:")
                print(f"   Original heruntergeladen: {len(successful_downloads)} Dateien")
                print(f"   Nach Duplikat-Bereinigung: {remaining_files} einzigartige Bilder")
                print(f"   Qualitätsverbesserung: Nur beste Versionen beibehalten")
            else:
                print("✅ Keine Duplikate gefunden - alle Bilder sind einzigartig")
                
        except ImportError:
            print("⚠️ Image Quality Manager nicht verfügbar - Duplikat-Bereinigung übersprungen")
        except Exception as e:
            print(f"⚠️ Fehler bei Duplikat-Bereinigung: {e}")
    
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
    
    print("🚀 ULTIMATIVE BILD-URL SUCHE GESTARTET!")
    print("=" * 60)
    
    # 1. Playwright-Crawl (Standard)
    print("\n[1/5] 🎭 Playwright-Crawl (JavaScript + Lazy Loading)...")
    await crawl_playwright()
    print(f"      ✅ {len(found_images)} URLs nach Playwright-Crawl")
    
    # 2. WordPress API Suche (ALLE Seiten)
    print("\n[2/6] 📡 WordPress REST API Suche (ALLE Medien)...")
    search_wordpress_api()
    search_additional_wordpress_endpoints()
    print(f"      ✅ {len(found_images)} URLs nach API-Suche")
    
    # 3. Sitemap-Durchsuchung
    print("\n[3/5] 🗺️  XML-Sitemap Durchsuchung...")
    search_sitemaps()
    print(f"      ✅ {len(found_images)} URLs nach Sitemap-Suche")
    
    # 4. Erweiterte Brute-Force + Historische Suche + Alternative Domains
    print("\n[4/7] 💥 ERWEITERTE Brute-Force (Historische Upload-Suche 2000-2030)...")
    brute_force_directories()
    search_historical_uploads()
    search_alternative_domains()
    print(f"      ✅ {len(found_images)} URLs nach Brute-Force + Historischer Suche + Alternative Domains")
    
    # 5. Finale Statistiken
    print("\n[5/7] 📊 Abschluss und Statistiken...")
    save_results()
    
    print("\n" + "=" * 60)
    print(f"🎉 ULTIMATIVE SUCHE ABGESCHLOSSEN!")
    print(f"⏱️  Gesamtzeit: {time.time()-start:.1f}s")
    print(f"🖼️  Insgesamt gefunden: {len(found_images)} Bild-URLs")
    print(f"📅 Zeitraum: 2000-2030 (31 Jahre abgedeckt)")
    print("=" * 60)
    
    # Automatisch Download starten wenn URLs gefunden
    if found_images:
        choice = input("\n💾 Möchten Sie ALLE Bilder jetzt herunterladen? (j/n): ").lower().strip()
        if choice in ['j', 'ja', 'y', 'yes', '']:
            await download_and_filter_images()
    
    print(f"\n✅ Fertig in {time.time()-start:.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
