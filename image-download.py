#!/usr/bin/env python3
"""
Intelligent Image Crawler

Combines:
- Playwright-based crawling (JS-rendering, lazy-load, network capture)  
- Recursion on internal links (up to max_depth)
- Bruteforce for typical upload directories
- Saves all found image URLs to 'all_images.txt'

USAGE: Only use on sites where you have permission!
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

# Import for metadata management
try:
    from image_metadata_utils import add_image_metadata, get_metadata_stats
except ImportError:
    # Fallback if utils not available
    def add_image_metadata(*args, **kwargs):
        pass
    def get_metadata_stats():
        return {}

# ------------- CONFIG -------------
BASE_URL = "https://example.com/"   # <-- CHANGE THIS TO YOUR TARGET WEBSITE
MAX_DEPTH = float('inf')  # No limit - crawl all pages!
CONCURRENT_PAGES = 3
NAV_TIMEOUT = 20000  # ms
OUTPUT_FILE = "all_images.txt"

# Download Configuration
DOWNLOAD_DIR = "data/images"  # Direct to data/images
MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024  # 50MB max per image
CONCURRENT_DOWNLOADS = 5
METADATA_FILE = "image_metadata.json"  # Stores URL mapping

# Global Lists
found_images = set()
visited = set()
errors = []
image_metadata = {}  # Stores URL -> local path mapping
queue = []  # Global queue for workers
saved_urls = set()  # Track already saved URLs

# Extended word list for upload directories (ALL possible terms)
UPLOAD_KEYWORDS = [
    "uploads", "upload", "media", "files", "images", "img", "pics", "pictures", 
    "photos", "gallery", "assets", "content", "wp-content", "resources", "static",
    "data", "storage", "documents", "docs", "downloads", "public", "shared",
    "user", "users", "member", "members", "profile", "profiles", "avatar", "avatars",
    "news", "articles", "blog", "post", "posts", "thumb", "thumbnails", "cache",
    "temp", "tmp", "archive", "backup", "old", "new", "current", "latest",
    # WordPress specific directories
    "wp-uploads", "wp-includes", "wp-admin", "themes", "plugins", "mu-plugins",
    # Other typical CMS directories
    "administrator", "admin", "manager", "cms", "system", "lib", "libraries",
    # Generate years dynamically (1980-2035 for broad coverage)
] + [str(year) for year in range(1980, 2036)] + [
    # Month-based structures  
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
    "january", "february", "march", "april", "may", "june", 
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # Event-based directories
    "events", "galleries", "slideshow", "slider", "carousel", "lightbox", "portfolio",
    "exhibitions", "awards", "concerts", "sports", "trips", "activities", "clubs",
    "academic", "curriculum", "boarding", "sixth-form", "prep", "senior",
    # Size variants
    "thumb", "thumbnail", "small", "medium", "large", "xl", "original",
    "150x150", "300x200", "1024x768", "scaled", "resized", "cropped",
    # Historical/Legacy terms
    "legacy", "archive", "historical", "oldsite", "backup", "migration"
]

# Additional WordPress-specific paths
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

# File extensions for aggressive search
IMAGE_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.tiff', '.tif',
    '.ico', '.avif', '.heic', '.heif', '.raw', '.dng', '.cr2', '.nef', '.arw'
]

def is_internal(url):
    """Checks if URL belongs to the same domain"""
    return urlparse(url).netloc in ['', urlparse(BASE_URL).netloc]

def is_related_image_url(url):
    """Checks if URL is a related image (including external CDNs/S3) - DYNAMIC"""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    
    # Internal URLs are always OK
    if domain in ['', urlparse(BASE_URL).netloc]:
        return True
    
    # Extract base domain for dynamic matching
    base_domain_parts = urlparse(BASE_URL).netloc.replace('www.', '').split('.')
    if len(base_domain_parts) >= 2:
        main_domain = base_domain_parts[-2]  # e.g., 'example' from 'subdomain.example.com'
        tld = base_domain_parts[-1]          # e.g., 'com' from 'subdomain.example.com'
    else:
        main_domain = base_domain_parts[0] if base_domain_parts else ''
        tld = ''
    
    # Dynamic AWS S3 patterns based on domain
    s3_patterns = [
        main_domain,  # Main domain name in S3 bucket
        main_domain.replace('-', ''),  # Without hyphens
        main_domain.replace('_', ''),  # Without underscores
    ]
    
    # Check for S3 URLs related to this domain
    if 's3' in domain:
        for pattern in s3_patterns:
            if pattern and pattern in domain:
                return True
    
    # CloudFront and other AWS CDNs
    if '.cloudfront.net' in domain:
        for pattern in s3_patterns:
            if pattern and pattern in domain:
                return True
    
    # CDNs that might belong to this website
    base_domain_full = urlparse(BASE_URL).netloc.replace('www.', '')
    if base_domain_full in domain or domain in base_domain_full:
        return True
    
    # Typical CDN patterns for this website (dynamic)
    cdn_indicators = [main_domain, 'cdn', 'assets', 'media', 'static', 'images']
    if any(indicator in domain for indicator in cdn_indicators if indicator):
        return True
    
    # WordPress specific patterns
    if any(wp_path in path for wp_path in ['wp-content', 'wp-uploads', 'uploads']):
        return True
    
    print(f"    ⚠️  External URL skipped: {domain}")
    return False

def normalize(url):
    """Normalizes URL (removes fragment)"""
    return urldefrag(url)[0]

def is_allowed_image(url):
    """Checks if URL has an allowed image format - extended version"""
    url_lower = url.lower()
    # Extended list of image formats
    return any(url_lower.endswith(ext) for ext in IMAGE_EXTENSIONS)

def get_potential_image_urls(base_url):
    """Generates potential image URLs based on typical structures"""
    base_parsed = urlparse(base_url)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    urls = set()
    
    # WordPress-specific paths
    for wp_path in WORDPRESS_PATHS:
        urls.add(f"{base_clean}/{wp_path}")
    
    # Dynamic year-based WordPress structures
    # Generate years from 1990 to 10 years in the future
    import datetime
    current_year = datetime.datetime.now().year
    year_range = range(1990, current_year + 11)  # 1990 to current_year + 10
    
    # Extract domain name for S3 pattern (dynamic)
    domain_parts = base_parsed.netloc.replace('www.', '').split('.')
    if len(domain_parts) >= 2:
        domain_name = domain_parts[0]  # e.g., 'example' from 'example.com'
        domain_base = '.'.join(domain_parts[-2:])  # e.g., 'example.com'
    else:
        domain_name = domain_parts[0] if domain_parts else 'site'
        domain_base = base_parsed.netloc
    
    for year in year_range:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            urls.add(f"{base_clean}/wp-content/uploads/{year}/{month_str}/")
            urls.add(f"{base_clean}/uploads/{year}/{month_str}/")
            urls.add(f"{base_clean}/media/{year}/{month_str}/")
            # Dynamic S3 structures based on domain
            if domain_name and domain_base:
                # Common S3 patterns
                s3_patterns = [
                    f"https://{domain_name}-media.s3.amazonaws.com/{domain_base}/wp-uploads/{year}/{month_str}/",
                    f"https://{domain_name}-uploads.s3.amazonaws.com/{year}/{month_str}/",
                    f"https://media-{domain_name}.s3.amazonaws.com/{year}/{month_str}/",
                ]
                urls.update(s3_patterns)
    
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

def analyze_website_patterns():
    """Analyzes website-specific upload patterns through initial reconnaissance"""
    global found_images
    
    print("[*] 🕵️ Analyzing website patterns...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    patterns_found = {}
    
    # Test most common WordPress/CMS endpoints for pattern recognition
    test_endpoints = [
        '/wp-json/wp/v2/media?per_page=1',  # WordPress detection
        '/api/v1/media',  # Generic CMS API
        '/admin/',        # Admin panel detection
        '/wp-admin/',     # WordPress Admin
        '/administrator/', # Joomla Admin
    ]
    
    cms_type = "unknown"
    
    for endpoint in test_endpoints:
        try:
            response = requests.get(f"{base_clean}{endpoint}", timeout=3)
            if response.status_code in [200, 403]:  # 403 = exists but no permission
                if 'wp-json' in endpoint:
                    cms_type = "wordpress"
                    patterns_found['wordpress_api'] = True
                elif 'wp-admin' in endpoint:
                    cms_type = "wordpress"  
                    patterns_found['wordpress_admin'] = True
                elif 'administrator' in endpoint:
                    cms_type = "joomla"
                    patterns_found['joomla'] = True
                print(f"    ✅ Detected: {endpoint}")
        except:
            continue
    
    # Derive website-specific patterns
    if cms_type == "wordpress":
        print("    🎯 WordPress detected - optimizing upload search")
        # Use WordPress-specific optimizations in later functions
        return {
            'cms': 'wordpress',
            'priority_paths': ['/wp-content/uploads/', '/wp-uploads/'],
            'api_available': patterns_found.get('wordpress_api', False),
            'year_structure': True  # WordPress uses year/month structure
        }
    elif cms_type == "joomla":
        print("    🎯 Joomla detected - adapted search")
        return {
            'cms': 'joomla', 
            'priority_paths': ['/images/', '/media/'],
            'api_available': False,
            'year_structure': False
        }
    else:
        print("    🔍 Unknown CMS - using generic search")
        return {
            'cms': 'generic',
            'priority_paths': ['/uploads/', '/media/', '/images/', '/files/'],
            'api_available': False,
            'year_structure': False
        }

def search_alternative_domains():
    """Searches for images on alternative/historical domains with intelligent adaptation"""
    global found_images
    
    print("[*] 🔍 Searching alternative domains and subdomains...")
    
    # First analyze website patterns
    website_patterns = analyze_website_patterns()
    
    # Alternative domains/subdomains to try
    base_domain = urlparse(BASE_URL).netloc.replace('www.', '')
    
    # Intelligent domain generation based on CMS type
    alternative_domains = [
        f"https://www.{base_domain}",
        f"https://{base_domain}",
    ]
    
    # Add CMS-specific subdomains
    if website_patterns['cms'] == 'wordpress':
        alternative_domains.extend([
            f"https://media.{base_domain}",
            f"https://cdn.{base_domain}",
            f"https://assets.{base_domain}",
        ])
    
    # Historical domains for all
    alternative_domains.extend([
        f"https://old.{base_domain}",
        f"https://archive.{base_domain}", 
        f"https://legacy.{base_domain}",
        f"https://backup.{base_domain}",
        f"https://images.{base_domain}",
        f"https://files.{base_domain}",
        f"https://static.{base_domain}",
        # Also try without HTTPS for legacy sites
        f"http://www.{base_domain}",
        f"http://{base_domain}",
        f"http://old.{base_domain}",
        f"http://archive.{base_domain}",
    ])
    
    for alt_domain in alternative_domains:
        try:
            # Test if domain is reachable
            response = requests.get(alt_domain, timeout=5, allow_redirects=True)
            if response.status_code == 200:
                print(f"    ✅ Alternative domain found: {alt_domain}")
                
                # Search typical upload directories on this domain
                for wp_path in WORDPRESS_PATHS:
                    test_url = f"{alt_domain}/{wp_path}"
                    try:
                        test_response = requests.get(test_url, timeout=3)
                        if test_response.status_code == 200:
                            print(f"        📁 {test_url}")
                            # Parse for images
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
    """Processes a single page"""
    global found_images, visited, errors, queue
    
    if url in visited:
        return
    
    print(f"[{depth}] {url}")
    visited.add(url)
    
    def on_response(response):
        """Interceptor for all network requests"""
        resp_url = response.url
        if is_allowed_image(resp_url) and is_related_image_url(resp_url):
            if resp_url not in found_images:  # Not found in this session yet
                found_images.add(resp_url)
                if save_image_url_live(resp_url):  # Only output if really newly saved
                    print(f"    🖼️  {resp_url}")
                else:
                    print(f"    🔄 {resp_url} (already exists)")
    
    try:
        page = await browser.new_page()
        page.on("response", on_response)
        
        # Load page
        await page.goto(url, timeout=NAV_TIMEOUT, wait_until='networkidle')
        
        # Auto-scroll for Lazy Loading
        await auto_scroll(page)
        
        # Parse HTML
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Collect all img tags (including various lazy-loading attributes)
        for img in soup.find_all('img'):
            # Various possible attributes for image URLs
            src_attrs = ['src', 'data-src', 'data-lazy-src', 'data-original', 'data-srcset', 'data-url']
            for attr in src_attrs:
                src = img.get(attr)
                if src:
                    # Handle multiple URLs in srcset
                    urls = [s.split()[0] for s in src.split(',')]
                    for url_part in urls:
                        full_url = urljoin(url, url_part.strip())
                        if is_allowed_image(full_url) and is_related_image_url(full_url):
                            if full_url not in found_images:
                                found_images.add(full_url)
                                if save_image_url_live(full_url):
                                    print(f"    🖼️  {full_url}")
                                else:
                                    print(f"    🔄 {full_url} (already exists)")
        
        # Search CSS background-images
        for element in soup.find_all(style=True):
            style = element.get('style', '')
            # Regex for background-image URLs
            bg_matches = re.findall(r'background-image:\s*url\(["\']?([^"\')\s]+)["\']?\)', style, re.I)
            for bg_url in bg_matches:
                full_url = urljoin(url, bg_url)
                if is_allowed_image(full_url) and is_related_image_url(full_url):
                    if full_url not in found_images:
                        found_images.add(full_url)
                        if save_image_url_live(full_url):
                            print(f"    🎨 {full_url} (CSS background)")
                        else:
                            print(f"    🔄 {full_url} (already exists)")
        
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
                        print(f"    🔄 {full_url} (already exists)")
        
        # Links für weitere Crawling sammeln (alle Tiefen erlaubt)
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = normalize(urljoin(url, href))
            
            if is_internal(full_url) and full_url not in visited:
                queue.append((full_url, depth + 1))
        
        await page.close()
        
    except Exception as e:
        errors.append(f"{url}: {e}")
        print(f"    ❌ Error: {e}")

async def auto_scroll(page, pause=0.2, max_scrolls=30):
    """Automatic scrolling for Lazy Loading"""
    for i in range(max_scrolls):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(pause)

def brute_force_directories():
    """Intelligent brute-force with adaptive search scope"""
    global found_images
    
    print("[*] 🚀 Starting INTELLIGENT brute-force scan...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    # Intelligent priority levels
    priority_patterns = {
        'high': ['/wp-content/uploads/', '/uploads/', '/media/', '/images/', '/files/'],
        'medium': ['/assets/', '/content/', '/gallery/', '/photos/', '/pictures/'],
        'low': ['/static/', '/resources/', '/data/', '/storage/', '/documents/']
    }
    
    scan_count = 0
    found_dirs = 0
    successful_patterns = []  # Track successful patterns
    
    # Stage 1: High-priority directories without years
    print("    🎯 Stage 1: High-priority directories...")
    for pattern in priority_patterns['high']:
        test_url = f"{base_clean}{pattern}"
        try:
            response = requests.get(test_url, timeout=3, allow_redirects=False)
            if response.status_code == 200:
                found_dirs += 1
                successful_patterns.append(pattern)
                print(f"    ✅ Base directory: {test_url}")
                scan_directory_for_images(test_url, response.text)
        except Exception:
            continue
        scan_count += 1
    
    # Stage 2: If base directories found, expand intelligently
    if successful_patterns:
        print("    🔍 Stage 2: Extended search in successful patterns...")
        
        # Dynamic year extension based on successful patterns
        import datetime
        current_year = datetime.datetime.now().year
        
        # Smart year range: Check probable years first
        smart_years = []
        
        # Modern years (last 10 years)
        smart_years.extend(range(current_year - 9, current_year + 2))
        
        # Historical milestones (WordPress era, Web 2.0, etc.)
        historical_years = [2003, 2004, 2005, 2008, 2010, 2012, 2015]  # WordPress + important web years
        smart_years.extend([y for y in historical_years if y not in smart_years])
        
        # If many successes: Full range
        if len(successful_patterns) >= 2:
            smart_years = list(range(1995, current_year + 3))
        
        for pattern in successful_patterns:
            for year in smart_years:
                for month in [1, 6, 12]:  # First test January, June, December
                    month_str = f"{month:02d}"
                    test_url = f"{base_clean}{pattern}{year}/{month_str}/"
                    
                    try:
                        response = requests.get(test_url, timeout=2, allow_redirects=False)
                        if response.status_code == 200:
                            found_dirs += 1
                            print(f"    ✅ Year directory: {test_url}")
                            scan_directory_for_images(test_url, response.text)
                            
                            # On success: Check all months of this year
                            for full_month in range(1, 13):
                                if full_month not in [1, 6, 12]:  # Skip already checked
                                    full_month_str = f"{full_month:02d}"
                                    full_test_url = f"{base_clean}{pattern}{year}/{full_month_str}/"
                                    try:
                                        full_response = requests.get(full_test_url, timeout=2)
                                        if full_response.status_code == 200:
                                            scan_directory_for_images(full_test_url, full_response.text)
                                    except:
                                        continue
                    except Exception:
                        continue
                    scan_count += 1
                    
                    if scan_count % 100 == 0:
                        print(f"    📊 {scan_count} directories checked, {found_dirs} found...")
    
    # Stage 3: Medium/Low Priority only if little success so far
    if found_dirs < 5:
        print("    🔍 Stage 3: Extended directories...")
        for priority, patterns in [('medium', priority_patterns['medium']), 
                                  ('low', priority_patterns['low'])]:
            for pattern in patterns:
                test_url = f"{base_clean}{pattern}"
                try:
                    response = requests.get(test_url, timeout=2)
                    if response.status_code == 200:
                        found_dirs += 1
                        scan_directory_for_images(test_url, response.text)
                except:
                    continue
                scan_count += 1
    
    print(f"[*] ✅ Intelligent brute-force: {found_dirs} directories found ({scan_count} checked)")

def scan_directory_for_images(directory_url, html_content):
    """Helper function: Scans a directory for images"""
    global found_images
    
    soup = BeautifulSoup(html_content, 'html.parser')
    images_found = 0
    
    for link in soup.find_all('a', href=True):
        href = link.get('href')
        if href and is_allowed_image(href):
            full_url = urljoin(directory_url, href)
            if is_related_image_url(full_url) and full_url not in found_images:
                found_images.add(full_url)
                images_found += 1
                if save_image_url_live(full_url):
                    print(f"        🖼️  {full_url}")
    
    if images_found > 0:
        print(f"        → {images_found} images found")

def search_wordpress_api():
    """Searches WordPress REST API for ALL media (with pagination)"""
    global found_images
    
    print("[*] 🔍 Searching WordPress REST API (ALL media)...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    total_media_found = 0
    page = 1
    per_page = 100  # Maximum per page
    
    while True:
        print(f"    📄 API page {page} being searched...")
        
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
                            print(f"        ✅ Page {page}: {media_found_this_page} media objects found")
                            
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
                    print(f"    ℹ️  Page {page}: No more pages available")
                    break
            except Exception as e:
                continue
        
        # If no media found on this page, we're done
        if media_found_this_page == 0:
            break
        
        page += 1
        
        # Sicherheits-Stop (falls API endlos läuft) - ERWEITERT für ALLE Bilder
        if page > 5000:  # Max 500.000 Medien (100 * 5000) - für ALLE Bilder
            print("    ⚠️  Safety stop at page 5000 reached (500,000+ media checked)")
            break
    
    print(f"    🎉 WordPress API completion: {total_media_found} media objects searched across {page-1} pages!")

def search_sitemaps():
    """Searches XML sitemaps for image URLs"""
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
                print(f"    ✅ Sitemap found: {sitemap_url}")
                
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
    """Intelligently searches historical WordPress upload structures"""
    global found_images
    
    print("[*] 📚 Searching HISTORICAL upload directories...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    # Intelligent year range determination
    import datetime
    current_year = datetime.datetime.now().year
    
    # Start with most likely years (estimate website age)
    # Check modern years first (last 15 years), then extend on success
    initial_years = list(range(max(1995, current_year - 15), current_year + 3))
    extended_years = list(range(1985, current_year + 11))  # Full range if needed
    
    months_to_check = list(range(1, 13))  # Alle Monate
    
    total_dirs_checked = 0
    found_dirs = 0
    
    # First round: Probable years
    print(f"    📅 First check: {len(initial_years)} probable years...")
    years_to_check = initial_years
    extend_search = False
    
    # Main loop for checking years
    def check_years(year_list, phase_name):
        nonlocal total_dirs_checked, found_dirs, extend_search
        
        for year in year_list:
            year_found = 0
            for month in months_to_check:
                month_str = f"{month:02d}"
                total_dirs_checked += 1
                
                # Various WordPress upload structures
                upload_patterns = [
                    f"{base_clean}/wp-content/uploads/{year}/{month_str}/",
                    f"{base_clean}/wp-uploads/{year}/{month_str}/",
                    f"{base_clean}/uploads/{year}/{month_str}/",
                    f"{base_clean}/media/{year}/{month_str}/",
                    f"{base_clean}/files/{year}/{month_str}/",
                ]
                
                # Add dynamic S3/CDN patterns based on domain
                domain_parts = urlparse(BASE_URL).netloc.replace('www.', '').split('.')
                if len(domain_parts) >= 2:
                    domain_name = domain_parts[0]  # e.g., 'example' from 'example.com'
                    domain_full = '.'.join(domain_parts)  # e.g., 'example.com'
                    
                    # Common S3 patterns for this domain
                    s3_patterns = [
                        f"https://{domain_name}-media.s3.amazonaws.com/{domain_full}/wp-uploads/{year}/{month_str}/",
                        f"https://{domain_name}-uploads.s3.amazonaws.com/{year}/{month_str}/",
                        f"https://media-{domain_name}.s3.amazonaws.com/{year}/{month_str}/",
                        # Regional S3 patterns
                        f"https://{domain_name}-media.s3.eu-west-1.amazonaws.com/{domain_full}/wp-uploads/{year}/{month_str}/",
                        f"https://{domain_name}-media.s3.us-east-1.amazonaws.com/{domain_full}/wp-uploads/{year}/{month_str}/",
                    ]
                    upload_patterns.extend(s3_patterns)
                
                for upload_url in upload_patterns:
                    try:
                        response = requests.get(upload_url, timeout=3, allow_redirects=False)
                        if response.status_code == 200:
                            found_dirs += 1
                            year_found += 1
                            extend_search = True  # Success found, extend search
                            print(f"        ✅ {year}/{month_str}: {upload_url}")
                            
                            # Parse Directory Listing
                            soup = BeautifulSoup(response.text, 'html.parser')
                            images_in_dir = 0
                            for link in soup.find_all('a', href=True):
                                href = link.get('href')
                                if href and is_allowed_image(href):
                                    full_url = urljoin(upload_url, href)
                                    if is_related_image_url(full_url) and full_url not in found_images:
                                        found_images.add(full_url)
                                        images_in_dir += 1
                                        if save_image_url_live(full_url):
                                            print(f"            📸 {full_url}")
                            
                            if images_in_dir > 0:
                                print(f"            → {images_in_dir} Bilder in {year}/{month_str}")
                            break  # First found pattern is enough
                            
                    except Exception:
                        continue
            
            if year_found > 0:
                print(f"    📈 Year {year}: {year_found} directories with content found")
            
            # Show progress
            if len(year_list) > 10 and (year - year_list[0] + 1) % 5 == 0:
                progress_pct = ((year - year_list[0] + 1) / len(year_list)) * 100
                print(f"    🔄 {phase_name}: {progress_pct:.0f}% completed...")
    
    # First phase: Probable years
    check_years(initial_years, "First Phase")
    
    # Second phase: Extended search if first phase was successful
    remaining_years = []
    if extend_search and len(initial_years) < len(extended_years):
        remaining_years = [y for y in extended_years if y not in initial_years]
        if remaining_years:
            print(f"    🎯 Success found! Extending to {len(remaining_years)} more years...")
            check_years(remaining_years, "Extended Search")
    
    all_years = initial_years + remaining_years
    print(f"    🎯 Historical search completed:")
    print(f"        - {total_dirs_checked} directories checked")
    print(f"        - {found_dirs} accessible directories found")
    print(f"        - Zeitraum: {min(all_years) if all_years else 'N/A'}-{max(all_years) if all_years else 'N/A'}")
    print(f"        - Intelligente Suche: {'Erweitert' if extend_search else 'Basis'}")

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
                        print(f"    ✅ Endpoint: {len(data)} objects found")
                        
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
    """Recursively extracts all URLs from JSON data"""
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
    
    return urls  # Silent errors for brute force

def load_existing_urls():
    """Loads already saved URLs from file."""
    global saved_urls
    
    if Path(OUTPUT_FILE).exists():
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_urls = {line.strip() for line in f if line.strip()}
                saved_urls.update(existing_urls)
                print(f"📄 {len(existing_urls)} already saved URLs loaded")
        except Exception as e:
            print(f"⚠️ Error loading existing URLs: {e}")
            saved_urls = set()
    else:
        saved_urls = set()

def save_image_url_live(url):
    """Saves found URL immediately to file (only if not already present)."""
    global saved_urls
    
    if url in saved_urls:
        return False  # Already saved
    
    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{url}\n")
        saved_urls.add(url)  # Track that URL was saved
        return True  # Successfully saved
    except Exception as e:
        print(f"    ⚠️ Error saving {url}: {e}")
        return False

def initialize_output_file():
    """Initializes output file and loads existing URLs."""
    global saved_urls
    
    # Load existing URLs if file exists
    if Path(OUTPUT_FILE).exists():
        load_existing_urls()
        print(f"📄 Extending existing file: {OUTPUT_FILE}")
    else:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("")  # Create empty file
            saved_urls = set()
            print(f"📄 New output file created: {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ Error creating output file: {e}")

def save_results():
    """Shows final statistics (file is already updated live)."""
    new_urls = len(found_images - saved_urls) if hasattr(save_results, '_initial_count') else len(found_images)
    print(f"\n✅ {len(saved_urls)} URLs total saved in '{OUTPUT_FILE}'")
    if new_urls > 0:
        print(f"🆕 {new_urls} new URLs found in this session")

def get_filename_from_url(url, website_name):
    """Extracts filename from URL with website prefix"""
    path = urlparse(url).path
    filename = os.path.basename(path)
    
    if not filename or '.' not in filename:
        # Fallback: Use URL hash
        hash_obj = hashlib.md5(url.encode())
        filename = f"image_{hash_obj.hexdigest()[:8]}.jpg"
    
    # Add prefix with website name
    name, ext = os.path.splitext(filename)
    return f"{website_name}_{name}{ext}"

async def download_image(session, url, download_dir, website_name):
    """Downloads a single image and saves metadata"""
    global image_metadata
    
    try:
        filename = get_filename_from_url(url, website_name)
        filepath = Path(download_dir) / website_name / filename
        
        # Create website-specific directory
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists
        if filepath.exists():
            # Save metadata anyway if not yet available
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
                    print(f"[!] Skipping {url} (too large: {content_length} bytes)")
                    return None
                
                content = await response.read()
                
                with open(filepath, 'wb') as f:
                    f.write(content)
                
                # Save metadata with utilities
                add_image_metadata(
                    local_path=str(filepath),
                    source_url=url,
                    website=website_name,
                    download_date=time.strftime('%Y-%m-%d %H:%M:%S'),
                    file_size=len(content)
                )
                
                # Also in global variable for compatibility
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
        print(f"[!] Download error {url}: {e}")
        return None

async def download_and_filter_images():
    """Downloads all images and saves metadata"""
    global image_metadata
    
    # Create directories
    Path(DOWNLOAD_DIR).mkdir(exist_ok=True)
    
    # Lade URLs aus Datei
    if not Path(OUTPUT_FILE).exists():
        print(f"[!] {OUTPUT_FILE} not found!")
        return
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        print("[!] No URLs found in file!")
        return
    
    # Website-Name aus BASE_URL extrahieren
    website_name = urlparse(BASE_URL).netloc.replace('.', '_').replace('-', '_')
    
    # Lade existierende Metadaten falls vorhanden
    metadata_path = Path(DOWNLOAD_DIR) / METADATA_FILE
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                image_metadata = json.load(f)
            print(f"[*] {len(image_metadata)} existing metadata loaded")
        except Exception as e:
            print(f"[!] Fehler beim Laden der Metadaten: {e}")
            image_metadata = {}
    
    print(f"[*] Starting download of {len(urls)} images to {DOWNLOAD_DIR}/{website_name}/...")
    
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
    
    # Filter for successful downloads
    successful_downloads = [f for f in downloaded_files if isinstance(f, Path) and f and f.exists()]
    print(f"[*] {len(successful_downloads)} images successfully downloaded")
    
    if successful_downloads:
        print("\n" + "="*50)
        print("DOWNLOAD COMPLETED")
        print("="*50)
        print(f"Downloaded images: {len(successful_downloads)}")
        print(f"Verzeichnis: {DOWNLOAD_DIR}/{website_name}/")
        print(f"Metadaten: {metadata_path}")
        print("="*50)
    else:
        print("[!] No images downloaded.")
    
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
                print(f"✅ {stats['files_to_remove']} duplicate images removed")
                print(f"💾 {mb_freed:.2f} MB storage space freed")
                print(f"📁 {stats['files_to_keep']} best quality versions kept")
                
                # Updated statistics
                remaining_files = len(successful_downloads) - stats['files_to_remove']
                print(f"\n📊 FINAL STATISTICS:")
                print(f"   Originally downloaded: {len(successful_downloads)} files")
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
    print(
        "The legacy image-download crawler is disabled because it does not "
        "enforce the current domain, robots.txt, and network-safety rules. "
        "Use the Streamlit Web import page for a bounded, consent-confirmed import."
    )
    return

    start = time.time()
    
    # Check if URLs already exist
    if Path(OUTPUT_FILE).exists():
        print(f"[*] {OUTPUT_FILE} found.")
        choice = input("Would you like to download the URLs? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '']:
            print("[*] Starting download...")
            await download_and_filter_images()
            print(f"Finished in {time.time()-start:.1f}s")
            return
    
    # Initialize output file for live updates
    initialize_output_file()
    
    print("🚀 INTELLIGENT IMAGE URL SEARCH STARTED!")
    print("=" * 60)
    
    # 0. Website analysis for optimized search
    print("\n[0/6] 🕵️ Website analysis and pattern recognition...")
    website_patterns = analyze_website_patterns()
    print(f"      ✅ CMS detected: {website_patterns['cms'].upper()}")
    
    # 1. Playwright crawl (standard)
    print("\n[1/6] 🎭 Playwright crawl (JavaScript + Lazy Loading)...")
    await crawl_playwright()
    print(f"      ✅ {len(found_images)} URLs after Playwright crawl")
    
    # 2. WordPress API search (if WordPress detected)
    if website_patterns['cms'] == 'wordpress' and website_patterns.get('api_available'):
        print("\n[2/6] 📡 WordPress REST API search (ALL media)...")
        search_wordpress_api()
        search_additional_wordpress_endpoints()
        print(f"      ✅ {len(found_images)} URLs after API search")
    else:
        print("\n[2/6] 📡 API search skipped (not available)")
    
    # 3. Sitemap search
    print("\n[3/6] 🗺️  XML sitemap search...")
    search_sitemaps()
    print(f"      ✅ {len(found_images)} URLs after sitemap search")
    
    # 4. Intelligent brute-force
    print("\n[4/6] 💥 INTELLIGENT brute-force (Adaptive search)...")
    brute_force_directories()
    print(f"      ✅ {len(found_images)} URLs after intelligent brute-force")
    
    # 5. Historical search (if year structure detected)
    if website_patterns.get('year_structure'):
        print("\n[5/6] 📚 HISTORICAL upload search (Dynamic timespan)...")
        search_historical_uploads()
        print(f"      ✅ {len(found_images)} URLs after historical search")
    else:
        print("\n[5/6] 📚 Historical search skipped (no year structure)")
    
    # 6. Alternative domains
    print("\n[6/6] 🔍 Alternative domains and subdomains...")
    search_alternative_domains()
    print(f"      ✅ {len(found_images)} URLs after domain search")
    
    # 7. Final statistics
    print("\n[7/6] 📊 Completion and statistics...")
    save_results()
    
    print("\n" + "=" * 60)
    print(f"🎉 INTELLIGENT SEARCH COMPLETED!")
    print(f"⏱️  Total time: {time.time()-start:.1f}s")
    print(f"🖼️  Total found: {len(found_images)} image URLs")
    print(f"🧠 CMS: {website_patterns['cms'].upper()}")
    
    # Dynamic timespan display based on actually found patterns
    import datetime
    current_year = datetime.datetime.now().year
    print(f"📅 Timespan coverage: 1990-{current_year + 10} (dynamically adapted)")
    print("=" * 60)
    
    # Automatically start download if URLs found
    if found_images:
        choice = input("\n💾 Would you like to download ALL images now? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '']:
            await download_and_filter_images()
    
    print(f"\n✅ Finished in {time.time()-start:.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
