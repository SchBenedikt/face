"""
Advanced Image scraper using Playwright for comprehensive image discovery and download
Includes recursive page crawling, sitemap parsing, and directory brute-forcing
"""
import asyncio
import aiohttp
import aiofiles
from playwright.async_api import async_playwright, Page, Browser
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
from urllib.parse import urljoin, urlparse, urlunsplit
import logging
from tqdm import tqdm
import time
import hashlib
import re
import xml.etree.ElementTree as ET

from config import SCRAPED_DIR, MAX_IMAGES_PER_SITE, SUPPORTED_IMAGE_FORMATS
from utils import is_valid_image_file, calculate_image_hash
from duplicate_detector import duplicate_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedImageScraper:
    """
    Advanced web scraper for comprehensive image discovery using Playwright
    Features:
    - Recursive page crawling through all site links
    - Sitemap.xml parsing for complete page discovery  
    - Common image directory brute-forcing (/wp-content/uploads/, /images/, etc.)
    - JavaScript-rendered content handling
    - WordPress-specific optimizations
    """
    
    def __init__(self):
        self.downloaded_hashes: Set[str] = set()
        self.visited_pages: Set[str] = set()
        self.discovered_urls: Set[str] = set()
        self.session_stats = {
            "pages_visited": 0,
            "pages_discovered": 0,
            "images_found": 0,
            "images_downloaded": 0,
            "duplicates_skipped": 0,
            "directories_scanned": 0,
            "errors": 0
        }
        
        # Common image directories to check
        self.common_image_paths = [
            '/wp-content/uploads/',
            '/wp-content/uploads/2024/',
            '/wp-content/uploads/2023/',
            '/wp-content/uploads/2022/',
            '/images/',
            '/img/', 
            '/assets/',
            '/assets/images/',
            '/media/',
            '/uploads/',
            '/files/',
            '/gallery/',
            '/photos/',
            '/pictures/',
            '/content/images/',
            '/static/images/',
            '/themes/images/',
        ]
    
    def is_direct_image_url(self, url: str) -> bool:
        """
        Check if URL is a direct link to an image file
        
        Args:
            url: URL to check
            
        Returns:
            True if URL points directly to an image file
        """
        try:
            parsed = urlparse(url.lower())
            path = parsed.path
            
            # Check for common image file extensions
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg', '.ico']
            return any(path.endswith(ext) for ext in image_extensions)
        except:
            return False
    
    async def setup_browser(self) -> tuple[Browser, Page]:
        """
        Setup Playwright browser and page with advanced settings
        
        Returns:
            Browser and page instances
        """
        try:
            playwright = await async_playwright().start()
            
            # Launch browser with optimized settings for comprehensive crawling
            browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-extensions',
                    '--disable-web-security',  # Allow cross-origin requests
                    '--allow-running-insecure-content',
                    '--ignore-certificate-errors',
                ]
            )
            
            # Create context with realistic user agent
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                java_script_enabled=True,
                accept_downloads=False,
                ignore_https_errors=True
            )
            
            # Create page from context
            page = await context.new_page()
            
            # Set longer timeout for complex sites
            page.set_default_timeout(60000)
            
            return browser, page
            
        except Exception as e:
            logger.error(f"Error setting up browser: {e}")
            raise
    
    async def discover_all_pages(self, base_url: str, page: Page) -> Set[str]:
        """
        Discover all pages on a website through multiple methods
        
        Args:
            base_url: Base website URL
            page: Playwright page instance
            
        Returns:
            Set of discovered page URLs
        """
        all_pages = set()
        
        try:
            # Method 1: Parse sitemap.xml
            sitemap_pages = await self._discover_from_sitemap(base_url)
            all_pages.update(sitemap_pages)
            logger.info(f"Found {len(sitemap_pages)} pages from sitemap.xml")
            
            # Method 2: Crawl starting from homepage
            crawled_pages = await self._crawl_pages_recursively(base_url, page, max_depth=3)
            all_pages.update(crawled_pages)
            logger.info(f"Found {len(crawled_pages)} pages from crawling")
            
            # Method 3: Common page patterns (WordPress, etc.)
            pattern_pages = await self._discover_common_pages(base_url)
            all_pages.update(pattern_pages)
            logger.info(f"Found {len(pattern_pages)} pages from common patterns")
            
            self.session_stats["pages_discovered"] = len(all_pages)
            logger.info(f"Total unique pages discovered: {len(all_pages)}")
            
            return all_pages
            
        except Exception as e:
            logger.error(f"Error discovering pages: {e}")
            return set()

    async def _discover_from_sitemap(self, base_url: str) -> Set[str]:
        """
        Discover pages from sitemap.xml
        
        Args:
            base_url: Base website URL
            
        Returns:
            Set of URLs from sitemap
        """
        sitemap_urls = set()
        
        try:
            # Common sitemap locations
            sitemap_paths = [
                '/sitemap.xml',
                '/sitemap_index.xml', 
                '/wp-sitemap.xml',
                '/sitemaps.xml',
                '/robots.txt'  # Will parse for sitemap references
            ]
            
            async with aiohttp.ClientSession() as session:
                for path in sitemap_paths:
                    try:
                        sitemap_url = urljoin(base_url, path)
                        async with session.get(sitemap_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                            if response.status == 200:
                                content = await response.text()
                                
                                if path == '/robots.txt':
                                    # Parse robots.txt for sitemap references
                                    sitemap_refs = re.findall(r'Sitemap:\s*(.+)', content, re.IGNORECASE)
                                    for ref in sitemap_refs:
                                        ref_urls = await self._parse_sitemap_xml(ref.strip())
                                        sitemap_urls.update(ref_urls)
                                else:
                                    # Parse XML sitemap
                                    urls = await self._parse_sitemap_xml_content(content)
                                    sitemap_urls.update(urls)
                                    
                    except Exception as e:
                        logger.debug(f"Could not access {path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error discovering from sitemap: {e}")
            
        return sitemap_urls

    async def _parse_sitemap_xml_content(self, xml_content: str) -> Set[str]:
        """
        Parse sitemap XML content to extract URLs
        
        Args:
            xml_content: XML content string
            
        Returns:
            Set of URLs
        """
        urls = set()
        
        try:
            root = ET.fromstring(xml_content)
            
            # Handle different namespaces
            namespaces = {
                'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                '': 'http://www.sitemaps.org/schemas/sitemap/0.9'
            }
            
            # Look for URL entries
            for url_elem in root.findall('.//url', namespaces) + root.findall('.//url'):
                loc_elem = url_elem.find('loc') or url_elem.find('.//loc', namespaces)
                if loc_elem is not None and loc_elem.text:
                    urls.add(loc_elem.text.strip())
            
            # Handle sitemap index files
            for sitemap_elem in root.findall('.//sitemap', namespaces) + root.findall('.//sitemap'):
                loc_elem = sitemap_elem.find('loc') or sitemap_elem.find('.//loc', namespaces)
                if loc_elem is not None and loc_elem.text:
                    # Recursively parse referenced sitemaps
                    try:
                        sub_urls = await self._parse_sitemap_xml(loc_elem.text.strip())
                        urls.update(sub_urls)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Error parsing sitemap XML: {e}")
            
        return urls

    async def _parse_sitemap_xml(self, sitemap_url: str) -> Set[str]:
        """
        Parse sitemap XML from URL
        
        Args:
            sitemap_url: URL of sitemap
            
        Returns:
            Set of URLs
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(sitemap_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        content = await response.text()
                        return await self._parse_sitemap_xml_content(content)
        except Exception as e:
            logger.debug(f"Error parsing sitemap {sitemap_url}: {e}")
            
        return set()

    async def _crawl_pages_recursively(self, base_url: str, page: Page, max_depth: int = 3) -> Set[str]:
        """
        Recursively crawl website pages to discover all internal links
        
        Args:
            base_url: Base website URL
            page: Playwright page instance
            max_depth: Maximum crawling depth
            
        Returns:
            Set of discovered URLs
        """
        discovered = set()
        to_visit = [(base_url, 0)]  # (url, depth)
        visited = set()
        
        domain = urlparse(base_url).netloc
        
        while to_visit and len(discovered) < 1000:  # Limit total pages
            current_url, depth = to_visit.pop(0)
            
            if current_url in visited or depth > max_depth:
                continue
                
            visited.add(current_url)
            discovered.add(current_url)
            
            try:
                logger.info(f"Crawling page (depth {depth}): {current_url}")
                await page.goto(current_url, wait_until="domcontentloaded", timeout=30000)
                
                # Find all links on current page
                links = await page.evaluate("""
                    () => {
                        const links = Array.from(document.querySelectorAll('a[href]'));
                        return links.map(link => link.href);
                    }
                """)
                
                # Filter and add internal links
                for link in links:
                    try:
                        parsed = urlparse(link)
                        if parsed.netloc == domain and link not in visited:
                            # Check if this is a direct image URL
                            if self.is_direct_image_url(link):
                                # Don't crawl image URLs, just add them to discovered for direct download
                                discovered.add(link)
                                continue
                            
                            # Skip certain file types and fragments
                            if not any(link.lower().endswith(ext) for ext in ['.pdf', '.doc', '.zip', '.exe']):
                                clean_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))
                                if clean_url not in visited:
                                    to_visit.append((clean_url, depth + 1))
                    except:
                        continue
                        
            except Exception as e:
                logger.error(f"Error crawling {current_url}: {e}")
                continue
                
        return discovered

    async def _discover_common_pages(self, base_url: str) -> Set[str]:
        """
        Check for common page patterns (WordPress, etc.)
        
        Args:
            base_url: Base website URL
            
        Returns:
            Set of discovered URLs
        """
        common_pages = set()
        
        # Common WordPress and CMS patterns
        common_paths = [
            '/blog/',
            '/news/',
            '/gallery/',
            '/photos/',
            '/images/',
            '/about/',
            '/contact/',
            '/events/',
            '/category/',
            '/archives/',
            '/page/',
            '/wp-content/uploads/',
        ]
        
        # Check which paths exist
        async with aiohttp.ClientSession() as session:
            for path in common_paths:
                try:
                    url = urljoin(base_url, path)
                    async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            common_pages.add(url)
                except:
                    continue
                    
        return common_pages

    async def find_images_from_directories(self, base_url: str) -> List[str]:
        """
        Discover images by checking common image directories
        
        Args:
            base_url: Base website URL
            
        Returns:
            List of discovered image URLs
        """
        discovered_images = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for path in self.common_image_paths:
                    try:
                        dir_url = urljoin(base_url, path)
                        logger.info(f"Checking directory: {dir_url}")
                        
                        # Try to access directory listing
                        async with session.get(dir_url, timeout=aiohttp.ClientTimeout(total=20)) as response:
                            if response.status == 200:
                                content = await response.text()
                                
                                # Parse HTML directory listing or try common patterns
                                images = self._extract_images_from_directory_html(content, dir_url)
                                discovered_images.extend(images)
                                
                                # Also try year-based patterns for WordPress uploads
                                if 'wp-content/uploads' in path:
                                    year_images = await self._scan_wordpress_uploads(base_url, session)
                                    discovered_images.extend(year_images)
                                    
                                self.session_stats["directories_scanned"] += 1
                                
                    except Exception as e:
                        logger.debug(f"Could not access directory {path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error discovering images from directories: {e}")
            
        return list(set(discovered_images))  # Remove duplicates
        
        # Apply quality prioritization to directory-discovered images
        return self._prioritize_highest_quality_images(list(set(discovered_images)))

    def _extract_images_from_directory_html(self, html_content: str, base_url: str) -> List[str]:
        """
        Extract image URLs from directory listing HTML
        
        Args:
            html_content: HTML content of directory listing
            base_url: Base URL for resolving relative paths
            
        Returns:
            List of image URLs
        """
        image_urls = []
        
        try:
            # Look for href attributes that point to images
            href_pattern = r'href=["\']([^"\']+\.(?:' + '|'.join(ext.lstrip('.') for ext in SUPPORTED_IMAGE_FORMATS) + r'))["\']'
            matches = re.findall(href_pattern, html_content, re.IGNORECASE)
            
            for match in matches:
                full_url = urljoin(base_url, match)
                if self._is_valid_image_url(full_url):
                    image_urls.append(full_url)
                    
            # Also look for direct image references in text
            img_pattern = r'(?:src|href)=["\']([^"\']*\.(?:' + '|'.join(ext.lstrip('.') for ext in SUPPORTED_IMAGE_FORMATS) + r'))["\']'
            img_matches = re.findall(img_pattern, html_content, re.IGNORECASE)
            
            for match in img_matches:
                full_url = urljoin(base_url, match)
                if self._is_valid_image_url(full_url):
                    image_urls.append(full_url)
                    
        except Exception as e:
            logger.error(f"Error extracting images from directory HTML: {e}")
            
        return image_urls

    async def _scan_wordpress_uploads(self, base_url: str, session: aiohttp.ClientSession) -> List[str]:
        """
        Scan WordPress uploads directory with year/month patterns
        
        Args:
            base_url: Base website URL
            session: aiohttp session
            
        Returns:
            List of discovered image URLs
        """
        images = []
        
        # Common WordPress upload patterns
        years = ['2024', '2023', '2022', '2021', '2020']
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        for year in years:
            for month in months:
                try:
                    upload_path = f'/wp-content/uploads/{year}/{month}/'
                    dir_url = urljoin(base_url, upload_path)
                    
                    async with session.get(dir_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            content = await response.text()
                            month_images = self._extract_images_from_directory_html(content, dir_url)
                            images.extend(month_images)
                            logger.info(f"Found {len(month_images)} images in {upload_path}")
                            
                except Exception as e:
                    logger.debug(f"Could not scan {year}/{month}: {e}")
                    continue
                    
        return images

    def _prioritize_highest_quality_images(self, image_urls: List[str]) -> List[str]:
        """
        Prioritize highest quality images by removing smaller versions of the same image
        
        Args:
            image_urls: List of image URLs
            
        Returns:
            List of highest quality image URLs
        """
        # Group images by base name (without size indicators)
        image_groups = {}
        
        for url in image_urls:
            try:
                # Extract base name by removing size indicators like -300x200, -768x1024, etc.
                parsed = urlparse(url)
                filename = Path(parsed.path).name
                
                # Remove size indicators and scaling suffixes
                base_name = re.sub(r'-(\d+x\d+|\d+w|\d+h|scaled|large|medium|small|thumb|thumbnail)', '', filename)
                base_name = re.sub(r'_(\d+x\d+)', '', base_name)
                
                if base_name not in image_groups:
                    image_groups[base_name] = []
                image_groups[base_name].append(url)
                
            except Exception as e:
                logger.debug(f"Could not process URL {url}: {e}")
                continue
        
        # Select highest quality version from each group
        prioritized_urls = []
        
        for base_name, urls in image_groups.items():
            if len(urls) == 1:
                prioritized_urls.append(urls[0])
            else:
                # Sort by likely quality/size (prefer larger dimensions, scaled versions, originals)
                sorted_urls = sorted(urls, key=lambda x: self._calculate_image_quality_score(x), reverse=True)
                prioritized_urls.append(sorted_urls[0])  # Take highest quality
                
                logger.debug(f"Selected {sorted_urls[0]} from {len(urls)} versions of {base_name}")
        
        logger.info(f"Quality filtering: {len(image_urls)} -> {len(prioritized_urls)} images")
        return prioritized_urls
    
    def _calculate_image_quality_score(self, url: str) -> int:
        """
        Calculate quality score for an image URL to prioritize higher quality versions
        
        Args:
            url: Image URL
            
        Returns:
            Quality score (higher is better)
        """
        score = 0
        url_lower = url.lower()
        
        # Prefer scaled/original versions
        if 'scaled' in url_lower:
            score += 1000
        if 'original' in url_lower:
            score += 900
        if 'large' in url_lower:
            score += 800
        if 'full' in url_lower:
            score += 700
        
        # Penalize small versions
        if any(size in url_lower for size in ['thumb', 'thumbnail', 'small', 'icon']):
            score -= 500
        if any(size in url_lower for size in ['66x66', '150x150', '200x']):
            score -= 300
        
        # Extract dimensions and add to score
        dimension_matches = re.findall(r'(\d+)x(\d+)', url)
        if dimension_matches:
            # Take the largest dimension pair found
            max_area = 0
            for width_str, height_str in dimension_matches:
                try:
                    width, height = int(width_str), int(height_str)
                    area = width * height
                    if area > max_area:
                        max_area = area
                except:
                    continue
            score += max_area // 1000  # Convert to reasonable score range
            
        return score
        
    async def find_highest_quality_images(self, page: Page, base_url: str) -> List[str]:
        """
        Find highest quality image URLs by prioritizing resolution and format
        
        Args:
            page: Playwright page instance
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of high-quality image URLs
        """
        try:
            # Enhanced JavaScript for quality detection
            image_data = await page.evaluate("""
                () => {
                    const images = [];
                    
                    // Helper to estimate quality score
                    const getQualityScore = (url, width, height, source) => {
                        let score = 0;
                        
                        // Prefer larger dimensions
                        score += (width || 0) * (height || 0) / 1000;
                        
                        // Prefer original/full-size indicators in URL
                        if (url.includes('original') || url.includes('full') || url.includes('large')) score += 100;
                        if (url.includes('2x') || url.includes('3x')) score += 50;
                        if (url.includes('1920') || url.includes('1080') || url.includes('2048')) score += 75;
                        
                        // Prefer better formats
                        if (url.includes('.webp')) score += 10;
                        if (url.includes('.png')) score += 5;
                        
                        // Source type priority
                        if (source.includes('picture')) score += 30;
                        if (source.includes('srcset')) score += 20;
                        if (source.includes('original')) score += 25;
                        
                        return score;
                    };
                    
                    // Method 1: Standard img tags with quality analysis
                    document.querySelectorAll('img').forEach(img => {
                        const addImageWithQuality = (url, source) => {
                            if (url && !url.startsWith('data:')) {
                                const width = img.naturalWidth || img.width || 0;
                                const height = img.naturalHeight || img.height || 0;
                                const quality = getQualityScore(url, width, height, source);
                                
                                images.push({
                                    url: url,
                                    width: width,
                                    height: height,
                                    quality: quality,
                                    source: source
                                });
                            }
                        };
                        
                        // Primary sources
                        if (img.src) addImageWithQuality(img.src, 'img-src');
                        if (img.dataset.src) addImageWithQuality(img.dataset.src, 'lazy-src');
                        if (img.dataset.original) addImageWithQuality(img.dataset.original, 'original');
                        if (img.dataset.fullSrc) addImageWithQuality(img.dataset.fullSrc, 'full-src');
                        
                        // Srcset - extract highest quality versions
                        if (img.srcset) {
                            img.srcset.split(',').forEach(entry => {
                                const parts = entry.trim().split(/\\s+/);
                                const url = parts[0];
                                const descriptor = parts[1] || '1x';
                                
                                let quality = getQualityScore(url, 0, 0, `srcset-${descriptor}`);
                                
                                // Boost quality based on descriptor
                                if (descriptor.endsWith('x')) {
                                    quality += parseFloat(descriptor.slice(0, -1)) * 25;
                                } else if (descriptor.endsWith('w')) {
                                    const width = parseInt(descriptor.slice(0, -1));
                                    quality += width / 10; // Higher resolution = higher quality
                                }
                                
                                images.push({
                                    url: url,
                                    width: descriptor.endsWith('w') ? parseInt(descriptor.slice(0, -1)) : 0,
                                    height: 0,
                                    quality: quality,
                                    source: `srcset-${descriptor}`
                                });
                            });
                        }
                    });
                    
                    // Method 2: Picture elements (usually high quality)
                    document.querySelectorAll('picture source').forEach(source => {
                        if (source.srcset) {
                            source.srcset.split(',').forEach(entry => {
                                const parts = entry.trim().split(/\\s+/);
                                const url = parts[0];
                                const descriptor = parts[1] || '1x';
                                
                                let quality = getQualityScore(url, 0, 0, `picture-${descriptor}`) + 30; // Picture elements are generally higher quality
                                
                                if (descriptor.endsWith('w')) {
                                    quality += parseInt(descriptor.slice(0, -1)) / 10;
                                }
                                
                                images.push({
                                    url: url,
                                    width: descriptor.endsWith('w') ? parseInt(descriptor.slice(0, -1)) : 0,
                                    height: 0,
                                    quality: quality,
                                    source: `picture-${descriptor}`
                                });
                            });
                        }
                    });
                    
                    // Method 3: Look for WordPress attachment URLs (usually full size)
                    document.querySelectorAll('a[href*="wp-content/uploads"]').forEach(link => {
                        if (link.href && (link.href.endsWith('.jpg') || link.href.endsWith('.jpeg') || link.href.endsWith('.png') || link.href.endsWith('.webp'))) {
                            images.push({
                                url: link.href,
                                width: 0,
                                height: 0,
                                quality: getQualityScore(link.href, 0, 0, 'wp-attachment') + 40, // WordPress attachments often full-size
                                source: 'wp-attachment'
                            });
                        }
                    });
                    
                    return images;
                }
            """)
            
            # Sort by quality and remove duplicates
            seen_urls = set()
            quality_urls = []
            
            # Sort by quality score (highest first)
            image_data.sort(key=lambda x: x['quality'], reverse=True)
            
            for img in image_data:
                url = img['url']
                if url in seen_urls:
                    continue
                    
                try:
                    full_url = urljoin(base_url, url)
                    if self._is_valid_image_url(full_url):
                        quality_urls.append(full_url)
                        seen_urls.add(url)
                except:
                    continue
            
            logger.info(f"Found {len(quality_urls)} quality-optimized images on page")
            return quality_urls
            
        except Exception as e:
            logger.error(f"Error finding high-quality images: {e}")
            return []

    async def find_image_urls(self, page: Page, base_url: str) -> List[str]:
        """
        Find all image URLs on a page using enhanced detection
        
        Args:
            page: Playwright page instance
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of image URLs
        """
        try:
            # Wait for page to load completely
            await page.wait_for_load_state("networkidle", timeout=30000)
            
            # Enhanced image detection with JavaScript
            image_urls = await page.evaluate("""
                () => {
                    const images = [];
                    
                    // Method 1: Standard img tags
                    document.querySelectorAll('img').forEach(img => {
                        if (img.src) images.push(img.src);
                        if (img.dataset.src) images.push(img.dataset.src); // Lazy loading
                        if (img.dataset.original) images.push(img.dataset.original); // Some lazy loading libraries
                        
                        // Parse srcset
                        if (img.srcset) {
                            const srcsetUrls = img.srcset.split(',').map(entry => {
                                return entry.trim().split(' ')[0];
                            });
                            images.push(...srcsetUrls);
                        }
                    });
                    
                    // Method 2: CSS background images
                    document.querySelectorAll('*').forEach(el => {
                        const style = window.getComputedStyle(el);
                        const bgImage = style.backgroundImage;
                        if (bgImage && bgImage !== 'none') {
                            const matches = bgImage.match(/url\\s*\\(['"]?([^'")]+)['"]?\\)/g);
                            if (matches) {
                                matches.forEach(match => {
                                    const url = match.match(/url\\s*\\(['"]?([^'")]+)['"]?\\)/)[1];
                                    images.push(url);
                                });
                            }
                        }
                    });
                    
                    // Method 3: Picture sources
                    document.querySelectorAll('picture source').forEach(source => {
                        if (source.srcset) {
                            const srcsetUrls = source.srcset.split(',').map(entry => {
                                return entry.trim().split(' ')[0];
                            });
                            images.push(...srcsetUrls);
                        }
                    });
                    
                    // Method 4: Data attributes and custom attributes
                    document.querySelectorAll('[data-bg], [data-background], [data-image]').forEach(el => {
                        if (el.dataset.bg) images.push(el.dataset.bg);
                        if (el.dataset.background) images.push(el.dataset.background);
                        if (el.dataset.image) images.push(el.dataset.image);
                    });
                    
                    return [...new Set(images)]; // Remove duplicates
                }
            """)
            
            # Resolve relative URLs and filter valid images
            resolved_urls = []
            for url in image_urls:
                try:
                    if url.startswith('data:'):
                        continue  # Skip data URLs
                    
                    full_url = urljoin(base_url, url)
                    if self._is_valid_image_url(full_url):
                        resolved_urls.append(full_url)
                except:
                    continue
            
            # Prioritize highest quality versions
            high_quality_urls = self._prioritize_highest_quality_images(resolved_urls)
            
            logger.info(f"Found {len(high_quality_urls)} high-quality image URLs on page (from {len(resolved_urls)} total)")
            return high_quality_urls
            
        except Exception as e:
            logger.error(f"Error finding image URLs: {e}")
            return []
        """
        Find all image URLs on a page
        
        Args:
            page: Playwright page instance
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of image URLs
        """
        try:
            # Wait for page to load
            await page.wait_for_load_state("networkidle", timeout=30000)
            
            # Find all img tags
            img_elements = await page.query_selector_all('img')
            
            image_urls = []
            for img in img_elements:
                src = await img.get_attribute('src')
                data_src = await img.get_attribute('data-src')  # Lazy loading
                srcset = await img.get_attribute('srcset')
                
                # Try different attributes
                url = src or data_src
                if url:
                    # Resolve relative URLs
                    full_url = urljoin(base_url, url)
                    
                    # Check if it's a valid image URL
                    if self._is_valid_image_url(full_url):
                        image_urls.append(full_url)
                
                # Handle srcset for high-resolution images
                if srcset:
                    srcset_urls = self._parse_srcset(srcset, base_url)
                    image_urls.extend(srcset_urls)
            
            # Find images in CSS backgrounds
            css_images = await self._find_css_background_images(page, base_url)
            image_urls.extend(css_images)
            
            # Remove duplicates
            unique_urls = list(set(image_urls))
            
            logger.info(f"Found {len(unique_urls)} unique image URLs on page")
            return unique_urls
            
        except Exception as e:
            logger.error(f"Error finding image URLs: {e}")
            return []
    
    async def _find_css_background_images(self, page: Page, base_url: str) -> List[str]:
        """
        Find images in CSS background-image properties
        
        Args:
            page: Playwright page instance
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of background image URLs
        """
        try:
            # Execute JavaScript to find background images
            bg_images = await page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('*');
                    const bgImages = [];
                    
                    elements.forEach(el => {
                        const style = window.getComputedStyle(el);
                        const bgImage = style.backgroundImage;
                        
                        if (bgImage && bgImage !== 'none') {
                            const matches = bgImage.match(/url\\s*\\(['"]?([^'")]+)['"]?\\)/g);
                            if (matches) {
                                matches.forEach(match => {
                                    const url = match.match(/url\\s*\\(['"]?([^'")]+)['"]?\\)/)[1];
                                    bgImages.push(url);
                                });
                            }
                        }
                    });
                    
                    return bgImages;
                }
            """)
            
            # Resolve relative URLs
            resolved_urls = []
            for url in bg_images:
                full_url = urljoin(base_url, url)
                if self._is_valid_image_url(full_url):
                    resolved_urls.append(full_url)
            
            return resolved_urls
            
        except Exception as e:
            logger.error(f"Error finding CSS background images: {e}")
            return []
    
    def _parse_srcset(self, srcset: str, base_url: str) -> List[str]:
        """
        Parse srcset attribute to extract image URLs
        
        Args:
            srcset: srcset attribute value
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of image URLs from srcset
        """
        urls = []
        try:
            # Parse srcset format: "url1 1x, url2 2x, url3 1024w"
            entries = srcset.split(',')
            for entry in entries:
                parts = entry.strip().split()
                if parts:
                    url = parts[0]
                    full_url = urljoin(base_url, url)
                    if self._is_valid_image_url(full_url):
                        urls.append(full_url)
        except Exception as e:
            logger.error(f"Error parsing srcset: {e}")
        
        return urls
    
    def _is_valid_image_url(self, url: str) -> bool:
        """
        Check if URL points to a valid image
        
        Args:
            url: URL to check
            
        Returns:
            True if valid image URL, False otherwise
        """
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False
            
            path_lower = parsed.path.lower()
            return any(path_lower.endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS)
            
        except Exception:
            return False
    
    async def download_image(self, session: aiohttp.ClientSession, url: str, output_dir: Path) -> bool:
        """
        Download a single image with exact original quality preservation
        
        Args:
            session: aiohttp session
            url: Image URL
            output_dir: Output directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create filename from URL
            parsed = urlparse(url)
            filename = Path(parsed.path).name
            if not filename or '.' not in filename:
                # Extract extension from content-type if possible
                filename = f"image_{hashlib.md5(url.encode()).hexdigest()[:8]}.jpg"
            
            output_path = output_dir / filename
            
            # Skip if file already exists (basic check)
            if output_path.exists():
                return False
            
            # Download image with proper headers to ensure original quality
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',  # Allow compression but preserve image quality
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive',
                'Referer': url  # Some sites require referer
            }
            
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    # Get the raw content WITHOUT any modifications
                    content = await response.read()
                    
                    # Validate minimum content size (avoid tiny images or errors)
                    if len(content) < 1024:  # Less than 1KB is probably not a real image
                        logger.debug(f"Image too small ({len(content)} bytes): {url}")
                        return False
                    
                    # Session-based duplicate detection (for current session)
                    content_hash = hashlib.md5(content).hexdigest()
                    if content_hash in self.downloaded_hashes:
                        self.session_stats["duplicates_skipped"] += 1
                        return False
                    
                    # Write content DIRECTLY to file without any processing
                    # This preserves the EXACT original quality and format
                    try:
                        async with aiofiles.open(output_path, 'wb') as f:
                            await f.write(content)
                        
                        # Verify the file was written correctly
                        if not output_path.exists() or output_path.stat().st_size != len(content):
                            logger.error(f"File write verification failed for {url}")
                            return False
                        
                        # Quick validation that it's a valid image file (without loading/processing)
                        if not self._quick_image_validation(output_path):
                            logger.debug(f"Image validation failed: {url}")
                            output_path.unlink()  # Remove invalid file
                            return False
                        
                        # Check for duplicates using file-based detection AFTER saving
                        if duplicate_detector.is_duplicate(output_path):
                            output_path.unlink()  # Remove duplicate
                            self.session_stats["duplicates_skipped"] += 1
                            return False
                        
                        # Add to duplicate detector database
                        duplicate_detector.add_file_hash(output_path)
                        self.downloaded_hashes.add(content_hash)
                        self.session_stats["images_downloaded"] += 1
                        
                        logger.debug(f"Successfully downloaded: {filename} ({len(content)} bytes)")
                        return True
                        
                    except Exception as write_error:
                        logger.error(f"Error writing file {output_path}: {write_error}")
                        if output_path.exists():
                            output_path.unlink()
                        return False
                
                else:
                    logger.debug(f"HTTP {response.status} for {url}")
                    return False
                
        except Exception as e:
            logger.error(f"Error downloading image {url}: {e}")
            self.session_stats["errors"] += 1
            return False
    
    def _quick_image_validation(self, file_path: Path) -> bool:
        """
        Quick validation that a file is likely a valid image without loading it
        
        Args:
            file_path: Path to image file
            
        Returns:
            True if likely valid image, False otherwise
        """
        try:
            # Check file size
            if file_path.stat().st_size < 100:  # Too small
                return False
            
            # Check magic bytes for common image formats
            with open(file_path, 'rb') as f:
                header = f.read(12)
            
            # JPEG magic bytes
            if header[:3] == b'\xff\xd8\xff':
                return True
            
            # PNG magic bytes
            if header[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
                return True
            
            # GIF magic bytes
            if header[:6] in (b'GIF87a', b'GIF89a'):
                return True
            
            # WebP magic bytes
            if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
                return True
            
            # BMP magic bytes
            if header[:2] == b'BM':
                return True
            
            # TIFF magic bytes
            if header[:4] in (b'II*\x00', b'MM\x00*'):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating image {file_path}: {e}")
            return False
    
    async def scrape_website(self, url: str, max_images: int = MAX_IMAGES_PER_SITE) -> Dict[str, Any]:
        """
        Comprehensive website scraping with advanced image discovery
        
        Args:
            url: Website URL to scrape
            max_images: Maximum number of images (0 = unlimited)
            
        Returns:
            Scraping results
        """
        try:
            # Create output directory
            domain = urlparse(url).netloc.replace('.', '_')
            output_dir = SCRAPED_DIR / domain
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Starting comprehensive scraping of {url}")
            
            # Setup browser
            browser, page = await self.setup_browser()
            
            try:
                # Phase 1: Discover all pages on the website
                logger.info("Phase 1: Discovering all pages...")
                all_pages = await self.discover_all_pages(url, page)
                logger.info(f"Discovered {len(all_pages)} total pages")
                
                # Phase 2: Find images from common directories (brute force)
                logger.info("Phase 2: Scanning common image directories...")
                directory_images = await self.find_images_from_directories(url)
                logger.info(f"Found {len(directory_images)} images from directories")
                
                # Phase 3: Crawl all discovered pages for images
                logger.info("Phase 3: Extracting images from all pages...")
                all_image_urls = set(directory_images)
                
                # Limit pages to crawl to prevent excessive scraping
                pages_to_crawl = list(all_pages)[:50]  # Limit to 50 pages
                
                for i, page_url in enumerate(pages_to_crawl):
                    try:
                        if page_url in self.visited_pages:
                            continue
                            
                        # Check if this is a direct image URL
                        if self.is_direct_image_url(page_url):
                            logger.info(f"Found direct image URL {i+1}/{len(pages_to_crawl)}: {page_url}")
                            all_image_urls.add(page_url)
                            continue
                            
                        logger.info(f"Crawling page {i+1}/{len(pages_to_crawl)}: {page_url}")
                        await page.goto(page_url, wait_until="domcontentloaded", timeout=30000)
                        
                        # Find images on this page with quality prioritization
                        page_images = await self.find_highest_quality_images(page, page_url)
                        if not page_images:
                            # Fallback to standard method if quality detection fails
                            page_images = await self.find_image_urls(page, page_url)
                        all_image_urls.update(page_images)
                        
                        self.visited_pages.add(page_url)
                        self.session_stats["pages_visited"] += 1
                        
                        # Small delay between pages
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error crawling page {page_url}: {e}")
                        self.session_stats["errors"] += 1
                        continue
                
                # Convert to list and update stats
                image_urls = list(all_image_urls)
                self.session_stats["images_found"] = len(image_urls)
                
                logger.info(f"Total unique images discovered: {len(image_urls)}")
                
                # Phase 4: Apply image limit if specified
                if max_images > 0 and len(image_urls) > max_images:
                    image_urls = image_urls[:max_images]
                    logger.info(f"Limiting to {max_images} images from {len(image_urls)} found")
                else:
                    logger.info(f"Downloading all {len(image_urls)} images found")
                
                # Phase 5: Download all discovered images with duplicate detection
                logger.info("Phase 5: Downloading images...")
                connector = aiohttp.TCPConnector(limit=10)
                timeout = aiohttp.ClientTimeout(total=300)
                
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    # Create download tasks
                    tasks = []
                    for img_url in image_urls:
                        task = self.download_image(session, img_url, output_dir)
                        tasks.append(task)
                    
                    # Execute downloads with progress tracking
                    successful_downloads = 0
                    for i, task in enumerate(asyncio.as_completed(tasks)):
                        try:
                            success = await task
                            if success:
                                successful_downloads += 1
                            
                            # Update progress every 25 downloads
                            if (i + 1) % 25 == 0:
                                logger.info(f"Downloaded {successful_downloads}/{i+1} images")
                                
                        except Exception as e:
                            logger.error(f"Download task failed: {e}")
                            self.session_stats["errors"] += 1
                
                logger.info(f"Scraping complete! Downloaded {successful_downloads} images")
                
                return {
                    "success": True,
                    "url": url,
                    "output_directory": str(output_dir),
                    "pages_discovered": len(all_pages),
                    "pages_crawled": len(self.visited_pages),
                    "images_found": len(image_urls),
                    "images_downloaded": successful_downloads,
                    "stats": self.session_stats.copy()
                }
                
            finally:
                await browser.close()
                
        except Exception as e:
            logger.error(f"Error scraping website {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "stats": self.session_stats.copy()
            }
    
    async def scrape_multiple_websites(self, urls: List[str], max_images_per_site: int = MAX_IMAGES_PER_SITE) -> List[Dict[str, Any]]:
        """
        Scrape images from multiple websites
        
        Args:
            urls: List of website URLs
            max_images_per_site: Maximum images per website
            
        Returns:
            List of scraping results
        """
        results = []
        
        for url in urls:
            logger.info(f"Scraping {url}...")
            
            # Reset stats for each site
            self.session_stats = {
                "pages_visited": 0,
                "images_found": 0,
                "images_downloaded": 0,
                "duplicates_skipped": 0,
                "errors": 0
            }
            
            result = await self.scrape_website(url, max_images_per_site)
            results.append(result)
            
            # Add delay between sites
            await asyncio.sleep(2)
        
        return results
    
    def get_scraped_images(self) -> List[Path]:
        """
        Get list of all scraped images
        
        Returns:
            List of scraped image paths
        """
        scraped_images = []
        
        if SCRAPED_DIR.exists():
            for image_file in SCRAPED_DIR.rglob('*'):
                if image_file.is_file() and is_valid_image_file(image_file):
                    scraped_images.append(image_file)
        
        return sorted(scraped_images)


    async def scrape_website_enhanced(self, url: str, max_images: int = 0, crawl_depth: int = 3, 
                                     quality_priority: str = "Highest", timeout: int = 30) -> Dict[str, Any]:
        """
        Enhanced website scraping with advanced options
        
        Args:
            url: Website URL to scrape
            max_images: Maximum number of images (0 = unlimited)
            crawl_depth: Maximum crawling depth for subpages
            quality_priority: Image quality priority ("Highest", "Balanced", "Speed")
            timeout: Page timeout in seconds
            
        Returns:
            Enhanced scraping results
        """
        try:
            # Create output directory
            domain = urlparse(url).netloc.replace('.', '_')
            output_dir = SCRAPED_DIR / domain
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Starting enhanced scraping of {url} (depth: {crawl_depth}, quality: {quality_priority})")
            
            # Setup browser with enhanced settings
            browser, page = await self.setup_browser()
            
            # Set timeout
            page.set_default_timeout(timeout * 1000)
            
            try:
                # Phase 1: Enhanced page discovery
                logger.info("Phase 1: Enhanced page discovery...")
                all_pages = await self.discover_all_pages_enhanced(url, page, crawl_depth)
                
                # Phase 2: Quality-focused image discovery
                logger.info("Phase 2: Quality-focused image discovery...")
                all_image_urls = set()
                
                # Discover from directories first (often highest quality)
                directory_images = await self.find_images_from_directories(url)
                all_image_urls.update(directory_images)
                
                # Crawl pages for images with quality priority
                pages_to_crawl = list(all_pages)[:min(50, len(all_pages))]
                
                for i, page_url in enumerate(pages_to_crawl):
                    try:
                        # Check if this is a direct image URL
                        if self.is_direct_image_url(page_url):
                            logger.info(f"Found direct image URL {i+1}/{len(pages_to_crawl)}: {page_url}")
                            all_image_urls.add(page_url)
                            continue
                        
                        logger.info(f"Crawling page {i+1}/{len(pages_to_crawl)}: {page_url}")
                        await page.goto(page_url, wait_until="domcontentloaded")
                        
                        if quality_priority == "Highest":
                            page_images = await self.find_highest_quality_images(page, page_url)
                        else:
                            page_images = await self.find_image_urls(page, page_url)
                        
                        all_image_urls.update(page_images)
                        self.session_stats["pages_visited"] += 1
                        
                        # Small delay
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Error crawling page {page_url}: {e}")
                        continue
                
                # Convert to list and apply quality sorting
                image_urls = await self.sort_images_by_quality(list(all_image_urls), quality_priority)
                
                # Apply limit if specified
                if max_images > 0:
                    image_urls = image_urls[:max_images]
                
                self.session_stats["images_found"] = len(image_urls)
                logger.info(f"Found {len(image_urls)} quality-sorted images")
                
                # Phase 3: Enhanced downloading
                logger.info("Phase 3: Enhanced downloading...")
                successful_downloads = await self.download_images_enhanced(image_urls, output_dir)
                
                return {
                    "success": True,
                    "url": url,
                    "output_directory": str(output_dir),
                    "pages_discovered": len(all_pages),
                    "pages_crawled": len(pages_to_crawl),
                    "images_found": len(image_urls),
                    "images_downloaded": successful_downloads,
                    "quality_priority": quality_priority,
                    "crawl_depth": crawl_depth,
                    "stats": self.session_stats.copy()
                }
                
            finally:
                await browser.close()
                
        except Exception as e:
            logger.error(f"Error in enhanced scraping: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "stats": self.session_stats.copy()
            }

    async def discover_all_pages_enhanced(self, base_url: str, page: Page, max_depth: int) -> Set[str]:
        """Enhanced page discovery with better depth control"""
        discovered = await self.discover_all_pages(base_url, page)
        
        # Additional discovery methods for enhanced mode
        try:
            # Check robots.txt for additional sitemaps
            robots_url = urljoin(base_url, "/robots.txt")
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            sitemap_urls = re.findall(r'Sitemap:\s*(.+)', robots_content, re.IGNORECASE)
                            for sitemap_url in sitemap_urls:
                                sitemap_pages = await self._parse_sitemap_xml(sitemap_url.strip())
                                discovered.update(sitemap_pages)
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Enhanced discovery error: {e}")
        
        return discovered

    async def sort_images_by_quality(self, image_urls: List[str], priority: str) -> List[str]:
        """Sort images by quality based on priority setting"""
        
        if priority == "Speed":
            return image_urls  # No sorting, return as-is for speed
        
        def quality_score(url: str) -> int:
            score = 0
            url_lower = url.lower()
            
            # File size indicators
            if 'original' in url_lower or 'full' in url_lower: score += 100
            if 'large' in url_lower or 'big' in url_lower: score += 80
            if 'medium' in url_lower: score += 40
            if 'small' in url_lower or 'thumb' in url_lower: score -= 50
            
            # Resolution indicators  
            if '2048' in url or '1920' in url or '1080' in url: score += 90
            if '1024' in url or '800' in url: score += 60
            if '400' in url or '300' in url: score += 30
            if '150' in url or '100' in url: score -= 20
            
            # Density indicators
            if '2x' in url or '3x' in url: score += 70
            if '@2x' in url or '@3x' in url: score += 70
            
            # Format preferences (for quality priority)
            if priority == "Highest":
                if url_lower.endswith('.png'): score += 20
                if url_lower.endswith('.webp'): score += 15
                if url_lower.endswith('.jpg') or url_lower.endswith('.jpeg'): score += 10
            
            return score
        
        # Sort by quality score (highest first)
        return sorted(image_urls, key=quality_score, reverse=True)

    async def download_images_enhanced(self, image_urls: List[str], output_dir: Path) -> int:
        """Enhanced image downloading with better progress tracking"""
        
        if not image_urls:
            return 0
            
        successful_downloads = 0
        connector = aiohttp.TCPConnector(limit=15)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            
            # Create download tasks
            for url in image_urls:
                task = self.download_image(session, url, output_dir)
                tasks.append(task)
            
            # Execute with progress tracking
            completed = 0
            for task in asyncio.as_completed(tasks):
                try:
                    success = await task
                    if success:
                        successful_downloads += 1
                    completed += 1
                    
                    # Log progress every 25 downloads
                    if completed % 25 == 0:
                        logger.info(f"Progress: {successful_downloads}/{completed} downloaded")
                        
                except Exception as e:
                    logger.error(f"Download task failed: {e}")
                    self.session_stats["errors"] += 1
        
        logger.info(f"Enhanced download complete: {successful_downloads}/{len(image_urls)} images")
        return successful_downloads


# Create global instance for use in other modules
advanced_scraper = AdvancedImageScraper()