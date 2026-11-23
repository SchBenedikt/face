"""Authorized web-image import page."""

import streamlit as st

from config import THUMBNAIL_SIZE
from ui_cache import face_thumbnail


def render_web_import_page(parse_scrape_urls, run_coroutine, display_image_provenance, process_existing_images):
    """Web scraping page"""

    st.header("🌐 Web Image Scraping")
    st.warning(
        "Only scrape images from domains and for purposes you are authorized to use. "
        "The crawler is restricted to the hostname entered, follows robots.txt, and blocks private-network targets."
    )

    tab1, tab2 = st.tabs(["Scrape New Sites", "Scraped Images"])

    with tab1:
        st.subheader("Scrape Images from Websites")

        # URL input
        urls_input = st.text_area(
            "Enter website URLs (one per line):",
            height=100,
            placeholder="https://example.com/gallery\nhttps://another-site.com/photos",
            help="Enter URLs of websites to scrape images from"
        )

        col1, col2 = st.columns(2)

        with col1:
            max_images = st.slider(
                "Maximum images per hostname",
                min_value=1,
                max_value=500,
                value=100,
                help="Web import has a finite limit to avoid unexpectedly large downloads.",
            )

        with col2:
            auto_process = st.checkbox(
                "Auto-process scraped images",
                value=False,
                help="Automatically process scraped images for face detection"
            )

        confirm_authorization = st.checkbox(
            "I confirm that I am authorized to collect images from the entered website(s).",
            value=False,
            key="scrape_authorization_confirmed",
        )

        # Show duplicate detection status.
        st.info("🔍 **Duplicate detection is on:** Existing images are skipped automatically")

        if st.button("🕷️ Start Scraping", type="primary"):
            urls, invalid_urls = parse_scrape_urls(urls_input)
            if invalid_urls:
                listed = ", ".join(f"`{value}`" for value in invalid_urls)
                st.error(
                    f"Invalid URL(s): {listed}. Enter complete URLs such as "
                    "https://example.com."
                )
            elif not urls:
                st.error("Please enter at least one URL")
            elif not confirm_authorization:
                st.error("Confirm that you are authorized to collect images before starting web import.")
            elif st.session_state.image_scraper is None:
                st.error(
                    "Web import is unavailable because old_image_scraper could not "
                    "be loaded. Check the missing dependency details above."
                )
            else:
                scrape_websites(urls, max_images, auto_process)

    with tab2:
        display_scraped_images(display_image_provenance, process_existing_images)

def scrape_websites(urls, max_images, auto_process, run_coroutine, process_existing_images):
    """Scrape images from websites"""

    with st.spinner("Scraping images from websites..."):
        try:
            # Run async scraping; asyncio.run() shuts the loop down cleanly
            # instead of closing it while Playwright tasks are still pending.
            results = run_coroutine(
                st.session_state.image_scraper.scrape_multiple_websites(
                    urls, max_images, authorization_confirmed=True
                )
            )

            # Display results
            success_count = sum(1 for r in results if r.get("success", False))
            total_downloaded = sum(r.get("images_downloaded", 0) for r in results)

            if success_count > 0:
                st.success(f"Successfully scraped {success_count}/{len(urls)} websites. Downloaded {total_downloaded} images.")

                # Show detailed results
                for result in results:
                    if result.get("success"):
                        st.info(f"✅ {result['url']}: Downloaded {result['images_downloaded']} images")
                    else:
                        st.error(f"❌ {result['url']}: {result.get('error', 'Unknown error')}")

                # Auto-process if requested
                if auto_process and total_downloaded > 0:
                    st.info("Auto-processing scraped images...")
                    scraped_images = st.session_state.image_scraper.get_scraped_images()
                    if scraped_images:
                        process_existing_images(scraped_images)

            else:
                st.error("Failed to scrape any websites. Please check the URLs and try again.")

        except Exception as e:
            st.error(f"Error during scraping: {e}")

def display_scraped_images(display_image_provenance, process_existing_images):
    """Display scraped images"""

    st.subheader("Scraped Images")

    scraped_images = st.session_state.image_scraper.get_scraped_images()

    if scraped_images:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Found {len(scraped_images)} scraped images**")

        with col2:
            show_all_scraped = st.toggle("Show All Scraped", value=True, help="Show all scraped images or limit to first 20")

        # Determine how many images to show
        images_to_show = scraped_images if show_all_scraped else scraped_images[:20]

        if not show_all_scraped and len(scraped_images) > 20:
            st.info(f"Showing first 20 of {len(scraped_images)} images. Toggle 'Show All Scraped' to see all.")

        # Display in grid
        cols_per_row = 5
        for i in range(0, len(images_to_show), cols_per_row):
            cols = st.columns(cols_per_row)

            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(images_to_show):
                    with col:
                        try:
                            image_path = images_to_show[idx]
                            thumbnail = face_thumbnail(image_path, None, THUMBNAIL_SIZE)
                            if thumbnail is not None:
                                st.image(thumbnail, width='stretch', alt=f"Scraped image: {image_path.name}")
                                st.caption(f"{image_path.name}")
                                display_image_provenance(image_path)
                        except Exception:
                            st.error("Failed to load")

        if len(scraped_images) > 20:
            st.info(f"Showing first 20 of {len(scraped_images)} images")

        # Process button
        if st.button("🔄 Process All Scraped Images"):
            process_existing_images(scraped_images)
    else:
        st.info("No scraped images found. Use the scraping tab to download images from websites.")
