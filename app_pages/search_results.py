"""Rendering helpers for local face-search results."""
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st

from config import RESULTS_PER_PAGE, THUMBNAIL_SIZE
from ui_cache import face_thumbnail
from utils import create_thumbnail, load_and_preprocess_image, parse_face_location


def display_search_performance(metrics, history):
    """Show per-search timings and a session-only view of recent searches."""
    if not metrics:
        return

    with st.expander("Search performance and score distribution", expanded=False):
        st.caption(
            "Timings are measured on this device for the current session. Detection "
            "runs before the search step; model, hardware, cache, and library size affect results."
        )
        timing_columns = st.columns(4)
        timing_values = [
            ("Face detection", "detection_ms"),
            ("Face embedding", "embedding_ms"),
            ("Vector search", "vector_search_ms"),
            ("Detection + search", "processing_ms"),
        ]
        for column, (label, key) in zip(timing_columns, timing_values):
            column.metric(label, f"{metrics.get(key, 0.0):.0f} ms")

        count_columns = st.columns(3)
        count_columns[0].metric("Candidates returned", metrics.get("candidates_returned", 0))
        count_columns[1].metric("After face filters", metrics.get("eligible_candidates", 0))
        count_columns[2].metric("Results returned", metrics.get("results_returned", 0))
        st.caption(
            f"Similarity cutoff used: {metrics.get('threshold', 0.0):.2f}. "
            f"Face filtering took {metrics.get('filtering_ms', 0.0):.0f} ms. "
            "Combined processing time includes detection and search, but excludes image upload and decoding. "
            "The cutoff is a ranking filter, not an identity-confidence level."
        )

        score_samples = metrics.get("score_samples", [])
        if score_samples:
            bucket_counts = [0] * 10
            for score in score_samples:
                bucket = min(9, max(0, int(float(score) * 10)))
                bucket_counts[bucket] += 1
            histogram = pd.DataFrame({
                "Similarity score range": [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)],
                "Candidates": bucket_counts,
            })
            st.caption(
                "Score distribution for candidates returned above the cutoff. "
                "It shows ranking values, not the chance of a correct identity match."
            )
            st.bar_chart(
                histogram,
                x="Similarity score range",
                y="Candidates",
                width="stretch",
                height=220,
            )

        if len(history) >= 2:
            history_frame = pd.DataFrame(history[-30:]).copy()
            history_frame.insert(0, "Search", range(1, len(history_frame) + 1))
            history_frame = history_frame.rename(columns={
                "processing_ms": "Detection + search processing",
            })
            st.caption(f"Recent detection + search processing times ({len(history_frame)} searches, up to 30 shown).")
            st.line_chart(
                history_frame,
                x="Search",
                y="Detection + search processing",
                width="stretch",
                height=240,
            )


def _display_image_source(metadata):
    """Display recorded source details without treating authorization as verified consent."""
    source_url = metadata.get("source_url")
    if not isinstance(source_url, str):
        return
    try:
        parsed = urlsplit(source_url)
    except ValueError:
        return
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return

    collected_at = metadata.get("collected_at")
    source_label = f"Image source: {parsed.hostname}"
    if isinstance(collected_at, str) and collected_at:
        source_label += f" · imported {collected_at[:19].replace('T', ' ')} UTC"
    st.caption(source_label)
    st.link_button("Open source page", source_url, width="stretch")
    if metadata.get("authorization_confirmed") is True:
        st.caption("The importer recorded an authorization confirmation.")


def display_search_results(results: List[Dict[str, Any]]):
    """Display enhanced search results with filtering options"""

    st.subheader(f"Enhanced Search Results ({len(results)} faces)")

    if not results:
        st.info("No results to display")
        return

    # Real-time filtering options
    with st.expander("🔧 Filter and sort results", expanded=False):
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            min_similarity_filter = st.slider(
                "Minimum cosine similarity score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Hide lower ranking scores. This value is not an identity-confidence cutoff."
            )

            min_size_filter = st.slider(
                "Min Face Size (px)",
                min_value=20,
                max_value=200,
                value=20,
                help="Hide small faces"
            )

        with col_f2:
            show_named_only = st.checkbox(
                "Named Faces Only",
                value=False,
                help="Show only faces with assigned names"
            )

        with col_f2:
            sort_order = st.selectbox(
                "Sort By",
                ["Similarity ↓", "Similarity ↑", "Size ↓", "Size ↑", "Name A-Z"],
                index=0,
                help="Change result ordering"
            )

    # Apply filters
    filtered_results = []
    for result in results:
        # Similarity filter
        if result['similarity'] < min_similarity_filter:
            continue

        # Size filter
        metadata = result.get('metadata', {})
        location_str = metadata.get('location', '0,0,0,0')
        try:
            coords = location_str.split(',')
            if len(coords) == 4:
                top, right, bottom, left = map(int, coords)
                face_size = min(right - left, bottom - top)
                if face_size < min_size_filter:
                    continue
        except:
            pass

        # Named faces filter
        if show_named_only:
            person_name = metadata.get('full_name', '')
            if not person_name:
                continue

        filtered_results.append(result)

    # Apply sorting
    if sort_order == "Similarity ↓":
        filtered_results = sorted(filtered_results, key=lambda x: x['similarity'], reverse=True)
    elif sort_order == "Similarity ↑":
        filtered_results = sorted(filtered_results, key=lambda x: x['similarity'])
    elif sort_order in ("Size ↓", "Size ↑"):
        def get_size(x):
            try:
                coords = x.get('metadata', {}).get('location', '0,0,0,0').split(',')
                if len(coords) == 4:
                    top, right, bottom, left = map(int, coords)
                    return (right - left) * (bottom - top)
                return 0
            except (TypeError, ValueError):
                return 0
        filtered_results = sorted(
            filtered_results, key=get_size, reverse=(sort_order == "Size ↓")
        )
    elif sort_order == "Name A-Z":
        filtered_results = sorted(filtered_results, key=lambda x: x.get('metadata', {}).get('full_name', 'zzz'))

    # Show filter results summary
    if len(filtered_results) != len(results):
        st.info(f"📋 Filtered: {len(filtered_results)} of {len(results)} results shown (filtered out {len(results) - len(filtered_results)})")
    if not filtered_results:
        st.info("No results match these filters. Lower the similarity or face-size filter to see more results.")
        return

    # Display options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{len(filtered_results)} results displayed**")

    with col2:
        show_all = st.toggle("Show All Images", value=False, help="Toggle pagination on/off")

    # Use filtered results for display
    results = filtered_results

    # Determine what to display
    if show_all:
        # Show all results
        page_results = results
        if len(results) > 20:
            st.info(f"Displaying all {len(results)} results")
    else:
        # Pagination
        total_pages = max(1, (len(results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
        current_page = min(
            max(1, int(st.session_state.get("current_page", 1))), total_pages
        )
        st.session_state.current_page = current_page

        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                if st.button("⬅️ Previous") and current_page > 1:
                    st.session_state.current_page = current_page - 1
                    st.rerun()

            with col2:
                st.write(f"Page {current_page} of {total_pages}")

            with col3:
                if st.button("➡️ Next") and current_page < total_pages:
                    st.session_state.current_page = current_page + 1
                    st.rerun()

        # Calculate page slice
        start_idx = (current_page - 1) * RESULTS_PER_PAGE
        end_idx = start_idx + RESULTS_PER_PAGE
        page_results = results[start_idx:end_idx]

    # Display results in grid
    cols = st.columns(4)

    for i, face_data in enumerate(page_results):
        with cols[i % 4]:
            try:
                metadata = face_data.get('metadata', {})
                image_path = Path(metadata.get('image_path', ''))
                location = metadata.get('location')
                box = parse_face_location(location)

                if not image_path.exists():
                    st.error("Image not found")
                    continue

                # Cached thumbnail: the full-resolution photo is decoded at most once
                thumbnail = face_thumbnail(image_path, location, THUMBNAIL_SIZE)
                if thumbnail is None and box is not None:
                    image = load_and_preprocess_image(image_path)
                    if image is not None:
                        top, right, bottom, left = box
                        thumbnail = create_thumbnail(image[top:bottom, left:right], THUMBNAIL_SIZE)

                if thumbnail is not None:
                    st.image(thumbnail, width='stretch', alt="Face thumbnail")

                face_width = face_height = 0
                if box is not None:
                    top, right, bottom, left = box
                    face_width = right - left
                    face_height = bottom - top
                face_size = min(face_width, face_height)
                face_area = face_width * face_height

                similarity_score = float(face_data['similarity'])

                quality_indicators = []
                if face_size >= 80:
                    quality_indicators.append("🔍 High Res")
                elif face_size >= 50:
                    quality_indicators.append("📏 Medium Res")
                else:
                    quality_indicators.append("📱 Low Res")
                if face_area > 6400:
                    quality_indicators.append("✨ Large")

                st.metric(
                    "Cosine similarity score",
                    f"{similarity_score:.3f} / 1.000",
                    help="A ranking value for face-embedding similarity, not a probability that the identities match.",
                )
                st.caption(f"Higher values rank closer · face size: {face_size}px")

                if quality_indicators:
                    st.caption(" | ".join(quality_indicators))

                st.write(f"**📁 {image_path.name}**")

                person_name = metadata.get('full_name', '')
                if person_name:
                    st.success(f"👤 **{person_name}**")

                _display_image_source(metadata)

                face_id = face_data.get('face_id', face_data.get('id', f"face_{i}"))
                if st.button("ℹ️ Details", key=f"search_result_details_{face_id}", help="Open full image, analyze face, or assign a name.", width='stretch'):
                    st.session_state.info_face_id = face_id
                    st.session_state.info_image_path = image_path
                    st.session_state.info_face_location = location
                    st.rerun()

            except Exception as e:
                st.error(f"Error displaying result: {e}")
