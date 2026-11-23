"""Person list and reviewed identity-group pages."""
from pathlib import Path
import json
import uuid
from typing import List

import streamlit as st

from settings_store import active_embedding_model, active_grouping_threshold
from ui_cache import face_thumbnail


def _assign_face_group(group, person_data, person_id, authenticated_actor):
    """Assign one reviewed group and return the IDs that were updated."""
    assigned: List[str] = []
    for face in group.get("faces", []):
        if st.session_state.vector_store.assign_person_name(
            face["face_id"],
            person_data,
            person_id,
            assignment_source="review_group",
            suggested_similarity=face.get("group_similarity"),
            actor=authenticated_actor(),
        ):
            assigned.append(face["face_id"])
    return assigned


def render_person_groups_page(get_cached_all_persons, invalidate_persons_cache, invalidate_gallery_cache, authenticated_actor):
    """Review identity groups of unnamed faces and assign them to a person."""
    st.header("Review face groups", icon=":material/groups:")
    st.markdown(
        "Unlabeled faces are grouped based on their embeddings. "
        "Suggestions are assigned to a person only after you confirm them."
    )

    notice = st.session_state.pop("person_groups_notice", None)
    if notice:
        st.success(notice)

    last_assignment = st.session_state.get("person_groups_last_assignment")
    if last_assignment:
        st.info(f"Last reviewed group assignment: {last_assignment.get('person_name', 'person')}.")
        if st.button("Undo last group assignment", key="undo_last_person_group"):
            outcome = st.session_state.vector_store.undo_person_assignments(
                last_assignment.get("events", {}), actor=authenticated_actor()
            )
            invalidate_persons_cache()
            invalidate_gallery_cache()
            st.session_state.person_groups_cache = None
            remaining = outcome.get("remaining", {})
            if remaining:
                last_assignment["events"] = remaining
                st.session_state.person_groups_last_assignment = last_assignment
            else:
                st.session_state.pop("person_groups_last_assignment", None)
            st.session_state.person_groups_notice = (
                f"Undid {len(outcome['undone'])} face assignment(s)."
                if outcome["undone"]
                else "No assignments were undone because the records had changed."
            )
            if outcome["skipped"] or outcome["errors"]:
                st.session_state.person_groups_notice += (
                    f" {len(outcome['skipped'])} changed face(s) were skipped; "
                    f"{len(outcome['errors'])} update(s) failed."
                )
            st.rerun()

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.30,
        max_value=0.90,
        value=active_grouping_threshold(),
        step=0.01,
        help=(
            "Cosine similarity. Starting points are 0.35 for Buffalo_L and 0.55 for ArcFace. "
            "Higher values create smaller, stricter groups. Review all suggestions before assigning names."
        ),
        key="person_groups_threshold",
    )

    if st.button("Recalculate groups", key="refresh_person_groups", icon=":material/refresh:"):
        st.session_state.person_groups_cache = None

    try:
        face_count = st.session_state.vector_store.collection.count()
    except Exception:  # noqa: BLE001 - the page still shows a readable error
        face_count = 0
    cache_key = f"{active_embedding_model()}:{threshold:.2f}:{face_count}"
    if (
        st.session_state.get("person_groups_cache") is None
        or st.session_state.get("person_groups_cache_key") != cache_key
    ):
        with st.spinner("Grouping faces..."):
            st.session_state.person_groups_cache = (
                st.session_state.vector_store.get_face_groups(threshold=threshold)
            )
            st.session_state.person_groups_cache_key = cache_key

    data = st.session_state.person_groups_cache
    if data.get("error"):
        st.error(f"Could not group faces: {data['error']}")
        return

    groups = data.get("groups", [])
    unnamed_faces = data.get("unnamed_faces", 0)
    grouped_faces = data.get("grouped_faces", 0)

    if unnamed_faces == 0:
        st.info("All faces are already assigned to a person.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Unassigned faces", unnamed_faces)
    col2.metric("Groups", len(groups))
    col3.metric("Faces in groups", grouped_faces)

    if not groups:
        st.info(
            "No group with at least two faces was found. Lower the threshold or assign names from the gallery."
        )
        return

    st.caption(
        f"{data.get('singletons', 0)} unassigned faces do not match a group and remain separate."
    )
    st.markdown("---")

    persons = get_cached_all_persons()
    person_choices = {"Create a new person": None}
    for person in persons:
        label = f"{person['full_name']} ({person['face_count']} faces)"
        person_choices[label] = person

    for index, group in enumerate(groups):
        group_key = group["face_ids"][0]
        with st.expander(
            f"Group {index + 1}: {group['size']} faces · "
            f"mean similarity {group['mean_similarity']:.0%}",
            expanded=index < 3,
        ):
            st.caption(
                f"Similarity to group center: {group['min_similarity']:.0%}–"
                f"{group['mean_similarity']:.0%} average. These are ranking scores, "
                "not identity probabilities."
            )
            for row_start in range(0, len(group["faces"]), 6):
                preview_columns = st.columns(min(6, len(group["faces"]) - row_start))
                for face_index, face in enumerate(
                    group["faces"][row_start:row_start + 6]
                ):
                    metadata = face.get("metadata", {})
                    image_path = metadata.get("image_path", "")
                    source_name = Path(image_path).name or face.get("face_id", "Face")
                    with preview_columns[face_index]:
                        thumbnail = face_thumbnail(
                            image_path,
                            metadata.get("location", ""),
                            (120, 120),
                        )
                        if thumbnail is not None:
                            st.image(
                                thumbnail,
                                caption=source_name,
                                width="stretch",
                                alt=f"Face from {source_name}",
                            )
                        else:
                            st.warning(f"Image unavailable: {source_name}")
                        st.caption(
                            f"{face.get('face_id', '')} · group-center similarity "
                            f"{face.get('group_similarity', 0):.0%}"
                        )

            suggestion = group.get("suggested_person")
            if suggestion:
                st.info(
                    f"Closest existing person: **{suggestion['full_name']}** "
                    f"({suggestion['similarity']:.0%}). Not assigned yet."
                )

            default_choice = 0
            if suggestion:
                for option_index, (_, person) in enumerate(person_choices.items()):
                    if person and person["person_id"] == suggestion["person_id"]:
                        default_choice = option_index
                        break
            choice = st.selectbox(
                "Assign group to",
                list(person_choices.keys()),
                index=default_choice,
                key=f"group_target_{group_key}",
            )
            first_name = ""
            last_name = ""
            if person_choices[choice] is None:
                name_col1, name_col2 = st.columns(2)
                first_name = name_col1.text_input(
                    "First name", key=f"group_first_{group_key}"
                )
                last_name = name_col2.text_input(
                    "Last name", key=f"group_last_{group_key}"
                )

            st.warning(
                f"Review all {group['size']} face crops and their source filenames. "
                "Confirming assigns this person to every face in the group. "
                "Similarity scores are not proof of identity."
            )
            if st.button(
                "Assign group",
                key=f"assign_group_{group_key}",
                type="primary",
            ):
                selected_person = person_choices[choice]
                if selected_person is None:
                    if not first_name.strip() and not last_name.strip():
                        st.warning("Enter at least a first or last name.")
                        continue
                    person_data = {
                        "first_name": first_name.strip(),
                        "middle_names": "",
                        "last_name": last_name.strip(),
                        "birth_date": "",
                        "birth_place": "",
                        "notes": "Assigned from review groups",
                    }
                    person_id = str(uuid.uuid4())
                else:
                    person_data = {
                        "first_name": selected_person.get("first_name", ""),
                        "middle_names": selected_person.get("middle_names", ""),
                        "last_name": selected_person.get("last_name", ""),
                        "birth_date": selected_person.get("birth_date", ""),
                        "birth_place": selected_person.get("birth_place", ""),
                        "notes": selected_person.get("notes", ""),
                    }
                    person_id = selected_person["person_id"]

                assigned_face_ids = _assign_face_group(group, person_data, person_id, authenticated_actor)
                if assigned_face_ids:
                    event_ids = {}
                    for face_id in assigned_face_ids:
                        current_face = st.session_state.vector_store.get_face_by_id(face_id)
                        raw_history = (current_face or {}).get("metadata", {}).get("assignment_history", "[]")
                        try:
                            history = json.loads(raw_history) if isinstance(raw_history, str) else []
                        except (TypeError, ValueError):
                            history = []
                        if history and history[-1].get("action") == "assigned":
                            event_ids[face_id] = str(history[-1].get("at", ""))
                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    st.session_state.person_groups_cache = None
                    st.session_state.person_groups_last_assignment = {
                        "person_name": f"{person_data['first_name']} {person_data['last_name']}".strip(),
                        "events": event_ids,
                    }
                    st.session_state.person_groups_notice = (
                        f"Assigned {len(assigned_face_ids)} faces to "
                        f"{person_data['first_name']} {person_data['last_name']}."
                    )
                    st.rerun()
                else:
                    st.error("❌ Could not assign any faces.")


def render_people_page(get_cached_all_persons, invalidate_persons_cache, invalidate_gallery_cache, authenticated_actor):
    """Name Gallery page to view and manage person names"""

    st.header("People", icon=":material/person:")
    st.markdown("Manage all named people in the database")

    try:
        # Get all persons from database with caching
        persons = get_cached_all_persons()

        if not persons:
            st.info("No names assigned yet. Open Face search or Face library to assign a name.")
            return

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👤 People", len(persons))
        with col2:
            total_assigned_faces = sum(p['face_count'] for p in persons)
            st.metric("Named faces", total_assigned_faces)
        with col3:
            avg_faces = total_assigned_faces / len(persons) if persons else 0
            st.metric("Average faces per person", f"{avg_faces:.1f}")

        st.markdown("---")

        # Search functionality
        col1, col2 = st.columns([2, 1])
        with col1:
            name_search = st.text_input("Search by name", placeholder="Enter a first or last name")
        with col2:
            sort_option = st.selectbox(
                "Sort by", ["Name (A–Z)", "Name (Z–A)", "Face count (ascending)", "Face count (descending)"]
            )

        # Filter persons based on search
        filtered_persons = persons
        if name_search.strip():
            search_term = name_search.lower()
            filtered_persons = [p for p in persons
                              if search_term in p['full_name'].lower()]

        # Sort persons
        if sort_option == "Name (A–Z)":
            filtered_persons.sort(key=lambda x: x['full_name'])
        elif sort_option == "Name (Z–A)":
            filtered_persons.sort(key=lambda x: x['full_name'], reverse=True)
        elif sort_option == "Face count (ascending)":
            filtered_persons.sort(key=lambda x: x['face_count'])
        elif sort_option == "Face count (descending)":
            filtered_persons.sort(key=lambda x: x['face_count'], reverse=True)

        if not filtered_persons:
            st.warning(f"No people found for '{name_search}'")
            return

        st.markdown(f"**{len(filtered_persons)} people found:**")

        # Display persons
        for person in filtered_persons:
            with st.expander(f"👤 **{person['full_name']}** ({person['face_count']} faces)", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write(f"**First name:** {person['first_name']}")
                    st.write(f"**Last name:** {person['last_name']}")
                    if person.get('birth_date'):
                        st.write(f"**Date of birth:** {person['birth_date']}")
                    st.write(f"**Person ID:** `{person['person_id']}`")

                with col2:
                    st.write(f"**Faces:** {person['face_count']}")
                    if st.button("Show all faces", key=f"show_faces_{person['person_id']}", icon=":material/visibility:"):
                        st.session_state.selected_person = person['person_id']
                        st.rerun()

                with col3:
                    # Simplified one-click deletion
                    if st.button("Remove name", key=f"remove_person_{person['person_id']}",
                                help="Remove this name from all faces assigned to this person.",
                                type="secondary"):
                        # Direct deletion without confirmation - as requested
                        with st.spinner("Removing the name from all faces..."):
                            removed_count = 0
                            for face_id in person['face_ids']:
                                if st.session_state.vector_store.remove_person_name(
                                    face_id, actor=authenticated_actor()
                                ):
                                    removed_count += 1

                            if removed_count > 0:
                                st.success(f"Removed the name from {removed_count} faces.")
                                # Invalidate cached names and gallery metadata
                                invalidate_persons_cache()
                                invalidate_gallery_cache()
                                st.rerun()
                            else:
                                st.error("❌ Could not remove the name.")

        # Display faces for selected person
        if st.session_state.get('selected_person'):
            person_id = st.session_state.selected_person
            person_faces = st.session_state.vector_store.get_faces_by_person_id(person_id)

            if person_faces:
                st.markdown("---")
                selected_person_name = next((p['full_name'] for p in persons if p['person_id'] == person_id), "Unknown")
                st.subheader(f"Faces assigned to **{selected_person_name}**")

                # Display faces in grid with enhanced functionality
                cols = st.columns(3)  # Use 3 columns for better layout with more buttons
                for idx, face in enumerate(person_faces):
                    with cols[idx % 3]:
                        metadata = face['metadata']
                        image_path = metadata.get('image_path', '')
                        face_location = metadata.get('location', '')
                        face_id = face['face_id']

                        try:
                            # Load and display face thumbnail
                            thumbnail = face_thumbnail(image_path, face_location, (150, 150))
                            if thumbnail is not None:
                                st.image(thumbnail, caption=f"Face {face_id[:8]}...", width="stretch", alt="Face assigned to this person")
                            else:
                                st.warning("Could not load face")

                        except Exception as e:
                            st.error(f"Error while loading: {e}")

                        # Face info
                        st.caption(f"**ID:** {face_id[:12]}...")
                        st.caption(f"**Image:** {Path(image_path).name}")

                        # Buttons in columns for better layout
                        btn_col1, btn_col2 = st.columns(2)

                        with btn_col1:
                            # Single unified info button
                            if st.button("Details", key=f"name_gallery_info_{face_id}", help="Open the full image, analyze the face, or change its assigned name."):
                                # Preserve selected_person state for Name Gallery
                                st.session_state.name_gallery_selected_person = st.session_state.selected_person
                                # Set face info data and trigger modal
                                st.session_state.info_face_id = face_id
                                st.session_state.info_image_path = image_path
                                st.session_state.info_face_location = face_location
                                st.rerun()

                        with btn_col2:
                            # Remove photo button
                            if st.button("Remove", key=f"remove_photo_{face_id}", help="Remove this photo from this person.", type="secondary"):
                                # Remove person name from this specific face
                                if st.session_state.vector_store.remove_person_name(
                                    face_id, actor=authenticated_actor()
                                ):
                                    st.success(f"Removed the photo from {selected_person_name}.")
                                    invalidate_persons_cache()
                                    invalidate_gallery_cache()
                                    # Refresh the view
                                    st.rerun()
                                else:
                                    st.error("❌ Could not remove the photo.")

                # Close view button
                st.markdown("---")
                if st.button("Close view", type="primary"):
                    del st.session_state.selected_person
                    st.rerun()

    except Exception as e:
        st.error(f"Could not load the people gallery: {e}")
