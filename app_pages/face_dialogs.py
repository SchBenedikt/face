"""Face detail, assignment, and full-image dialogs."""
from pathlib import Path
import uuid

import cv2
import streamlit as st

from app_pages.facial_analysis import show_analysis_tab
from utils import create_thumbnail, load_and_preprocess_image


@st.dialog("Assign a name", width="large")
def render_name_assignment_modal(get_cached_all_persons, invalidate_persons_cache, invalidate_gallery_cache, authenticated_actor):
    """Enhanced name assignment interface with extended fields and existing person selection"""

    # Required keys for name assignment
    required_keys = ['name_assign_face_id', 'name_assign_image_path', 'name_assign_face_location']
    missing_keys = [key for key in required_keys if key not in st.session_state]

    if missing_keys:
        st.error(f"Missing face data for name assignment: {', '.join(missing_keys)}")
        st.info("Try opening the name assignment again.")
        if st.button("Close"):
            # Clear any partial session state
            for key in required_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return

    face_id = st.session_state.name_assign_face_id
    image_path = st.session_state.name_assign_image_path
    face_location = st.session_state.name_assign_face_location

    # Ensure image_path is a Path object
    if isinstance(image_path, str):
        image_path = Path(image_path)

    # Header info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Face ID:** `{face_id}`")
        st.markdown(f"**File:** `{image_path.name}`")

    # Show face preview
    with col2:
        try:
            full_image = load_and_preprocess_image(image_path)
            face_crop = None
            if full_image is not None:
                if isinstance(face_location, str):
                    coords = face_location.split(',')
                    if len(coords) == 4:
                        top, right, bottom, left = map(int, coords)
                        face_crop = full_image[top:bottom, left:right]
                elif isinstance(face_location, (tuple, list)) and len(face_location) == 4:
                    top, right, bottom, left = face_location
                    face_crop = full_image[top:bottom, left:right]

                if face_crop is not None and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    st.image(face_crop, caption="Face to assign", width=150, alt="Selected face")
        except Exception as e:
            st.warning(f"Could not display the face: {e}")

    st.markdown("---")

    # Get existing person data if available
    existing_face = st.session_state.vector_store.get_face_by_id(face_id)
    existing_data = {}
    if existing_face and existing_face.get('metadata', {}).get('person_id'):
        metadata = existing_face['metadata']
        existing_data = {
            'first_name': metadata.get('first_name', ''),
            'middle_names': metadata.get('middle_names', ''),
            'last_name': metadata.get('last_name', ''),
            'full_name': metadata.get('full_name', ''),
            'birth_date': metadata.get('birth_date', ''),
            'birth_place': metadata.get('birth_place', ''),
            'notes': metadata.get('notes', ''),
            'person_id': metadata.get('person_id', '')
        }
        st.info(f"✅ **Current name:** {existing_data['full_name']}")
        replace_existing_identity = st.checkbox(
            "I confirm that changing this face's person assignment is intentional.",
            key=f"confirm_reassign_{face_id}",
            help="Without confirmation, this face keeps its current person label.",
        )
    else:
        replace_existing_identity = False

    # Get all existing persons for selection
    all_persons = get_cached_all_persons()  # Use cached version

    # Choice between new person or existing person
    st.subheader("Select a person")
    st.caption(
        "The name will be assigned only to this face. Review similar faces "
        "separately in **Review groups**."
    )

    assignment_type = st.radio(
        "How would you like to assign the name?",
        ["Create a new person", "Select an existing person"],
        key="assignment_type",
        horizontal=True
    )

    if assignment_type == "Select an existing person" and all_persons:
        # Show existing persons selection
        st.subheader("Existing people")

        # Create searchable list of persons
        person_options = {"-- Select a person --": None}
        for person in all_persons:
            display_name = f"{person['full_name']}"
            if person.get('birth_date'):
                display_name += f" (*{person['birth_date']})"
            display_name += f" ({person['face_count']} faces)"
            person_options[display_name] = person

        selected_person_display = st.selectbox(
            "Select a person:",
            options=list(person_options.keys()),
            key="selected_existing_person"
        )

        if selected_person_display and person_options[selected_person_display] is not None:
            selected_person = person_options[selected_person_display]

            # Ensure selected_person is not None
            if selected_person is None:
                st.error("Could not load person data.")
                return

            # Show person details in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Full name:**")
                st.write(f"*{selected_person['full_name']}*")
                st.write(f"**First name:** {selected_person['first_name']}")
                if selected_person.get('middle_names'):
                    st.write(f"**Middle names:** {selected_person['middle_names']}")
                st.write(f"**Last name:** {selected_person['last_name']}")

            with col2:
                st.write(f"**Date of birth:** {selected_person.get('birth_date', '-')}")
                st.write(f"**Place of birth:** {selected_person.get('birth_place', '-')}")
                st.write(f"**Faces:** {selected_person['face_count']}")

            with col3:
                if selected_person.get('notes'):
                    st.write("**Notes:**")
                    st.write(selected_person['notes'])

            # Assign button
            if st.button("Assign this person", type="primary", width='stretch', icon=":material/person_add:"):
                person_data = {
                    'first_name': selected_person['first_name'],
                    'middle_names': selected_person.get('middle_names', ''),
                    'last_name': selected_person['last_name'],
                    'birth_date': selected_person.get('birth_date', ''),
                    'birth_place': selected_person.get('birth_place', ''),
                    'notes': selected_person.get('notes', '')
                }

                result = st.session_state.vector_store.assign_person_name(
                    face_id,
                    person_data,
                    selected_person['person_id'],
                    allow_reassignment=replace_existing_identity,
                    actor=authenticated_actor(),
                )

                if result:
                    st.success("Person assigned.")

                    # Invalidate cached names and gallery metadata
                    invalidate_persons_cache()
                    invalidate_gallery_cache()

                    # Clear session state and close dialog
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error("❌ Could not assign the person.")

    elif assignment_type == "Select an existing person" and not all_persons:
        st.info("No existing people found. Create a new person instead.")

    else:
        # New person creation form
        st.subheader("Create a new person")

        # Extended person form in columns
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Basic information:**")
            first_name = st.text_input(
                "First name *",
                value=existing_data.get('first_name', '') or '',
                key="new_first_name"
            )
            middle_names = st.text_input(
                "Middle names",
                value=existing_data.get('middle_names', '') or '',
                help="For example: Moritz Luitpold Damian",
                key="new_middle_names"
            )
            last_name = st.text_input(
                "Last name *",
                value=existing_data.get('last_name', '') or '',
                key="new_last_name"
            )

        with col2:
            st.write("**Additional information:**")
            birth_date = st.text_input(
                "Date of birth",
                value=existing_data.get('birth_date', ''),
                help="For example: 1985-03-15 or 15.03.1985",
                key="new_birth_date"
            )
            birth_place = st.text_input(
                "Place of birth",
                value=existing_data.get('birth_place', ''),
                help="For example: Munich, Germany",
                key="new_birth_place"
            )

        notes = st.text_area(
            "Notes:",
            value=existing_data.get('notes', ''),
            help="Additional notes about this person.",
            key="new_notes",
            height=80
        )

        # Preview full name
        if first_name or middle_names or last_name:
            name_parts = [p.strip() for p in [first_name or '', middle_names or '', last_name or ''] if p and p.strip()]
            full_name = ' '.join(name_parts)
            st.success(f"**Full name:** {full_name}")

        # Save new person button
        if st.button("Save person", type="primary", width='stretch', icon=":material/save:"):
            if (first_name and first_name.strip()) or (last_name and last_name.strip()):
                # Create person data
                person_data = {
                    'first_name': (first_name or '').strip(),
                    'middle_names': (middle_names or '').strip(),
                    'last_name': (last_name or '').strip(),
                    'birth_date': (birth_date or '').strip(),
                    'birth_place': (birth_place or '').strip(),
                    'notes': (notes or '').strip()
                }

                result = st.session_state.vector_store.assign_person_name(
                    face_id,
                    person_data,
                    allow_reassignment=replace_existing_identity,
                    actor=authenticated_actor(),
                )

                if result:
                    st.success("Person created and assigned.")

                    # Invalidate cached names and gallery metadata
                    invalidate_persons_cache()
                    invalidate_gallery_cache()

                    # Clear session state and close dialog
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error(
                        "Could not create the person. A record with the same name "
                        "and birth details may already exist; select that person instead."
                    )
            else:
                st.warning("⚠️ Enter at least a first or last name.")

    # Action buttons at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if existing_data.get('full_name'):
            if st.button("Remove name", width='stretch'):
                if st.session_state.vector_store.remove_person_name(
                    face_id, actor=authenticated_actor()
                ):
                    st.success("✅ Name removed.")
                    # Invalidate cached names and gallery metadata
                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    _clear_name_assignment_session()
                    st.rerun()
                else:
                    st.error("❌ Could not remove the name.")

    with col3:
        if st.button("Cancel", width='stretch'):
            _clear_name_assignment_session()
            st.rerun()


def _clear_name_assignment_session():
    """Helper function to clear name assignment session state"""
    keys_to_clear = [
        'name_assign_face_id', 'name_assign_image_path', 'name_assign_face_location',
        'assignment_type', 'selected_existing_person', 'new_first_name',
        'new_middle_names', 'new_last_name', 'new_birth_date',
        'new_birth_place', 'new_notes'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Restore Name Gallery state if it was preserved
    if 'name_gallery_selected_person' in st.session_state:
        st.session_state.selected_person = st.session_state.name_gallery_selected_person
        del st.session_state.name_gallery_selected_person


@st.dialog("Face information", width="large")
def render_face_info_modal(display_image_provenance, show_assignment_history, get_cached_all_persons, invalidate_persons_cache, invalidate_gallery_cache, authenticated_actor):
    """Unified modal for all face information: Full Image, Analysis, and Name Assignment"""

    # Make the dialog content even wider using custom CSS
    st.markdown("""
    <style>
    .stDialog > div:first-child > div:first-child > div:first-child {
        width: 95vw !important;
        max-width: 1600px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Required keys for face info display
    required_keys = ['info_face_id', 'info_image_path', 'info_face_location']
    missing_keys = [key for key in required_keys if key not in st.session_state]

    if missing_keys:
        st.error(f"Missing face data: {', '.join(missing_keys)}")
        if st.button("Close"):
            for key in required_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return

    face_id = st.session_state.info_face_id
    image_path = st.session_state.info_image_path
    face_location = st.session_state.info_face_location

    # Ensure image_path is a Path object
    if isinstance(image_path, str):
        image_path = Path(image_path)

    # Header info
    st.markdown(f"**🆔 Face ID:** `{face_id[:20]}...`")
    st.markdown(f"**File:** `{image_path.name}`")
    display_image_provenance(image_path)
    show_assignment_history(face_id)

    # Tabs for different information types
    tab1, tab2, tab3 = st.tabs(["Full image", "Face analysis", "Assign name"])

    with tab1:
        show_full_image_tab(image_path, face_location)

    with tab2:
        show_analysis_tab(image_path, face_location, face_id)

    with tab3:
        show_name_assignment_tab(face_id, image_path, face_location, get_cached_all_persons, invalidate_persons_cache, invalidate_gallery_cache, authenticated_actor)

    # Close button
    st.markdown("---")
    if st.button("Close", key="close_face_info_modal", type="primary", width='stretch'):
        # Clear face info data
        for key in required_keys:
            if key in st.session_state:
                del st.session_state[key]

        # Restore Name Gallery state if it was preserved
        if 'name_gallery_selected_person' in st.session_state:
            st.session_state.selected_person = st.session_state.name_gallery_selected_person
            del st.session_state.name_gallery_selected_person

        st.rerun()


def show_full_image_tab(image_path, face_location):
    """Display full image with face bounding box in tab"""
    try:
        # Load the full image
        full_image = load_and_preprocess_image(image_path)
        if full_image is None:
            st.error("❌ Could not load the image.")
            return

        # Parse face location if provided
        image_with_box = full_image.copy()
        if face_location:
            if isinstance(face_location, str):
                coords = face_location.split(',')
                if len(coords) == 4:
                    top, right, bottom, left = map(int, coords)
                    # Draw bounding box on the image
                    cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
                    st.success(f"Face marked at: top={top}, right={right}, bottom={bottom}, left={left}")
            else:
                top, right, bottom, left = face_location
                cv2.rectangle(image_with_box, (left, top), (right, bottom), (0, 255, 0), 3)
                st.success(f"Face marked at: top={top}, right={right}, bottom={bottom}, left={left}")

        # Display the image with maximum width
        caption = f"Full image - {image_path.name}"
        if face_location:
            caption += " (face highlighted)"

        st.image(image_with_box, caption=caption, width='stretch', alt=caption)

        # Image info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Image size", f"{full_image.shape[1]} × {full_image.shape[0]} px")

        with col2:
            file_size = image_path.stat().st_size if image_path.exists() else 0
            st.metric("File size", f"{file_size / 1024:.1f} KB")

        with col3:
            if face_location:
                st.metric("Face detected", "Yes")
            else:
                st.metric("Face detected", "No")

    except Exception as e:
        st.error(f"❌ Could not display the full image: {str(e)}")


def show_name_assignment_tab(face_id, image_path, face_location, get_cached_all_persons, invalidate_persons_cache, invalidate_gallery_cache, authenticated_actor):
    """Display name assignment interface in tab"""

    # Show face preview
    try:
        full_image = load_and_preprocess_image(image_path)
        if full_image is not None and face_location:
            if isinstance(face_location, str):
                coords = face_location.split(',')
                if len(coords) == 4:
                    top, right, bottom, left = map(int, coords)
                    face_crop = full_image[top:bottom, left:right]
                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        thumbnail = create_thumbnail(face_crop, (150, 150))
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(thumbnail, caption="Face for name assignment", alt="Selected face")
    except:
        pass

    # Get existing person data if available
    existing_face = st.session_state.vector_store.get_face_by_id(face_id)
    existing_data = {}
    if existing_face and existing_face.get('metadata', {}).get('person_id'):
        metadata = existing_face['metadata']
        existing_data = {
            'first_name': metadata.get('first_name', ''),
            'middle_names': metadata.get('middle_names', ''),
            'last_name': metadata.get('last_name', ''),
            'full_name': metadata.get('full_name', ''),
            'birth_date': metadata.get('birth_date', ''),
            'birth_place': metadata.get('birth_place', ''),
            'notes': metadata.get('notes', ''),
            'person_id': metadata.get('person_id', '')
        }
        st.info(f"✅ **Current name:** {existing_data['full_name']}")
        replace_existing_identity = st.checkbox(
            "I confirm that changing this face's person assignment is intentional.",
            key=f"tab_confirm_reassign_{face_id}",
            help="Without confirmation, this face keeps its current person label.",
        )
    else:
        replace_existing_identity = False

    # Get all existing persons for selection
    all_persons = get_cached_all_persons()  # Use cached version

    # Choice between new person or existing person
    st.subheader("🎯 Select a person")
    st.caption(
        "The name is assigned only to this face. Review similar faces "
        "in **Review groups**."
    )

    assignment_type = st.radio(
        "How would you like to assign the name?",
        ["Create a new person", "Select an existing person"],
        key="tab_assignment_type",
        horizontal=True
    )

    if assignment_type == "Select an existing person" and all_persons:
        # Existing person selection logic (simplified)
        person_options = {"-- Select a person --": None}
        for person in all_persons:
            display_name = f"{person['full_name']} ({person['face_count']} faces)"
            person_options[display_name] = person

        selected_person_display = st.selectbox(
            "Select a person:",
            options=list(person_options.keys()),
            key="tab_selected_person"
        )

        if selected_person_display and person_options[selected_person_display] is not None:
            selected_person = person_options[selected_person_display]

            # Ensure selected_person is not None
            if selected_person is None:
                st.error("Could not load person data.")
                return

            # Show person details
            st.success(f"**Name:** {selected_person['full_name']}")
            st.write(f"**Faces:** {selected_person['face_count']}")

            # Assign button
            if st.button("✅ Assign this person", type="primary", width='stretch', key="tab_assign_existing"):
                person_data = {
                    'first_name': selected_person['first_name'],
                    'middle_names': selected_person.get('middle_names', ''),
                    'last_name': selected_person['last_name'],
                    'birth_date': selected_person.get('birth_date', ''),
                    'birth_place': selected_person.get('birth_place', ''),
                    'notes': selected_person.get('notes', ''),
                }

                result = st.session_state.vector_store.assign_person_name(
                    face_id,
                    person_data,
                    selected_person['person_id'],
                    allow_reassignment=replace_existing_identity,
                    actor=authenticated_actor(),
                )

                if result:
                    st.success(f"Name assigned to {selected_person['full_name']}.")

                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    st.balloons()
                else:
                    st.error("❌ Could not assign the name.")

    else:
        # New person creation form (simplified)
        st.subheader("📝 Create a new person")

        col1, col2 = st.columns(2)

        with col1:
            first_name = st.text_input("First name *", value=existing_data.get('first_name', ''), key="tab_first_name")
            last_name = st.text_input("Last name *", value=existing_data.get('last_name', ''), key="tab_last_name")

        with col2:
            birth_date = st.text_input("Date of birth", value=existing_data.get('birth_date', ''), key="tab_birth_date")
            birth_place = st.text_input("Place of birth", value=existing_data.get('birth_place', ''), key="tab_birth_place")

        notes = st.text_area("Notes", value=existing_data.get('notes', ''), key="tab_notes", height=60)

        # Preview full name
        if first_name or last_name:
            name_parts = [p.strip() for p in [(first_name or ''), (last_name or '')] if p and p.strip()]
            full_name = ' '.join(name_parts)
            st.success(f"**Full name:** {full_name}")

        # Save button
        if st.button("💾 Save person", type="primary", width='stretch', key="tab_save_person"):
            if (first_name and first_name.strip()) or (last_name and last_name.strip()):
                person_data = {
                    'first_name': (first_name or '').strip(),
                    'last_name': (last_name or '').strip(),
                    'birth_date': (birth_date or '').strip(),
                    'birth_place': (birth_place or '').strip(),
                    'notes': (notes or '').strip()
                }

                full_name = ' '.join([p for p in [person_data['first_name'], person_data['last_name']] if p])

                person_id = str(uuid.uuid4())

                result = st.session_state.vector_store.assign_person_name(
                    face_id,
                    person_data,
                    person_id,
                    allow_reassignment=replace_existing_identity,
                    actor=authenticated_actor(),
                )

                if result:
                    st.success(f"Person '{full_name}' created and assigned.")

                    invalidate_persons_cache()
                    invalidate_gallery_cache()
                    st.balloons()
                else:
                    st.error(
                        "Could not create the person. A record with the same name "
                        "and birth details may already exist; select that person instead."
                    )
            else:
                    st.error("❌ Enter at least a first or last name.")
