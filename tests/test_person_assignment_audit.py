import json


def _clear_person_assignment(store, face_id):
    metadata = (store.get_face_by_id(face_id) or {}).get("metadata") or {}
    if metadata.get("person_id"):
        assert store.remove_person_name(face_id)


def test_person_assignment_and_removal_keep_review_history(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    face_id = "face_3_0"
    _clear_person_assignment(store, face_id)
    assert store.assign_person_name(
        face_id,
        {"first_name": "Alex", "last_name": "Example"},
        person_id="person-audit-test",
        actor="alice@example.com",
    ) == "person-audit-test"

    assigned = store.get_face_by_id(face_id)["metadata"]
    first_history = json.loads(assigned["assignment_history"])
    assert len(first_history) == 1
    assert first_history[0]["reviewed"] is True
    assert first_history[0]["from_name"] == ""
    assert first_history[0]["to_name"] == "Alex Example"
    assert first_history[0]["source"] == "manual_review"
    assert first_history[0]["actor"] == "alice@example.com"

    assert store.remove_person_name(face_id, actor="alice@example.com") is True
    removed = store.get_face_by_id(face_id)["metadata"]
    history = json.loads(removed["assignment_history"])
    assert len(history) == 2
    assert history[1]["action"] == "removed"
    assert history[1]["actor"] == "alice@example.com"
    assert history[1]["from_name"] == "Alex Example"
    assert removed["full_name"] == ""


def test_undo_assignment_restores_previous_person_fields(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    face_id = "face_3_0"
    _clear_person_assignment(store, face_id)
    assert store.update_face_metadata(face_id, {
        "first_name": "Before",
        "last_name": "Label",
        "full_name": "Before Label",
        "person_id": "previous-person",
        "birth_date": "1980-01-02",
        "birth_place": "Berlin",
        "notes": "Keep this note",
    })
    assert store.assign_person_name(
        face_id,
        {"first_name": "After", "last_name": "Review"},
        person_id="new-person",
        allow_reassignment=True,
    ) == "new-person"
    assigned = store.get_face_by_id(face_id)["metadata"]
    assigned_event = json.loads(assigned["assignment_history"])[-1]

    outcome = store.undo_person_assignments(
        {face_id: assigned_event["at"]}, actor="alice@example.com"
    )

    assert outcome["undone"] == [face_id]
    restored = store.get_face_by_id(face_id)["metadata"]
    assert restored["person_id"] == "previous-person"
    assert restored["first_name"] == "Before"
    assert restored["full_name"] == "Before Label"
    assert restored["birth_date"] == "1980-01-02"
    assert restored["birth_place"] == "Berlin"
    assert restored["notes"] == "Keep this note"
    undone_event = json.loads(restored["assignment_history"])[-1]
    assert undone_event["action"] == "undo"
    assert undone_event["actor"] == "alice@example.com"


def test_old_undo_cannot_overwrite_a_later_assignment(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    face_id = "face_3_0"
    _clear_person_assignment(store, face_id)
    assert store.assign_person_name(
        face_id,
        {"first_name": "First", "last_name": "Person"},
        person_id="first-person",
    ) == "first-person"
    first = store.get_face_by_id(face_id)["metadata"]
    stale_event_at = json.loads(first["assignment_history"])[-1]["at"]
    assert store.assign_person_name(
        face_id,
        {"first_name": "Later", "last_name": "Person"},
        person_id="later-person",
        allow_reassignment=True,
    ) == "later-person"

    stale_undo = store.undo_person_assignments({face_id: stale_event_at})
    assert stale_undo["undone"] == []
    assert stale_undo["remaining"] == {face_id: stale_event_at}
    current = store.get_face_by_id(face_id)["metadata"]
    assert current["person_id"] == "later-person"
    assert current["full_name"] == "Later Person"

    repeated_stale_undo = store.undo_person_assignments(stale_undo["remaining"])
    assert repeated_stale_undo["undone"] == []
    current = store.get_face_by_id(face_id)["metadata"]
    assert current["person_id"] == "later-person"


def test_person_reassignment_requires_explicit_confirmation(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    face_id = "face_3_0"
    _clear_person_assignment(store, face_id)
    assert store.assign_person_name(
        face_id,
        {"first_name": "First", "last_name": "Person"},
        person_id="first-person",
    ) == "first-person"

    assert store.assign_person_name(
        face_id,
        {"first_name": "Second", "last_name": "Person"},
        person_id="second-person",
    ) is None
    assert store.get_face_by_id(face_id)["metadata"]["person_id"] == "first-person"

    assert store.assign_person_name(
        face_id,
        {"first_name": "Second", "last_name": "Person"},
        person_id="second-person",
        allow_reassignment=True,
    ) == "second-person"
    history = json.loads(store.get_face_by_id(face_id)["metadata"]["assignment_history"])
    assert history[-1]["from_person_id"] == "first-person"
    assert history[-1]["to_person_id"] == "second-person"


def test_one_person_id_cannot_silently_split_person_details(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    person_data = {"first_name": "Alex", "last_name": "Example"}
    _clear_person_assignment(store, "face_3_0")
    _clear_person_assignment(store, "face_3_1")
    assert store.assign_person_name(
        "face_3_0", person_data, person_id="consistent-person"
    ) == "consistent-person"

    assert store.assign_person_name(
        "face_3_1",
        {"first_name": "Different", "last_name": "Example"},
        person_id="consistent-person",
    ) is None
    assert store.assign_person_name(
        "face_3_1", person_data, person_id="consistent-person"
    ) == "consistent-person"
    assert store.get_face_by_id("face_3_1")["metadata"]["full_name"] == "Alex Example"


def test_duplicate_name_and_birth_details_require_existing_person_selection(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    _clear_person_assignment(store, "face_3_0")
    _clear_person_assignment(store, "face_3_1")
    person_data = {
        "first_name": "Jordan",
        "last_name": "Example",
        "birth_date": "1990-04-05",
        "birth_place": "Berlin",
    }
    assert store.assign_person_name(
        "face_3_0", person_data, person_id="jordan-existing"
    ) == "jordan-existing"

    assert store.assign_person_name(
        "face_3_1", person_data, person_id="jordan-duplicate"
    ) is None
    assert not store.get_face_by_id("face_3_1")["metadata"].get("person_id")

    different_person = {**person_data, "birth_date": "1994-04-05"}
    assert store.assign_person_name(
        "face_3_1", different_person, person_id="jordan-different"
    ) == "jordan-different"
