import json


def test_person_assignment_and_removal_keep_review_history(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    face_id = "face_3_0"
    assert store.assign_person_name(
        face_id,
        {"first_name": "Alex", "last_name": "Example"},
        person_id="person-audit-test",
    ) == "person-audit-test"

    assigned = store.get_face_by_id(face_id)["metadata"]
    first_history = json.loads(assigned["assignment_history"])
    assert len(first_history) == 1
    assert first_history[0]["reviewed"] is True
    assert first_history[0]["from_name"] == ""
    assert first_history[0]["to_name"] == "Alex Example"

    assert store.remove_person_name(face_id) is True
    removed = store.get_face_by_id(face_id)["metadata"]
    history = json.loads(removed["assignment_history"])
    assert len(history) == 2
    assert history[1]["action"] == "removed"
    assert history[1]["from_name"] == "Alex Example"
    assert removed["full_name"] == ""


def test_undo_assignment_restores_previous_person_fields(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    face_id = "face_3_0"
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
    ) == "new-person"
    assigned = store.get_face_by_id(face_id)["metadata"]
    assigned_event = json.loads(assigned["assignment_history"])[-1]

    outcome = store.undo_person_assignments({face_id: assigned_event["at"]})

    assert outcome["undone"] == [face_id]
    restored = store.get_face_by_id(face_id)["metadata"]
    assert restored["person_id"] == "previous-person"
    assert restored["first_name"] == "Before"
    assert restored["full_name"] == "Before Label"
    assert restored["birth_date"] == "1980-01-02"
    assert restored["birth_place"] == "Berlin"
    assert restored["notes"] == "Keep this note"
    assert json.loads(restored["assignment_history"])[-1]["action"] == "undo"


def test_old_undo_cannot_overwrite_a_later_assignment(seeded_data):
    from vector_store import FaceVectorStore

    store = FaceVectorStore()
    face_id = "face_3_0"
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

