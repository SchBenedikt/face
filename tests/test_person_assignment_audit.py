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

