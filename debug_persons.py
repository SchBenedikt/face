#!/usr/bin/env python3
"""
Debug script to check person assignment in ChromaDB
"""

import logging
from vector_store import FaceVectorStore
import config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    print("=== Debug Persons in ChromaDB ===")
    
    # Initialize vector store
    vector_store = FaceVectorStore()
    
    print("\n1. Getting all persons...")
    persons = vector_store.get_all_persons()
    print(f"Found {len(persons)} persons")
    
    if not persons:
        print("\n2. Let's check raw metadata to see what we have...")
        
        # Get raw data from collection
        results = vector_store.collection.get(include=['metadatas'])
        
        print(f"Total faces in database: {len(results.get('metadatas', []))}")
        
        # Sample first 10 faces
        print("\n3. Sample of face metadata (first 10):")
        for i, metadata in enumerate(results.get('metadatas', [])[:10]):
            print(f"\nFace {i}:")
            print(f"  Keys: {list(metadata.keys())}")
            
            if 'person_id' in metadata:
                print(f"  person_id: {metadata['person_id']} (type: {type(metadata['person_id'])})")
            if 'first_name' in metadata:
                print(f"  first_name: {metadata['first_name']}")
            if 'last_name' in metadata:  
                print(f"  last_name: {metadata['last_name']}")
        
        # Check for any faces with person_id
        print("\n4. Looking for faces with person_id...")
        person_id_count = 0
        for i, metadata in enumerate(results.get('metadatas', [])):
            if 'person_id' in metadata and metadata['person_id'] is not None:
                raw_person_id = metadata['person_id']
                person_id_str = str(raw_person_id).strip()
                
                if person_id_str and person_id_str not in ["None", "null", "", "nan"]:
                    person_id_count += 1
                    if person_id_count <= 5:  # Show first 5
                        print(f"  Face {i}: person_id = '{person_id_str}' (first_name: {metadata.get('first_name', 'N/A')})")
        
        print(f"\nTotal faces with valid person_id: {person_id_count}")
        
        if person_id_count == 0:
            print("\n5. No faces have person_id assigned! This explains why get_all_persons returns 0.")
            print("   You need to assign names to faces first.")
            
            # Let's find a face to test with
            print("\n6. Finding a face to test assignment...")
            if results.get('ids'):
                test_face_id = results['ids'][0]
                print(f"   Test face ID: {test_face_id}")
                
                # Test assigning a person
                test_person_data = {
                    'first_name': 'Test',
                    'middle_names': '',
                    'last_name': 'Person',
                    'birth_date': '1980-01-01',
                    'birth_place': 'München',
                    'notes': 'Debug test person'
                }
                
                print(f"   Assigning test person to face {test_face_id}...")
                result = vector_store.assign_person_name(test_face_id, test_person_data)
                
                if result:
                    print(f"   ✅ Test assignment successful!")
                    
                    # Now try get_all_persons again
                    persons_after = vector_store.get_all_persons()
                    print(f"   After assignment: {len(persons_after)} persons found")
                    
                    if persons_after:
                        print(f"   First person: {persons_after[0]}")
                else:
                    print(f"   ❌ Test assignment failed!")
    
    else:
        print(f"\n✅ Found {len(persons)} persons:")
        for person in persons[:5]:  # Show first 5
            print(f"  - {person['full_name']} ({person['face_count']} faces)")
            
            # Test get_faces_by_person_id for this person
            print(f"    Testing get_faces_by_person_id for {person['person_id']}...")
            person_faces = vector_store.get_faces_by_person_id(person['person_id'])
            print(f"    Found {len(person_faces)} faces for this person")
            
            if person_faces:
                first_face = person_faces[0]
                print(f"    First face ID: {first_face['face_id']}")
                print(f"    First face metadata keys: {list(first_face.get('metadata', {}).keys())}")
            else:
                print(f"    ❌ No faces returned by get_faces_by_person_id!")
            break  # Just test first person

if __name__ == "__main__":
    main()