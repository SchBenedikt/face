# Manual Testing Guide for Person Name Assignment System

## 🧪 Testing Workflow

### Prerequisites
1. Start the application: `streamlit run app.py`
2. Have some faces already in the database (either from uploads or web scraping)

## Test Scenarios

### 1. Name Assignment in Face Search
1. Go to **🔍 Face Search**
2. Upload an image with a face
3. When results appear, click **🏷️ Namen zuweisen** on any face
4. **Expected**: Modal dialog opens with:
   - Face preview image
   - Face ID and filename displayed
   - Two text fields: "Vorname" and "Nachname"
   - "💾 Namen speichern" button
   - "❌ Schließen" button

5. Enter "John" in Vorname, "Doe" in Nachname
6. Click "💾 Namen speichern"
7. **Expected**: 
   - Success message appears
   - Modal closes automatically
   - Face now shows "👤 **John Doe**" below the image
   - Similar faces (>80% match) automatically get the same name

### 2. Name Assignment in Face Gallery
1. Go to **👥 Face Gallery**
2. Find any face without a name
3. Click **🏷️ Namen** button
4. **Expected**: Same modal as above opens
5. Test the same workflow as Face Search

### 3. Name Gallery Page
1. Go to **🏷️ Name Gallery** (new page in navigation)
2. **Expected to see**:
   - Header: "🏷️ Name Gallery"
   - Statistics: Number of persons, named faces, average faces per person
   - Search box for names
   - Sort dropdown (Name A-Z, Z-A, Faces ascending/descending)
   - Expandable sections for each person showing:
     - Person's full name
     - Number of faces
     - "🔍 Alle Gesichter anzeigen" button
     - "🗑️ Namen löschen" button

3. Click "🔍 Alle Gesichter anzeigen" for a person
4. **Expected**: 
   - Grid of all faces for that person
   - Each face shows thumbnail and basic info
   - "🗑️ Name entfernen" button for individual faces
   - "❌ Ansicht schließen" button

### 4. Automatic Name Propagation Test
1. Assign a name to a face in Face Search
2. Look for the success message mentioning auto-assigned faces
3. Go to Name Gallery and check if the person has multiple faces
4. **Expected**: Similar faces (>80% similarity) should automatically receive the same name

### 5. Name Removal Test
1. In Name Gallery, click "🗑️ Namen löschen" for a person
2. **Expected**: Warning message asking for confirmation
3. Click again to confirm
4. **Expected**: All faces of that person lose their name assignment
5. Person disappears from Name Gallery

### 6. Individual Face Name Removal
1. In Name Gallery, expand a person with multiple faces
2. Click "🗑️ Name entfernen" on one face
3. **Expected**: Only that face loses the name, others keep it

## Expected UI Changes

### Face Search Results
- **Before**: [Thumbnail] [🖼️ Ganzes Bild] [🧬 Analyse]
- **After**: [Thumbnail] [👤 **John Doe**] [🖼️ Ganzes Bild] [🧬 Analyse] [🏷️ Namen zuweisen]

### Face Gallery
- **Before**: [Thumbnail] [🖼️ Ganzes Bild] [🧬 Analyse] [🗑️ Löschen]
- **After**: [Thumbnail] [👤 **John Doe**] [🖼️ Ganzes Bild] [🧬 Analyse] [🏷️ Namen] [🗑️ Löschen]

### Navigation Sidebar
- **Added**: "🏷️ Name Gallery" between "👥 Face Gallery" and "🔧 Duplicate Manager"

## Error Scenarios to Test

### 1. Empty Name Assignment
1. Try to save without entering any name
2. **Expected**: Warning "⚠️ Bitte mindestens Vor- oder Nachname eingeben"

### 2. Name Removal Confirmation
1. Try to remove a name from a person with many faces
2. **Expected**: Confirmation dialog prevents accidental removal

### 3. Missing Face Data
1. Try to access Name Gallery with no named faces
2. **Expected**: Info message suggesting to assign names first

## Performance Tests

### 1. Large Database
1. With 1000+ faces in database
2. Assign names to various faces
3. **Expected**: Name Gallery loads quickly and search works smoothly

### 2. Auto-Assignment
1. Assign name to face with many similar faces
2. **Expected**: Auto-assignment happens quickly, success message shows count

## Database Verification

After testing, check that:
1. Face metadata includes `person_id`, `first_name`, `last_name`, `full_name`
2. Multiple faces can share the same `person_id`
3. Names can be removed (fields set to None/null)
4. Person IDs are unique UUIDs

## Success Criteria

✅ All modals open and close properly  
✅ Names are saved and displayed correctly  
✅ Auto-assignment works for similar faces (>80%)  
✅ Name removal works for individuals and entire persons  
✅ Name Gallery displays all persons with correct statistics  
✅ Search and sort functions work in Name Gallery  
✅ UI shows person names prominently in search results and gallery  
✅ Error handling works for edge cases  
✅ Performance is acceptable with large datasets  

## Known Limitations

- Auto-assignment threshold is fixed at 80% (could be made configurable)
- Names are stored as simple text (no special characters validation)
- Person IDs are not exposed in UI (internal use only)
- No export function for person name data (future enhancement)