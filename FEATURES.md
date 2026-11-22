# Face Recognition System - Features & Documentation

## ✅ Currently Implemented Features

### 🔍 **Face Search**
- Upload reference image with face detection
- Facenet512 embedding search with adjustable cosine-distance threshold (default: 0.70)
- Trefferwerte zeigen Cosine-Ähnlichkeit, keine Match-Wahrscheinlichkeit
- Multi-face detection with selection interface
- Full-screen view with highlighted face
- Detailed facial analysis (age, gender, emotions, ethnicity)
- **New: Assign person names** to found faces

### 👥 **Face Gallery**
- Overview of all faces in the database
- Pagination with customizable view (20-500 faces per page)
- Metadata display with file info and position
- Delete functions for incorrectly detected faces:
  - Individual deletion with confirmation
  - Batch deletion of entire pages
  - Intelligent quality control (automatically detect low quality)
- **New: Assign person names** in the gallery

### 🏷️ **Name Gallery (Namen-Galerie)** - NEU!
- Übersicht aller vergebenen Personen-Namen
- Anzeige aller Gesichter pro Person
- Namen-Suche und Sortierung
- Namen von einzelnen Gesichtern oder ganzen Personen entfernen
- Statistiken über benannte Gesichter

### 🧩 **Personengruppen** - NEU!
- Gruppiert unbenannte Gesichter anhand ihrer Embeddings (Cosine-Ähnlichkeit)
- Standard-Schwelle 0,68 Cosine-Ähnlichkeit (Facenet512); im UI einstellbar
- Zeigt Gruppen mit Vorschaubildern, Gruppengröße und durchschnittlicher Ähnlichkeit
- Schlägt eine passende bestehende Person vor, ändert aber nichts ohne Bestätigung
- Weist eine bestätigte Gruppe gesammelt einer neuen oder bestehenden Person zu

### 🧬 **Facial Attribute Analysis (Gesichtsanalyse)**
- **Alter**: Modellschätzung in Jahren; keine garantierte Genauigkeit
- **Geschlecht**: Modellschätzung mit den vom Analysemodell unterstützten Klassen
- **Emotionen**: Modellschätzungen für mehrere Emotionsklassen
- **Ethnizität**: Modellschätzung aus den vom Analysemodell unterstützten Kategorien
- Modal-Popup für übersichtliche Darstellung aller Analysedaten

### 🏷️ **Person Namen System** - NEU!
- **Eindeutige Person IDs**: Jede Person wird intern über eine UUID identifiziert
- **Vor- und Nachname**: Separate Eingabefelder für strukturierte Namensvergabe
- **Automatische Namensverbreitung**: Gesichter ab etwa 0,68 Cosine-Ähnlichkeit (Facenet512) erhalten den gleichen Namen; bei ähnlichen Treffern zu einer anderen benannten Person wird nichts zugewiesen
- **Namen-Verwaltung**: Namen können jederzeit entfernt oder geändert werden
- **Such-Integration**: Namen werden in Face Search und Gallery angezeigt

### 🔧 **Löschfunktionen**
- **Einzellöschung**: Mit doppelter Bestätigung
- **Batch-Löschung**: Ganze Seiten oder nach Qualitätskriterien
- **Qualitätskontrolle**: Automatische Erkennung problematischer Gesichter:
  - Zu kleine Gesichter (< 30x30 Pixel)
  - Ungewöhnliche Proportionen
  - Sehr kleine Flächen (< 900 Pixel²)

## 🎯 Verwendung des Namen-Systems

### Namen zuweisen:
1. Gehen Sie zu **Face Search** oder **Face Gallery**
2. Klicken Sie bei einem Gesicht auf **"🏷️ Namen zuweisen"**
3. Geben Sie Vor- und Nachname in separate Felder ein
4. Klicken Sie **"💾 Namen speichern"**
5. Ähnliche Gesichter (Standard: ≥0,68 Cosine) erhalten automatisch denselben Namen

### Namen verwalten:
1. Gehen Sie zur **🏷️ Name Gallery**
2. Sehen Sie alle Personen und deren Gesichter
3. Entfernen Sie Namen bei Fehlzuweisungen
4. Durchsuchen Sie alle Gesichter einer Person

## ⚠️ Wichtige Hinweise

### Gesichtsanalyse
Die KI-basierte Gesichtsanalyse dient ausschließlich zu **Demonstrations- und Forschungszwecken**. 
Die Ergebnisse sind Schätzungen und sollten nicht für Identifikation, Diskriminierung oder Entscheidungsfindung verwendet werden.

### Datenschutz und Rechtsrahmen
- Gesichtsvektoren werden im konfigurierten Datenverzeichnis gespeichert.
- Die Anwendung prüft weder Einwilligungen noch rechtliche Voraussetzungen.
- Kläre vor der Verarbeitung fremder Bilder, welche Regeln für deinen Einsatz gelten.

## 🚀 Technische Details

### Datenbank-Schema
- **Face IDs**: Eindeutige Identifikation jedes erkannten Gesichts
- **Person IDs**: UUID-basierte Personenidentifikation (unabhängig vom Namen)
- **Metadaten**: Speicherung von Namen, Bildpfad, Position, Zeitstempel
- **Embeddings**: 512-dimensionale Facenet512-Vektoren für die aktive Ähnlichkeitssuche

### Ähnlichkeitsberechnung
- Die aktive Suche vergleicht unit-normalisierte Facenet512-Vektoren mit Cosine-Ähnlichkeit
- Suchstandard: 0,70 Cosine; automatische Namenszuweisung: 0,68, per `FACE_AUTO_ASSIGN_THRESHOLD` übersteuerbar
- Ähnlichkeitswerte sind Ranking-Scores, keine Wahrscheinlichkeiten oder bestätigten Identitäten

Gesichts- und Attributerkennung kann falsche Ergebnisse liefern. Für Benchmark-Angaben siehe nur reproduzierbare Messungen mit einem geeigneten, beschrifteten Datensatz.
