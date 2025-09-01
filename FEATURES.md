# Face Recognition App - Features & Documentation

## ✅ Aktuell implementierte Funktionen

### 🔍 **Face Search (Gesichtssuche)**
- Upload eines Referenzbildes mit Gesichtserkennung
- Ähnlichkeitssuche in der Datenbank mit einstellbarem Threshold
- Mehrfach-Gesichtserkennung mit Auswahlmöglichkeit
- Vollbild-Ansicht mit markiertem Gesicht
- Detaillierte Gesichtsanalyse (Alter, Geschlecht, Emotionen, Ethnie)
- **Neu: Person Namen zuweisen** für gefundene Gesichter

### 👥 **Face Gallery (Gesichts-Galerie)**
- Übersicht aller Gesichter in der Datenbank
- Paginierung und anpassbare Ansicht (20-500 Gesichter pro Seite)
- Metadaten-Anzeige mit Dateinfo und Position
- Löschfunktionen für falsch erkannte Gesichter:
  - Einzellöschung mit Bestätigung
  - Batch-Löschung ganzer Seiten
  - Intelligente Qualitätskontrolle (niedrige Qualität automatisch erkennen)
- **Neu: Person Namen zuweisen** in der Galerie

### 🏷️ **Name Gallery (Namen-Galerie)** - NEU!
- Übersicht aller vergebenen Personen-Namen
- Anzeige aller Gesichter pro Person
- Namen-Suche und Sortierung
- Namen von einzelnen Gesichtern oder ganzen Personen entfernen
- Statistiken über benannte Gesichter

### 🧬 **Facial Attribute Analysis (Gesichtsanalyse)**
- **Alter**: Geschätztes Alter in Jahren (±4.65 Jahre Genauigkeit)
- **Geschlecht**: Männlich/Weiblich Wahrscheinlichkeiten (97.44% Genauigkeit)
- **Emotionen**: 7 Emotionsklassen mit Konfidenzwerten
- **Ethnische Herkunft**: 6 Hauptkategorien mit Wahrscheinlichkeitsverteilung
- Modal-Popup für übersichtliche Darstellung aller Analysedaten

### 🏷️ **Person Namen System** - NEU!
- **Eindeutige Person IDs**: Jede Person wird intern über eine UUID identifiziert
- **Vor- und Nachname**: Separate Eingabefelder für strukturierte Namensvergabe
- **Automatische Namensverbreitung**: Gesichter mit >80% Ähnlichkeit erhalten automatisch den gleichen Namen
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
5. Ähnliche Gesichter (>80%) erhalten automatisch denselben Namen

### Namen verwalten:
1. Gehen Sie zur **🏷️ Name Gallery**
2. Sehen Sie alle Personen und deren Gesichter
3. Entfernen Sie Namen bei Fehlzuweisungen
4. Durchsuchen Sie alle Gesichter einer Person

## ⚠️ Wichtige Hinweise

### Gesichtsanalyse
Die KI-basierte Gesichtsanalyse dient ausschließlich zu **Demonstrations- und Forschungszwecken**. 
Die Ergebnisse sind Schätzungen und sollten nicht für Identifikation, Diskriminierung oder Entscheidungsfindung verwendet werden.

### Rechtliche Hinweise
- EU/Deutschland: Verarbeitung biometrischer Daten kann der DSGVO (Art. 9) unterliegen
- Nutzung nur mit entsprechenden Einwilligungen und Rechtsgrundlagen
- Bei Unsicherheit Rechtsberatung einholen

## 🚀 Technische Details

### Datenbank-Schema
- **Face IDs**: Eindeutige Identifikation jedes erkannten Gesichts
- **Person IDs**: UUID-basierte Personenidentifikation (unabhängig vom Namen)
- **Metadaten**: Speicherung von Namen, Bildpfad, Position, Zeitstempel
- **Embeddings**: 128/512-dimensionale Gesichtsvektoren für Ähnlichkeitssuche

### Ähnlichkeitsberechnung
- Ensemble-Ansatz mit mehreren Metriken (Cosine, Euclidean, Correlation)
- Konfigurierbare Schwellenwerte (Standard: 80% für Auto-Namenszuweisung)
- Konfidenzwerte für verbesserte Genauigkeit

Alle Funktionen sind vollständig implementiert und einsatzbereit! 🎉