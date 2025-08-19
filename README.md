
# Face

Kompakte und pragmatische Gesichtserkennungs- und Suchanwendung (Streamlit-Frontend). Dieses Repository enthält Tools zum Extrahieren von Gesichts-Embeddings, zum Aufbau einer Vektor-Datenbank und zur Suche nach ähnlichen Gesichtern.

Kurz und knapp:
- Ziel: Lokale Recherche & Batch-Verarbeitung von Bildern zur Ähnlichkeitssuche.
- Oberfläche: `app.py` (Streamlit).

Installation
------------
1. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

2. Anwendung starten:

```bash
streamlit run app.py
```

Wichtige Dateien
---------------
- `app.py` – Streamlit UI
- `face_recognition_engine.py` – Embedding- und Matching-Logik
- `vector_store.py` – Speicherung/Suche von Embeddings
- `config.py` – Einstellungen
- `fast_process.py` – Batch-Verarbeitung

Tests
-----
Kurze Testskripte sind im Repo enthalten. Beispiel:

```bash
python test_basic_functionality.py
```

Rechtliche Hinweise (Kurzfassung)
--------------------------------
Diese Anwendung lädt und verarbeitet Bilder, die personenbezogene Daten (Gesichter) enthalten können. Die folgenden Hinweise sind keine Rechtsberatung:

- EU / Deutschland: Verarbeitung biometrischer Daten kann der DSGVO (Art. 9) unterliegen und erfordert in vielen Fällen eine ausdrückliche Einwilligung.
- USA: Es gibt keine einheitliche Regelung; einzelne Bundesstaaten (z. B. Illinois) haben strenge Biometrie-Gesetze (BIPA).
- Urheberrecht: Bilder sind häufig urheberrechtlich geschützt; prüfen Sie Lizenzen vor Download/Weiterverwendung.

Best Practices
--------------
1. Nutzen Sie nur Bilder, für die Sie Rechtssicherheit haben (Einwilligung, Lizenz).
2. Minimieren Sie Datenspeicherung und pseudonymisieren Sie Daten, wenn möglich.
3. Dokumentieren Sie Zweck, Rechtsgrundlage und Sicherheitsmaßnahmen.
4. Respektieren Sie die Nutzungsbedingungen (ToS) von Webseiten und APIs.
5. Bei Unsicherheit Rechtsberatung einholen.

Mitwirken
---------
PRs willkommen. Bitte erstellen Sie Issues für größere Änderungen.

Lizenz
------
Siehe `LICENSE` (falls vorhanden) oder kontaktieren Sie die/den Projektverantwortliche/n.

---

Wenn du eine noch knappere Fassung oder zusätzlich eine englische Übersetzung möchtest, sage kurz Bescheid.


## 🤝 Beitrag leisten

Verbesserungsvorschläge und Pull Requests sind willkommen!

## 📄 Lizenz

Siehe LICENSE-Datei für Details.