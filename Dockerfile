# Utilise directement l'image hapnan qui fonctionne
FROM hapnan/whisperx-worker:v1.0.6

# Remplacer seulement les fichiers modifiés
COPY src/rp_handler.py /rp_handler.py
COPY src/rp_schema.py /rp_schema.py

CMD ["python3", "-u", "/rp_handler.py"]
