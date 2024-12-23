# Stappenplan: Automatiseren met Selenium, OpenCV en YOLO

1. **Open een browservenster**  
   - Gebruik Selenium Firefox Driver.  
   - Zorg dat geckodriver correct is ingesteld.  

2. **Capture het scherm**  
   - Maak een screenshot van het browservenster met OpenCV en PIL.  
   - Verwerk de screenshot naar een RGB-frame voor beeldherkenning.  

3. **Start het spel**  
   - Automatiseer het vinden en klikken van de startknop met Selenium.  

4. **Model herkent de huidige staat**  
   - Gebruik een YOLOv11-model (Ultralytics) om de huidige staat van het spel te analyseren.  
   - Detecteer objecten en bepaal klassen zoals obstakels

5. **Bereken de actie**  
   - Analyseer de detectieresultaten om de te nemen actie te bepalen (bijvoorbeeld springen of hurken).  

6. **Voer de actie uit**  
   - Gebruik Selenium om toetsen zoals spatiebalk of pijltoetsen te simuleren.  

## Benodigde libraries
- Selenium  
- OpenCV  
- NumPy  
- PIL (Python Imaging Library)  
- Ultralytics (YOLOv8)