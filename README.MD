# Keystroke Injection Attack
Key files and folders:
 - keystroke_inject_attack.py - Keystroke Injetion Attack library that directly modifies preprocessed keystroke sequences
 - malware_emulator folder - This folder contains all malware emulators used during the empirical analysis.
 - keystroke_models - This folder contain codes to train and test models, and the three trained models.
 - feature_extraction - This folder contains all the code used to extract features from the dataset collected by Dhakal et al. This folder is a bit messy.

Please reach out to me if you would like to have access to the preprocessed datasets. 


# Keystroke Event Viewer
1. Setting up the **Chrome Browser Extension** to inject keystrokes
2. Starting the **Web-App** that detects and stores keystrokes
3. Running the **Feature Extraction** script

## Chrome Broswer Extension
The Chrome Broswer Extension injects keystrokes. Currently, when you type any number keys, it simulates keydown and keyup of key 'q'. 
1. Go to ```chrome://extensions/``` on Google Chrome browser.
2. Toggle ```developer mode``` on.
3. Click ```Load unpacked``` and select ```key_inject_brows_extension``` folder.

## Web-App
The Web-App consists a server that stores your keystrokes every 10 seconds and stores it inside ```keystrokes.csv``` file.
1. Create virtual environment to run the python server.
    * ```rm -rf venv```
    * ```python3 -m venv venv```
    * Linux or Mac: ```source venv/bin/activate``` Windows: ```venv\Scripts\activate```
2. Install requirements in venv ```pip install -r requirements.txt``` or ```pip3 install -r requirements.txt```.
3. Run the server ```python backend.py```.
4. Open ```index.html```.
5. Any keystrokes made to this web-app is stored into ```keystrokes.csv``` file.

## Feature Extraction
After collecting keystrokes (```keystrokes.csv```), keystroke features can be extracted with ```keystroke_extraction.py```.
1. Run ```python feature_extraction.py```.
2. Keystroke features are extracted into ```keystroke_features.csv``` file.