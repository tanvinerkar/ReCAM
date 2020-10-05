cd trail_data
python split_data.py
cd ..
pip install -r requirements.txt
python -m spacy download en
python Run_GAReader.py --epoch_num 1