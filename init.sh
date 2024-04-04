echo "---------------- Virtual Environment Creation Started ----------------"
python -m venv .venv
source .venv/Scripts/activate
echo "---------------- Virtual Environment Created and Activated ----------------"

echo "---------------- Package installation Started ----------------"
pip -q install -r requirements.txt
echo "---------------- Package installation Finished  ----------------"

echo "---------------- Running the Code  ----------------"
python main.py
echo "---------------- Exiting the Code  ----------------"

