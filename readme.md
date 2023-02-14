python -m venv venv
./venv/Scripts/Activate.pst
python -r requirements.txt
python enthropies.py
python assignment_2.py
cd web
python -m mkdocs build
to update pages run  python -m mkdocs serve for online edit/view 
result is in site directory