.venv\Scripts\Activate.ps1

$env:API_URL = "http://localhost:80"; $env:API_TOKEN = "18_YgH-4oOQjFe6Ph0FtgUzM_oMolrnz"; $env:PYTHONIOENCODING = "utf-8"; & ".venv\Scripts\python.exe" documentation\Scripts\iris\delete_iris.py --yes; & ".venv\Scripts\python.exe" documentation\Scripts\wine\delete_wine.py --yes; & ".venv\Scripts\python.exe" documentation\Scripts\cancer\delete_cancer.py --yes
$env:API_URL = "http://localhost:80"; $env:API_TOKEN = "18_YgH-4oOQjFe6Ph0FtgUzM_oMolrnz"; $env:PYTHONIOENCODING = "utf-8"; & ".venv\Scripts\python.exe" init_data\seed_sample_data.py
