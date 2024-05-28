import pandas as pd
import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import gspread
import os.path
import importlib.resources as pkg_resources
from sportstradamus import creds

# Authorize the gspread API
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
]
cred = None

# Check if token.json file exists and load credentials
if os.path.exists((pkg_resources.files(creds) / "token.json")):
    cred = Credentials.from_authorized_user_file(
        (pkg_resources.files(creds) / "token.json"), SCOPES
    )

# If no valid credentials found, let the user log in
if not cred or not cred.valid:
    if cred and cred.expired and cred.refresh_token:
        cred.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            (pkg_resources.files(creds) / "credentials.json"), SCOPES
        )
        cred = flow.run_local_server(port=0)

    # Save the credentials for the next run
    with open((pkg_resources.files(creds) / "token.json"), "w") as token:
        token.write(cred.to_json())

gc = gspread.authorize(cred)

url = "https://stats.underdogfantasy.com/v1/slates/71dbd531-99ea-4a1a-b7b4-f4f514957d7e/scoring_types/ccf300b0-9197-5951-bd96-cba84ad71e86/appearances"

res = requests.get(url).json()

api_df = pd.json_normalize(res, record_path='appearances', record_prefix='appearances.')
api_df['appearances.badges'] = ""
api_df = api_df[[col for col in api_df.columns if col not in ['appearances.score', 'appearances.team_id']]+['appearances.score', 'appearances.team_id']]
api_df.update(api_df.apply(pd.to_numeric, errors='coerce'))

wks = gc.open("Best Ball Made EZ").worksheet("UD Apps")
wks.clear()
wks.update([api_df.columns.values.tolist()] + api_df.values.tolist())