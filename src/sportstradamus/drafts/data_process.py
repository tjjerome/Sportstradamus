# %%
import pandas as pd
from sportstradamus.drafts import data
# import data
import importlib.resources as pkg_resources
import numpy as np
from datetime import datetime
import nfl_data_py as nfl
from tqdm import tqdm
import pickle

# %%
rosters = nfl.import_rosters([2021, 2022])
rosters = rosters.loc[rosters.position.isin(['QB', 'WR', 'TE', 'RB'])]
rosters = rosters[['player_name', 'season', 'team']]
rosters = rosters.set_index(
    rosters['player_name'] + ', ' + rosters['season'].astype(str))
rosters = rosters.to_dict()['team']

# %%
schedules = nfl.import_schedules([2021, 2022])
schedules = schedules.loc[schedules.week.isin([15, 16, 17])]
schedules = schedules[['season', 'week', 'away_team', 'home_team']]

# %%
data_list = [f.name for f in pkg_resources.files(
    data).iterdir() if ".csv" in f.name and "Regular" in f.name]
# %%
df = pd.DataFrame()
for data_str in tqdm(data_list, desc='Loading Data Files', unit='file'):
    if df.empty:
        df = pd.read_csv(pkg_resources.files(data) / data_str)
    else:
        df = pd.concat([df, pd.read_csv(pkg_resources.files(data) / data_str)])

# %%
df['season'] = df['draft_time'].str[:4]
df['team'] = (df['player_name'] + ', ' + df['season']
              ).map(rosters).fillna("None")
# %%
picks = df.overall_pick_number.replace(0, 216).to_numpy()
value = df.pick_points.to_numpy()
fit = np.polyfit(np.log(picks), value, 1)
df['draft_capital_spent'] = fit[0]*np.log(df['overall_pick_number'])+fit[1]
df['projection_draft_capital'] = fit[0] * \
    np.log(df['projection_adp'].replace(0, 216))+fit[1]
df['draft_capital_gained'] = df['projection_draft_capital'] - \
    df['draft_capital_spent']

# %%
draft_date = pd.to_datetime(df['draft_time'].str[:10], format='%Y-%m-%d')
df['days_until_season_start'] = (datetime.strptime(
    '2022-09-08', '%Y-%m-%d') - draft_date).dt.days
df.loc[df['days_until_season_start'] > 200, 'days_until_season_start'] = \
    df.loc[df['days_until_season_start'] >
           200, 'days_until_season_start'] - 363

# %%
teams = df.groupby(['draft_id', 'pick_order'])
teams = teams[['player_name', 'position_name', 'team', 'team_pick_number', 'roster_points', 'draft_capital_spent',
               'draft_capital_gained', 'days_until_season_start']]


# %%
team_df = {}
with tqdm(total=teams.ngroups, desc="Analyzing Teams", unit="team") as pbar:
    for order, team in teams:
        team = team.sort_values('team_pick_number')
        QBTeams = team.loc[team['position_name'] == 'QB', 'team'].to_list()[:3]
        if len(QBTeams) < 3:
            QBTeams = QBTeams + ['NA']*(3-len(QBTeams))

        QB1WR = team.loc[(team['team'] == QBTeams[0]) & (
            team['position_name'] == 'WR'), 'draft_capital_spent']
        QB1RB = team.loc[(team['team'] == QBTeams[0]) & (
            team['position_name'] == 'RB'), 'draft_capital_spent']
        QB1TE = team.loc[(team['team'] == QBTeams[0]) & (
            team['position_name'] == 'TE'), 'draft_capital_spent']
        QB2WR = team.loc[(team['team'] == QBTeams[1]) & (
            team['position_name'] == 'WR'), 'draft_capital_spent']
        QB2RB = team.loc[(team['team'] == QBTeams[1]) & (
            team['position_name'] == 'RB'), 'draft_capital_spent']
        QB2TE = team.loc[(team['team'] == QBTeams[1]) & (
            team['position_name'] == 'TE'), 'draft_capital_spent']
        QB3WR = team.loc[(team['team'] == QBTeams[2]) & (
            team['position_name'] == 'WR'), 'draft_capital_spent']
        QB3RB = team.loc[(team['team'] == QBTeams[2]) & (
            team['position_name'] == 'RB'), 'draft_capital_spent']
        QB3TE = team.loc[(team['team'] == QBTeams[2]) & (
            team['position_name'] == 'TE'), 'draft_capital_spent']

        teamStacks = team.loc[~team['team'].isin(QBTeams)].groupby(
            'team').filter(lambda x: x['player_name'].count() > 1)

        team_df[order] = {
            'daysFromSeasonStart': team['days_until_season_start'].mean(),
            'draftValue0106': team.iloc[:6]['draft_capital_gained'].sum(),
            'draftValue0712': team.iloc[6:12]['draft_capital_gained'].sum(),
            'draftValue1318': team.iloc[12:]['draft_capital_gained'].sum(),
            'NumQB': team.loc[team['position_name'] == 'QB', 'draft_capital_spent'].count(),
            'NumWR': team.loc[team['position_name'] == 'WR', 'draft_capital_spent'].count(),
            'NumRB': team.loc[team['position_name'] == 'RB', 'draft_capital_spent'].count(),
            'NumTE': team.loc[team['position_name'] == 'TE', 'draft_capital_spent'].count(),
            'PriceQB': team.loc[team['position_name'] == 'QB', 'draft_capital_spent'].sum(),
            'PriceWR': team.loc[team['position_name'] == 'WR', 'draft_capital_spent'].sum(),
            'PriceRB': team.loc[team['position_name'] == 'RB', 'draft_capital_spent'].sum(),
            'PriceTE': team.loc[team['position_name'] == 'TE', 'draft_capital_spent'].sum(),
            'QB1WRNum': QB1WR.count(),
            'QB1RBNum': QB1RB.count(),
            'QB1TENum': QB1TE.count(),
            'QB1WRVal': QB1WR.sum(),
            'QB1RBVal': QB1RB.sum(),
            'QB1TEVal': QB1TE.sum(),
            'QB2WRNum': QB2WR.count(),
            'QB2RBNum': QB2RB.count(),
            'QB2TENum': QB2TE.count(),
            'QB2WRVal': QB2WR.sum(),
            'QB2RBVal': QB2RB.sum(),
            'QB2TEVal': QB2TE.sum(),
            'QB3WRNum': QB3WR.count(),
            'QB3RBNum': QB3RB.count(),
            'QB3TENum': QB3TE.count(),
            'QB3WRVal': QB3WR.sum(),
            'QB3RBVal': QB3RB.sum(),
            'QB3TEVal': QB3TE.sum(),
            'NonQBStackNum': teamStacks.groupby('team').count()['draft_capital_spent'].max(),
            'NonQBStackVal': teamStacks.groupby('team').sum()['draft_capital_spent'].max(),
            'Points': team['roster_points'].mean()
        }
        pbar.update(1)
    pbar.close()

# %%
with open(pkg_resources.files(data) / 'BBM_teams.dat', 'wb') as outfile:
    pickle.dump(team_df, outfile, -1)
