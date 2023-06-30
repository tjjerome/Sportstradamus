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
schedules = schedules.loc[schedules.week < 18]
schedules = schedules[['season', 'week', 'away_team', 'home_team']]
schedules = schedules.groupby(['season', 'week']).apply(
    lambda x: {i[2]: i[3] for i in x.values} | {i[3]: i[2] for i in x.values})
schedules = schedules.to_dict()

# %%
divisions = {
    'MIN': 'NFCN',
    'GB': 'NFCN',
    'DET': 'NFCN',
    'CHI': 'NCFN',
    'PHI': 'NFCE',
    'DAL': 'NFCE',
    'WSH': 'NFCE',
    'NYG': 'NCFE',
    'NO': 'NFCS',
    'TB': 'NFCS',
    'CAR': 'NFCS',
    'ATL': 'NCFS',
    'SF': 'NFCW',
    'SEA': 'NFCW',
    'LAR': 'NFCW',
    'ARI': 'NCFW',
    'CLE': 'AFCN',
    'CIN': 'AFCN',
    'BAL': 'AFCN',
    'PIT': 'ACFN',
    'NE': 'AFCE',
    'MIA': 'AFCE',
    'BUF': 'AFCE',
    'NYJ': 'ACFE',
    'HOU': 'AFCS',
    'TEN': 'AFCS',
    'JAX': 'AFCS',
    'IND': 'ACFS',
    'KC': 'AFCW',
    'LAC': 'AFCW',
    'LV': 'AFCW',
    'DEN': 'ACFW'
}

# %%
weekly = nfl.import_weekly_data([2021, 2022], [
                                'player_display_name', 'position', 'season', 'week', 'fantasy_points', 'receptions'], True)
weekly['fantasy_points'] = weekly['fantasy_points'] + weekly['receptions']/2
weekly = weekly.loc[(weekly['position'].isin(
    ['QB', 'WR', 'RB', 'TE'])) & (weekly['week'] < 18)]
weekly = weekly.groupby(['season', 'week']).apply(
    lambda x: {i[0]: i[4] for i in x.values})
weekly = weekly.to_dict()

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


picks = df.overall_pick_number.replace(0, 216).to_numpy()
value = df.pick_points.to_numpy()
fit = np.polyfit(np.log(picks), value, 1)

# %%
for data_str in tqdm(data_list, desc='Loading Data Files', unit='file'):
    df = pd.read_csv(pkg_resources.files(data) / data_str)

    df['season'] = df['draft_time'].str[:4]
    df['team'] = (df['player_name'] + ', ' + df['season']
                  ).map(rosters).fillna("None")
    df['division'] = df['team'].map(divisions)

    df['draft_capital_spent'] = fit[0]*np.log(df['overall_pick_number'])+fit[1]
    df['projection_draft_capital'] = fit[0] * \
        np.log(df['projection_adp'].replace(0, 216))+fit[1]
    df['draft_capital_gained'] = df['projection_draft_capital'] - \
        df['draft_capital_spent']

    draft_date = pd.to_datetime(df['draft_time'].str[:10], format='%Y-%m-%d')
    df['days_until_season_start'] = (datetime.strptime(
        '2022-09-08', '%Y-%m-%d') - draft_date).dt.days
    df.loc[df['days_until_season_start'] > 200, 'days_until_season_start'] = \
        df.loc[df['days_until_season_start'] >
               200, 'days_until_season_start'] - 363

    teams = df.groupby(['draft_id', 'pick_order'])
    teams = teams[['player_name', 'position_name', 'team', 'team_pick_number', 'roster_points', 'draft_capital_spent',
                   'draft_capital_gained', 'days_until_season_start', 'season', 'division', 'bye']]

    team_df = {}
    weekly_df = {}
    with tqdm(total=teams.ngroups, desc="Analyzing Teams", unit="team") as pbar:
        for order, team in teams:
            team = team.sort_values('team_pick_number')
            QBTeams = team.loc[team['position_name'] == 'QB', 'team'].to_list()[
                :3]
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

            teamStacks = team.loc[~team['team'].isin(QBTeams)].groupby('team')
            teamStacks = teamStacks.filter(
                lambda x: x['player_name'].count() == np.min([teamStacks.size().max(), 2]))
            if teamStacks.empty:
                StackedTeam = "NA"
            else:
                StackedTeam = teamStacks.sum()['draft_capital_spent'].idxmax()
            teamStacks = team.loc[team['team'] ==
                                  StackedTeam, 'draft_capital_spent']

            divStack = team.groupby('division').filter(
                lambda x: x['team'].unique().count() > 2)
            divStack = divStack.filter(
                lambda x: x['player_name'].count() == np.min([divStack.size().max(), 2]))
            if divStack.empty:
                StackedDiv = "NA"
            else:
                StackedDiv = divStack.sum()['draft_capital_spent'].idxmax()
            divStack = team.loc[team['division'] ==
                                StackedDiv, 'draft_capital_spent']

            byeStack = team.groupby('bye').filter(
                lambda x: x['team'].unique().count() > 2)
            byeStack = byeStack.filter(
                lambda x: x['player_name'].count() == np.min([divStack.size().max(), 2]))
            if byeStack.empty:
                StackedBye = 0
            else:
                StackedBye = byeStack.sum()['draft_capital_spent'].idxmax()
            byeStack = team.loc[team['bye'] ==
                                StackedBye, 'draft_capital_spent']

            team_df[order] = {
                'daysFromSeasonStart': int(team['days_until_season_start'].mean()),
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
                'NonQBStackNum': teamStacks.count(),
                'NonQBStackVal': teamStacks.sum(),
                'DivStackNum': divStack.count(),
                'DivStackVal': divStack.sum(),
                'ByeStackNum': byeStack.count(),
                'ByeStackVal': byeStack.sum(),
                'Points': team['roster_points'].mean()
            }
            for week in np.arange(1, 18):
                team['player_points'] = team['player_name'].map(
                    weekly[(int(team['season'].max()), week)])
                QBPoints = team.loc[team['position_name'] == 'QB', 'player_points'].sort_values(
                    ascending=False).iloc[:1].sum()
                activePlayers = team.loc[team['position_name']
                                         == 'QB'].index.to_list()
                WRPoints = team.loc[team['position_name'] == 'WR', 'player_points'].sort_values(
                    ascending=False).iloc[:3].sum()
                activePlayers.extend(team.loc[team['position_name'] == 'WR', 'player_points'].sort_values(
                    ascending=False).iloc[:3].index.to_list())
                RBPoints = team.loc[team['position_name'] == 'RB', 'player_points'].sort_values(
                    ascending=False).iloc[:2].sum()
                activePlayers.extend(team.loc[team['position_name'] == 'RB', 'player_points'].sort_values(
                    ascending=False).iloc[:2].index.to_list())
                TEPoints = team.loc[team['position_name'] == 'TE', 'player_points'].sort_values(
                    ascending=False).iloc[:1].sum()
                activePlayers.extend(team.loc[team['position_name'] == 'TE', 'player_points'].sort_values(
                    ascending=False).iloc[:1].index.to_list())
                FLXPoints = team.loc[~team.index.isin(
                    activePlayers), 'player_points'].max()

                team_df[order][f"W{week}_Points"] = QBPoints + \
                    WRPoints + RBPoints + TEPoints + FLXPoints
                # QB1OPP = team.loc[(team['team'] == schedules[(
                #     int(team['season'].max()), week)].get(QBTeams[0])), 'draft_capital_spent']
                # QB2OPP = team.loc[(team['team'] == schedules[(
                #     int(team['season'].max()), week)].get(QBTeams[1])), 'draft_capital_spent']
                # QB3OPP = team.loc[(team['team'] == schedules[(
                #     int(team['season'].max()), week)].get(QBTeams[2])), 'draft_capital_spent']
                # teamStackOPP = team.loc[(team['team'] == schedules[(
                #     int(team['season'].max()), week)].get(StackedTeam)), 'draft_capital_spent']

                # if not (int(team['season'].max()), week) in weekly_df:
                #     weekly_df[(int(team['season'].max()), week)] = {}

                # weekly_df[(int(team['season'].max()), week)][order] = team_df[order] | {
                #     'QB1OPPNum': QB1OPP.count(),
                #     'QB2OPPNum': QB2OPP.count(),
                #     'QB3OPPNum': QB3OPP.count(),
                #     'QB1OPPVal': QB1OPP.sum(),
                #     'QB2OPPVal': QB2OPP.sum(),
                #     'QB3OPPVal': QB3OPP.sum(),
                #     'NonQBOPPNum': teamStackOPP.count(),
                #     'NonQBOPPVal': teamStackOPP.sum(),
                #     'Points': QBPoints + WRPoints + RBPoints + TEPoints + FLXPoints
                # }
            pbar.update(1)
        pbar.close()

    team_df = pd.DataFrame.from_dict(team_df, orient='index')
    team_df.to_csv(pkg_resources.files(data) /
                   data_str.replace(".csv", "_WWteams.csv"))

    # for k in weekly_df.keys():
    #     df = pd.DataFrame.from_dict(weekly_df[k], orient='index')
    #     df.to_csv(pkg_resources.files(data) /
    #               data_str.replace(".csv", f"_W{k[1]}_teams.csv"))

    del df
    del team_df
    del weekly_df

# %%
# with open(pkg_resources.files(data) / 'BBM_teams.dat', 'wb') as outfile:
#     pickle.dump(team_df, outfile, -1)
