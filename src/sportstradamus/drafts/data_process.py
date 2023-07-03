# %%
import pandas as pd
from sportstradamus.drafts import data
# import data
import importlib.resources as pkg_resources
import numpy as np
from datetime import datetime
import nfl_data_py as nfl
from tqdm import tqdm

# %%
rosters = {2021: {}, 2022: {}}
for y in rosters.keys():
    depth_df = pd.read_csv(
        'https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{year}.csv'.format(year=y))
    depth_df = depth_df[['full_name', 'team', 'position', 'week']]
    depth_df = depth_df.loc[depth_df['week'] == 1]
    depth_df = depth_df.loc[depth_df['position'].isin(
        ['QB', 'TE', 'WR', 'RB'])]
    depth_df = depth_df[['full_name', 'team']]
    depth_df = depth_df.set_index(depth_df['full_name'])[['team']]
    rosters[y] = depth_df.to_dict()['team']

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
    'CHI': 'NFCN',
    'PHI': 'NFCE',
    'DAL': 'NFCE',
    'WAS': 'NFCE',
    'NYG': 'NFCE',
    'NO': 'NFCS',
    'TB': 'NFCS',
    'CAR': 'NFCS',
    'ATL': 'NFCS',
    'SF': 'NFCW',
    'SEA': 'NFCW',
    'LA': 'NFCW',
    'ARI': 'NFCW',
    'CLE': 'AFCN',
    'CIN': 'AFCN',
    'BAL': 'AFCN',
    'PIT': 'AFCN',
    'NE': 'AFCE',
    'MIA': 'AFCE',
    'BUF': 'AFCE',
    'NYJ': 'AFCE',
    'HOU': 'AFCS',
    'TEN': 'AFCS',
    'JAX': 'AFCS',
    'IND': 'AFCS',
    'KC': 'AFCW',
    'LAC': 'AFCW',
    'LV': 'AFCW',
    'DEN': 'AFCW'
}

# %%
byes = {}
for s, w in schedules.keys():
    teams = list(set(divisions.keys())-set(schedules[(s, w)].keys()))
    if s not in byes:
        byes[s] = {}
    for team in teams:
        byes[s][team] = int(w)

# %%
weekly = nfl.import_weekly_data([2021, 2022], [
                                'player_display_name', 'position', 'season', 'week', 'receptions', 'receiving_tds',
                                'receiving_yards', 'rushing_tds', 'rushing_yards', 'passing_tds', 'passing_yards',
                                'interceptions', 'passing_2pt_conversions', 'rushing_2pt_conversions',
                                'receiving_2pt_conversions', 'rushing_fumbles_lost', 'receiving_fumbles_lost',
                                'sack_fumbles_lost'], True)
weekly = weekly.loc[(weekly['position'].isin(
    ['QB', 'WR', 'RB', 'TE'])) & (weekly['week'] < 18)]
weekly['fantasy_points'] = weekly['receptions']/2 + weekly['receiving_tds']*6 + weekly['rushing_tds']*6\
    + weekly['passing_tds']*4 + weekly['receiving_yards']/10 + weekly['rushing_yards']/10 + weekly['passing_yards']/25\
    + weekly['receiving_2pt_conversions']*2 + weekly['rushing_2pt_conversions']*2 + weekly['passing_2pt_conversions']*2\
    - weekly['interceptions'] - weekly['rushing_fumbles_lost']*2\
    - weekly['receiving_fumbles_lost']*2 - weekly['sack_fumbles_lost']*2
weekly = weekly[['player_display_name', 'season', 'week', 'fantasy_points']].groupby(['season', 'week']).apply(
    lambda x: {i[0]: np.round(i[3], 2) for i in x.values})
weekly = weekly.to_dict()

# %%
data_list = [f.name for f in pkg_resources.files(
    data).iterdir() if ".csv" in f.name and "Regular" in f.name and "teams" not in f.name]
# %%
# df = pd.DataFrame()
# for data_str in tqdm(data_list, desc='Loading Data Files', unit='file'):
#     if df.empty:
#         df = pd.read_csv(pkg_resources.files(data) / data_str)
#     else:
#         df = pd.concat([df, pd.read_csv(pkg_resources.files(data) / data_str)])


# picks = df.overall_pick_number.replace(0, 216).to_numpy()
# value = df.pick_points.to_numpy()
# fit = np.polyfit(np.log(picks), value, 1) # 257.4327 - 39.4727ln(x)
fit = [-39.4727, 257.4327]


# %%
def analyze_teams(teams):
    team_df = {}
    with tqdm(total=teams.ngroups, desc="Analyzing Teams", unit="team") as pbar:
        for order, team in teams:
            team = team.sort_values('team_pick_number')
            QBTeams = team.loc[team['position_name'] == 'QB', 'team'].to_list()[:3]
            QBTeams.extend(['NA'] * (3 - len(QBTeams)))

            qb_filter = team['position_name'] == 'QB'
            wr_filter = team['position_name'] == 'WR'
            rb_filter = team['position_name'] == 'RB'
            te_filter = team['position_name'] == 'TE'

            QB1WR = team.loc[(team['team'] == QBTeams[0]) &
                             wr_filter, 'draft_capital_spent']
            QB1RB = team.loc[(team['team'] == QBTeams[0]) &
                             rb_filter, 'draft_capital_spent']
            QB1TE = team.loc[(team['team'] == QBTeams[0]) &
                             te_filter, 'draft_capital_spent']
            QB2WR = team.loc[(team['team'] == QBTeams[1]) &
                             wr_filter, 'draft_capital_spent']
            QB2RB = team.loc[(team['team'] == QBTeams[1]) &
                             rb_filter, 'draft_capital_spent']
            QB2TE = team.loc[(team['team'] == QBTeams[1]) &
                             te_filter, 'draft_capital_spent']
            QB3WR = team.loc[(team['team'] == QBTeams[2]) &
                             wr_filter, 'draft_capital_spent']
            QB3RB = team.loc[(team['team'] == QBTeams[2]) &
                             rb_filter, 'draft_capital_spent']
            QB3TE = team.loc[(team['team'] == QBTeams[2]) &
                             te_filter, 'draft_capital_spent']
            try:
                StackedTeam = team.loc[~team['team'].isin(QBTeams), 'team'].value_counts().idxmax()
            except:
                StackedTeam = 'NA'

            teamStack = team.loc[team['team'] == StackedTeam, 'draft_capital_spent']

            try:
                StackedDiv = team['division'].value_counts().idxmax()
            except:
                StackedDiv = 'NA'

            divStack = team.loc[team['division'] == StackedDiv, 'draft_capital_spent']

            try:
                StackedBye = team['bye'].value_counts().idxmax()
            except:
                StackedBye = 0

            byeStack = team.loc[team['bye'] == StackedBye, 'draft_capital_spent']

            team_df[order] = {
                'daysFromSeasonStart': int(team['days_until_season_start'].mean()),
                'draftValue0106': team.iloc[:6]['draft_capital_gained'].sum(),
                'draftValue0712': team.iloc[6:12]['draft_capital_gained'].sum(),
                'draftValue1318': team.iloc[12:]['draft_capital_gained'].sum(),
                'NumQB': qb_filter.sum(),
                'NumWR': wr_filter.sum(),
                'NumRB': rb_filter.sum(),
                'NumTE': te_filter.sum(),
                'PriceQB': team[qb_filter]['draft_capital_spent'].sum(),
                'PriceWR': team[wr_filter]['draft_capital_spent'].sum(),
                'PriceRB': team[rb_filter]['draft_capital_spent'].sum(),
                'PriceTE': team[te_filter]['draft_capital_spent'].sum(),
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
                'NonQBStackNum': teamStack.count(),
                'NonQBStackVal': teamStack.sum(),
                'DivStackNum': divStack.count(),
                'DivStackVal': divStack.sum(),
                'ByeStackNum': byeStack.count(),
                'ByeStackVal': byeStack.sum(),
                'SeasonPoints': team['roster_points'].mean()
            }

            for week in np.arange(1, 18):
                QB1OPP = team.loc[(team['team'] == schedules[(
                    int(team['season'].max()), week)].get(QBTeams[0])), 'draft_capital_spent']
                QB2OPP = team.loc[(team['team'] == schedules[(
                    int(team['season'].max()), week)].get(QBTeams[1])), 'draft_capital_spent']
                QB3OPP = team.loc[(team['team'] == schedules[(
                    int(team['season'].max()), week)].get(QBTeams[2])), 'draft_capital_spent']
                teamStackOPP = team.loc[(team['team'] == schedules[(
                    int(team['season'].max()), week)].get(StackedTeam)), 'draft_capital_spent']
                team_points = team[f"player_points_{week}"].sort_values(ascending=False)
                QBPoints = team_points.loc[qb_filter].iloc[:1]
                activePlayers = team.loc[qb_filter].index.to_list()
                WRPoints = team_points.loc[wr_filter].iloc[:3]
                activePlayers.extend(WRPoints.index.to_list())
                RBPoints = team_points.loc[rb_filter].iloc[:2]
                activePlayers.extend(RBPoints.index.to_list())
                TEPoints = team_points.loc[te_filter].iloc[:1]
                activePlayers.extend(TEPoints.index.to_list())
                FLXPoints = team_points.loc[~team.index.isin(activePlayers)].iloc[:1]

                team_df[order][f"W{week}_QB1OPPNum"] = QB1OPP.count()
                team_df[order][f"W{week}_QB1OPPVal"] = QB1OPP.sum()
                team_df[order][f"W{week}_QB2OPPNum"] = QB2OPP.count()
                team_df[order][f"W{week}_QB2OPPVal"] = QB2OPP.sum()
                team_df[order][f"W{week}_QB3OPPNum"] = QB3OPP.count()
                team_df[order][f"W{week}_QB3OPPVal"] = QB3OPP.sum()
                team_df[order][f"W{week}_NonQBOPPNum"] = teamStackOPP.count()
                team_df[order][f"W{week}_NonQBOPPVal"] = teamStackOPP.sum()
                team_df[order][f"W{week}_Points"] = QBPoints.sum() + WRPoints.sum() + \
                    RBPoints.sum() + TEPoints.sum() + FLXPoints.sum()

            pbar.update(1)
        pbar.close()

    team_df = pd.DataFrame.from_dict(team_df, orient='index').round(2)
    team_df.to_csv(pkg_resources.files(data) /
                   data_str.replace(".csv", "_teams.csv"))

# %%


for data_str in tqdm(data_list, desc='Loading Data Files', unit='file'):
    df = pd.read_csv(pkg_resources.files(data) / data_str, index_col=0)

    df['season'] = df['draft_time'].str[:4]
    df['team'] = df['player_name'].replace(".", "").map(
        rosters[int(df.iloc[0]['season'])]).fillna("None")
    df['division'] = df['team'].map(divisions)
    df['bye'] = df['team'].map(byes[int(df.iloc[0]['season'])])

    for week in np.arange(1,18):
        df[f"player_points_{week}"] = df['player_name'].replace(".", "").map(
            weekly[(int(df.iloc[0]['season']), week)]).fillna(0.0)

    df['draft_capital_spent'] = np.round(fit[0]*np.log(df['overall_pick_number'])+fit[1], 2)
    df['projection_draft_capital'] = np.round(fit[0]*np.log(df['projection_adp'].replace(0, 216))+fit[1], 2)
    df['draft_capital_gained'] = df['projection_draft_capital'] - df['draft_capital_spent']

    draft_date = pd.to_datetime(df['draft_time'].str[:10], format='%Y-%m-%d')
    df['days_until_season_start'] = (datetime.strptime(
        '2022-09-08', '%Y-%m-%d') - draft_date).dt.days
    df.loc[df['days_until_season_start'] > 200, 'days_until_season_start'] = \
        df.loc[df['days_until_season_start'] >
               200, 'days_until_season_start'] - 363

    teams = df.groupby(['draft_id', 'pick_order'])

    analyze_teams(teams)
