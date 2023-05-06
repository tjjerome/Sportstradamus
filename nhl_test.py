import pickle
import requests
import random
from tqdm import tqdm

def get_pp():
    offers = []
    count = 0
    while True:
        """
        try:
            leagues = requests.get(
                'https://api.prizepicks.com/leagues', headers=random.choice(headers), proxies=proxy)
            if leagues.status_code == 200:
                leagues = leagues.json()
                break
            else:
                count += 1
                print("Attempt " + str(count) +
                      ": PrizePicks Error " + str(leagues.status_code))
                if count == 5:
                    print("Could not receive offer data")
                    return []
                else:
                    sleep(random.uniform(3, 5))
        except:
            count += 1
            print("Attempt " + str(count) + ": Proxy Login Error")
            if count == 5:
                print("Could not receive offer data")
                return []
            else:
                sleep(random.uniform(3, 5))

    leagues = [i['id'] for i in leagues['data']
               if i['attributes']['projections_count'] > 0]
   """
    leagues = [2,7,8]

    print("Processing PrizePicks offers")
    for l in tqdm(leagues):
        count = 0
        params = {
            'api_key': '82ccbf28-ddd6-4e37-b3a1-0097b10fd412',
            'url': f"https://api.prizepicks.com/projections?league_id={l}",
            'bypass': 'cloudflare'
        }
        while True:
            try:
                api = requests.get("https://proxy.scrapeops.io/v1/", params=params)
                if api.status_code == 200:
                    api = api.json()
                    break
                else:
                    count += 1
                    print("Attempt " + str(count) + ": League " + str(l) +
                          ": PrizePicks Error " + str(api.status_code))
                    if count == 5:
                        print("Could not receive offer data")
                        break
                    else:
                        sleep(random.uniform(3, 5))
            except:
                count += 1
                print("Attempt " + str(count) + ": Proxy Login Error")
                if count == 5:
                    print("Could not receive offer data")
                    break
                else:
                    sleep(random.uniform(3, 5))

        if count >= 5:
            continue

        player_ids = {}
        for p in api['included']:
            if p['type'] == 'new_player':
                player_ids[p['id']] = {
                    'Name': p['attributes']['name'].replace('\t', ''),
                    'Team': p['attributes']['team']
                }
            elif p['type'] == 'league':
                league = p['attributes']['name']

        print("Getting offers for " + league)
        for o in tqdm(api['data']):
            n = {
                'Player': remove_accents(player_ids[o['relationships']['new_player']['data']['id']]['Name']),
                'League': league,
                'Team': player_ids[o['relationships']['new_player']['data']['id']]['Team'],
                'Market': o['attributes']['stat_type'],
                'Line': o['attributes']['line_score'],
                'Opponent': o['attributes']['description']
            }
            if o['attributes']['is_promo']:
                n['Line'] = o['attributes']['flash_sale_line_score']
            offers.append(n)

    print(str(len(offers)) + " offers found")
    return offers
    
pp_offers = get_pp()