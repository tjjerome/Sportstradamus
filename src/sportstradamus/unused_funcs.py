
def find_bets(C, M, EV, model_odds, book_odds, bet_size, df, info={}, current_group=[], prev_searched=[], pbar=None):
    qualifying_groups = []
    searched = []
    stop_search = False

    # Base case: If the current group size is equal to the desired group size
    if len(current_group) == bet_size:
        bet = itemgetter(*current_group)(df)
        payout = payout_table[platform][bet_size-2]
        if len(set([leg["Team"] for leg in bet])) < 2:
            return [], False
        
        # p = np.product([leg["Boosted Model"] for leg in bet])
        # pb = np.product([leg["Boosted Books"] for leg in bet])

        # if p*payout < (league_cutoff_model[0]*bet_size+league_cutoff_model[1]) or pb*payout < (league_cutoff_books[0]*bet_size+league_cutoff_books[1]):
        #     return [], False
        
        if pbar:
            pbar.update(1)

        SIG = C[np.ix_(current_group, current_group)]
        boost = np.product(M[np.ix_(current_group, current_group)][np.triu_indices(bet_size,1)])
        payout = np.clip(payout*boost, 1, 100)
        try:
            p = payout*multivariate_normal.cdf(model_odds[current_group], np.zeros(bet_size), SIG)
        except:
            return [], False
        
        if p > 1.5:
            pb = payout*(3*multivariate_normal.cdf(book_odds[current_group], np.zeros(bet_size), SIG)+np.product(norm.cdf(book_odds[current_group])))/4
            units = (p - 1)/(payout - 1)/0.05
            if pb > 1.01 and units > .9:
                units = (p - 1)/(payout - 1)/0.05
                parlay = info | {
                    "Model EV": p,
                    "Books EV": pb,
                    "Boost": boost,
                    "Rec Bet": units,
                    "Leg 1": "",
                    "Leg 2": "",
                    "Leg 3": "",
                    "Leg 4": "",
                    "Leg 5": "",
                    "Leg 6": "",
                    "Legs": ", ".join([leg["Desc"] for leg in bet]),
                    # "Players": {f"{leg['Player']} {leg['Bet']}" for leg in bet},
                    "Markets": [(market, "Under" if leg["Bet"] == "Over" else "Over") if i == 2 and "vs." in leg["Player"] else (market, leg["Bet"]) for leg in bet for i, market in enumerate(leg["cMarket"])],
                    "Fun": np.sum([3-(np.abs(leg["Line"])/stat_std.get(league, {}).get(leg["Market"], 1)) if ("H2H" in leg["Desc"]) else 2 - 1/stat_cv.get(league, {}).get(leg["Market"], 1) + leg["Line"]/stat_std.get(league, {}).get(leg["Market"], 1) for leg in bet if (leg["Bet"] == "Over") or ("H2H" in leg["Desc"])]),
                    "Bet Size": bet_size,
                    "Mean EV": np.mean(EV[np.ix_(current_group, current_group)][np.triu_indices(bet_size,1)])
                }
                for i in np.arange(bet_size):
                    parlay["Leg " + str(i+1)] = bet[i]["Desc"]

                return [parlay], False
            else:
                return [], False
            
        else:
            return [], True

    # Recursive case: Try to add each of the remaining variables
    if len(current_group) == 0:
        for i in range(C.shape[0]):
            searched.append(i)
            search, stop_search = find_bets(C, M, EV, model_odds, book_odds, bet_size, df, info, [i], searched, pbar)
            # if pbar:
            #     pbar.update(1)
            if stop_search:
                # if pbar:
                #     pbar.update(len(df)-i-1)
                break

            qualifying_groups.extend(search)

        return qualifying_groups
    else:
        search_space = np.argsort(np.sum(EV[np.ix_(current_group)],0))[::-1]
        for idx in search_space:
            if idx in prev_searched or idx in current_group:
                continue
            if np.product(EV[np.ix_(current_group, [idx])]) == 0:
                continue
            new_group = current_group + [idx]
            searched.append(idx)
            search, stop_search = find_bets(C, M, EV, model_odds, book_odds, bet_size, df, info, new_group, prev_searched+searched, pbar)
            if stop_search:
                break

            qualifying_groups.extend(search)

        return qualifying_groups, (stop_search and len(searched)<=1)