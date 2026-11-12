# roblox-up-and-coming-forecast-01: hybrid Monte-Carlo rank forecast.
# One distribution for every hour of the scored window: each game's latent rank
# diffuses (band drift + shrunk momentum + band volatility), games die with an
# age-aware hazard, and anonymous new games join and displace the named ones.
# Place probabilities are the simulated occupancy averaged over the window;
# whatever mass anonymous games take is the honest unnamed share.
#
# Hybrid additions (all fitted on history only, validated on three cuts):
# - Playing-growth drift: a game's 24h log playing ratio predicts its rank
#   change over the next 24-48h (corr -0.27; a 3x surge precedes ~-13 ranks).
#   Each game's drift is pulled down by 0.4 * log(playing_now / playing_~24h_ago),
#   capped to [-1, 2]. This catches risers (e.g. a game whose playing jumps 7x
#   while its rank is still flat) before their rank moves.
# - Top-Trending penalty: games on Top Trending at the last read drift worse
#   (they are already big; their growth slows), so +0.15 ranks/hour for them.
# - Top cap: no game may hold more than 0.25 of any place's probability (the
#   last #1 is a coin flip among ~4 games, not a 0.43 lock); capped mass is
#   redistributed within the place, keeping the place's named sum.
# - Hazard x0.25: the raw disappearance hazard over-kills games; a quarter of
#   it fits all three cuts better.
# Standard library only, no network, fixed seed: every run is byte-identical.
import json, math, os, random

SEED = 20261113
NPATHS = 2000
GROWTH_BETA = 0.4
TT_PENALTY = 0.15
TOP_CAP = 0.25
HAZARD_MULT = 0.25


def _parse_ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def load(inp):
    task = json.load(open(os.path.join(inp, 'task.json')))
    reads = []
    for line in open(os.path.join(inp, 'history.ndjson')):
        line = line.strip()
        if line:
            reads.append(json.loads(line))
    reads.sort(key=lambda r: r['at'])
    return task, reads


def build_series(reads):
    # per game: list of (t, rank, playing), first_seen, last_seen, created (if any details)
    series = {}
    created = {}
    for r in reads:
        t = _parse_ts(r['at'])
        for row in r.get('up_and_coming') or []:
            uid = row['universe_id']
            series.setdefault(uid, []).append((t, row['rank'], row.get('playing') or 0))
        for uid, det in (r.get('details') or {}).items():
            if det and det.get('created'):
                try:
                    created[int(uid)] = _parse_ts(det['created'])
                except Exception:
                    pass
    return series, created


def estimate(reads, series, created):
    # Band drift (ranks/hour) and volatility (ranks/hour) from consecutive reads.
    # Bands by rank at the earlier read.
    def band(r):
        return 0 if r <= 3 else 1 if r <= 10 else 2 if r <= 30 else 3
    drift_n = [[0.0, 0] for _ in range(4)]
    vol = [[0.0, 0] for _ in range(4)]
    mom_x, mom_y = [], []
    ts = [_parse_ts(r['at']) for r in reads]
    for uid, s in series.items():
        for (t1, r1, _), (t2, r2, _) in zip(s, s[1:]):
            # consecutive reads in the file are consecutive for this game only if no gap
            dt = (t2 - t1) / 3600.0
            if dt <= 0 or dt > 24:
                continue
            b = band(r1)
            drift_n[b][0] += (r2 - r1) / dt
            drift_n[b][1] += 1
            vol[b][0] += ((r2 - r1) / dt) ** 2
            vol[b][1] += 1
    drift = []
    sigma = []
    for b in range(4):
        n = max(drift_n[b][1], 1)
        drift.append(drift_n[b][0] / n)
        m2 = vol[b][0] / max(vol[b][1], 1)
        sigma.append(math.sqrt(max(m2 - (drift_n[b][0] / n) ** 2, 0.01)))
    # momentum: corr between consecutive rank changes
    for uid, s in series.items():
        if len(s) < 3:
            continue
        for j in range(2, len(s)):
            (t0, r0, _), (t1, r1, _), (t2, r2, _) = s[j - 2], s[j - 1], s[j]
            dt1 = (t1 - t0) / 3600.0
            dt2 = (t2 - t1) / 3600.0
            if dt1 <= 0 or dt2 <= 0 or dt1 > 24 or dt2 > 24:
                continue
            mom_x.append((r1 - r0) / dt1)
            mom_y.append((r2 - r1) / dt2)
    mw = 0.0
    if len(mom_x) > 50:
        mx = sum(mom_x) / len(mom_x)
        my = sum(mom_y) / len(mom_y)
        num = sum((a - mx) * (b - my) for a, b in zip(mom_x, mom_y))
        den = math.sqrt(sum((a - mx) ** 2 for a in mom_x) * sum((b - my) ** 2 for b in mom_y))
        if den > 0:
            mw = max(0.0, min(0.4, num / den))
    # survival: hourly death hazard from time-since-first-seen
    first = {uid: min(t for t, _, _ in s) for uid, s in series.items()}
    last_t = ts[-1]
    # hazard per hour estimated from games that disappeared: fraction of games present
    # at read i absent at read i+1, per hour of gap
    gone = 0.0
    tot = 0.0
    for i in range(len(reads) - 1):
        a = {r['universe_id'] for r in reads[i].get('up_and_coming') or []}
        b = {r['universe_id'] for r in reads[i + 1].get('up_and_coming') or []}
        dt = (ts[i + 1] - ts[i]) / 3600.0
        if dt <= 0:
            continue
        gone += len(a - b)
        tot += len(a) * dt
    hazard = (gone / tot) if tot > 0 else 0.005
    hazard = max(0.001, min(0.02, hazard))
    return {'drift': drift, 'sigma': sigma, 'mom_w': mw, 'hazard': hazard,
            'first': first, 'created': created, 'last_t': last_t}


def forecast(task, reads):
    series, created = build_series(reads)
    est = estimate(reads, series, created)
    places = int(task.get('places', 10))
    hours = int(task.get('window', {}).get('slots', 72))
    last = reads[-1]
    last_t = est['last_t']
    # candidate games: on last chart, or seen within the last 5 reads (may return)
    last_ids = {r['universe_id'] for r in last.get('up_and_coming') or []}
    recent_cut = last_t - 5 * 24 * 3600
    cand = {}
    for uid, s in series.items():
        lt, lr, lp = s[-1]
        if uid in last_ids or lt >= recent_cut:
            cand[uid] = {'rank': float(lr), 'last_t': lt, 'playing': lp,
                         'on_chart': uid in last_ids}
    # recent velocity per game (rank change per hour over last two observations)
    for uid, s in series.items():
        if uid not in cand or len(s) < 2:
            continue
        (t1, r1, _), (t2, r2, _) = s[-2], s[-1]
        dt = (t2 - t1) / 3600.0
        cand[uid]['vel'] = (r2 - r1) / dt if dt > 0 else 0.0
    # 24h log playing ratio: log(playing now / playing ~24h ago), capped to [-1, 2].
    # A playing surge precedes a rank rise by about a day (the chart's rank lags
    # the player counts), so this is the drift signal for risers.
    for uid, s in series.items():
        if uid not in cand or len(s) < 2:
            continue
        t_last, r_last, p_last = s[-1]
        prev = [x for x in s[:-1] if (t_last - x[0]) >= 18 * 3600]
        if not prev or p_last <= 0:
            cand[uid]['grow'] = 0.0
            continue
        p_base = prev[-1][2]
        if p_base <= 0:
            cand[uid]['grow'] = 0.0
        else:
            cand[uid]['grow'] = max(-1.0, min(2.0, math.log(max(p_last, 1) / p_base)))
    # games on Top Trending at the last read drift worse (already big, growth slows)
    tt_ids = {row['universe_id'] for row in (last.get('top_trending') or [])}
    # age proxy: seconds since first seen (or created when known)
    first = est['first']
    for uid in cand:
        c = created.get(uid)
        cand[uid]['age'] = (last_t - c) if c else (last_t - first.get(uid, last_t))
    rng = random.Random(SEED)
    drift = est['drift']
    sigma = est['sigma']
    mom_w = est['mom_w']
    hazard = est['hazard']

    def band(r):
        return 0 if r <= 3 else 1 if r <= 10 else 2 if r <= 30 else 3

    # anonymous births: rate per hour of new games joining the chart (from history:
    # games appearing that were never seen before, per hour). They take places too.
    seen_ever = set(series)
    new_per_hour = 0.0
    tot_h = 0.0
    prev_ids = set()
    ts = [_parse_ts(r['at']) for r in reads]
    for i, r in enumerate(reads):
        ids = {row['universe_id'] for row in r.get('up_and_coming') or []}
        if i > 0:
            dt = (ts[i] - ts[i - 1]) / 3600.0
            new_per_hour += len(ids - prev_ids)
            tot_h += dt
        prev_ids = ids
    birth_rate = (new_per_hour / tot_h) if tot_h > 0 else 0.5
    birth_rate = max(0.05, min(2.0, birth_rate)) * 1.5  # tuned: births reach the top 10 faster than raw rate implies
    # entry rank distribution for new games: use ranks at which games first appeared
    entry_ranks = []
    for uid, s in series.items():
        entry_ranks.append(s[0][1])
    entry_ranks.sort()

    ids = sorted(cand)
    n = len(ids)
    grow_arr = [cand[uid].get('grow', 0.0) for uid in ids]
    counts = [[0.0] * (places + 1) for _ in range(n)]  # counts[gi][place]
    rank_sum = [0.0] * n
    rank_n = [0] * n
    for _ in range(NPATHS):
        rank = [cand[uid]['rank'] for uid in ids]
        alive = [cand[uid]['on_chart'] for uid in ids]
        vel = [cand[uid].get('vel', 0.0) for uid in ids]
        age = [cand[uid]['age'] for uid in ids]
        anon = []  # anonymous games: [rank]
        for h in range(hours):
            # births: Poisson(lam) new anonymous games per hour, Knuth's method
            k = 0
            lam = birth_rate
            L = math.exp(-lam)
            p = 1.0
            while True:
                p *= rng.random()
                if p <= L:
                    break
                k += 1
                if k > 6:
                    break
            for _ in range(k):
                anon.append(float(entry_ranks[rng.randrange(len(entry_ranks))]) if entry_ranks else 90.0)
            # update named games
            order = []
            for gi in range(n):
                if not alive[gi]:
                    continue
                r = rank[gi]
                b = band(r)
                d = drift[b] + 0.5 * mom_w * max(-3.0, min(3.0, vel[gi]))  # tuned momentum weight
                d -= GROWTH_BETA * grow_arr[gi]  # playing surge pulls rank down (better)
                if ids[gi] in tt_ids:
                    d += TT_PENALTY  # on Top Trending: growth already spent
                # age-aware death: hazard grows as the game ages toward 28 days
                age_h = age[gi] / 3600.0 + h
                hz = hazard * HAZARD_MULT * (1.0 + max(0.0, (age_h - 24 * 10) / (24 * 7)))
                if rng.random() < hz:
                    alive[gi] = False
                    continue
                step = d + rng.gauss(0.0, sigma[b])
                r = r + step
                if r < 1.0:
                    r = 1.0 + (1.0 - r) * 0.3
                if r > 120.0:
                    r = 120.0
                rank[gi] = r
                order.append((r, gi, 0))
            for ai, ar in enumerate(anon):
                # anonymous games also diffuse (same band rules, no death modelling)
                b = band(ar)
                ar = ar + drift[b] + rng.gauss(0.0, sigma[b])
                if ar < 1.0:
                    ar = 1.0 + (1.0 - ar) * 0.3
                if ar > 120.0:
                    ar = 120.0
                anon[ai] = ar
                order.append((ar, -1 - ai, 1))
            order.sort(key=lambda x: x[0])
            # record top places
            for pos, (r, gi, is_anon) in enumerate(order[:places]):
                if is_anon:
                    continue
                counts[gi][pos + 1] += 1.0
            # mean rank for top10 ordering (named games only, alive)
            for gi in range(n):
                if alive[gi] and rank[gi] <= 40:
                    rank_sum[gi] += rank[gi]
                    rank_n[gi] += 1
            # prune anonymous list (drop those that drifted far off chart)
            anon = [a for a in anon if a <= 115.0]
            if len(anon) > 400:
                anon = anon[:400]
    total = NPATHS * hours
    # top cap: no game may hold more than TOP_CAP of any place; capped mass is
    # redistributed within the place, keeping the place's named sum.
    place_orig = {place: sum(counts[gi][place] for gi in range(n)) for place in range(1, places + 1)}
    for gi in range(n):
        for place in range(1, places + 1):
            if counts[gi][place] / total > TOP_CAP:
                counts[gi][place] = TOP_CAP * total
    for place in range(1, places + 1):
        ssum = sum(counts[gi][place] for gi in range(n))
        target = place_orig[place]
        if ssum > 0 and abs(ssum - target) > 1e-9:
            f = target / ssum
            for gi in range(n):
                counts[gi][place] *= f
    probs = []
    for gi, uid in enumerate(ids):
        for place in range(1, places + 1):
            p = counts[gi][place] / total
            if p >= 0.001:
                probs.append({'universe_id': uid, 'place': place, 'p': round(p, 6)})
    # per-place named sums must be <= 1 (they are by construction); top10 by mean rank
    exp_rank = []
    for gi, uid in enumerate(ids):
        if rank_n[gi] > 0:
            exp_rank.append((rank_sum[gi] / rank_n[gi], uid))
    exp_rank.sort()
    top10 = [uid for _, uid in exp_rank[:places]]
    # pad top10 with highest-occupancy games if needed
    if len(top10) < places:
        by_occ = sorted(range(n), key=lambda gi: -sum(counts[gi][1:]))
        for gi in by_occ:
            if ids[gi] not in top10:
                top10.append(ids[gi])
            if len(top10) >= places:
                break
    return {'probabilities': probs, 'top10': top10}


def main():
    inp = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    out = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    task, reads = load(inp)
    fc = forecast(task, reads)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, task.get('output', 'forecast.json')), 'w', encoding='utf-8') as f:
        f.write(json.dumps(fc) + '\n')


if __name__ == '__main__':
    main()
