# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import logging
import re

logger = logging.getLogger(__name__)


def add_ch_winter_import_limit(n, snapshots, limit_twh, months=(10, 11, 12, 1, 2, 3),
                               country="CH"):
    """Limit the NET cross-border electricity import of `country` over `months`.

    Net import [MWh] = sum over winter snapshots, weighted by snapshot hours, of
    (flows into CH on border lines) - (flows out of CH on border lines), constrained
    to <= limit_twh. Expressed via the line-flow variable `Line-s` (bus0->bus1),
    so it is independent of the sector-coupling complexity. CH = several AC nodes
    (adm clustering); border lines = lines with exactly one end on a CH AC bus.
    """
    b = n.buses
    ch_ac = b.index[(b.carrier == "AC") & (b.country == country)] # all import buses (AC) in Switzerland
    assert len(ch_ac) > 0, "no buses found"
    ch_set = set(ch_ac)
    L = n.lines
    border = L.index[L.bus0.isin(ch_set) ^ L.bus1.isin(ch_set)]  # all lines with exactly one end in CH
    assert len(border) > 0, "no border lines found"
    into = [l for l in border if L.at[l, "bus1"] in ch_set]   # flow bus0->bus1 = IMPORT
    outof = [l for l in border if L.at[l, "bus0"] in ch_set]  # flow bus0->bus1 = EXPORT

    sns = snapshots[snapshots.month.isin(months)]
    w = n.snapshot_weightings.generators.loc[sns]             # hours per snapshot
    s = n.model["Line-s"]
    lhs = (s.loc[sns, into] * w).sum() - (s.loc[sns, outof] * w).sum()   # net import [MWh]
    n.model.add_constraints(lhs <= limit_twh * 1e6, name="ch_winter_net_import_limit")
    logger.info("CH winter net-import limit active: <= %.1f TWh, months %s, %d border lines "
                "(%d into / %d out of CH).", limit_twh, list(months), len(border),
                len(into), len(outof))


def add_ch_summer_import_limit(n, snapshots, limit_twh, months=(4, 5, 6, 7, 8, 9),
                               country="CH"):
    """Limit the NET cross-border electricity import of `country` over summer `months`.

    Companion to add_ch_winter_import_limit. With limit_twh = 0 this forbids CH from
    being a NET electricity importer in summer (Apr-Sep): CH must be net-balanced or a
    net exporter. Rationale: without it the optimiser imports cheap summer PV from DE/FR
    to charge the (unconstrained) seasonal H2/heat storage for winter, which masks the
    Swiss winter supply gap and understates the value of alpine PV. Net import [MWh] =
    (flows into CH) - (flows out of CH) over summer snapshots, via `Line-s`, <= limit_twh.
    """
    b = n.buses
    ch_ac = b.index[(b.carrier == "AC") & (b.country == country)] # all import buses (AC) in Switzerland
    assert len(ch_ac) > 0, "no buses found"
    ch_set = set(ch_ac)
    L = n.lines
    border = L.index[L.bus0.isin(ch_set) ^ L.bus1.isin(ch_set)]  # all lines with exactly one end in CH
    assert len(border) > 0, "no border lines found"
    into = [l for l in border if L.at[l, "bus1"] in ch_set]   # flow bus0->bus1 = IMPORT
    outof = [l for l in border if L.at[l, "bus0"] in ch_set]  # flow bus0->bus1 = EXPORT

    sns = snapshots[snapshots.month.isin(months)]
    w = n.snapshot_weightings.generators.loc[sns]             # hours per snapshot
    s = n.model["Line-s"]
    lhs = (s.loc[sns, into] * w).sum() - (s.loc[sns, outof] * w).sum()   # net import [MWh]
    n.model.add_constraints(lhs <= limit_twh * 1e6, name="ch_summer_net_import_limit")
    logger.info("CH summer net-import limit active: <= %.1f TWh, months %s, %d border lines "
                "(%d into / %d out of CH).", limit_twh, list(months), len(border),
                len(into), len(outof))


def add_green_import_limit(n, snapshots, limit_twh):
    """Limit green energy imports (carriers containing 'import', mainly 'import H2').

    Mirrors solve_network.add_import_limit_constraint but parameter-driven, so it works
    in a stand-alone re-optimize where n.config may not be set. Caps the annual energy
    of all import generators + import links to limit_twh [TWh/yr]. Fossil carriers
    ('oil'/'gas'/'methanol' without 'import') are NOT matched; the 'import gas/oil/methanol'
    links currently carry 0 TWh.
    """
    nyears = n.snapshot_weightings.generators.sum() / 8760
    gi = n.generators.index[n.generators.carrier.str.contains("import", na=False)]
    li = n.links.index[n.links.carrier.str.contains("import", na=False)]
    if len(gi) == 0 and len(li) == 0:
        logger.warning("No import generators/links found - green import limit skipped.")
        return
    w = n.snapshot_weightings.generators.loc[snapshots]
    lhs = 0
    if len(gi):
        lhs = lhs + (n.model["Generator-p"].loc[snapshots, gi] * w).sum()
    if len(li):
        eff = n.links.loc[li, "efficiency"]
        lhs = lhs + (n.model["Link-p"].loc[snapshots, li] * eff * w).sum()
    n.model.add_constraints(lhs <= limit_twh * 1e6 * nyears, name="green_import_limit")
    logger.info("Green energy-import limit active: <= %.1f TWh/yr (%d gens, %d links).",
                limit_twh, len(gi), len(li))


def custom_extra_functionality(n, snapshots, snakemake):
    """Read import limits from the run name and apply them.

    'imp<N>'  -> CH winter net electricity import <= N TWh   (e.g. '1718_imp5')
    'simp<N>' -> CH summer net electricity import <= N TWh   (e.g. '1718_simp0')
    '_g<N>'   -> green energy imports <= N TWh/yr             (e.g. '1718_imp5_g5')
    Use 'p' for a decimal point: 'imp4p5' -> 4.5. Without a matching token the run is
    left unchanged (existing unconstrained runs stay untouched).
    """
    run = "{} {}".format(snakemake.config.get("run", {}).get("name", ""),
                         snakemake.config.get("run", {}).get("prefix", ""))
    me = re.search(r"(?<![a-zA-Z])imp(\d+(?:p\d+)?)", run)   # winter; lookbehind: NOT 'simp'
    ms = re.search(r"simp(\d+(?:p\d+)?)", run)               # summer
    mg = re.search(r"_g(\d+(?:p\d+)?)", run)
    if me:
        add_ch_winter_import_limit(n, snapshots, float(me.group(1).replace("p", ".")))
    if ms:
        add_ch_summer_import_limit(n, snapshots, float(ms.group(1).replace("p", ".")))
    if mg:
        add_green_import_limit(n, snapshots, float(mg.group(1).replace("p", ".")))
